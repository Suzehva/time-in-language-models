# https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html

import os
import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib as mpl
from pyvene import embed_to_distrib, top_vals, format_token
from pyvene import (
    IntervenableModel,
    VanillaIntervention, Intervention,
    RepresentationConfig,
    IntervenableConfig,
    ConstantSourceIntervention,
    LocalistRepresentationIntervention,
    create_olmo
)

# TODO remove plotnine im not using
from plotnine import (
    ggplot,
    geom_tile,
    aes,
    theme,
    element_text,
    xlab, ylab, 
    ggsave
)
from plotnine.scales import scale_y_reverse, scale_fill_cmap
from tqdm import tqdm

# use this to control whether you compute and plot stuff besides the block output
plot_only_block_outputs = True


from dataclasses import dataclass

@dataclass
class Prompt:
    prompt: str
    prompt_len: int
    dim_corrupted_tokens: int
    corrupted_tokens_indices: list[list]
    list_of_soln: list[list[str]]
    descriptive_label: str
    custom_labels: list[str]
    breaks: list[int]
    year: int


# Plotting params #################################################

mpl.rcParams['figure.figsize'] = (6, 4)  # Set default figure size
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVGs

titles={
    "block_output": "single restored layer in OLMo 1B",
    "mlp_activation": "center of interval of 10 patched mlp layer",
    "attention_output": "center of interval of 10 patched attn layer"
}
colors={
    "block_output": "Purples",
    "mlp_activation": "Greens",
    "attention_output": "Reds"
} 



class CausalTracer:
    def __init__(self, model_id, folder_path="pyvene_data_ct_olmo", device=None):
        self.model_id = model_id
        # Initialize the device (GPU or MPS [for apple silicon] or CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" 
        DEVICE = self.device # TODO: fix this so it's not necessary 
        print(f'Using device: {self.device}')
        if self.model_id == "allenai/OLMo-1B-hf":
            self.config, self.tokenizer, self.model = create_olmo(name=self.model_id) # create_gpt2(name="gpt2-xl")
        else:
           os.error(f'only olmo is supported at this time') 
        self.model.to(self.device)

        self.prompts = []
        self.folder_path = folder_path

    def add_prompt(self, prompt: str, dim_corrupted_words: int, list_of_soln: list[str], descriptive_label: str, year: int):
        # TODO might need to take care of choosing the corrupted tokens manually...
        prompt_list = self.num_tokens_in_prompt(prompt.split(" "))
        prompt_len = len(self.num_tokens_in_prompt(prompt.split(" ")))

        token_list = self.num_tokens_in_prompt((prompt.split(" ")[0:dim_corrupted_words]))  # takes the first dct words we want to corrupt
        dim_corrupted_tokens = len(token_list) 

        corrupted_tokens_indices = [[[i for i in range(dim_corrupted_tokens)]]]
        custom_labels = self.add_asterisks(prompt_list, dim_corrupted_tokens)
        breaks = list(range(prompt_len))

        # add a prompt we want to test in the format of a prompt tuple:
        # (prompt, prompt_len, dim_corrupted_tokens, corrupted_tokens, list_of_soln, custom_labels, breaks)
        # for example:
        # ("In 2050 there", 3, 2, [[[0, 1]]], [ [" was", " were"], [" will"]], ["In*", "2050*", "there"], [0, 1, 2], 1980)
        prompt_obj = Prompt(prompt, prompt_len, dim_corrupted_tokens, corrupted_tokens_indices, list_of_soln, descriptive_label, custom_labels, breaks, year) 
                
        print("\nPROMPT OBJ\n" + str(prompt_obj))
        self.prompts.append(prompt_obj)
        
    def get_prompts(self):
        return self.prompts

    def add_asterisks(self, prompt, n):
        return [word + "*" if i < n else word for i, word in enumerate(prompt)]

    def num_tokens_in_prompt(self, prompt: list[str]):
        """
        Returns a list of strings based on how the model tokenizes the prompt
        Note: there could be multiple token id's for a word
        """
        token_list = []
        
        for word in prompt:
            # get id's that correspond to a word and its subwords
            token_ids = self.tokenizer(word, add_special_tokens=False)["input_ids"] 
            # get tokens that each id corresponds to. 
            token_pieces = self.tokenizer.convert_ids_to_tokens(token_ids) 

            token_list += token_pieces

        # TODO: look into this. why is the attention sometimes being returned as [1,1] and sometimes [] (usually each entry is [1])
        # print("TOKENIZER: " +str(self.tokenizer(prompt)))#, return_tensors="pt")))
        # print("TOKEN PIECES: " +str(token_list))

        return token_list
    
    def list_from_prompt(self, prompt: str):
        return prompt.split(" ")

    def factual_recall(self, prompt: Prompt):

        print("FACTUAL RECALL:")
        print(prompt.prompt)

        inputs = [
            self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device),
        ]
        res = self.model.model(**inputs[0])  # use self.model.model to get the BASE output instead of the CAUSAL output
        distrib = embed_to_distrib(self.model, res.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10) # prints top 10 results from distribution

    def corrupted_run(self, prompt: Prompt):

        print("CORRUPTED RUN: ")
        print(prompt.prompt)

        base = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device)
        
        NoiseIntervention.dim_corrupted_tokens = prompt.dim_corrupted_tokens
        NoiseIntervention.DEVICE = self.device
        config = IntervenableConfig(
            model_type=type(self.model_id),
            representations=[
                RepresentationConfig(
                    0,              # layer
                    "block_input",  # intervention type
                ),
            ],
            intervention_types=NoiseIntervention,
        )

        intervenable = IntervenableModel(config, self.model)
        _, counterfactual_outputs = intervenable(
            base, unit_locations={"base": (prompt.corrupted_tokens_indices)}  # defines which positions get corrupted
        )
        # TODO: counterfactual_outputs.hidden_states[-1] is sketchy, fix it!
        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
        distrib = embed_to_distrib(self.model, counterfactual_outputs.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10)
        return base

    def restore_run(self, prompt: Prompt, timestamp: str, run_type="default", relative_prompt_focus=" was"):
        # @param base : use the base from the related corrupted_run
        # @param run_type : if "default", just do regular plots. if "relative", compute difference betweenn all other words in all possible solns

        print("RESTORED RUN:")
        print(prompt.prompt)

        base = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device)
        for i in range(len(prompt.list_of_soln)):
            solns = prompt.list_of_soln[i]
            print("\ntense:  " + str(solns))
            if run_type=="relative":
                if relative_prompt_focus not in solns:
                    print("skipped!")
                    continue  # TODO: can check if its relative, and then continue if its a tense we dont care about
             
            tokens = [self.tokenizer.encode(soln)[0] for soln in solns]

            for stream in ["block_output", "mlp_activation", "attention_output"]:
                # don't plot anything besides block output if we dont want it
                if plot_only_block_outputs and stream != "block_output":  
                    continue
                
                data = []
                for layer_i in tqdm(range(self.model.config.num_hidden_layers)):
                    for pos_i in range(prompt.prompt_len):
                        config = restore_corrupted_with_interval_config(
                            layer=layer_i,
                            device=self.device, 
                            dim_corrupted_tokens=prompt.dim_corrupted_tokens,
                            stream=stream, 
                            window=1 if stream == "block_output" else 10, 
                            num_layers=self.model.config.num_hidden_layers,      
                        )
                        n_restores = len(config.representations) - 1
                        intervenable = IntervenableModel(config, self.model)
                        _, counterfactual_outputs = intervenable(
                            base,
                            [None] + [base]*n_restores,
                            {
                                "sources->base": (
                                    [None] + [[[pos_i]]]*n_restores,
                                    prompt.corrupted_tokens_indices + [[[pos_i]]]*n_restores,
                                )
                            },
                        )

                        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1]
                        distrib = embed_to_distrib(
                            self.model, counterfactual_outputs.last_hidden_state, logits=False
                        )

                        # can sum over multiple words' tokens instead of just one
                        prob = sum(distrib[0][-1][token].detach().cpu().item() for token in tokens)
                        if (run_type == "relative"): # subtract away the other tense words
                            subt_words = [item for j, sublist in enumerate(prompt.list_of_soln) if j != i for item in sublist] # all items we're not interested in
                            subt_tokens = [self.tokenizer.encode(w)[0] for w in subt_words] # tokenize
                            prob_subtr = sum(distrib[0][-1][word].detach().cpu().item() for word in subt_tokens) # sum all their probabilities                 
                            prob = prob - prob_subtr

                        data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
                df = pd.DataFrame(data) 

                os.makedirs(self.folder_path, exist_ok=True)
                soln_txt = ''.join([s.replace(' ', '_') for s in solns])
                filepath = "./"+self.folder_path+"/"+run_type+"_"+prompt.descriptive_label+"_"+soln_txt+"_"+stream+"_"+timestamp

                df.to_csv(filepath+".csv") # write csv
                self.plot(prompt, soln_txt, timestamp, stream, filepath)

    def plot(self, prompt: Prompt, soln_txt: str, timestamp: str, stream: str, filepath:str):
        df = pd.read_csv(filepath+".csv")  # read csv
        df["layer"] = df["layer"].astype(int)
        df["pos"] = df["pos"].astype(int)
        df["p("+soln_txt+")"] = df["prob"].astype(float)

        custom_labels = prompt.custom_labels
        breaks = prompt.breaks

        plot = (
            ggplot(df, aes(x="layer", y="pos"))    

            + geom_tile(aes(fill="p("+soln_txt+")"))
            + scale_fill_cmap(colors[stream]) + xlab(titles[stream])
            + scale_y_reverse(
                limits = (-0.5, len(breaks)+.5),   # aditi edit! previously, hardcoded (-.5, 6.5)
                breaks=breaks, labels=custom_labels) 
            + theme(figure_size=(5, 4)) + ylab("") 
            + theme(axis_text_y  = element_text(angle = 90, hjust = 1))
        )
        ggsave(
            plot, filename=filepath+".pdf", dpi=200 # write pdf graph # TODO: how to save as png??
        )
        print(plot)

def restore_corrupted_with_interval_config(
    layer, device, dim_corrupted_tokens, stream="mlp_activation", window=10, num_layers=48):

    start = max(0, layer - window // 2)
    end = min(num_layers, layer - (-window // 2))

    NoiseIntervention.dim_corrupted_tokens = dim_corrupted_tokens
    NoiseIntervention.DEVICE = device

    config = IntervenableConfig(
        representations=[
            RepresentationConfig(
                0,              # layer
                "block_input",  # intervention type
            ),
        ] + [
            RepresentationConfig(
                i,       # layer
                stream,  # intervention type
        ) for i in range(start, end)],
        intervention_types=\
            [NoiseIntervention]+[VanillaIntervention]*(end-start),
    )
    return config

class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    dim_corrupted_tokens = None
    DEVICE = None

    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)

        if NoiseIntervention.dim_corrupted_tokens is None or NoiseIntervention.DEVICE is None:
            print("dim_corrupted_tokens and DEVICE must be set before instantiating NoiseIntervention.")

        self.noise = torch.from_numpy(
            prng(1,NoiseIntervention.dim_corrupted_tokens, embed_dim)
            ).to(NoiseIntervention.DEVICE)
        self.noise_level = 0.13462981581687927

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention(embed_dim={self.embed_dim})"

def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tracer = CausalTracer(model_id="allenai/OLMo-1B-hf")  # can also pass an arg specifying the folder 

    # TODO: avg over multiple prompt templates... longer prompt template

    # create all the prompts we wanna use
    YEARS = [1980, 2000, 2020, 2050]
    # PROMPTS: prompt, dim words to corrupt, and a descriptive name for generated files
    # NOTE!! no commas in prompts. it breaks.
    # NOTE!! make sure to put a space at the end of the prompt!!
    PROMPTS = [("On a gloomy day in [[YEAR]] there ", 6, "gloomy"), ("On a rainy day in [[YEAR]] there ", 6, "rainy"), 
                ("On a beautiful day in [[YEAR]] there ", 6, "beautiful"), 
               ("In [[YEAR]] there ", 2, "there"), ("As of [[YEAR]] it ", 3, "asof"), ("In [[YEAR]] they ", 2, "they")]
               
    TENSES = [[" was", " were"], [" will"], [" is", " are"]]
    for y in YEARS:
        for (prompt_template, num_words, descr) in PROMPTS:
            prompt_real = prompt_template[:prompt_template.find("[[")] + str(y) + prompt_template[prompt_template.find("]]")+2:]
            len_words_prompt = len(prompt_real.split(" ")) - 1 # always have an extra space @ the end of the prompt
            descr_label = str(y) + "_" + descr 
            # dim_corrupted_words is observed to usually be len_words_prompt-1 because the year is usually the 2nd to last word
            tracer.add_prompt(prompt=prompt_real, dim_corrupted_words=num_words, 
                              list_of_soln=TENSES, descriptive_label=descr_label, year=y)

    # suze's method of passing 
    # prob_to_plot=[[(" was", 1), (" were", 1), (" will", -1), (" is", -1), (" are", -1)]]
    # TODO: aditi incorporate this method 

    # # YEARS = [1980, 2000, 2020, 2050]
    # descr_label = str(y)+"_"
    # tracer.add_prompt(prompt="In 1980 there", dim_corrupted_words=2, prob_to_plot=[[(" was", 1), (" were", 1), (" will", -1), (" is", -1), (" are", -1)]], descriptive_label="1980")
    # #tracer.add_prompt(prompt="In 2020 there", dim_corrupted_words=2, prob_to_plot=[[(" was", -1), (" were", -1), (" will", -1), (" is", 1), (" are", 1)]], descriptive_label="2020")
    # #tracer.add_prompt(prompt="In 2050 there", dim_corrupted_words=2, prob_to_plot=[[(" was", -1), (" were", -1), (" will", 1), (" is", -1), (" are", -1)]], descriptive_label="2050")

    # loop over every prompt to run pyvene
    for p in tracer.get_prompts():
        print("prompt is: " + p.prompt)
        # tracer.factual_recall(prompt=p)
        # tracer.corrupted_run(prompt=p)

        # control which year we want to focus on for restore run. this is only relavant with relative runs
        relative_prompt_focus = " was" # past
        if p.year > 2005:
            relative_prompt_focus=" is" # present
        if p.year > 2030:
            relative_prompt_focus=" will" # future

        # tracer.restore_run(prompt=p, timestamp=timestamp)  # regular run for each tense

        tracer.restore_run(prompt=p, timestamp=timestamp, run_type="relative", relative_prompt_focus=relative_prompt_focus)  # with subtraction



if __name__ == "__main__":
    main()


#-------------------------------


# Seattle PROMPT CONSTS
# PROMPT = "The Space Needle is in downtown"
# PROMPT_LEN = 7 # needle splits into need + le
# DIM_CORRUPTED_TOKENS = 4
# CORRUPTED_TOKENS = [[[0, 1, 2, 3]]]
# SOLUTION = " Seattle"
# CUSTOM_LABELS = ["The*", "Space*", "Need*", "le*", "is", "in", "downtown"]
# BREAKS = [0, 1, 2, 3, 4, 5, 6]

# 1980 PROMPT CONSTS
# PROMPT = "In 1980 there"
# PROMPT_LEN = 3
# DIM_CORRUPTED_TOKENS = 2
# CORRUPTED_TOKENS = [[[0, 1]]]
# SOLUTION = " was"
# CUSTOM_LABELS = ["In*", "1980*", "there"]
# BREAKS = [0, 1, 2]


# 2020 PROMPT CONSTS
# PROMPT = "In 2020 there"
# PROMPT_LEN = 3
# DIM_CORRUPTED_TOKENS = 2
# CORRUPTED_TOKENS = [[[0, 1]]]
# SOLUTION = " is"
# CUSTOM_LABELS = ["In*", "2020*", "there"]
# BREAKS = [0, 1, 2]

# 2050 PROMPT CONSTS
# PROMPT = "In 2050 there"
# PROMPT_LEN = 3
# DIM_CORRUPTED_TOKENS = 2
# CORRUPTED_TOKENS = [[[0, 1]]]
# SOLUTION = " will"
# CUSTOM_LABELS = ["In*", "2050*", "there"]
# BREAKS = [0, 1, 2]
