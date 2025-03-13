# aditi todo :)
# TODO continue adding support for llama!! 

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
    create_olmo,
    create_gpt2

)

from PIL import Image

# TODO remove plotnine im not using
from plotnine import (
    ggplot,
    geom_tile,
    aes,
    scale_fill_gradient,
    theme,
    element_text,
    xlab, ylab, 
    ggsave,
    facet_wrap,
    labs
)

import patchworklib as pw  # Import patchworklib for arranging plots

# from plotnine.ggplot import plot_grid
from plotnine.scales import scale_y_reverse, scale_fill_cmap
from plotnine.scales import scale_x_continuous, scale_y_continuous
from tqdm import tqdm

# use this to control whether you compute and plot stuff besides the block output
plot_only_block_outputs = False


from dataclasses import dataclass

@dataclass
class Prompt:
    prompt: str
    len_prompt_tokens: int
    len_corrupted_tokens: int
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
        print(f'Using device: {self.device}')
        if self.model_id == "allenai/OLMo-1B-hf":
            self.config, self.tokenizer, self.model = create_olmo(name=self.model_id) 
        elif self.model_id == "gpt2":
            self.config, self.tokenizer, self.model = create_gpt2()
        elif self.model_id == "meta-llama/Llama-3.2-1B":
            # bit hacky but oh well
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            # bit hacky
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
            self.config = AutoConfig.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map=self.device, config=self.config)  
        else:
           raise Exception(f'only olmo, gpt2, llama is supported at this time') 

        self.model.to(self.device)

        self.prompts = []
        self.folder_path = folder_path

    def add_prompt(self, prompt: str, dim_corrupted_words: int, list_of_soln: list[str], descriptive_label: str, year: int):
        # TODO might need to take care of choosing the corrupted tokens manually...
        # if (prompt[-1] != " "):
        #     prompt+=" " # in case the user did not add a space at the end of the prompt. not adding a space messes up the prompt list calculation below.
        
        _, prompt_tokens = self.string_to_token_ids_and_tokens(prompt)
        len_prompt_tokens = len(prompt_tokens)

        to_corrupt = " ".join(prompt.split(" ")[:dim_corrupted_words])  # the part of the sentence we want to corrupt (the first dim_corrupted_words words)
        _, to_corr_tokens = self.string_to_token_ids_and_tokens(to_corrupt)
        len_corrupted_tokens = len(to_corr_tokens) 

        corrupted_tokens_indices = [[[i for i in range(len_corrupted_tokens)]]]
       
        custom_labels = self.add_asterisks(prompt_tokens, len_corrupted_tokens) # add asteriks to the first len_corrupted_tokens in the prompt tokens
        breaks = list(range(len_prompt_tokens))

        # add a prompt we want to test in the format of a prompt tuple:
        # (prompt, len_prompt_tokens, len_corrupted_tokens, corrupted_tokens, list_of_soln, custom_labels, breaks)
        # for example:
        # ("In 2050 there", 3, 2, [[[0, 1]]], [ [" was", " were"], [" will"]], ["In*", "2050*", "there"], [0, 1, 2], 1980)
        prompt_obj = Prompt(prompt, len_prompt_tokens, len_corrupted_tokens, corrupted_tokens_indices, list_of_soln, descriptive_label, custom_labels, breaks, year) 
                
        print("\nPROMPT OBJ\n" + str(prompt_obj))
        self.prompts.append(prompt_obj)


    def get_prompts(self):
        return self.prompts

    def add_asterisks(self, prompt, n):
        return [word + "*" if i < n else word for i, word in enumerate(prompt)]


    def string_to_token_ids_and_tokens(self, s: str):
        # token ids's are the numbers, tokens are the text pieces
        token_ids = self.tokenizer(s, return_tensors="pt").to(self.device)
        """
        note: this^ returns more than just token_ids, also info about attention masks
        e.g:
        {
            'input_ids': tensor([[1437, 318, 12696, 2952, 30]]), 
            'attention_mask': tensor([[1, 1, 1, 1, 1]])
        }
        """
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids['input_ids'][0])
        print(f"{len(tokens)} tokens in '{s}': {tokens}")
        return token_ids, tokens

    
    def list_from_prompt(self, prompt: str):
        return prompt.split(" ")

    def factual_recall(self, prompt: str):
        print("FACTUAL RECALL:")
        print(prompt)

        inputs, _ = self.string_to_token_ids_and_tokens(prompt)
        if self.model_id == "allenai/OLMo-1B-hf":
            res = self.model.model(**inputs) # removed [0] from **inputs[0] because input is now not a list
        elif self.model_id == "gpt2":
            res = self.model(**inputs[0])
        elif self.model_id == "meta-llama/Llama-3.2-1B":
            res = self.model(**inputs, output_hidden_states=True) 
            res.last_hidden_state = res.hidden_states[-1] #this seems to work
        else:
            raise Exception(f'only olmo, gpt2, llama is supported at this time') 

        distrib = embed_to_distrib(self.model, res.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10) # prints top 10 results from distribution


    def corrupted_run(self, prompt: Prompt):

        print("CORRUPTED RUN: ")
        print(prompt.prompt)

        base = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device)
        
        NoiseIntervention.dim_corrupted_tokens = prompt.len_corrupted_tokens
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
        print("\nDEBUG base: " + str(base))

        filepaths=[]
        for i in range(len(prompt.list_of_soln)): # pylint: disable=consider-using-enumerate
            solns = prompt.list_of_soln[i]
            print("\ntense:  " + str(solns))
            if run_type=="relative":
                if relative_prompt_focus not in solns:
                    print("skipped!")
                    continue  # check if its relative, and continue if its a tense we dont care about
             
            tokens = [self.tokenizer.encode(soln)[0] for soln in solns]  # this provides same results as suze's tokenizer method

            for stream in ["block_output", "mlp_activation", "attention_output"]:
                # don't plot anything besides block output if we dont want it
                if not plot_only_block_outputs and stream == "block_output":    # aditi edit to only plot the other 2. change back to !=
                    continue
                
                data = []
                for layer_i in tqdm(range(self.model.config.num_hidden_layers)):
                    for pos_i in range(prompt.len_prompt_tokens):
                        config = restore_corrupted_with_interval_config(
                            layer=layer_i,
                            device=self.device, 
                            dim_corrupted_tokens=prompt.len_corrupted_tokens,
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
                soln_txt = ''.join([s.replace(' ', ',') for s in solns])[1:]
                filepath = "./"+self.folder_path+"/"+run_type+"_"+prompt.descriptive_label+"_"+soln_txt+"_"+stream+"_"+timestamp
                df.to_csv(filepath+".csv") # write csv

                self.plot(prompt, soln_txt, stream, filepath)
                filepaths.append(filepath+".png")
        
        outputfilepath="./"+self.folder_path+"/"+"combined_"+(prompt.prompt.replace(' ', '_'))+"_"+timestamp+".png"
        self.merge_images_horizontally(filepaths, outputfilepath)

    
    def plot(self, prompt: Prompt, soln_txt: str, stream: str, filepath:str):
        df = pd.read_csv(filepath+".csv")  # read csv
        df["layer"] = df["layer"].astype(int)
        df["pos"] = df["pos"].astype(int)
        df["p("+soln_txt+")"] = df["prob"].astype(float)

        custom_labels = prompt.custom_labels
        breaks = prompt.breaks

        plot = (
            ggplot(df)
            + geom_tile(aes(x="pos", y="layer", fill="p("+soln_txt+")"))
            + scale_fill_gradient(low="white", high="purple", limits=(0, 1))  # Fixes 0 to light, 1 to dark
            + theme(
                figure_size=(4, 5),
                axis_text_x=element_text(rotation=90)
            )
            + scale_x_continuous(
                breaks=breaks,
                labels=custom_labels
            )
            + scale_y_continuous(
                breaks=list(range(self.model.config.num_hidden_layers)),  # num hidden layers in model
                labels=[str(i) for i in range(self.model.config.num_hidden_layers)]  # Convert to strings for labels
            )
            + labs(
                title=f"{prompt.prompt}",
                y=f"Restored layer in {self.model_id}",
                fill="p("+soln_txt+")",
            )
        )

        ggsave(
            plot, filename=filepath+".png", dpi=200 # write pdf graph # TODO: how to save as png??
        )
        # print(plot)


    def merge_images_horizontally(self, image_paths, output_path="merged.png"):
        images = [Image.open(img) for img in image_paths]  # Open images
        
        # Get total width and max height
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create blank image
        merged_image = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))

        # Paste images side by side
        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
            img.close()

        # delete all the extra images and csv files
        for img_path in image_paths:
            os.remove(img_path)  # Delete the individual img file
            os.remove(img_path[:-4] + ".csv")

        # Save merged image
        merged_image.save(output_path)



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


# aditi's mini-experiment to see whether the year affects the output, or if there's something else at play here...
def add_prompts_for_experimental_runs(tracer: CausalTracer):
    
    # tracer.add_prompt(prompt="In 1980 there", dim_corrupted_words=2, 
    #                     list_of_soln=TENSES, descriptive_label="ctrl_there", year=1980)                 # this is our usual "there" test
    # tracer.add_prompt(prompt="Before 1980 there", dim_corrupted_words=2, 
    #                     list_of_soln=TENSES, descriptive_label="ctrl_before_there", year=1980)          # this tests if its relative to 1980 -- what happens now?
    # tracer.add_prompt(prompt="After 1980 there", dim_corrupted_words=2, 
    #                     list_of_soln=TENSES, descriptive_label="ctrl_after_there", year=1980)           # parallels before
    
    # tracer.add_prompt(prompt="On a beautiful day in 1980 there", dim_corrupted_words=6, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_beautiful", year=1980)       # slighty longer prompt for 1980
    # tracer.add_prompt(prompt="On a beautiful day in summer there", dim_corrupted_words=6, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_summer", year=1980)          # replace 1980 with summmer -- time of year
    # tracer.add_prompt(prompt="On a beautiful day in Elmsville there", dim_corrupted_words=6, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_elmsville", year=1980)       # replace 1980 with Elmsville -- fictional place
    
    tracer.add_prompt(prompt="In 1980 on a beautiful day there", dim_corrupted_words=2, 
                            list_of_soln=TENSES, descriptive_label="ctrl_bkw_beautiful", year=1980)       # slighty longer prompt after 1980
    # tracer.add_prompt(prompt="In summer on a beautiful day there", dim_corrupted_words=2, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_bkw_summer", year=1980)          # replace 1980 with summmer -- time of year
    tracer.add_prompt(prompt="In Elmsville on a beautiful day there", dim_corrupted_words=2, 
                            list_of_soln=TENSES, descriptive_label="ctrl_bkw_elmsville", year=1980)       # replace 1980 with Elmsville -- fictional place
    
    # tracer.add_prompt(prompt="In 2020 on a beautiful day there", dim_corrupted_words=2, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_bkw_beautiful_2020", year=2020)   
    # tracer.add_prompt(prompt="2020 on a beautiful day there", dim_corrupted_words=2, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_bkw_beautiful_2020_NO_IN", year=2020)    
    # tracer.add_prompt(prompt="In 2030 on a beautiful day there", dim_corrupted_words=2, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_bkw_beautiful_2030", year=2030)    
 
    # tracer.add_prompt(prompt="In 2050 on a beautiful day there", dim_corrupted_words=2, 
    #                         list_of_soln=TENSES, descriptive_label="ctrl_bkw_beautiful_2050", year=2050)   
    # tracer.add_prompt(prompt="In 2000 on a beautiful day there", dim_corrupted_words=2, 
    #                     list_of_soln=TENSES, descriptive_label="ctrl_bkw_beautiful_2000", year=2000)   
   


### defs for prompts

# create all the prompts we wanna use
YEARS = [1980, 2000, 2020, 2050] 
# PROMPTS: prompt, dim words to corrupt, and a descriptive name for generated files
# NOTE!! no commas allowed in prompts. it breaks sometimes.
PROMPTS = [("In [[YEAR]] on a beautiful day there", 2, "beautiful_bkw"), 
           ("In [[YEAR]] there", 2, "there"), ("As of [[YEAR]] it", 3, "asof"),
            ("On a beautiful day in [[YEAR]] there", 6, "beautiful"),
           ("In [[YEAR]] on a gloomy day there", 2, "gloomy"), ("In [[YEAR]] on a rainy day here", 2, "rainy")]
            # ("In [[YEAR]] they ", 2, "they") --- BAD according to what it gets from factual recall and corrupted run steps. it tries to return \n ???
TENSES = [[" was", " were"], [" is", " are"], [" will"]]
    # should i also include "was" and "were" without the space??


def add_prompts_over_years(tracer: CausalTracer, years=YEARS, prompts=PROMPTS, tenses=TENSES):
    # used to generate lots of graphs
    for y in years:
        for (prompt_template, num_words_to_corrupt, descr) in prompts:
            prompt_real = prompt_template[:prompt_template.find("[[")] + str(y) + prompt_template[prompt_template.find("]]")+2:]
            descr_label = str(y) + "_" + descr 
            # dim_corrupted_words is observed to usually be len_words_prompt-1 because the year is usually the 2nd to last word
            tracer.add_prompt(prompt=prompt_real, dim_corrupted_words=num_words_to_corrupt, 
                            list_of_soln=tenses, descriptive_label=descr_label, year=y)




def relative_2020_beautiful(tracer: CausalTracer):
    prompt="In 2020 on a beautiful day there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=2, 
                            list_of_soln=TENSES, descriptive_label="2020_beautiful_day", year=2020)
    # NOTES: make sure to set it to relative in main
    # also set the correct relative_prompt_focus term (is vs will) in the restore_run call


########################################################################################################################


def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tracer = CausalTracer(model_id="allenai/OLMo-1B-hf")  # can also pass an arg specifying the folder 

    
    # DO THIS: set the appropriate test
    add_prompts_for_experimental_runs(tracer)
    # add_prompts_over_years(tracer)  # adds a lot of prompts -- loops over years and prompt structures


    # for the single relative generated graph
    # relative_2020_beautiful(tracer)

    # DO THIS: set relative to true if you want the relative plots
    relative = False


    # loop over every prompt to run pyvene
    for p in tracer.get_prompts():

        # part 1        
        # tracer.factual_recall(prompt=p)  

        # part 2
        # tracer.corrupted_run(prompt=p)   

        # part 3: regular run over all tenses
        if (not relative):
            tracer.restore_run(prompt=p, timestamp=timestamp)

        if (relative):
            # relative runs:
            # control which year we want to focus on for restore run. 
            relative_prompt_focus = " was" # past
            if p.year > 2005:
                relative_prompt_focus=" is" # present
            if p.year > 2025:
                relative_prompt_focus=" will" # future

            tracer.restore_run(prompt=p, timestamp=timestamp, run_type="relative", relative_prompt_focus=relative_prompt_focus)  # with subtraction




if __name__ == "__main__":
    main()

