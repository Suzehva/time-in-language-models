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

from plotnine import (
    ggplot,
    geom_tile,
    aes,
    facet_wrap,
    theme,
    element_text,
    geom_bar,
    geom_hline,
    scale_y_log10,
    xlab, ylab, ylim,
    scale_y_discrete, scale_y_continuous, ggsave
)
from plotnine.scales import scale_y_reverse, scale_fill_cmap
from tqdm import tqdm


from dataclasses import dataclass

@dataclass
class Prompt:
    prompt: str
    prompt_len: int
    dim_corrupted_tokens: int
    corrupted_tokens: list[list]
    soln: str
    list_of_soln: list[list[str]]
    descriptive_label: str
    custom_labels: list[str]
    breaks: list[int]


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

    def add_prompt(self, prompt_tuple: tuple):
        # add a prompt we want to test
        # format of a prompt tuple:
        # (prompt, prompt_len, dim_corrupted_tokens, corrupted_tokens, soln, custom_labels, breaks)
        # for example:
        # ("In 2050 there", 3, 2, [[[0, 1]]], " will", ["In*", "2050*", "there"], [0, 1, 2])

        print("adding prompts\n")
        prompt_obj = Prompt(*prompt_tuple)  # Unpack tuple and pass as args to Prompt
        self.prompts.append(prompt_obj)

    def get_prompts(self):
        return self.prompts

    def num_tokens_in_vocab(self, prompt: list[str]):
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
            base, unit_locations={"base": (prompt.corrupted_tokens)}  # defines which positions get corrupted
        )
        # TODO: counterfactual_outputs.hidden_states[-1] is sketchy, fix it!
        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
        distrib = embed_to_distrib(self.model, counterfactual_outputs.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10)
        return base

    def restore_run(self, prompt: Prompt, timestamp: str):
        # @param base : use the base from the related corrupted_run

        print("RESTORED RUN:")
        print(prompt.prompt)

        base = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device)
        for solns in prompt.list_of_soln:
            tokens = [self.tokenizer.encode(soln)[0] for soln in solns]
            print("\n\n\nsolns pre-encoded" + str(solns))
            print("\n\n\ntoken encoded" + str(tokens))

            for stream in ["block_output", "mlp_activation", "attention_output"]:
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
                                    prompt.corrupted_tokens + [[[pos_i]]]*n_restores,
                                )
                            },
                        )

                        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1]
                        distrib = embed_to_distrib(
                            self.model, counterfactual_outputs.last_hidden_state, logits=False
                        )

                        # can sum over multiple words' tokens instead of just one
                        prob = sum(distrib[0][-1][token].detach().cpu().item() for token in tokens)
                        data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
                df = pd.DataFrame(data) 

                os.makedirs(self.folder_path, exist_ok=True)
                soln_txt = ''.join([s.replace(' ', '_') for s in solns])
                filepath = "./"+self.folder_path+"/"+prompt.descriptive_label+soln_txt+"_"+stream+"_"+timestamp

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
                limits = (-0.5, 6.5), 
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
                0,       # layer
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


# TODO: How do i pass it dim_corrupted_tokens and DEVICE ?
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

    # we will need to manually add the prompts like this... for now
    # prompt: str
    # prompt_len: int
    # dim_corrupted_tokens: int
    # corrupted_tokens: list[list]
    # soln: str
    # list_of_soln: list[str]
    # descriptive_label: str
    # custom_labels: list[str]
    # breaks: list[int]

    tracer.add_prompt(("In 1980 there", 3, 2, [[[0, 1]]], 
                       " was", [[" was", " were", " had", " have", "wasn"], [" will", " would"], [" is", " are"]], 
                       "1980_list", ["In*", "1980*", "there"], [0, 1, 2]))

    # loop over every prompt to run pyvene
    for p in tracer.get_prompts():
        print("prompt is: " + p.prompt)
        tracer.factual_recall(prompt=p)
        tracer.corrupted_run(prompt=p)
        tracer.restore_run(prompt=p, timestamp=timestamp)


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
PROMPT = "In 2020 there"
PROMPT_LEN = 3
DIM_CORRUPTED_TOKENS = 2
CORRUPTED_TOKENS = [[[0, 1]]]
SOLUTION = " is"
CUSTOM_LABELS = ["In*", "2020*", "there"]
BREAKS = [0, 1, 2]

# 2050 PROMPT CONSTS
# PROMPT = "In 2050 there"
# PROMPT_LEN = 3
# DIM_CORRUPTED_TOKENS = 2
# CORRUPTED_TOKENS = [[[0, 1]]]
# SOLUTION = " will"
# CUSTOM_LABELS = ["In*", "2050*", "there"]
# BREAKS = [0, 1, 2]
