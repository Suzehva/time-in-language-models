# https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html

import os # aditi addition
import datetime # suze addition
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

# for corrupted run
class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)
        self.noise = torch.from_numpy(
            prng(1, DIM_CORRUPTED_TOKENS, embed_dim)).to(device)
        self.noise_level = 0.13462981581687927

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention(embed_dim={self.embed_dim})"


class CausalTracer:
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        # Initialize the device (GPU or MPS [for apple silicon] or CPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f'Using device: {self.device}')
        if self.model_id == "allenai/OLMo-1B-hf":
            self.config, self.tokenizer, self.model = create_olmo(name=self.model_id) # create_gpt2(name="gpt2-xl")
        else:
           os.error(f'only olmo is supported at this time') 
        self.model.to(self.device)


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

    def factual_recall(self, prompt:str):
        print("FACTUAL RECALL:")
        inputs = [
            self.tokenizer(prompt, return_tensors="pt").to(self.device),
        ]
        res = self.model.model(**inputs[0])  # use self.model.model to get the BASE output instead of the CAUSAL output
        print(prompt)
        distrib = embed_to_distrib(self.model, res.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10) # prints top 10 results from distribution

    def corrupted_run(self, prompt: str, corrupted_tokens):
        print("CORRUPTED RUN: ")
        base = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
        intervenable = IntervenableModel(config, self.model_id)
        _, counterfactual_outputs = intervenable(
            base, unit_locations={"base": (corrupted_tokens)}  # defines which positions get corrupted
        )
        # TODO: counterfactual_outputs.hidden_states[-1] is sketchy, fix it!
        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
        distrib = embed_to_distrib(self.model_id, counterfactual_outputs.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10)

    def restore_run(self, timestamp: str, ):

        # should finish within 1 min with a standard 12G GPU
        token = self.tokenizer.encode(SOLUTION)[0]  # 16335
        print(token)

        for stream in ["block_output", "mlp_activation", "attention_output"]:
            data = []
            for layer_i in tqdm(range(olmo.config.num_hidden_layers)):  # aditi modif num_hidden_layers
                for pos_i in range(PROMPT_LEN):
                    config = restore_corrupted_with_interval_config(
                        layer_i, stream, 
                        window=1 if stream == "block_output" else 10, 
                        num_layers=olmo.config.num_hidden_layers
                    )
                    n_restores = len(config.representations) - 1
                    intervenable = IntervenableModel(config, olmo)
                    _, counterfactual_outputs = intervenable(
                        base,
                        [None] + [base]*n_restores,
                        {
                            "sources->base": (
                                [None] + [[[pos_i]]]*n_restores,
                                CORRUPTED_TOKENS + [[[pos_i]]]*n_restores,
                            )
                        },
                    )

                    counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1]
                    distrib = embed_to_distrib(
                        olmo, counterfactual_outputs.last_hidden_state, logits=False
                    )
                    prob = distrib[0][-1][token].detach().cpu().item()
                    data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
            df = pd.DataFrame(data) 

            os.makedirs(folder_path, exist_ok=True)
            df.to_csv(f"./"+folder_path+"/"+PROMPT+stream+timestamp+".csv")
            self.plot(self, timestamp, stream)

    def plot(self, timestamp: str, stream: str):
        df = pd.read_csv(f"./"+folder_path+"/"+PROMPT+stream+timestamp+".csv")
        df["layer"] = df["layer"].astype(int)
        df["pos"] = df["pos"].astype(int)
        df["p("+SOLUTION+")"] = df["prob"].astype(float)

        custom_labels = CUSTOM_LABELS
        breaks = BREAKS

        plot = (
            ggplot(df, aes(x="layer", y="pos"))    

            + geom_tile(aes(fill="p("+SOLUTION+")"))
            + scale_fill_cmap(colors[stream]) + xlab(titles[stream])
            + scale_y_reverse(
                limits = (-0.5, 6.5), 
                breaks=breaks, labels=custom_labels) 
            + theme(figure_size=(5, 4)) + ylab("") 
            + theme(axis_text_y  = element_text(angle = 90, hjust = 1))
        )

        ggsave(
            plot, filename=f"./"+folder_path+"/"+PROMPT+stream+timestamp+".pdf", dpi=200 # suze edit
        )
        print(plot)




def restore_corrupted_with_interval_config(
    layer, stream="mlp_activation", window=10, num_layers=48):
    start = max(0, layer - window // 2)
    end = min(num_layers, layer - (-window // 2))
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

        
def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tracer = CausalTracer(model_id="allenai/OLMo-1B-hf")
    tracer.factual_recall(prompt="The Space Needle is in downtown")
    tracer.corrupted_run(prompt=?, corrupted_tokens=[[[0, 1, 2, 3]]])

if __name__ == "__main__":
    main()


#-------------------------------

mpl.rcParams['figure.figsize'] = (6, 4)  # Set default figure size
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVGs

folder_path = "pyvene_data_olmo_time"

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

<<<<<<< HEAD


def corrupted_config(model_type):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                0,              # layer
                "block_input",  # intervention type
            ),
        ],
        intervention_types=NoiseIntervention,
    )
    return config



# see edits in intervenable_base.py
# counterfactual_outputs.hidden_states[-1] is sketchy and NOTE to fix it!
counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
distrib = embed_to_distrib(olmo, counterfactual_outputs.last_hidden_state, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)
=======
##########################################
print("## PART THREE: RESTORED RUN ##")
##########################################

def restore_corrupted_with_interval_config(
    layer, stream="mlp_activation", window=10, num_layers=48):
    start = max(0, layer - window // 2)
    end = min(num_layers, layer - (-window // 2))
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

# should finish within 1 min with a standard 12G GPU
token = tokenizer.encode(SOLUTION)[0]  # 16335
print(token)

for stream in ["block_output", "mlp_activation", "attention_output"]:
    data = []
    for layer_i in tqdm(range(olmo.config.num_hidden_layers)):  # aditi modif num_hidden_layers
        for pos_i in range(PROMPT_LEN):
            config = restore_corrupted_with_interval_config(
                layer_i, stream, 
                window=1 if stream == "block_output" else 10, 
                num_layers=olmo.config.num_hidden_layers
            )
            n_restores = len(config.representations) - 1
            intervenable = IntervenableModel(config, olmo)
            _, counterfactual_outputs = intervenable(
                base,
                [None] + [base]*n_restores,
                {
                    "sources->base": (
                        [None] + [[[pos_i]]]*n_restores,
                        CORRUPTED_TOKENS + [[[pos_i]]]*n_restores,
                    )
                },
            )

            counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1]
            distrib = embed_to_distrib(
                olmo, counterfactual_outputs.last_hidden_state, logits=False
            )
            prob = distrib[0][-1][token].detach().cpu().item()
            data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
    df = pd.DataFrame(data) 

    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # suze addition
    df.to_csv(f"./"+folder_path+"/"+PROMPT+stream+timestamp+".csv")


    ###############################################
    print("## PLOTTING :) ##")
    ###############################################

    df = pd.read_csv(f"./"+folder_path+"/"+PROMPT+stream+timestamp+".csv")
    df["layer"] = df["layer"].astype(int)
    df["pos"] = df["pos"].astype(int)
    df["p("+SOLUTION+")"] = df["prob"].astype(float)

    custom_labels = CUSTOM_LABELS
    breaks = BREAKS

    plot = (
        ggplot(df, aes(x="layer", y="pos"))    

        + geom_tile(aes(fill="p("+SOLUTION+")"))
        + scale_fill_cmap(colors[stream]) + xlab(titles[stream])
        + scale_y_reverse(
            limits = (-0.5, 6.5), 
            breaks=breaks, labels=custom_labels) 
        + theme(figure_size=(5, 4)) + ylab("") 
        + theme(axis_text_y  = element_text(angle = 90, hjust = 1))
    )

    ggsave(
        plot, filename=f"./"+folder_path+"/"+PROMPT+stream+timestamp+".pdf", dpi=200 # suze edit
    )
    print(plot)
>>>>>>> 3d8b6a9888b7042196133add56f1bd54b49d28a0

