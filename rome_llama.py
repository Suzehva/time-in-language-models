# https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html
import os # aditi addition
import datetime # suze addition

# ------------------------------------------------------------------------------

import torch
import pandas as pd
import numpy as np
from pyvene import embed_to_distrib, top_vals, format_token
from pyvene import (
    IntervenableModel,
    VanillaIntervention, Intervention,
    RepresentationConfig,
    IntervenableConfig,
    ConstantSourceIntervention,
    LocalistRepresentationIntervention
)
from pyvene import create_llama

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (6, 4)  # Set default figure size
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVGs

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

folder_path = "pyvene_data_llama_time"


titles={
    "block_output": "single restored layer in LLAMA 1B",
    "mlp_activation": "center of interval of 10 patched mlp layer",
    "attention_output": "center of interval of 10 patched attn layer"
}

colors={
    "block_output": "Purples",
    "mlp_activation": "Greens",
    "attention_output": "Reds"
} 

##########################################
print("## PART ONE: FACTUAL RECALL ##")
##########################################

model_name = "meta-llama/Llama-3.2-1B"  # autoregressive model 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
config, tokenizer, llama = create_llama(name=model_name)

llama.to(device)

base = "The Space Needle is in downtown"
inputs = [
    tokenizer(base, return_tensors="pt").to(device),
]
res = llama.model(**inputs[0])  # use llama.model to get the BASE output instead of the CAUSAL output
distrib = embed_to_distrib(llama, res.last_hidden_state, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)


##########################################
print("## PART TWO: CORRUPTED RUN ##")
##########################################

class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)
        self.noise = torch.from_numpy(
            prng(1, 4, embed_dim)).to(device)
        self.noise_level = 0.13462981581687927

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention(embed_dim={self.embed_dim})"

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

base = tokenizer("The Space Needle is in downtown", return_tensors="pt").to(device)
config = corrupted_config(type(llama))
intervenable = IntervenableModel(config, llama)

_, counterfactual_outputs = intervenable(
    base, unit_locations={"base": ([[[0, 1, 2, 3]]])}
)

# see edits in intervenable_base.py
# counterfactual_outputs.hidden_states[-1] is sketchy and NOTE to fix it!
counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
distrib = embed_to_distrib(llama, counterfactual_outputs.last_hidden_state, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)


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
token = tokenizer.encode(" Seattle")[0]  # 16335
print(token)

for stream in ["block_output", "mlp_activation", "attention_output"]:
    data = []
    for layer_i in tqdm(range(llama.config.num_hidden_layers)):  # aditi modif num_hidden_layers
        for pos_i in range(7):
            config = restore_corrupted_with_interval_config(
                layer_i, stream, 
                window=1 if stream == "block_output" else 10, 
                num_layers=llama.config.num_hidden_layers
            )
            n_restores = len(config.representations) - 1
            intervenable = IntervenableModel(config, llama)
            _, counterfactual_outputs = intervenable(
                base,
                [None] + [base]*n_restores,
                {
                    "sources->base": (
                        [None] + [[[pos_i]]]*n_restores,
                        [[[0, 1, 2, 3]]] + [[[pos_i]]]*n_restores,
                    )
                },
            )

            counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
            distrib = embed_to_distrib(
                llama, counterfactual_outputs.last_hidden_state, logits=False
            )
            prob = distrib[0][-1][token].detach().cpu().item()
            data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
    df = pd.DataFrame(data) 

    os.makedirs(folder_path, exist_ok=True) # aditi addition
    df.to_csv(f"./"+folder_path+"/pyvene_rome_"+stream+".csv")


###############################################
print("## LAST BUT NOT LEAST: PLOTTING :) ##")
###############################################

for stream in ["block_output", "mlp_activation", "attention_output"]:
    df = pd.read_csv(f"./"+folder_path+"/pyvene_rome_"+stream+".csv")
    df["layer"] = df["layer"].astype(int)
    df["pos"] = df["pos"].astype(int)
    df["p(Seattle)"] = df["prob"].astype(float)

    custom_labels = ["The*", "Space*", "Need*", "le*", "is", "in", "downtown"]
    breaks = [0, 1, 2, 3, 4, 5, 6]

    plot = (
        ggplot(df, aes(x="layer", y="pos"))    

        + geom_tile(aes(fill="p(Seattle)"))
        + scale_fill_cmap(colors[stream]) + xlab(titles[stream])
        + scale_y_reverse(
            limits = (-0.5, 6.5), 
            breaks=breaks, labels=custom_labels) 
        + theme(figure_size=(5, 4)) + ylab("") 
        + theme(axis_text_y  = element_text(angle = 90, hjust = 1))
    )
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # suze addition

    ggsave(
        plot, filename=f"./"+folder_path+"/pyvene_rome_"+stream+timestamp+".pdf", dpi=200 # suze edit
    )
    print(plot)

