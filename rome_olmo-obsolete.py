# https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

from torch import nn
sm = nn.Softmax(dim=-1)
lsm = nn.LogSoftmax(dim=-1)

# %config InlineBackend.figure_formats = ['svg']
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

titles={
    "block_output": "single restored layer in GPT2-XL",
    "mlp_activation": "center of interval of 10 patched mlp layer",
    "attention_output": "center of interval of 10 patched attn layer"
}

colors={
    "block_output": "Purples",
    "mlp_activation": "Greens",
    "attention_output": "Reds"
} 

device = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL SETUP
model_name = "allenai/OLMo-1B-hf"  # autoregressive model 
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

##########################################################################
##  PART 1: FACTUAL RECALL 
##########################################################################
print("\nBEGINNING PART 1: FACTUAL RECALL\n")

base = "The Space Needle is in downtown"
inputs = [tokenizer(base, return_tensors="pt").to(device),]
res = model(**inputs[0])

# instead of using the provided distr function (doesn't work on tiny LLAMA or OLMO)
#  distrib = embed_to_distrib(model, res.logits, logits=False) # for generation models # aditi added this 
logits = res.logits[0, -1, :]  # Get the last token's logits (for "downtown")
distrib = sm(res.logits) 

top_vals(tokenizer, distrib[0, -1], n=10)

##########################################################################
##  PART 2: CORRUPTED RUN
##########################################################################
print("\nBEGINNING PART 2: CORRUPTED RUN\n")

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
config = corrupted_config(type(model))
config.output_hidden_states = True # aditi addition

intervenable = IntervenableModel(config, model)

# IMPLEMENT THIS HACK IN THE PYVENE LIBRARY!!
# my env is called time, so for me its under:
#   /Users/aditi/Documents/cs224n-time/time/lib/python3.12/site-packages/pyvene
# I modified line 1952 and 1953 in intervenable_base.py (imported file) like so:
#    if 'output_hidden_states' in self.model.config:  # aditi addition 
#        model_kwargs["output_hidden_states"] = True  # aditi addition 
#    icounterfactual_outputs = self.model(**base, **model_kwargs)

_, counterfactual_outputs = intervenable(
    base, unit_locations={"base": ([[[0, 1, 2, 3]]])}
)

print("COUNTERFACTUAL OUTPUTS", counterfactual_outputs)
# get the LAST hidden state. 
last_hidden_state = counterfactual_outputs.hidden_states[-1]

print( "last_hidden_state", last_hidden_state)
print( "model", model)

distrib = embed_to_distrib(model, last_hidden_state, logits=False)
print("DISTRIB",distrib)

top_vals(tokenizer, distrib[0][-1], n=10)

##########################################################################
##   PART 3: RESTORED RUN
##########################################################################
print("\nBEGINNING PART 3: RESTORED RUN\n")

## CONTINUE NOTE:
# Currently on part 3
# the number of hidden layers in the model don't match up

from pyvene.models.interventions import NoiseIntervention

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
token = tokenizer.encode(" Seattle")[0]  # 128000
print(token)

for stream in ["block_output", "mlp_activation", "attention_output"]:
    data = []
    use_range = (model.config.num_hidden_layers) # for llama # aditi addition
    # range(gpt.config.n_layer) for gpt
    for layer_i in tqdm(range(min(use_range, 16))):  # TODO GET RID OF 16 bc its hacky !?
        for pos_i in range(7):
            config = restore_corrupted_with_interval_config(
                layer_i, stream, 
                window=1 if stream == "block_output" else 10
            )
            config.output_hidden_states = True # aditi addition

            n_restores = len(config.representations) - 1
            intervenable = IntervenableModel(config, model)
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
            last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
            distrib = embed_to_distrib(
                model, last_hidden_state, logits=False
            )
            prob = distrib[0][-1][token].detach().cpu().item()
            data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
    df = pd.DataFrame(data)
    import os  # aditi addition to create dir
    os.makedirs("tutorial_data", exist_ok=True) # aditi addition to create dir
    df.to_csv(f"./tutorial_data/pyvene_rome_{stream}.csv")

# Graphing
for stream in ["block_output", "mlp_activation", "attention_output"]:
    df = pd.read_csv(f"./tutorial_data/pyvene_rome_{stream}.csv")
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
    ggsave(
        plot, filename=f"./tutorial_data/pyvene_rome_{stream}.pdf", dpi=200
    )
    print(plot)
