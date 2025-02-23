# https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html
import pyvene

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

# from pyvene.models.interventions import NoiseIntervention

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
model_name = "meta-llama/Llama-3.2-1B"  # autoregressive model 
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


##  PART 1: FACTUAL RECALL 
'''
base = "The Space Needle is in downtown"
inputs = [tokenizer(base, return_tensors="pt").to(device),]
res = model(**inputs[0])

# instead of using the provided distr function (doesn't work on tiny LLAMA or OLMO)
#  distrib = embed_to_distrib(model, res.logits, logits=False) # for generation models # aditi added this 
logits = res.logits[0, -1, :]  # Get the last token's logits (for "downtown")
distrib = sm(res.logits) 

top_vals(tokenizer, distrib[0, -1], n=10)
'''


##  PART 2: CORRUPTED RUN
# slightly working...
        # Intervention key: layer_0_comp_block_input_unit_pos_nunit_1#0
        # _and                 0.09548791497945786
        # _is                  0.06283295899629593
        # _downtown            0.0499887578189373
        # _in                  0.04853437840938568
        # _department          0.030326135456562042
        # _right               0.016650980338454247
        # _are                 0.014145351946353912
        # _city                0.013409560546278954
        # _today               0.010750866495072842
        # _world               0.009167429059743881

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

gpt=model
base = tokenizer("The Space Needle is in downtown", return_tensors="pt").to(device)
config = corrupted_config(type(gpt))
config.output_hidden_states = True # aditi addition
config.last_hidden_state = True

intervenable = IntervenableModel(config, gpt)

_, counterfactual_outputs = intervenable(
    base, unit_locations={"base": ([[[0, 1, 2, 3]]])}
)
# get the LAST hidden state. modified line 1952 and 1953 in intervenable_base.py (imported file) like so:
#    if 'output_hidden_states' in self.model.config:  # aditi addition 
#        model_kwargs["output_hidden_states"] = True  # aditi addition 
#    icounterfactual_outputs = self.model(**base, **model_kwargs)

last_hidden_state = counterfactual_outputs.hidden_states[-1]

def embed_to_distrib_aditi(model, embed, log=False, logits=False):
    """Convert an embedding to a distribution over the vocabulary"""
    if "gpt2" in model.config.architectures[0].lower():
        with torch.inference_mode():
            vocab = torch.matmul(embed, model.wte.weight.t())
    elif "llama" in model.config.architectures[0].lower():  ## llama case modified by aditi
        with torch.inference_mode():
            vocab = torch.matmul(embed, model.lm_head.weight.t()) ## equivalent of GPT's model.wte
    else:
        raise ValueError("Unsupported model architecture")

    if logits:
        return vocab
    return lsm(vocab) if log else sm(vocab)

distrib = embed_to_distrib_aditi(gpt, last_hidden_state, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)



