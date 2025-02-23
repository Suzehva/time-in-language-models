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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "allenai/OLMo-1B-hf"  # autoregressive model 
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

embedding_dim = model.get_input_embeddings().weight.shape[1]
# print("Embedding Dimension:", embedding_dim)

# base = tokenizer("test", return_tensors="pt").to(device)
# with torch.no_grad():
#     output = model(**base)
# print("Hidden State Shape:", output.logits.shape)


##  VERSION 1 FACTUAL RECALL 
'''
base = "The Space Needle is in downtown"
inputs = [
    tokenizer(base, return_tensors="pt").to(device),
]
# print(base)
res = model(**inputs[0])

# instead of using the provided distr function (doesn't work on tiny LLAMA or OLMO)
#  distrib = embed_to_distrib(model, res.logits, logits=False) # for generation models # aditi added this 
logits = res.logits[0, -1, :]  # Get the last token's logits (for "downtown")
distrib = sm(res.logits) 

# top_k = torch.topk(logits, 10)  # Get top 10 predictions
# top_tokens = [tokenizer.decode([idx]) for idx in top_k.indices.tolist()]

top_vals(tokenizer, distrib[0, -1], n=10)
'''

##  VERSION 2 CORRUPTED RUN
# '''

# class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
#     def __init__(self, embed_dim, **kwargs):
#         super().__init__()
#         self.interchange_dim = embed_dim
#         rs = np.random.RandomState(1)
#         prng = lambda *shape: rs.randn(*shape)
#         second_dim = 4  # tried 7 and it kinda worked...
#         self.noise = torch.from_numpy(prng(1, second_dim, embed_dim)).to(device) 
#         self.noise_level = 0.13462981581687927

#     def forward(self, base, source=None, subspaces=None):
#         # if self.noise.shape[1] != base.shape[1]:  # Ensure matching shape
#             # rs = np.random.RandomState(1)
#             # prng = lambda *shape: rs.randn(*shape)
#             # self.noise = torch.from_numpy(prng(1, base.shape[1], self.interchange_dim)).to(device) 
#             # self.noise = torch.from_numpy(self.prng(1, , self.interchange_dim)).to(device)
#         base[..., : self.interchange_dim] += self.noise * self.noise_level
#         return base

#     def __str__(self):
#         return f"NoiseIntervention(embed_dim={self.embed_dim})"
    

# def corrupted_config(model_type):
#     config = IntervenableConfig(
#         model_type=model_type,
#         representations=[
#             RepresentationConfig(
#                 0,              # layer
#                 "block_input",  # intervention type
#             ),
#         ],
#         intervention_types = NoiseIntervention,
#     )
#     return config

# base = tokenizer("The Space Needle is in downtown", return_tensors="pt").to(device)
# config = corrupted_config(type(model))
# intervenable = IntervenableModel(config, model)
# _, counterfactual_outputs = intervenable(
#     base, unit_locations={"base": ([[[0, 1, 2, 3]]])}
# )

# # distrib = embed_to_distrib(model, counterfactual_outputs.last_hidden_state, logits=False)
# # res = model(**intervenable)
# res = intervenable(base)
# # print(res)
# res = res[1]
# logits = res.logits[0, -1, :]  # Get the last token's logits (for "downtown")
# distrib = sm(res.logits) 

# top_vals(tokenizer, distrib[0][-1], n=10)
# # '''