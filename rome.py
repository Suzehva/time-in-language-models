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
from pyvene import create_gpt2


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
# config, tokenizer, gpt = create_gpt2(name="gpt2-xl")
# gpt.to(device)

model_name = "allenai/OLMo-1B-hf"  # autoregressive model 
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


base = "The Space Needle is in downtown"
inputs = [
    tokenizer(base, return_tensors="pt").to(device),
]
print(base)
res = model(**inputs[0])

# provided code, installed at /Users/aditi/Documents/cs224n-time/time/lib/python3.12/site-packages/pyvene/
# distrib = embed_to_distrib(model, res.last_hidden_state, logits=False) # for encoder models 

# distrib = embed_to_distrib(model, res.logits, logits=False) # for generation models # aditi added this 
# def embed_to_distrib(model, embed, log=False, logits=False):
#     """Convert an embedding to a distribution over the vocabulary"""
#     if "gpt2" in model.config.architectures[0].lower():
#         with torch.inference_mode():
#             vocab = torch.matmul(embed, model.wte.weight.t())
#             if logits:
#                 return vocab
#             return lsm(vocab) if log else sm(vocab)
#     elif "llama" in model.config.architectures[0].lower():
#         assert False, "Support for LLaMA is not here yet"


# print(res.logits.shape) 

# def top_vals(tokenizer, res, n=10, return_results=False):
#     """Pretty print the top n values of a distribution over the vocabulary"""
#     top_values, top_indices = torch.topk(res, n)
#     ret = []
#     for i, _ in enumerate(top_values):
#         tok = format_token(tokenizer, top_indices[i].item())
#         ret += [(tok, top_values[i].item())]
#         if not return_results:
#             print(f"{tok:<20} {top_values[i].item()}")
#     if return_results:
#         return ret


# chat code
logits = res.logits[0, -1, :]  # Get the last token's logits (for "downtown")
top_k = torch.topk(logits, 10)  # Get top 10 predictions
top_tokens = [tokenizer.decode([idx]) for idx in top_k.indices.tolist()]
distrib = sm(res.logits) 

top_vals(tokenizer, distrib[0, -1], n=10)
# print("Top predictions:", top_tokens)

