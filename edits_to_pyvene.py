# EDITS BY ADITI. these should be completed to rune rome_olmo.py and rome_llama.py

# TODO edit model.config for olmo

# in pyvene's init.py, around line 55
from .models.olmo.modelings_intervenable_olmo import create_olmo  # aditi update

# in pyvene's basic_utils.py, around line 44
def embed_to_distrib(model, embed, log=False, logits=False):
    """Convert an embedding to a distribution over the vocabulary"""
    if "gpt2" in model.config.architectures[0].lower():
        with torch.inference_mode():
            vocab = torch.matmul(embed, model.wte.weight.t())
            if logits:
                return vocab
            return lsm(vocab) if log else sm(vocab)
    elif "olmo" in model.config.architectures[0].lower(): # this if statement is an aditi addition
        with torch.inference_mode():
            vocab = torch.matmul(embed, model.lm_head.weight.t()) # modified to lm_head
            if logits:
                return vocab
            return lsm(vocab) if log else sm(vocab)
    elif "llama" in model.config.architectures[0].lower():
        with torch.inference_mode():
            vocab = torch.matmul(embed, model.lm_head.weight.t()) # modified to lm_head by aditi
            if logits:
                return vocab
            return lsm(vocab) if log else sm(vocab)
    else:  # this if statement is an aditi addition
        assert False, "Support for model is not here yet"


# in pyvene's intervenable_base.py, around line 1945, in the forward function
# run intervened forward
model_kwargs = {}
if labels is not None: # for training
    model_kwargs["labels"] = labels
if use_cache is not None and 'use_cache' in self.model.config.to_dict(): # for transformer models
    model_kwargs["use_cache"] = use_cache

model_kwargs["output_hidden_states"] = True   # aditi addition
# print("model_kwargs", model_kwargs)  # aditi addition

counterfactual_outputs = self.model(**base, **model_kwargs)
# print("counterfactual_outputs", counterfactual_outputs)  # aditi addition





