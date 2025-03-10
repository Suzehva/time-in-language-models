# based off of https://stanfordnlp.github.io/pyvene/tutorials/basic_tutorials/Basic_Intervention.html#interchange-intervention 
import pandas as pd
import pyvene
from pyvene import embed_to_distrib, top_vals, format_token
from pyvene import RepresentationConfig, IntervenableConfig, IntervenableModel
from pyvene import (
    IntervenableModel,
    VanillaIntervention, Intervention,
    RepresentationConfig,
    IntervenableConfig,
    ConstantSourceIntervention,
    LocalistRepresentationIntervention,
    create_olmo
)
from tqdm import tqdm
import torch
import os
import datetime
import time
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
# note: this is using GPT

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
)


class InterchangeIntervention:
    def __init__(self, model_id, folder_path="pyvene_data_interchange_intervention_olmo", device=None):
        self.model_id = model_id
        # Initialize the device (GPU or MPS [for apple silicon] or CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" 
        print(f'Using device: {self.device}')
        if self.model_id == "allenai/OLMo-1B-hf":
            self.config, self.tokenizer, self.model = create_olmo(name=self.model_id) 
        else:
           os.error(f'only olmo is supported at this time') 
        self.model.to(self.device)

        self.prompts = []
        self.folder_path = folder_path

    def factual_recall(self, prompt: str):
        print("FACTUAL RECALL:")
        print(prompt)

        inputs = [
            self.tokenizer(prompt, return_tensors="pt").to(self.device),
        ]
        res = self.model.model(**inputs[0])  # use self.model.model to get the BASE output instead of the CAUSAL output
        distrib = embed_to_distrib(self.model, res.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10) # prints top 10 results from distribution

    def simple_position_config(self, component, layer):
        config = IntervenableConfig(
            model_type=type(self.model_id),
            representations=[
                RepresentationConfig(
                    layer,              # layer
                    component,          # component
                    "pos",              # intervention unit
                    1,                  # max number of unit
                ),
            ],
            intervention_types=VanillaIntervention,
        )
        return config

    def intervene(self, base: str, sources: list[str], output_to_measure: list[str]):
        base_tokenized = self.tokenizer("The capital of Spain is", return_tensors="pt")
        sources_tokenized = [self.tokenizer("The capital of Italy is", return_tensors="pt")]
        tokens = [self.tokenizer.encode(word) for word in output_to_measure]

        intervention_data = []

        for layer_i in tqdm(range(self.model.config.num_hidden_layers)): # looping over all hidden layers in model
            config = self.simple_position_config("mlp_output", layer_i)
            intervenable = IntervenableModel(config, self.model)
            for pos_i in range(len(base_tokenized.input_ids[0])): # looping over all the tokens in the base sentence
                _, counterfactual_outputs = intervenable(
                    base_tokenized, sources_tokenized, {"sources->base": pos_i}
                )
                counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # TODO: there must be a better way
                distrib = embed_to_distrib(
                    self.model, counterfactual_outputs.last_hidden_state, logits=False
                )
                for token in tokens:
                    intervention_data.append(
                        {
                            "token": format_token(self.tokenizer, token),
                            "prob": float(distrib[0][-1][token]),
                            "layer": f"f{layer_i}",
                            "pos": pos_i,
                            "type": "mlp_output",
                        }
                    )

            config = self.simple_position_config("attention_input", layer_i)
            intervenable = IntervenableModel(config, self.model)
            for pos_i in range(len(base_tokenized.input_ids[0])):
                _, counterfactual_outputs = intervenable(
                    base_tokenized, sources_tokenized, {"sources->base": pos_i}
                )
                counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # TODO: there must be a better way
                distrib = embed_to_distrib(
                    self.model, counterfactual_outputs.last_hidden_state, logits=False
                )
                for token in tokens:
                    intervention_data.append(
                        {
                            "token": format_token(self.tokenizer, token),
                            "prob": float(distrib[0][-1][token]),
                            "layer": f"a{layer_i}",
                            "pos": pos_i,
                            "type": "attention_input",
                        }
                    )
        df = pd.DataFrame(intervention_data)

        os.makedirs(self.folder_path, exist_ok=True)
        base_txt = base.replace(' ', '_')
        output_prob_text = '(' + ''.join([s.replace(' ', '_') for s in output_to_measure]) + ')'
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = "./"+self.folder_path+"/"+base_txt+output_prob_text+"_"+timestamp
        print(f"saving interchange intervention data to {filepath}.pdf")
        df.to_csv(filepath+".csv")
        return df

    def heatmap_plot(self, df, base: str, sources: list[str], output_to_measure: list[str]):
        df["layer"] = df["layer"].astype("category")
        df["token"] = df["token"].astype("category")
        nodes = []
        for l in range(self.model.config.num_hidden_layers - 1, -1, -1):
            nodes.append(f"f{l}")
            nodes.append(f"a{l}")
        df["layer"] = pd.Categorical(df["layer"], categories=nodes[::-1], ordered=True)

        plot_heat = (
            ggplot(df)
            + geom_tile(aes(x="pos", y="layer", fill="prob", color="prob"))
            + facet_wrap("~token")
            + theme(axis_text_x=element_text(rotation=90))
        )
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_txt = base.replace(' ', '_')
        output_prob_text = '(' + ''.join([s.replace(' ', '_') for s in output_to_measure]) + ')'
        filepath = "./"+self.folder_path+"/"+base_txt+output_prob_text+"_"+timestamp
        print(f"saving interchange intervention heatmap to {filepath}.pdf")
        ggsave(
            plot_heat, filename=filepath+".pdf", dpi=200 # write pdf graph # TODO: how to save as png??
        )
    
    def bar_plot(self, df):
        pass



def main():
    base_prompt = "The capital of Spain is" # sentence where part of residual stream will be replaced
    source_prompts = ["The capital of Italy is"] # sentence from which we take the replacement
    interchange_intervention = InterchangeIntervention(model_id="allenai/OLMo-1B-hf")
    output_to_measure = [" Madrid", " Rome"]
    #interchange_intervention.factual_recall(prompt=base_prompt)
    #for s_p in source_prompts:
    #    interchange_intervention.factual_recall(prompt=s_p)
    results_df = interchange_intervention.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    interchange_intervention.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)


if __name__ == "__main__":
    main()

# -----------

"""
filtered = df
filtered = filtered[filtered["pos"] == 4]
plot_bar = (
    ggplot(filtered)
    + geom_bar(aes(x="layer", y="prob", fill="token"), stat="identity")
    + theme(axis_text_x=element_text(rotation=90), legend_position="none")
    + scale_y_log10()
    + facet_wrap("~token", ncol=1)
)
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filepath = "./"+folder_path+"/"+timestamp
print(f"saving file to {filepath}.pdf")
ggsave(
    plot_bar, filename=filepath+".pdf", dpi=200 # write pdf graph # TODO: how to save as png??
"""