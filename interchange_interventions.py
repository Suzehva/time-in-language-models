# based off of https://stanfordnlp.github.io/pyvene/tutorials/basic_tutorials/Basic_Intervention.html#interchange-intervention 
import pandas as pd
from pyvene import embed_to_distrib, top_vals, format_token
from pyvene import RepresentationConfig, IntervenableConfig, IntervenableModel
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
    scale_y_discrete, scale_y_continuous, ggsave,
    labs,
    scale_x_discrete, 
    scale_x_continuous
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
    def __init__(self, model_id, folder_path: str, device=None):
        self.model_id = model_id
        print(f"Running with model {self.model_id}")
        # Initialize the device (GPU or MPS [for apple silicon] or CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" 
        print(f'Using device: {self.device}')
        if self.model_id == "allenai/OLMo-1B-hf":
            self.config, self.tokenizer, self.model = create_olmo(name=self.model_id) 
        elif self.model_id == "gpt2":
            self.config, self.tokenizer, self.model = create_gpt2()
        else:
           os.error(f'only olmo, gpt2 is supported at this time') 
        self.model.to(self.device)

        self.prompts = []
        self.folder_path = folder_path

    def factual_recall(self, prompt: str):
        print("FACTUAL RECALL:")
        print(prompt)

        inputs, _ = self.string_to_token_ids_and_tokens(prompt)
        if self.model_id == "allenai/OLMo-1B-hf":
            res = self.model.model(**inputs) # removed [0] from **inputs[0] because input is now not a list
        else:
            # assuming this is gpt2
            res = self.model(**inputs[0]) 
        distrib = embed_to_distrib(self.model, res.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10) # prints top 10 results from distribution

    def simple_position_config(self, component, layer):
        config = IntervenableConfig(
            model_type=type(self.model),
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

    def intervene(self, base: str, sources: list[str], output_to_measure: list[str], component: str):
        self.base_ids, self.base_tokens = self.string_to_token_ids_and_tokens(base)
        self.component = component
        sources_ids, sources_tokens = self.string_to_token_ids_and_tokens(sources)
        if (len(self.base_ids.input_ids[0]) != len(sources_ids.input_ids[0])):
            raise Exception(f"number of tokens in source ({len(sources_ids.input_ids[0])}) are not the same as number of tokens in the base ({len(self.base_ids.input_ids[0])}). Source tokens: {sources_tokens}, Base tokens: {self.base_tokens}")
        self.sources_ids, self.sources_tokens = [sources_ids], [sources_tokens] # for some reason the input needs to be a list
        tokens = [self.tokenizer.encode(word) for word in output_to_measure] # tokenizer.encode returns list of token ids

        intervention_data = []
        output_intervention_data = []

        for layer_i in tqdm(range(self.model.config.num_hidden_layers)): # looping over all hidden layers in model (every layer is an MLP?)
            config = self.simple_position_config(self.component, layer_i)
            intervenable = IntervenableModel(config, self.model)
            for pos_i in range(len(self.base_ids.input_ids[0])): # looping over all the tokens in the base sentence
                _, counterfactual_outputs = intervenable( 
                    # counterfactual_outputs stores lots of things, amongst which hidden states of the base model with the mlp_output 
                    # at position i replaced with the mlp_output of source
                    self.base_ids, self.sources_ids, {"sources->base": pos_i} # TODO: why can we pass in a list of sources??
                )
                counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # TODO: there must be a better way
                distrib = embed_to_distrib(
                    self.model, counterfactual_outputs.last_hidden_state, logits=False
                )
                # if (layer_i == self.model.config.num_hidden_layers - 1) and (pos_i == len(self.base_ids.input_ids[0]) - 1):
                #     # this is the last hidden state
                #     self.output_token = top_vals(self.tokenizer, distrib[0][-1], n=1, return_results=True)
                    
                for token in tokens:
                    intervention_data.append(
                        {
                            "token": format_token(self.tokenizer, token), # this is an actual word piece
                            "prob": float(distrib[0][-1][token]),
                            "layer": layer_i,
                            "pos": pos_i,
                            "type": self.component,
                        }
                    )
                
                top_token_info = top_vals(self.tokenizer, distrib[0][-1], n=1, return_results=True)
                output_intervention_data.append(
                    {
                        "token": top_token_info[0][0], # this is an actual word piece
                        "prob": top_token_info[0][1], # probability of the token that was outputted
                        "layer": layer_i,
                        "pos": pos_i,
                        "type": self.component,
                    }
                )
        df_intervention_data = pd.DataFrame(intervention_data)
        df_output_intervention_data = pd.DataFrame(output_intervention_data)

        os.makedirs(self.folder_path, exist_ok=True)
        base_txt = base.replace(' ', '_')
        output_prob_text = '(' + ''.join([s.replace(' ', '_') for s in output_to_measure]) + ')'
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = "./"+self.folder_path+"/"+base_txt+output_prob_text+"_"+timestamp
        print(f"saving interchange intervention data to {filepath}.pdf")
        df_intervention_data.to_csv(filepath+".csv")
        df_output_intervention_data.to_csv(filepath+"_output.csv")
        return df_intervention_data, df_output_intervention_data

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


    def heatmap_plot(self, df, output_df, base: str, sources: list[str], output_to_measure: list[str]):
        df["layer"] = df["layer"].astype(int)
        df["token"] = df["token"].astype("category")
        breaks, labels = list(range(len(self.base_tokens))), self.base_tokens
        print(f"breaks: {breaks}, labels: {labels}")
        # Example:
        # labels: ['The', 'capital', 'of', 'Spain', 'is']
        # breaks: [0, 1, 2, 3, 4]

        plot_heat = (
            ggplot(df)
            + geom_tile(aes(x="pos", y="layer", fill="prob", color="prob"))
            + facet_wrap("~token") # splits the graph into multiple graphs, one for each token
            + theme(axis_text_x=element_text(rotation=90))
            + scale_x_continuous(
                breaks=breaks,
                labels=labels
            )
            + scale_y_continuous(
                breaks=list(range(self.model.config.num_hidden_layers))
            )
            + labs(
                title=f"Base: {base}, Source: {sources}", # TODO
                y=f"Restored {self.component} for layer in {self.model_id}",
                fill="Probability Scale",
                color="Probability Scale"
            )
        )
        
        base_txt = base.replace(' ', '_')
        output_prob_text = '(' + ''.join([s.replace(' ', '_') for s in output_to_measure]) + ')'

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = "./"+self.folder_path+"/heat_"+base_txt+output_prob_text+"_"+timestamp
        print(f"saving interchange intervention heatmap to {filepath}.pdf")
        ggsave(
            plot_heat, filename=filepath+".pdf", dpi=200 # write pdf graph # TODO: how to save as png??
        )

        # --------------
        output_df["layer"] = output_df["layer"].astype(int)
        token_pivot = output_df.pivot(index='layer', columns='pos', values='token')

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = "./"+self.folder_path+"/token_text_"+base_txt+"_"+timestamp
        token_pivot.to_csv(filepath+"_tokens.csv")

        # Create a text annotation dataframe
        # We need to melt the pivot table back to long format for plotting
        text_data = token_pivot.reset_index().melt(
            id_vars='layer', 
            var_name='pos', 
            value_name='token'
        )
        text_data['pos'] = text_data['pos'].astype(int)
        text_data['dummy_color'] = 1

        plot_text = (
            ggplot(text_data)
            + geom_tile(aes(x="pos", y="layer", fill="dummy_color"), alpha=0.3)
            + geom_text(aes(x="pos", y="layer", label="token"), size=8)
            + theme(axis_text_x=element_text(rotation=90))
            + scale_x_continuous(
                breaks=breaks,
                labels=labels
            )
            + scale_y_continuous(
                breaks=list(range(self.model.config.num_hidden_layers))
            )
            + labs(
                title=f"Top Predicted Tokens - Base: {base}, Source: {sources}",
                y=f"Layer in {self.model_id}",
                x="Position"
            )
            # Hide the fill legend since it's just a dummy
            + theme(legend_position="none")
        )
        print(f"saving token text heatmap to {filepath}.pdf")
        ggsave(
            plot_text, filename=filepath+".pdf", dpi=300, width=12, height=10
        )
    
    
    def bar_plot(self, df, base: str, sources: list[str], output_to_measure: list[str], layer_to_filter: int):
        if layer_to_filter > self.model.config.num_hidden_layers:
            print(f"cannot make bar plot for {layer_to_filter} because the model ({self.model_id}) only has {self.model.config.num_hidden_layers} layers")
        filtered = df[df["pos"] == layer_to_filter]

        plot_bar = (
            ggplot(filtered)
            + geom_bar(aes(x="layer", y="prob", fill="token"), stat="identity")
            + theme(axis_text_x=element_text(rotation=90), legend_position="none")
            #+ scale_y_log10()
            + facet_wrap("~token", ncol=1)
            + labs(
                title=f"Base: {base}, Source: {sources}",
                y=f"Probability Scale",
                x=f"Restored {self.component} for layer ('{self.base_tokens[layer_to_filter]}') in {self.model_id}",
                fill="Probability Scale",
                color="Probability Scale"
            )
        )
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_txt = base.replace(' ', '_')
        output_prob_text = '(' + ''.join([s.replace(' ', '_') for s in output_to_measure]) + ')'
        filepath = "./"+self.folder_path+"/bar_"+base_txt+output_prob_text+"_"+timestamp
        print(f"saving interchange intervention bar plot to {filepath}.pdf")
        ggsave(
            plot_bar, filename=filepath+".pdf", dpi=200 # write pdf graph # TODO: how to save as png??
        )
        

def main():

    # tutorial inputs
    # base_prompt = "The capital of Spain is" # sentence where part of residual stream will be replaced
    # source_prompts = ["The capital of Italy is"] # sentence from which we take the replacement
    # interchange_intervention = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="pyvene_data_interchange_intervention_olmo") # options: allenai/OLMo-1B-hf or gpt2
    # output_to_measure = [" Rome", " Madrid"]
    # interchange_intervention.factual_recall(prompt=base_prompt)
    # for s_p in source_prompts:
    #     interchange_intervention.factual_recall(prompt=s_p)
    # results_df = interchange_intervention.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    # interchange_intervention.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    # interchange_intervention.bar_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, layer_to_filter=4)


    # time inputs
    # TODO: what if base and source prompts don't have the same amount of tokens?
    # base_prompt = "On a beautiful day in 1980 there" # sentence where part of residual stream will be replaced
    # source_prompts = ["On a beautiful day in 2020 there"] # sentence from which we take the replacement
    base_prompt = "In 1980 on a beautiful day there" # sentence where part of residual stream will be replaced
    source_prompts = ["In 2030 on a beautiful day there"] # sentence from which we take the replacement
    interchange_intervention = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="pyvene_data_interchange_intervention_olmo") # options: allenai/OLMo-1B-hf or gpt2
    #interchange_intervention = InterchangeIntervention(model_id="gpt2", folder_path="pyvene_data_interchange_intervention_gpt2") # options: allenai/OLMo-1B-hf or gpt2
    output_to_measure = [" was", " will"] # Make sure to include space at the beginning!
    interchange_intervention.factual_recall(prompt=base_prompt)
    for s_p in source_prompts:
        interchange_intervention.factual_recall(prompt=s_p)
    results_df, output_results_df = interchange_intervention.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
    interchange_intervention.heatmap_plot(df=results_df, output_df=output_results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)

    interchange_intervention.bar_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, layer_to_filter=6)

if __name__ == "__main__":
    main()