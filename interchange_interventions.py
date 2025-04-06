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
    scale_fill_gradient,
    element_line,
    element_rect,
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
    geom_text,
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
        elif self.model_id == "meta-llama/Llama-3.2-1B":
            # bit hacky but oh well
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            # bit hacky
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
            self.config = AutoConfig.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map=self.device, config=self.config)
            
        else:
           raise Exception(f'only olmo, gpt2, llama is supported at this time') 
        self.model.to(self.device)

        self.prompts = []
        self.folder_path = folder_path

    def factual_recall(self, prompt: str):
        print("FACTUAL RECALL:")
        print(prompt)

        inputs, _ = self.string_to_token_ids_and_tokens(prompt)
        if self.model_id == "allenai/OLMo-1B-hf":
            res = self.model.model(**inputs) # removed [0] from **inputs[0] because input is now not a list
        elif self.model_id == "gpt2":
            res = self.model(**inputs[0])
        elif self.model_id == "meta-llama/Llama-3.2-1B":
            res = self.model(**inputs, output_hidden_states=True) 
            res.last_hidden_state = res.hidden_states[-1] #this seems to work
        else:
            raise Exception(f'only olmo, gpt2 is supported at this time') 

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
 
                for token in tokens:
                    # if token is a list, it means the words we are measuring are getting split up into multiple tokens. 
                    # # To plot them, I just multiply their probabilities since we're dealing with conditional probability, 
                    # TODO: should verify if that's okay
                    if self.model_id == "meta-llama/Llama-3.2-1B" and len(token) == 2 and token[0] == 128000: # 128000 is <|begin_of_text|> token for llama which we want to ignore TODO don't hardcode this
                        token = [token[1]]
                    if len(token) > 1:
                        # this happens for llama NEVERMIND THAT IS THE <|begin_of_text|> SO THIS SHOULD NEVER HAPPEN
                        raise Exception("token should not be more than one token")
                    else:
                        prob = float(distrib[0][-1][token])

                    intervention_data.append(
                        {
                            "token": format_token(self.tokenizer, token), # this is an actual word piece
                            "prob": prob,
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
        print(f"saving interchange intervention data to {filepath}.png")
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
        print(f"{len(tokens)} tokens in '{s}': {tokens} with token_ids: {token_ids}")
        return token_ids, tokens


    def heatmap_plot(self, df, base: str, sources: list[str], output_to_measure: list[str]):
        df["layer"] = df["layer"].astype(int)
        formatted_output_to_measure = [word.replace(" ", "_").replace("\n", "\\n") for word in output_to_measure] # based off of format_token funtion in basic_utils
        df["token"] = pd.Categorical(
            df["token"],
            categories=formatted_output_to_measure,
            ordered=True
        )
        #raise Exception(f"df[token]: {df["token"]}, edited version: {test}")
        breaks, labels = list(range(len(self.base_tokens))), self.base_tokens
        print(f"breaks: {breaks}, labels: {labels}")
        # Example:
        # labels: ['The', 'capital', 'of', 'Spain', 'is']
        # breaks: [0, 1, 2, 3, 4]
        

        # change color based on component
        hi_color = "#00CCFF"    # blue
        if(self.component=="mlp_output"):
            hi_color="#006600"  # teal
        if(self.component=="attention_output"):
            hi_color="#009980"  # green
        
        title_text = f"Base: {base}, Source: {sources[0]}"
        y_text = f"Single {self.component} layer restored in {self.model_id}"

        plot_heat = (
            ggplot(df)
            + geom_tile(aes(x="pos", y="layer", fill="prob"))
            + facet_wrap("~token", ncol=len(output_to_measure)) # splits the graph into multiple graphs, one for each token
            + scale_fill_gradient(low="white", high=hi_color, limits=(0, 1))  # Fixes 0 to light, 1 to dark

            + theme(
                axis_text_x=element_text(rotation=90),
                plot_title=element_text(size=len(title_text)//8),  # make sure title fits
                axis_text_y=element_text(size=len(y_text)//8)
            )
            + scale_x_continuous(
                breaks=breaks,
                labels=labels
            )
            + scale_y_continuous(
                breaks=list(range(self.model.config.num_hidden_layers))
            )
            + labs(
                title=title_text, 
                y=y_text,
                x=" ",
                fill="Probability",
            )
        )
        
        base_txt = base.replace(' ', '_')
        output_prob_text = '(' + ''.join([s.replace(' ', '_') for s in output_to_measure]) + ')'

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = "./"+self.folder_path+"/heat_"+base_txt+output_prob_text+"_"+timestamp
        print(f"saving interchange intervention heatmap to {filepath}.png")
        ggsave(
            plot_heat, filename=filepath+".png", dpi=200 
        )

    def text_heatmap_plot(self, output_df, base: str, sources: list[str]):
        base_txt = base.replace(' ', '_')
        breaks, labels = list(range(len(self.base_tokens))), self.base_tokens
        output_df["layer"] = output_df["layer"].astype(int)
        
        # Create pivot tables for both tokens and probabilities
        token_pivot = output_df.pivot(index='layer', columns='pos', values='token')
        prob_pivot = output_df.pivot(index='layer', columns='pos', values='prob')
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filepath = "./"+self.folder_path+"/token_text_"+base_txt+"_"+timestamp
        token_pivot.to_csv(filepath+"_tokens.csv")
        
        # Melt both pivot tables to long format
        text_data = token_pivot.reset_index().melt(
            id_vars='layer', 
            var_name='pos', 
            value_name='token'
        )
        prob_data = prob_pivot.reset_index().melt(
            id_vars='layer',
            var_name='pos',
            value_name='prob'
        )
        
        # Merge the two dataframes to get both token and probability in one
        merged_data = pd.merge(text_data, prob_data, on=['layer', 'pos'])
        merged_data['pos'] = merged_data['pos'].astype(int)

        # change color based on component
        hi_color = "#00CCFF"    # blue
        if(self.component=="mlp_output"):
            hi_color="#006600"  # teal
        if(self.component=="attention_output"):
            hi_color="#009980"  # green
        
        title_text = f"Base: {base}, Source: {sources[0]}"
        y_text = f"Single {self.component} layer restored in {self.model_id}"
        
        plot_text = (
            ggplot(merged_data)
            + geom_tile(aes(x="pos", y="layer", fill="prob"), color="white")
            + geom_text(aes(x="pos", y="layer", label="token"), size=8, color="black")
            + scale_fill_gradient(low="white", high=hi_color, limits=(0, 1))  # Fixes 0 to light, 1 to dark
            + theme(
                axis_text_x=element_text(rotation=90),
                axis_text_y=element_text(size=len(y_text)//6),
                plot_title=element_text(size=len(title_text)//6), # make sure title fits,
                panel_grid_major=element_line(color="white", size=0.5),
                panel_grid_minor=element_line(color="white", size=0.25),
                panel_background=element_rect(fill="white")
            )
            + scale_x_continuous(
                breaks=breaks,
                labels=labels
            )
            + scale_y_continuous(
                breaks=list(range(self.model.config.num_hidden_layers))
            )
            + labs(
                title=f"Top Predicted Tokens - Base: {base}, Source: {sources}",
                y=y_text,
                x=" ",
                fill="Probability"  # Correct legend title
            )
        )
        
        print(f"saving token text heatmap to {filepath}.png")
        ggsave(
            plot_text, filename=filepath+".png", dpi=300, width=12, height=10
        )
    
    
    def bar_plot(self, df, base: str, sources: list[str], output_to_measure: list[str], layer_to_filter: int):
        # this function isn't used so i stopped updating it
        if layer_to_filter > self.model.config.num_hidden_layers:
            print(f"cannot make bar plot for {layer_to_filter} because the model ({self.model_id}) only has {self.model.config.num_hidden_layers} layers")
        filtered = df[df["pos"] == layer_to_filter]

        plot_bar = (
            ggplot(filtered)
            + geom_bar(aes(x="layer", y="prob", fill="token"), stat="identity")
            + theme(
                axis_text_x=element_text(rotation=90), 
                legend_position="none",
            )
            + facet_wrap("~token", ncol=1) # ncol or nrow?
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
        print(f"saving interchange intervention bar plot to {filepath}.png")
        ggsave(
            plot_bar, filename=filepath+".png", dpi=200 
        )


def fact_recall_meas():
    # Factual recall measurements
    interchange_intervention = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="pyvene_data_interchange_intervention_llama") # for if you want to use gpt
    #interchange_intervention = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="pyvene_data_interchange_intervention_olmo") # options: allenai/OLMo-1B-hf or gpt2
    
   
    prompt_list = [
        "In contrast on a beautiful day there",
        "In summary on a beautiful day there",
        "In addition on a beautiful day there",
        "In response on a beautiful day there",
    ]

    for prompt in prompt_list:
        interchange_intervention.factual_recall(prompt=prompt)

def run_ii_experiment():
    # working with sentences of 10 tokens
    prompt_combos_llama = [
        ("In 1980 on a beautiful day there", ["In 2050 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 2020 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 2000 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In Elmsville on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In just three hours on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In addition to this on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In contrast to this on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In response to this on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In summary of events on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In conclusion to that on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 1980 on a beautiful day there"]), # nothing should change in output

        ("In 2030 on a beautiful day there", ["In 2050 on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In 2030 on a beautiful day there"]), # nothing should change in output
        ("In 2030 on a beautiful day there", ["In 2020 on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In 2000 on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In Elmsville on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In just three hours on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In addition to this on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In contrast to this on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In response to this on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In summary of events on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In conclusion to that on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In 1980 on a beautiful day there"]),

        # now reverse
        ("In 2050 on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In 2020 on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In 2000 on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In just three hours on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In Elmsville on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In addition to this on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In contrast to this on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In response to this on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In summary of events on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In conclusion to that on a beautiful day there", ["In 1980 on a beautiful day there"]),

        ("In 2050 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In 2020 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In 2000 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In just three hours on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In addition to this on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In contrast to this on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In response to this on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In summary of events on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In conclusion to that on a beautiful day there", ["In 2030 on a beautiful day there"]),

        # unrelated experiment: try different prompt template to show results are robust
        ("On a beautiful day in 1980 there", ["On a beautiful day in 2030 there"]),
        ("In 1980 there", ["In 2030 there"]),
        ("On a beautiful day in 2030 there", ["On a beautiful day in 1980 there"]),
        ("In 2030 there", ["In 1980 there"]),
    ]


    # working with sentences of 7 tokens
    prompt_combos_olmo = [ # use summer instead of elmsville (to make sentences the same amount of tokens) and don't use 2050 bc it gets split up into two tokens
        ("In 1980 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 2020 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 2000 on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In summer on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In contrast on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In addition on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In response on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In summary on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In conclusion on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["Tomorrow on a beautiful day there"]),
        ("In 1980 on a beautiful day there", ["In 1980 on a beautiful day there"]), # nothing should change in output

        ("In 2030 on a beautiful day there", ["In 2030 on a beautiful day there"]), # nothing should change in output
        ("In 2030 on a beautiful day there", ["In 2020 on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In 2000 on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In summer on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In contrast on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In addition on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In response on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In summary on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In conclusion on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["Tomorrow on a beautiful day there"]),
        ("In 2030 on a beautiful day there", ["In 1980 on a beautiful day there"]),

        # now reverse
        ("In 2020 on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In 2000 on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In summer on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In addition on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In response on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In summary on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("In conclusion on a beautiful day there", ["In 1980 on a beautiful day there"]),
        ("Tomorrow on a beautiful day there", ["In 1980 on a beautiful day there"]),

        ("In 2020 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In 2000 on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In summer on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In addition on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In response on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In summary on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("In conclusion on a beautiful day there", ["In 2030 on a beautiful day there"]),
        ("Tomorrow on a beautiful day there", ["In 2030 on a beautiful day there"]),

        # unrelated experiment: try different prompt template to show results are robust
        ("On a beautiful day in 1980 there", ["On a beautiful day in 2030 there"]),
        ("In 1980 there", ["In 2030 there"]),
        ("On a beautiful day in 2030 there", ["On a beautiful day in 1980 there"]),
        ("In 2030 there", ["In 1980 there"]),
    ]


    output_to_measure = [" was", " is", " will"] # Make sure to include space at the beginning!

    # block_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground_march_13/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="ii_playground_march_13/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # mlp_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground_march_13/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="ii_playground_march_13/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # attention_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground_march_13/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="ii_playground_march_13/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)


def run_ii_experiment_v2():

    prompt_combos_llama = [
        ("Compared to 1980, now there", ["Compared to 2030, now there"]),
        ("Compared to 2030, now there", ["Compared to 1980, now there"]),

        ("After there", ["Before there"]),
        ("After on a beautiful day there", ["Before on a beautiful day there"]),
        ("Before there", ["After there"]),
        ("Before on a beautiful day there", ["After on a beautiful day there"]),

        ("Before on a beautiful day there", ["Now on a beautiful day there"]),
        ("Now on a beautiful day there", ["Before on a beautiful day there"]),

        ("After on a beautiful day there", ["Now on a beautiful day there"]),
        ("Now on a beautiful day there", ["After on a beautiful day there"]),
    ]
    prompt_combos_olmo = prompt_combos_llama

    output_to_measure = [" was", " is", " will"] # Make sure to include space at the beginning!

    # block_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground_march_13/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="ii_playground_march_13/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # mlp_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground_march_13/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="ii_playground_march_13/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # attention_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground_march_13/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="ii_playground_march_13/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

def run_ii_experiment_v3():

    prompt_combos_olmo = [
        #("In 1980 on a beautiful day there", ["In Rome on a beautiful day there"]),
        #("In Rome on a beautiful day there", ["In 1980 on a beautiful day there"]),
        #("In 2030 on a beautiful day there", ["In Rome on a beautiful day there"]),
        #("In Rome on a beautiful day there", ["In 2030 on a beautiful day there"]),
        #("In 1980 on a beautiful day there", ["Last week on a beautiful day there"]),
        #("In 2030 on a beautiful day there", ["Last week on a beautiful day there"]),
        #("In 1980 on a beautiful day there", ["Next week on a beautiful day there"]),
        #("In 2030 on a beautiful day there", ["Next week on a beautiful day there"]),
        #("Compared to 1980 on a beautiful day there", ["Compared to 2030 on a beautiful day there"]),
        #("Compared to 2030 on a beautiful day there", ["Compared to 1980 on a beautiful day there"]),
        #("Compared to 1980 there", ["Compared to 2030 there"]),
        #("Compared to 2030 there", ["Compared to 1980 there"]),
        # ("In 1980 on a beautiful day he", ["In 2030 on a beautiful day he"]),
        # ("In 2030 on a beautiful day he", ["In 1980 on a beautiful day he"]),
        ("In contrast on a beautiful day there", ["In summary on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In addition on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In response on a beautiful day there"]),
        
        
    ]
    prompt_combos_llama = [
        # ("Compared to 1980 on a beautiful day there", ["Compared to 2030 on a beautiful day there"]),
        # ("Compared to 2030 on a beautiful day there", ["Compared to 1980 on a beautiful day there"]),
        # ("In 1980 on a beautiful day he", ["In 2030 on a beautiful day he"]),
        # ("In 2030 on a beautiful day he", ["In 1980 on a beautiful day he"]),
        ("In contrast on a beautiful day there", ["In summary on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In addition on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In response on a beautiful day there"]),
    ]

    output_to_measure = [" was", " is", " will"] # Make sure to include space at the beginning!

    # block_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="final_project_data/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="final_project_data/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # mlp_output
    #ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="final_project_data/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    #ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="final_project_data/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # attention_output
    #ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="final_project_data/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    #ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="final_project_data/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

def run_ii_experiment_v4():
    prompt_combos_olmo = [
        #("In 2030 on a beautiful day there", ["In 1980 on a beautiful day there"]),
        #("In contrast on a beautiful day there", ["In response on a beautiful day there"]),
        #("In 2030 on a beautiful day there", ["In Rome on a beautiful day there"]),
    ]
    prompt_combos_llama = [
        ("In 2030 on a beautiful day there", ["In Elmsville on a beautiful day there"]),
        ("In contrast on a beautiful day there", ["In response on a beautiful day there"]),
        
    ]

    output_to_measure = [" was", " is", " will"] # Make sure to include space at the beginning!

    # block_output
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="final_project_data/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="final_project_data/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # mlp_output
    #ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="final_project_data/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    #ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="final_project_data/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    # attention_output
    #ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="final_project_data/olmo")
    for base_prompt, source_prompts in prompt_combos_olmo:
        results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    #ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="final_project_data/llama")
    for base_prompt, source_prompts in prompt_combos_llama:
        results_df, output_results_df = ii_llama.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
        ii_llama.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
        #ii_llama.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

def test_plots():
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="ii_playground/olmo")
    base_prompt = "In 1980 on a beautiful day there" # sentence where part of residual stream will be replaced
    source_prompts = ["In 2030 on a beautiful day there"] # sentence from which we take the replacement
    output_to_measure = [" was", " is", " will"]
    
    
    results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
    ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

    results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="mlp_output") # options: attention_input, mlp_output, block_output
    ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)


    results_df, output_results_df = ii_olmo.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="attention_output") # options: attention_input, mlp_output, block_output
    ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)

def main():
    """
    3 options for models: llama, gpt2 or olmo
    ii_llama = InterchangeIntervention(model_id="meta-llama/Llama-3.2-1B", folder_path="pyvene_data_interchange_intervention_llama") # for if you want to use gpt
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="pyvene_data_interchange_intervention_olmo") # options: allenai/OLMo-1B-hf or gpt2
    ii_gpt2 = InterchangeIntervention(model_id="gpt2", folder_path="pyvene_data_interchange_intervention_gpt2") # for if you want to use gpt

    3 options (or more?) for components: attention_input, mlp_output, block_output
    
    Example usage:
    ii_olmo = InterchangeIntervention(model_id="allenai/OLMo-1B-hf", folder_path="pyvene_data_interchange_intervention_olmo") # options: allenai/OLMo-1B-hf or gpt2
    base_prompt = "In 1980 on a beautiful day there" # sentence where part of residual stream will be replaced
    source_prompts = ["In 2030 on a beautiful day there"] # sentence from which we take the replacement
    output_to_measure = [" was", " is", " will"] # Make sure to include space at the beginning!

    ii_olmo.factual_recall(prompt=base_prompt)
    for s_p in source_prompts:
        ii_olmo.factual_recall(prompt=s_p)
    results_df, output_results_df = interchange_intervention.intervene(base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, component="block_output") # options: attention_input, mlp_output, block_output
    ii_olmo.heatmap_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure)
    ii_olmo.text_heatmap_plot(output_df=output_results_df, base=base_prompt, sources=source_prompts)
    ii_olmo.bar_plot(df=results_df, base=base_prompt, sources=source_prompts, output_to_measure=output_to_measure, layer_to_filter=6)
    """

    #fact_recall_meas()
    #run_ii_experiment()
    #run_ii_experiment_v2()
    #run_ii_experiment_v3()
    run_ii_experiment_v4()
    #test_plots()

    # TODO: add it so folder gets added automatically instead of requiring user to pre-make it
    # TODO: change file names to be more readable
    # TODO: run "yesterday" and more relative time stuff
    

if __name__ == "__main__":
    main()
