# aditi todo :)
# summer/elmsville on II
# add gpt/llama support (on causal_tracing.py file)
 

# https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html

import os
import datetime
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
    create_olmo,
    create_gpt2
)

from PIL import Image

# TODO remove plotnine im not using
from plotnine import (
    ggplot,
    geom_tile,
    aes,
    scale_fill_gradient,
    theme,
    element_text,
    xlab, ylab, 
    ggsave,
    facet_wrap,
    labs
)

import patchworklib as pw  # Import patchworklib for arranging plots

# from plotnine.ggplot import plot_grid
from plotnine.scales import scale_y_reverse, scale_fill_cmap
from plotnine.scales import scale_x_continuous, scale_y_continuous
from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class Prompt:
    prompt: str
    len_prompt_tokens: int
    len_corrupted_tokens: int
    corrupted_tokens_indices: list[list]
    list_of_soln: list[list[str]]
    descriptive_label: str
    custom_labels: list[str]
    breaks: list[int]
    year: int


# Plotting params #################################################

mpl.rcParams['figure.figsize'] = (6, 4)  # Set default figure size
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVGs

titles={
    "block_output": "single restored layer in OLMo 1B",
    "mlp_activation": "center of interval of 10 patched mlp layer",
    "attention_output": "center of interval of 10 patched attn layer"
}




class CausalTracer:
    def __init__(self, model_id, folder_path="pyvene_data_ct_olmo", device=None):
        self.model_id = model_id
        print(f"Running with model {self.model_id}")
        # Initialize the device (GPU or MPS [for apple silicon] or CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" 
        print(f'Using device: {self.device}')
        if self.model_id == "allenai/OLMo-1B-hf":
            self.config, self.tokenizer, self.model = create_olmo(name=self.model_id) 
            self.name = "olmo"
        elif self.model_id == "gpt2":
            self.config, self.tokenizer, self.model = create_gpt2()
            self.name= "gpt2"
        elif self.model_id == "meta-llama/Llama-3.2-1B":
            # bit hacky but oh well
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            # bit hacky
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
            self.config = AutoConfig.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map=self.device, config=self.config)
            self.name = "llama"
            
        else:
           raise Exception(f'only olmo, gpt2, llama is supported at this time') 
        self.model.to(self.device)

        self.prompts = []
        self.folder_path = folder_path

    def add_prompt(self, prompt: str, dim_corrupted_words: int, list_of_soln: list[str], descriptive_label: str, year=2000):
        # TODO might need to take care of choosing the corrupted tokens manually...
        # if (prompt[-1] != " "):
        #     prompt+=" " # in case the user did not add a space at the end of the prompt. not adding a space messes up the prompt list calculation below.
        # only need a year for relative stuff

        _, prompt_tokens = self.string_to_token_ids_and_tokens(prompt)
        len_prompt_tokens = len(prompt_tokens)

        to_corrupt = " ".join(prompt.split(" ")[:dim_corrupted_words])  # the part of the sentence we want to corrupt (the first dim_corrupted_words words)
        _, to_corr_tokens = self.string_to_token_ids_and_tokens(to_corrupt)
        len_corrupted_tokens = len(to_corr_tokens) 

        corrupted_tokens_indices = [[[i for i in range(len_corrupted_tokens)]]]
       
        custom_labels = self.add_asterisks(prompt_tokens, len_corrupted_tokens) # add asteriks to the first len_corrupted_tokens in the prompt tokens
        breaks = list(range(len_prompt_tokens))

        # add a prompt we want to test in the format of a prompt tuple:
        # (prompt, len_prompt_tokens, len_corrupted_tokens, corrupted_tokens, list_of_soln, custom_labels, breaks)
        # for example:
        # ("In 2050 there", 3, 2, [[[0, 1]]], [ [" was", " were"], [" will"]], ["In*", "2050*", "there"], [0, 1, 2], 1980)
        prompt_obj = Prompt(prompt, len_prompt_tokens, len_corrupted_tokens, corrupted_tokens_indices, list_of_soln, descriptive_label, custom_labels, breaks, year) 
                
        # print("\nPROMPT OBJ\n" + str(prompt_obj))
        self.prompts.append(prompt_obj)


    def get_prompts(self):
        return self.prompts

    def add_asterisks(self, prompt, n):
        return [word + "*" if i < n else word for i, word in enumerate(prompt)]


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

    
    def list_from_prompt(self, prompt: str):
        return prompt.split(" ")
    

    def factual_recall(self, prompt: Prompt):
        print("FACTUAL RECALL:")
        print(str(prompt.prompt))

        inputs, _ = self.string_to_token_ids_and_tokens(prompt.prompt)
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

    def corrupted_run(self, prompt: Prompt):

        print("CORRUPTED RUN: ")
        print(prompt.prompt)

        base = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device)
        
        NoiseIntervention.dim_corrupted_tokens = prompt.len_corrupted_tokens
        NoiseIntervention.DEVICE = self.device
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

        intervenable = IntervenableModel(config, self.model)
        _, counterfactual_outputs = intervenable(
            base, unit_locations={"base": (prompt.corrupted_tokens_indices)}  # defines which positions get corrupted
        )
        # TODO: counterfactual_outputs.hidden_states[-1] is sketchy, fix it!
        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1] # aditi addition
        distrib = embed_to_distrib(self.model, counterfactual_outputs.last_hidden_state, logits=False)
        top_vals(self.tokenizer, distrib[0][-1], n=10)
        return base

    def restore_run(self, prompt: Prompt, timestamp: str, run_type="default", relative_prompt_focus=" was", plot_only_block_outputs=True):
        # @param base : use the base from the related corrupted_run
        # @param run_type : if "default", just do regular plots. if "relative", compute difference betweenn all other words in all possible solns

        print("RESTORED RUN:")
        print(prompt.prompt)

        base = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.device)
        print("\nDEBUG base: " + str(base))

        filepaths=[]

        for stream in ["block_output", "mlp_activation", "attention_output"]:
            filepaths=[] # reset filepaths for each stream
            
            # don't plot anything besides block output if we dont want it
            if plot_only_block_outputs and stream != "block_output":    # only run block outputs
                continue
            elif not plot_only_block_outputs and stream == "block_output":    # run non-block outputs (mlp and attention)
                continue
        
            for i in range(len(prompt.list_of_soln)): # pylint: disable=consider-using-enumerate
                solns = prompt.list_of_soln[i]
                print("\ntense:  " + str(solns))
                if run_type=="relative":
                    if relative_prompt_focus not in solns:
                        print("skipped!")
                        continue  # check if its relative, and continue if its a tense we dont care about
                
                print("solutions: " + str(solns) +"\n")
                tokens = [self.tokenizer.encode(soln) for soln in solns]  # this provides same results as suze's tokenizer method
                # tokens = [self.tokenizer.encode(soln)[0] for soln in solns]  # this provides same results as suze's tokenizer method
                print("\n\nin restore run, tokens is:\n" + str(tokens))
                
                data = []
                for layer_i in tqdm(range(self.model.config.num_hidden_layers)):
                    for pos_i in range(prompt.len_prompt_tokens):
                        config = restore_corrupted_with_interval_config(
                            layer=layer_i,
                            device=self.device, 
                            dim_corrupted_tokens=prompt.len_corrupted_tokens,
                            stream=stream, 
                            window=1 if stream == "block_output" else 10, 
                            num_layers=self.model.config.num_hidden_layers,      
                        )
                        n_restores = len(config.representations) - 1
                        intervenable = IntervenableModel(config, self.model)

                        _, counterfactual_outputs = intervenable(
                            base,
                            [None] + [base]*n_restores,
                            {
                                "sources->base": (
                                    [None] + [[[pos_i]]]*n_restores,
                                    prompt.corrupted_tokens_indices + [[[pos_i]]]*n_restores,
                                )
                            },
                        )

                        counterfactual_outputs.last_hidden_state = counterfactual_outputs.hidden_states[-1]
                        distrib = embed_to_distrib(
                            self.model, counterfactual_outputs.last_hidden_state, logits=False
                        )

                        # can sum over multiple words' tokens instead of just one
                        prob = 0
                        for token in tokens:
                            if self.model_id == "meta-llama/Llama-3.2-1B" and len(token) == 2 and token[0] == 128000: 
                                # 128000 is <|begin_of_text|> token for llama which we want to ignore TODO don't hardcode this
                                token = [token[1]]
                            if len(token) > 1:
                                print("token is "+ str(token) + "\n")
                                # this happens for llama NEVERMIND THAT IS THE <|begin_of_text|> SO THIS SHOULD NEVER HAPPEN
                                raise Exception("token should not be more than one token")
                            else:
                                # this happens for gpt2, olmo
                                prob += distrib[0][-1][token].detach().cpu().item() # for token in tokens)

                                if (run_type == "relative"): # subtract away the other tense words
                                    subt_words = [item for j, sublist in enumerate(prompt.list_of_soln) if j != i for item in sublist] # all items we're not interested in
                                    subt_tokens = [self.tokenizer.encode(w)[0] for w in subt_words] # tokenize
                                    prob_subtr = sum(distrib[0][-1][word].detach().cpu().item() for word in subt_tokens) # sum all their probabilities                 
                                    prob = prob - prob_subtr

                        data.append({"layer": layer_i, "pos": pos_i, "prob": prob})
                df = pd.DataFrame(data) 

                os.makedirs(self.folder_path, exist_ok=True)
                soln_txt = ''.join([s.replace(' ', ',') for s in solns])[1:]
                filepath = "./"+self.folder_path+"/"+run_type+"_"+prompt.descriptive_label+"_"+soln_txt+"_"+stream+"_"+timestamp
                df.to_csv(filepath+".csv") # write csv

                self.plot(prompt, soln_txt, stream, filepath) # read csv to create a png
                filepaths.append(filepath+".png")
        
            outputfilepath="./"+self.folder_path+"/"+"combined_"+(prompt.prompt.replace(' ', '_'))+"_"+self.name+"_"+stream+timestamp+"_"+".png"
            self.merge_images_horizontally(filepaths, outputfilepath)


    def plot(self, prompt: Prompt, soln_txt: str, stream: str, filepath:str):
        df = pd.read_csv(filepath+".csv")  # read csv
        df["layer"] = df["layer"].astype(int)
        df["pos"] = df["pos"].astype(int)
        df["p("+soln_txt+")"] = df["prob"].astype(float)

        custom_labels = prompt.custom_labels
        breaks = prompt.breaks

        # change color based on stream
        hi_color = "purple" # "#660099" 
        if(stream=="mlp_activation"):
            hi_color="pink"  
        if(stream=="attention_output"):
            hi_color="#FF9900"  # orange 


        prompt_len=len(prompt.prompt)
        font_size = 6
        if prompt_len < 20:
            font_size = 14
        if prompt_len < 40:
            font_size = 12
        elif prompt_len < 60:
            font_size = 10.5
        elif prompt_len < 75:
            font_size = 8

        plot = (
            ggplot(df)
            + geom_tile(aes(x="pos", y="layer", fill="p("+soln_txt+")"))
            + scale_fill_gradient(low="white", high=hi_color, limits=(0, 1))  # Fixes 0 to light, 1 to dark
            + theme(
                figure_size=(4, 5),
                axis_text_x=element_text(rotation=90, size=9.5),
                plot_title=element_text(size=font_size), # make sure title fits
            )
            + scale_x_continuous(
                breaks=breaks,
                labels=custom_labels
            )
            + scale_y_continuous(
                breaks=list(range(self.model.config.num_hidden_layers)),  # num hidden layers in model
                labels=[str(i) for i in range(self.model.config.num_hidden_layers)]  # Convert to strings for labels
            )
            + labs(
                title=f"{prompt.prompt}",
                x="",  # Remove x-axis label
                y=f"Restored {stream} layer in {self.name}",
                fill="p("+soln_txt+")",
            )
        )

        print(df.columns)
        print(df.head())
        ggsave(
            plot, filename=filepath+".png", dpi=200 # write pdf graph # TODO: how to save as png??
        )
        # print(plot)


    def merge_images_horizontally(self, image_paths, output_path="merged.png"):
        images = [Image.open(img) for img in image_paths]  # Open images
        
        # Get total width and max height
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create blank image
        merged_image = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 0))

        # Paste images side by side
        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
            img.close()

        # delete all the extra images and csv files
        for img_path in image_paths:
            os.remove(img_path)  # Delete the individual img file
            os.remove(img_path[:-4] + ".csv")

        # Save merged image
        merged_image.save(output_path)



def restore_corrupted_with_interval_config(
    layer, device, dim_corrupted_tokens, stream="mlp_activation", window=10, num_layers=48):

    start = max(0, layer - window // 2)
    end = min(num_layers, layer - (-window // 2))

    NoiseIntervention.dim_corrupted_tokens = dim_corrupted_tokens
    NoiseIntervention.DEVICE = device

    config = IntervenableConfig(
        representations=[
            RepresentationConfig(
                0,              # layer
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

class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    dim_corrupted_tokens = None
    DEVICE = None

    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)

        if NoiseIntervention.dim_corrupted_tokens is None or NoiseIntervention.DEVICE is None:
            print("dim_corrupted_tokens and DEVICE must be set before instantiating NoiseIntervention.")

        self.noise = torch.from_numpy(
            prng(1,NoiseIntervention.dim_corrupted_tokens, embed_dim)
            ).to(NoiseIntervention.DEVICE)
        self.noise_level = 0.13462981581687927

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention(embed_dim={self.embed_dim})"


##################################################################################################
##################################################################################################

def add_prompts_for_beautiful_day_mlp_attention(tracer: CausalTracer):
    tracer.add_prompt(prompt="In 1980 on a beautiful day there", dim_corrupted_words=2, 
            list_of_soln=TENSES, descriptive_label="beautiful_1980") 
    tracer.add_prompt(prompt="In 2030 on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_2030")   
    tracer.add_prompt(prompt="In Elmsville on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_elmsville")   
    

def add_prompts_for_beautiful_day(tracer: CausalTracer):
    tracer.add_prompt(prompt="In 1980 on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_1980")  
    tracer.add_prompt(prompt="In 2000 on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_2000")  
    tracer.add_prompt(prompt="In 2020 on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_2020")  
    tracer.add_prompt(prompt="In 2030 on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_2030")   
    tracer.add_prompt(prompt="In 2050 on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_2050")
    tracer.add_prompt(prompt="In Elmsville on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_elmsville")   
    tracer.add_prompt(prompt="In summer on a beautiful day there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="beautiful_summer")   


def add_prompts_for_1980(tracer: CausalTracer):
    tracer.add_prompt(prompt="In 1980 there", dim_corrupted_words=2, 
                list_of_soln=TENSES, descriptive_label="1980_there")   
    tracer.add_prompt(prompt="On a beautiful day in 1980 there", dim_corrupted_words=6, 
                list_of_soln=TENSES, descriptive_label="beautiful_end_1980")    
    tracer.add_prompt(prompt="On a gloomy day in 1980 there", dim_corrupted_words=6, 
                list_of_soln=TENSES, descriptive_label="gloomy_end_1980")   

def add_prompts_for_relative(tracer: CausalTracer):
    tracer.add_prompt(prompt="Tomorrow on a beautiful day there", dim_corrupted_words=1, 
                        list_of_soln=TENSES, descriptive_label="beautiful_tmr")
    tracer.add_prompt(prompt="In just three hours on a beautiful day there", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="beautiful_just_three_hours")
 
def add_prompts_for_task1d(tracer: CausalTracer):
    tracer.add_prompt(prompt="The year is 1980, and COVID", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="1980_descr_covid")  
    tracer.add_prompt(prompt="The year is 1980, and WiFi", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="1980_descr_wifi")   
    tracer.add_prompt(prompt="The year is 1980, and planes", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="1980_descr_planes")  
    tracer.add_prompt(prompt="The year is 2030, and COVID", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="2030_descr_covid")  
    tracer.add_prompt(prompt="The year is 2030, and WiFi", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="2030_descr_wifi")   
    tracer.add_prompt(prompt="The year is 2030, and planes", dim_corrupted_words=4, 
                        list_of_soln=TENSES, descriptive_label="2030_descr_planes")  
    
def add_prompts_for_thirty_years_before(tracer: CausalTracer):
    tracer.add_prompt(prompt="Thirty years before 2060 on a beautiful day there", dim_corrupted_words=4, 
                            list_of_soln=TENSES, descriptive_label="30_2060_beautiful")   
    tracer.add_prompt(prompt="Thirty years before 2020 on a beautiful day there", dim_corrupted_words=4, 
                            list_of_soln=TENSES, descriptive_label="30_2020_beautiful")   
    tracer.add_prompt(prompt="Thirty years before 1980 on a beautiful day there", dim_corrupted_words=4, 
                            list_of_soln=TENSES, descriptive_label="30_1980_beautiful")  
    
def add_prompts_for_in_addition(tracer: CausalTracer):
    tracer.add_prompt(prompt="In addition on a beautiful day there",                         
        dim_corrupted_words=2, list_of_soln=TENSES, descriptive_label="in_addition")   
    tracer.add_prompt(prompt="In contrast on a beautiful day there", dim_corrupted_words=2, 
        list_of_soln=TENSES, descriptive_label="in_contrast")   
    tracer.add_prompt(prompt="In response on a beautiful day there", dim_corrupted_words=2, 
        list_of_soln=TENSES, descriptive_label="in_responset")   
    tracer.add_prompt(prompt="In conclusion on a beautiful day there", dim_corrupted_words=2, 
        list_of_soln=TENSES, descriptive_label="in_conclusion")   
    tracer.add_prompt(prompt="In summary on a beautiful day there", dim_corrupted_words=2, 
        list_of_soln=TENSES, descriptive_label="in_summary")   



def relative_2020_beautiful(tracer: CausalTracer):
    prompt="In 2020 on a beautiful day there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=2, 
                            list_of_soln=TENSES, descriptive_label="2020_beautiful_day", year=2020)
    # NOTES: make sure to set it to relative in main
    # also set the correct relative_prompt_focus term (is vs will) in the restore_run call


def add_prompts_for_compared_test(tracer: CausalTracer):
    prompt="Compared to 1980, now there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="1980_now_there")
    prompt="Compared to 2030, now there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="2030_now_there")
    prompt="Compared to 2050, now there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="2050_now_there")


def add_prompts_for_now_there(tracer: CausalTracer):
    prompt="Now there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="now_there")
    prompt="Before there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="before_there")
    prompt="Afterwards there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="afterwards_there")
    prompt="After there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="after_there")
    prompt="Now on a beautiful day there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="now_beautiful_there")
    prompt="Before on a beautiful day there"
    tracer.add_prompt(prompt=prompt, dim_corrupted_words=3, 
                            list_of_soln=TENSES, descriptive_label="before_beautiful_there")


### defs for prompts

# create all the prompts we wanna use
YEARS = [1980, 2000, 2020, 2050] 
# PROMPTS: prompt, dim words to corrupt, and a descriptive name for generated files
# NOTE!! no commas allowed in prompts. it breaks sometimes.
PROMPTS = [("In [[YEAR]] on a beautiful day there", 2, "beautiful_bkw"), 
           ("In [[YEAR]] there", 2, "there"), ("As of [[YEAR]] it", 3, "asof"),
            ("On a beautiful day in [[YEAR]] there", 6, "beautiful"),
           ("In [[YEAR]] on a gloomy day there", 2, "gloomy"), ("In [[YEAR]] on a rainy day here", 2, "rainy")]
            # ("In [[YEAR]] they ", 2, "they") --- BAD according to what it gets from factual recall and corrupted run steps. it tries to return \n ???
TENSES = [[" was", " were"], [" is", " are"], [" will"]]
    # should i also include "was" and "were" without the space??


def add_prompts_over_years(tracer: CausalTracer, years=YEARS, prompts=PROMPTS, tenses=TENSES):
    # used to generate lots of graphs
    for y in years:
        for (prompt_template, num_words_to_corrupt, descr) in prompts:
            prompt_real = prompt_template[:prompt_template.find("[[")] + str(y) + prompt_template[prompt_template.find("]]")+2:]
            descr_label = str(y) + "_" + descr 
            # dim_corrupted_words is observed to usually be len_words_prompt-1 because the year is usually the 2nd to last word
            tracer.add_prompt(prompt=prompt_real, dim_corrupted_words=num_words_to_corrupt, 
                            list_of_soln=tenses, descriptive_label=descr_label, year=y)





########################################################################################################################



def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    models = [("allenai/OLMo-1B-hf", "pyvene_causal_tracing/olmo"), ("meta-llama/Llama-3.2-1B", "pyvene_causal_tracing/llama")]
    for model, folder in models:
        tracer = CausalTracer(model_id=model, folder_path=folder) 
        plot_only_block_outputs = True

        # DO THIS: set the appropriate test

        # 1. 
        add_prompts_for_relative(tracer)
        # add_prompts_for_task1d(tracer)
        
        # # 2. 
        # add_prompts_for_beautiful_day(tracer)

        # # 3. 
        # add_prompts_for_1980(tracer)

        # # 4.
        # # DO THIS: use this to control whether you plot only residuals vs mlp/attention
        # # plot_only_block_outputs = False  
        # # add_prompts_for_beautiful_day_mlp_attention(tracer)

        # # 5. 
        # add_prompts_for_in_addition(tracer)

        # # 6.  NEVER RUN
        # # add_prompts_for_thirty_years_before(tracer)

        # # 7.  
        # add_prompts_for_now_there(tracer)

        # # 8.  
        # add_prompts_for_compared_test(tracer)

        # set runtype="relative" for relative plots
        relative=False


        print("\n\n\nSWITCHING MODELS!!!!!!!!!\n\n\n")

        # loop over every prompt to run pyvene
        for p in tracer.get_prompts():
            # part 1        
            tracer.factual_recall(prompt=p)  

            # part 3: regular run over all tenses
            # tracer.restore_run(prompt=p, timestamp=timestamp, plot_only_block_outputs=plot_only_block_outputs)


if __name__ == "__main__":
    main()

