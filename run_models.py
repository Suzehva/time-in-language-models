from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import re

class MultiModelManager:
    def __init__(self, device=None):
        # Initialize the device (GPU or MPS [for apple silicon] or CPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f'Using device: {self.device}')
        
        # Dictionary to hold multiple models and tokenizers
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_id: str):
        # Load a model and its tokenizer if not already loaded
        if model_id not in self.models:
            print(f"Loading model {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # TODO get gemma running!? low_cpu_mem_usage=True  # reduce the precision for gemma to reduce memory usage
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device) 

            # Explicitly limit batch size (if applicable) -- for gemma bc otherwise it gets stuck
            if hasattr(model, "config"):
                model.config.max_batch_size = 1  # Some models use this for internal batching

            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
        else:
            print(f"Model {model_id} already loaded.")
    
    def generate_text_from_file(self, model_id: str, filename: str, max_new_tokens, years_included=False):
    def generate_text_from_file_simple(self, model_id: str, filename: str, max_new_tokens):
        """
        Reads input text from a file, generates one token per input, and returns structured data.

        :param model_id: The model ID being used
        :param filename: The file containing input texts
        :return: List of tuples (model_id, input_text, generated_token)
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not loaded yet. Please load it first.")

        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]

        generated_data = []  # List of tuples (model_id, input_text, generated_token)

        with open(filename, "r") as file:
            for line in file:
                input_text = line.strip()
                if input_text:
                    inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
                    
                    # Generate only 1 token
                    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    
                    # Decode only the newly generated token
                    generated_token = tokenizer.decode(output[0][-max_new_tokens:], skip_special_tokens=True)

                    # Store structured data (model, input, output)
                    generated_data.append((model_id, input_text, generated_token))

        return generated_data  # Return structured data

    def generate_text_from_file(self, model_id: str, filename: str, max_new_tokens):
        """
        Reads input text from a file, generates one token per input, and returns structured data.

        :param model_id: The model ID being used
        :param filename: The file containing input texts
        :return: List of tuples (model_id, input_text, generated_token)
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not loaded yet. Please load it first.")

        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]

        generated_data = []  # List of tuples (model_id, input_text, generated_token)

        with open(filename, "r") as file:
            for line in file:
                input_text = line.strip()
                parts = input_text.split(',,,')
                
                input_year = parts[1].strip()
                input_text = parts[0].strip()

                if input_text:
                    inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
                    #Structure: {
                    # 'input_ids': tensor([[128000,  15724,    358,   9687,    922,   1380,    380,   1105, 13, 578,   1060,    374]], device='mps:0'), 
                    # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='mps:0')
                    # }
                    input_ids = inputs["input_ids"]

                    generated_tokens = []
                    token_logits = []

                    for _ in range(max_new_tokens):
                        outputs = model(**inputs)  # Forward pass to get logits
                        #print(outputs)
                        # output.logits has shape (batch_size, sequence_length, config.vocab_size)
                        logits = outputs.logits[:, -1, :]  # Get logits for last token
                        #print(logits) # tensor([[10.6421,  8.9314,  6.6964,  ...,  0.7085,  0.7083,  0.7086]], device='mps:0', grad_fn=<SliceBackward0>)
                        #print(len(logits[0])) # 128256
                        #print(model.config.vocab_size) # # 128256
                        token_id = torch.argmax(logits, dim=-1)  # Select most probable token; look at last dimension bc that is vocab size
                        #print(token_id) # e.g. tensor([264], device='mps:0')

                        generated_tokens.append(token_id.item())
                        logits_processed = logits.squeeze(0).cpu().detach().numpy() # NumPy does not work with GPU tensors so we move to cpu
                        token_logits.append(logits_processed.tolist())

                        # Append new token to input for next step
                        input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)
                        inputs = {"input_ids": input_ids}

                        # Stop if we generate an EOS token
                        if token_id == tokenizer.eos_token_id:
                            break
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # token_logits is an array where every entry has all the logits for one token generated. Every entry has 128256 numbers, which is the size of our vocab
                if years_included:
                    generated_data.append((model_id, input_year, input_text, generated_text, token_logits))
                else:
                    generated_data.append((model_id, input_text, generated_text, token_logits))

                # IN CASE YOU WANT TO USE MODEL.GENERATE
                #inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
                #generate_output = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, top_k=0, top_p=1.0, temperature=1.0, no_repeat_ngram_size=0)
                #generate_output = tokenizer.decode(generate_output[0][-max_new_tokens:], skip_special_tokens=True)

                # notes
                # attach the expected output to input prompt and use the forward pass to find its probability  (teacher forcing)
                # use probability distr instead of top token
                # track probabiliy change across is/was/will/etc

        return generated_data

    def learn_relevant_vocab(self, model_id: str, words_of_interest:str):
        """
        Returns a list of tuples where every tuple has three entries:
        1. the input word
        2. the input word split up into its tokens
        3. the indices of these tokens (= index into vocab)
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} is not loaded yet. Please load it first.")

        tokenizer = self.tokenizers[model_id]
        token_id_map = []
        
        # note: there could me multiple token ids for a word
        for word in words_of_interest:
            # get the id's that correspond to a word and its subwords
            token_ids = tokenizer(word, add_special_tokens=False)["input_ids"] 
            # get the pieces that each id correspond to. 
            # helps understand how the word was split when making tokens
            token_pieces = tokenizer.convert_ids_to_tokens(token_ids) 

            token_id_map.append((word, token_pieces, token_ids))
        return token_id_map


    def store_output_to_csv(self, generated_data, output_directory: str, headers=None):
        """
        Saves generated tokens to a CSV file with model, input, and output info.

        :param generated_data: List of tuples (model_id, input_text, generated_token)
        :param output_directory: Directory where the CSV file will be saved
        """
        if not generated_data:
            print("No data to save.")
            return

        # Ensure the directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"generated_output_{timestamp}.csv"
        output_filepath = os.path.join(output_directory, output_filename)

        # Determine headers
        if headers is None:
            if isinstance(generated_data[0], dict):
                headers = list(generated_data[0].keys())  # Extract keys from the first dictionary
            else:
                headers = [f"column_{i+1}" for i in range(len(generated_data[0]))]  # Generic column names

        # Write the data to a CSV file
        with open(output_filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)  # Write the header row

            for row in generated_data:
                if isinstance(row, dict):
                    writer.writerow([row.get(header, "") for header in headers])  # Extract values in header order
                else:
                    writer.writerow(row)  # Assume it's a list/tuple and write directly

        print(f"Generated data has been saved to {output_filepath}.")
        return output_filepath

def run_task_1a():
    max_new_tokens = 1
    input_data_path = 'task1a/task1a.data'

def run_task_1b():
    max_new_tokens = 5
    input_data_path = 'task1b/task1b.data'

def run_task_1c():
    max_new_tokens = 10
    input_data_path = 'task1c/task1c.data'

def run_task_1d():
    max_new_tokens = 10
    input_data_path = 'task1d/task1d.data'

def run_task_2a(manager, model_id):
    max_new_tokens = 10
    input_data_path = 'task2a/task2a.data'
    solns_data_path = 'task2a/task2a-with-solns.data'
    # get next token generation and its logits
    generated_texts = manager.generate_text_from_file_simple(model_id=model_id, filename=input_data_path, max_new_tokens=max_new_tokens) 
    generated_path = manager.store_output_to_csv(generated_texts, "task2a/" + model_id)
    
    return generated_path, solns_data_path


import pandas as pd
def test_task_2a(generated_path, solns_path, model_id, manager):
    
    # Load solns
    solutions_dict = {}
    with open(solns_path, "r") as file:
        for line in file:
            if "," in line:
                prompt, expected_answer = line.rsplit(",", 1)  # Split at last comma
                solutions_dict[prompt.strip()] = expected_answer.strip()
    
    # Load the AI-generated CSV
    df = pd.read_csv(generated_path)
    print(df.columns)
    # Convert to dictionary { input_text: generated_token }
    # column_1 = model column_2 = input text, column_3 = gen token
    model_dict = dict(zip(df["column_2"], df["column_3"]))

    print('SOLNS DICT\n\n' , solutions_dict)
    print('MODEL DICT\n\n' , model_dict)
    # Function to compare answers
    def evaluate_answers(solutions_dict, model_dict):
        results = []
        for prompt, expected in solutions_dict.items():
            generated = model_dict.get(prompt, "N/A")
            
            # ADITI!! TODO continue here

            expected_tokens = expected.lower().split()
            generated_lower = generated.lower()
            match = any(token in generated_lower for token in expected_tokens)
            if match:
                match_status = "🟢 Match"
            else:
                match_status = "🔴 Incorrect"
        
            results.append((prompt, expected, generated, match_status))
    
        return pd.DataFrame(results, columns=["Prompt", "Expected", "Generated", "Status"])

    # Run evaluation
    report = evaluate_answers(solutions_dict, model_dict)

    # Save or display report
    manager.store_output_to_csv(report, "task2a/" + model_id + "_report")

def run_task_2b():
    max_new_tokens = 10
    input_data_path = 'task2b/task2b.data'
    

# ADITI's IMPLM OF MAIN
def main():
    model_ids = [
        "meta-llama/Llama-3.2-1B",
        "allenai/OLMo-1B-hf",
        # "google/gemma-2-2b"
    ]
    
    # Create the MultiModelManager instance
    manager = MultiModelManager()

    for model_id in model_ids:
        manager.load_model(model_id)

        # task-specific
        gen, soln = run_task_2a(manager, model_id)
        test_task_2a(gen, soln, model_id, manager)




# SUZE's IMPLM OF MAIN
'''


    def create_probability_plots_1a(self, task1a_result: List[Tuple[str, int, str, str, List[Tuple[str, float, float]]]],
                            output_dir: str, plot_softmax=True):
        """
        Create separate plots for each sweep of years from 1950-2050 in the task1a_result data.
        
        Args:
            task1a_result: List of tuples containing 
                        (model_id, input_year, input_text, generated_text, relevant_words_logits)
                        where relevant_words_logits is a list of (word, logit, softmaxed_logit) tuples
            output_dir: Directory to save plot files
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Group data by sweep
        sweeps = []
        current_sweep = []
        prev_year = None
        
        for entry in task1a_result:
            #print(entry)
            model_id, input_year, input_text, generated_text, relevant_words_logits = entry
            
            # Check if we've started a new sweep
            if prev_year is not None and int(input_year) == 1950:
                # We've wrapped around to a new sweep
                if current_sweep:
                    sweeps.append(current_sweep)
                    current_sweep = []
            
            current_sweep.append((model_id, input_year, input_text, generated_text, relevant_words_logits))
            prev_year = input_year
        
        # Add the last sweep
        if current_sweep:
            sweeps.append(current_sweep)
        
        # Create a plot for each sweep
        for sweep_idx, sweep_data in enumerate(sweeps):
            self.create_sweep_plot(sweep_data, sweep_idx, output_dir, plot_softmax=plot_softmax)
            
        

    def create_sweep_plot(self, sweep_data, sweep_idx, output_dir, plot_softmax):
        """Create a plot for a single sweep of the year range."""
        # Initialize word probability data
        word_probs = {}
        years = []
        
        # Extract data for each year
        for model_id, input_year, input_text, generated_text, relevant_words_logits in sweep_data:
            years.append(input_year)
            
            # Store probabilities for each word
            if plot_softmax:
                for word, _, prob in relevant_words_logits:
                    if word not in word_probs:
                        word_probs[word] = []
                    word_probs[word].append(prob)
            else:
                # plot raw digits
                for word, raw, _ in relevant_words_logits:
                    if word not in word_probs:
                        word_probs[word] = []
                    word_probs[word].append(raw)

        
        # Sort years and corresponding probabilities to ensure proper ordering
        sorted_indices = np.argsort(years)
        years = [years[i] for i in sorted_indices]
        
        for word in word_probs:
            word_probs[word] = [word_probs[word][i] for i in sorted_indices]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot each word with a different color
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, (word, probs) in enumerate(word_probs.items()):
            color = colors[i % len(colors)]
            plt.plot(years, probs, marker='o', label=word, color=color, linewidth=2)
        
        # Customize the plot
        plt.xlabel('Year')
        plt.ylabel('Logit softmaxed relative to other verbs')
        input_text_general = re.sub(r'\b\d{4}\b', '[year]', input_text)
        plt.title(f'Word Probabilities Over Time - prompt: {input_text_general} - model: {model_id}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set x-axis ticks with appropriate intervals
        if len(years) > 20:
            # If we have many years, show every 5th or 10th year
            step = max(1, len(years) // 10)
            tick_positions = years[::step]
            plt.xticks(tick_positions, rotation=45)
        else:
            # If we have fewer years, show all of them
            plt.xticks(years, rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"word_probabilities_sweep_{sweep_idx+1}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Created plot for sweep {sweep_idx+1}: {filepath}")



def main():
    model_ids = [
        "meta-llama/Llama-3.2-1B",
        "allenai/OLMo-1B-hf",
        "google/gemma-2-2b"
    ]

    # task : max num of new tokens (max_new_tokens)
    # tasks = {  
    #     "task1a":1,
    #     "task1b":5,
    #     "task1c":10,
    #     "task1d":10
    # }
    
    # Create the MultiModelManager instance
    manager = MultiModelManager()

    for model_id in model_ids:
        # load model
        manager.load_model(model_id)

        # task 1a:
        input_data_path = 'task1a/task1a.data'

        # get next token generation and its logits
        generated_texts = manager.generate_text_from_file(model_id=model_id, filename=input_data_path, max_new_tokens=1, years_included=True) # only generate 1 token
        #manager.store_output_to_csv(generated_texts, "task1a/" + model_id)

        # get indices of words we care about 
        relevant_words = ["was", "will", "is", "were", "are"]
        relevant_words_info = manager.learn_relevant_vocab(model_id=model_id, words_of_interest=relevant_words)

        # for every entry, get list of logits for the relevant words
        task1a_result = []
        task1a_result_total = {}
        task1a_result_total_list = []
        # has tuples with model_id, input text, generated text, list of tuples where (word, logits, logits_softmaxed)
        for _, input_year_from_entry, input_text_from_entry, generated_text_from_entry, token_logits_from_entry in generated_texts:
            if input_year_from_entry not in task1a_result_total:
                task1a_result_total[input_year_from_entry] = {}
                task1a_result_total[input_year_from_entry]['was/were'] = 0
                task1a_result_total[input_year_from_entry]['is/are'] = 0
                task1a_result_total[input_year_from_entry]['will'] = 0
                task1a_result_total[input_year_from_entry]['was/were_soft'] = 0
                task1a_result_total[input_year_from_entry]['is/are_soft'] = 0
                task1a_result_total[input_year_from_entry]['will_soft'] = 0


            # compute list of relevant logits using token_ids
            relevant_words_logits = []
            for word, _, token_ids in relevant_words_info:
                logits = 1
                for i, token_id in enumerate(token_ids):
                    # conditional probability so we multiply 
                    logits *= token_logits_from_entry[i][token_id]
                relevant_words_logits.append((word, logits))
            
            # Extract logits and apply softmax
            raw_logits = np.array([logit for _, logit in relevant_words_logits])
            softmaxed_logits = np.exp(raw_logits) / np.sum(np.exp(raw_logits))

            # Combine the logits for "was" and "were", and "is" and "are"
            combined_logits = {}
            for word, raw_logit, softmaxed_logit in zip([w[0] for w in relevant_words_logits], raw_logits, softmaxed_logits):
                if word in ['was', 'were']:
                    combined_word = 'was/were'
                    if combined_word not in combined_logits:
                        combined_logits[combined_word] = {'raw': 0, 'softmax': 0}
                    combined_logits[combined_word]['raw'] += raw_logit
                    combined_logits[combined_word]['softmax'] += softmaxed_logit
                    task1a_result_total[input_year_from_entry][combined_word] += raw_logit

                elif word in ['is', 'are']:
                    combined_word = 'is/are'
                    if combined_word not in combined_logits:
                        combined_logits[combined_word] = {'raw': 0, 'softmax': 0}
                    combined_logits[combined_word]['raw'] += raw_logit
                    combined_logits[combined_word]['softmax'] += softmaxed_logit
                    task1a_result_total[input_year_from_entry][combined_word] += raw_logit
                else:
                    # Keep other words as they are
                    if word not in combined_logits:
                        combined_logits[word] = {'raw': 0, 'softmax': 0}
                    combined_logits[word]['raw'] += raw_logit
                    combined_logits[word]['softmax'] += softmaxed_logit
                    task1a_result_total[input_year_from_entry][word] += raw_logit
                
            # Process the combined logits to compute averages for "softmaxed_logit" --> fo one year entry!
            final_relevant_words_logits = [
                (word, 
                combined_data['raw'].item(), 
                combined_data['softmax'].item())
                for word, combined_data in combined_logits.items()
            ]
            # Add the updated relevant words logits with combined results to the task1a_result list
            task1a_result.append((model_id, input_year_from_entry, input_text_from_entry, generated_text_from_entry, final_relevant_words_logits))
        # task1a_result will now contain the updated list with combined logits for "was/were" and "is/are"
        manager.store_output_to_csv(task1a_result, "task1a/" + model_id + "_logits")
        manager.create_probability_plots_1a(task1a_result, "task1a/", plot_softmax=False)

        #print(task1a_result_total) # this still has correct number but not when gone through softmax
        for year, _ in task1a_result_total.items():
            arr_to_softmax = [task1a_result_total[year]['was/were'], task1a_result_total[year]['is/are'], task1a_result_total[year]['will'] ]
            exp_logits = np.exp(arr_to_softmax)
            softmaxed_logits = exp_logits / np.sum(exp_logits)
            # print(arr_to_softmax)
            # print(exp_logits)
            # print(softmaxed_logits)
            task1a_result_total[year]['was/were_soft'], task1a_result_total[year]['is/are_soft'], task1a_result_total[year]['will_soft'] = softmaxed_logits[0], softmaxed_logits[1], softmaxed_logits[2]
        for year, d in task1a_result_total.items():
            final_relevant_words_logits_total = [
                ('was/were', d['was/were'].item(), d['was/were_soft'].item()), ('is/are', d['is/are'].item(), d['is/are_soft'].item()), ('will', d['will'].item(), d['will_soft'].item())
            ]
            task1a_result_total_list.append((model_id, year, "[all prompts]", "-", final_relevant_words_logits_total))
        # print(task1a_result_total_list[0])
        # print(task1a_result_total_list[-1])
        manager.create_probability_plots_1a(task1a_result_total_list, "task1a/", plot_softmax=False)


        return 1

        # task 1b: TODO

        # task 1c
        input_data_path = 'task1c/task1c.data'

        # get next token generation and its logits
        generated_texts = manager.generate_text_from_file(model_id=model_id, filename=input_data_path, max_new_tokens=2, years_included=False) # only generate 2 tokens, one for each

        # get indices of words we care about 
        relevant_words = [str(year) for year in range(1400, 2201)]
        relevant_words_info = manager.learn_relevant_vocab(model_id=model_id, words_of_interest=relevant_words)

        # TODO
      


