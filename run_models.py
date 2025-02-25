from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import datetime
import csv
import numpy as np

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


    def store_output_to_csv(self, generated_data, output_directory: str, headers=None, delim=","):
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
            writer = csv.writer(csvfile, delimiter=delim) # allows usage of different delimiters
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
    # define these constants for task 2a !!
    MAX_NEW_TOKENS = 25
    input_data_path = 'task2a/task2a.data'
    solns_data_path = 'task2a/task2a-with-solns.data'

    # get next token generation and its logits
    generated_texts = manager.generate_text_from_file_simple(model_id=model_id, filename=input_data_path, max_new_tokens=MAX_NEW_TOKENS) 
    generated_path = manager.store_output_to_csv(generated_texts, "task2a/" + model_id, delim="|")

    return generated_path, solns_data_path

def test_task_2a(generated_path, solns_path, model_id, manager):
    # Load solutions
    solutions_dict = {}
    with open(solns_path, "r") as file:
        for line in file:
            if "|" in line:
                prompt, expected_answer = line.rsplit("|", 1)
                solutions_dict[prompt.strip()] = expected_answer.strip()

    # Load the AI-generated CSV manually
    model_dict = {}
    with open(generated_path, "r") as file:
        lines = file.readlines()
        headers = lines[0].strip().split("|")
        column_2_index = headers.index("column_2")
        column_3_index = headers.index("column_3")
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split("|")
            if len(parts) > max(column_2_index, column_3_index):
                model_dict[parts[column_2_index]] = parts[column_3_index]

    # Function to compare answers
    def evaluate_answers(solutions_dict, model_dict):
        results = []
        for prompt, expected in solutions_dict.items():
            generated = model_dict.get(prompt, "N/A")
            expected_tokens = expected.lower().split()
            generated_lower = generated.lower()
            match = any(token in generated_lower for token in expected_tokens)
            match_status = "🟢 Match" if match else "🔴 Incorrect"
            results.append({"Prompt": prompt, "Expected": expected, "Generated": generated, "Status": match_status})
        return results

    # Run evaluation
    report = evaluate_answers(solutions_dict, model_dict)
    print("REPORT", report)

    csv_headers = ["Prompt", "Expected", "Generated", "Status"]
    manager.store_output_to_csv(report, "task2a/" + model_id + "_report", headers=csv_headers, delim="|")

def run_task_2b(manager, model_id):

    # define these constants for task 2a !!
    MAX_NEW_TOKENS = 25
    input_data_path = 'task2b/task2b.data'
    solns_data_path = 'task2b/task2b-with-solns.data'

    # get next token generation and its logits
    generated_texts = manager.generate_text_from_file_simple(model_id=model_id, filename=input_data_path, max_new_tokens=MAX_NEW_TOKENS) 
    generated_path = manager.store_output_to_csv(generated_texts, "task2b/" + model_id, delim="|")

    # CONTINUE!! 
    return generated_path, solns_data_path


def test_task_2b(generated_path, solns_path, model_id, manager):
    # Load solutions
    solutions_dict = {}
    with open(solns_path, "r") as file:
        for line in file:
            if "|" in line:
                prompt, expected_answer = line.rsplit("|", 1)
                solutions_dict[prompt.strip()] = expected_answer.strip()

    # Load the AI-generated CSV manually
    model_dict = {}
    with open(generated_path, "r") as file:
        lines = file.readlines()
        headers = lines[0].strip().split("|")
        column_2_index = headers.index("column_2")
        column_3_index = headers.index("column_3")
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split("|")
            if len(parts) > max(column_2_index, column_3_index):
                model_dict[parts[column_2_index]] = parts[column_3_index]

    # Function to compare answers
    def evaluate_answers(solutions_dict, model_dict):
        results = []
        for prompt, expected in solutions_dict.items():
            generated = model_dict.get(prompt, "N/A")
            expected_tokens = expected.lower().split()
            generated_lower = generated.lower()
            match = any(token in generated_lower for token in expected_tokens)
            match_status = "🟢 Match" if match else "🔴 Incorrect"
            results.append({"Prompt": prompt, "Expected": expected, "Generated": generated, "Status": match_status})
        return results

    # Run evaluation
    report = evaluate_answers(solutions_dict, model_dict)
    print("REPORT", report)

    csv_headers = ["Prompt", "Expected", "Generated", "Status"]
    manager.store_output_to_csv(report, "task2b/" + model_id + "_report", headers=csv_headers, delim="|")


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

        # task-specific for 2a
        gen, soln = run_task_2b(manager, model_id)
        test_task_2b(gen, soln, model_id, manager)




# SUZE's IMPLM OF MAIN
'''
def main():
    model_ids = [
        "meta-llama/Llama-3.2-1B",
        #"allenai/OLMo-1B-hf",
        # "google/gemma-2-2b"
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
        generated_texts = manager.generate_text_from_file(model_id=model_id, filename=input_data_path, max_new_tokens=1) # only generate 1 token
        manager.store_output_to_csv(generated_texts, "task1a/" + model_id)

        # get indices of words we care about 
        relevant_words = ["was", "will", "is", "were", "are"]
        relevant_words_info = manager.learn_relevant_vocab(model_id=model_id, words_of_interest=relevant_words)

        # for every entry, get list of logits for the relevant words
        task1a_result = []
        # has tuples with model_id, input text, generated text, list of tuples where (word, logits, logits_softmaxed)
        for _, input_text_from_entry, generated_text_from_entry, token_logits_from_entry in generated_texts:
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

            # add softmaxed logits to list
            relevant_words_logits = [
                (word, logit, softmaxed_logit.item())  # Convert np.float64 to a regular Python float
                for (word, logit), softmaxed_logit in zip(relevant_words_logits, softmaxed_logits)
            ]

            task1a_result.append((model_id, input_text_from_entry, generated_text_from_entry, relevant_words_logits))
        manager.store_output_to_csv(task1a_result, "task1a/" + model_id + "_logits")
            
            

        
     '''

if __name__ == "__main__":
    main()




   
        # for task, max_new_tokens in tasks.items():
        #     # generate next token(s) from a file of prompts
        #     task_prompts = task + "/" + task + ".data"  # eg. task1a/task1a.data

        #     generated_texts = manager.generate_text_from_file(model_id, task_prompts, max_new_tokens=max_new_tokens)
            
        #     # TODO: look into sampling algorithm (probably using greedy right now)
        #     # TODO: add option to use model.generator OR forward etc?

        #     # store generated output to file TURNED OF FOR DEBUGGING
        #     file = task + "/" + model_id
        #     manager.store_output_to_csv(generated_texts, file) # it automatically saves to the model's output folder

