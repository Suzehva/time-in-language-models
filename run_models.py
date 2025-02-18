from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import datetime
import csv

class MultiModelManager:
    def __init__(self, device=None):
        # Initialize the device (CPU or GPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        
        # Dictionary to hold multiple models and tokenizers
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_id: str):
        # Load a model and its tokenizer if not already loaded
        if model_id not in self.models:
            print(f"Loading model {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
        else:
            print(f"Model {model_id} already loaded.")
    
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
                    
                    # Generate only 1 token
                    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    # TODO: want to run forward function instead (look at logits)
                    # attach the expected output to input prompt and use the forward pass to find its probability  (teacher forcing)
                    
                    # Decode only the newly generated token
                    generated_token = tokenizer.decode(output[0][-1], skip_special_tokens=True)

                    # Store structured data (model, input, output)
                    generated_data.append((model_id, input_text, generated_token))

        return generated_data  # Return structured data


    def store_output_to_csv(self, generated_data, output_directory: str):
        """
        Saves generated tokens to a CSV file with model, input, and output info.

        :param generated_data: List of tuples (model_id, input_text, generated_token)
        :param output_directory: Directory where the CSV file will be saved
        """
        # Ensure the directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"generated_output_{timestamp}.csv"
        output_filepath = os.path.join(output_directory, output_filename)
        
        # Write the data to a CSV file
        with open(output_filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["model", "input_text", "generated_token"])
            
            # Write data rows
            for model_id, input_text, generated_token in generated_data:
                writer.writerow([model_id, input_text, generated_token])
        
        print(f"Generated tokens have been saved to {output_filepath}.")




def main():
    model_ids = [
        # "meta-llama/Llama-3.2-1B",
        "allenai/OLMo-1B-hf",
        "google/gemma-2-2b",
    ]
    
    # Create the MultiModelManager instance
    manager = MultiModelManager()

    # Load the models
    for model_id in model_ids:
        manager.load_model(model_id)

        # Example of generating text from file
        generated_texts = manager.generate_text_from_file(model_id, "task1a/task1a.data", max_new_tokens=1)
        
        # TODO: look into sampling algorithm (probably using greedy right now)

        # TODO: call model's forward function to get logits (loss, logits are both returned from hugging face)
        # use probability distr instead of top token
        # track probabiliy change across is/was/will/etc

        # Now store the generated output to a file
        manager.store_output_to_csv(generated_texts, "task1a_"+model_id)

if __name__ == "__main__":
    main()


