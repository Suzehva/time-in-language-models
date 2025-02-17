from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

file = "task1a/task1a.data"
generation_length = 10
with open(file, "r") as file:
    for line in file:
        line.strip()
        input_text = line

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate output
        output = model.generate(**inputs, max_length=generation_length)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print(generated_text)
        print("\n")


print("ALL DONE!")

