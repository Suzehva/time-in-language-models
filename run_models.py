# import torch
# from transformers import pipeline


# pipe = pipeline(
#     "text-generation", 
#     model=model_id, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

# pipe("The key to life is")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

input_text = "Once upon a time,"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
output = model.generate(**inputs, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)



# def main():
#     # todo

# if '__name__' == main:
#     main()


