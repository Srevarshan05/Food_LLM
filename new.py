from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define the model name and your Hugging Face access token (replace with your own token)
model_name = "BioGPT"  # You can change this to a different model if you prefer
access_token = "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"  # Replace with your Hugging Face access token

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=access_token)

# Sample medical-related prompt
prompt = """
You are a medical assistant. Please provide a detailed analysis of the following medical case:
- Patient: John Doe
- Age: 45
- Condition: Hypertension, Type 2 Diabetes
- Medications: Metformin, Amlodipine

Please give advice on managing this condition based on current medical guidelines.
"""

# Encode the input prompt
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

# Generate a response from the model
with torch.no_grad():  # Turn off gradient calculation for inference
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)

# Decode the model's response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the model's output
print("Model Response:\n", response)
