import ollama

# Specify the model stored in your .ollama folder (replace 'llama3.2' with the actual model name)
model_name = "llama3.2"  # Replace with your actual model's name

# Ask the model the question about the capital of France
response = ollama.chat(model=model_name, messages=[{"role": "user", "content": "what is capital of India?"}])

# Extract and print only the answer from the response
answer = response.get('message', {}).get('content', 'No answer found')
print(answer)
