import ollama
from paddleocr import PaddleOCR

# Initialize PaddleOCR (English language)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Replace with your image path
image_path = 'C:\\Users\\Srevarshan\\OneDrive\\Desktop\\OCR new\\Ex3.jpg'

# Run OCR on the image
result = ocr.ocr(image_path, cls=True)

# Extract the text from the OCR result
extracted_text = "\n".join([line[1][0] for line in result[0]])

# Specify the model stored in your .ollama folder
model_name = "llama3.2"  # Use your actual model name here

# Prepare the prompt for refining the extracted text
prompt = f"""
Here is the extracted text from an image of nutritional information. Please refine it into a more readable format 
and structure it as nutritional facts. Remove any irrelevant or unclear information, and organize it clearly 
to resemble the format used on nutrition labels.

Extracted Text:
{extracted_text}

Refined Nutritional Facts:
"""

# Using the chat method to pass the prompt to your local Llama model
try:
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    # Extract and print the refined answer
    refined_text = response.get('message', {}).get('content', 'No refined output found')
    print("Refined Nutritional Facts:")
    print(refined_text)

except Exception as e:
    print("An error occurred:", e)
