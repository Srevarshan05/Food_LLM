import ollama
import pandas as pd
from paddleocr import PaddleOCR

# Initialize PaddleOCR (English language)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path for your image
image_path = 'C:\\Users\\Srevarshan\\OneDrive\\Desktop\\OCR new\\ingred4.jpg'

# Run OCR on the image
result = ocr.ocr(image_path, cls=True)

# Extract the text from the OCR result
extracted_text = "\n".join([line[1][0] for line in result[0]])

# Path to the CSV file
csv_path = "C:\\Users\\Srevarshan\\OneDrive\\Desktop\\OCR new\\data.csv"

# Read the CSV file
data = pd.read_csv(csv_path)

# Assuming we are only interested in the first person's data in the CSV
person_data = data.iloc[0].to_dict()

# Prepare the prompt with more personalized, accurate, and professional language
prompt = f"""
Dear {person_data['Name']},

Based on the comprehensive health details provided and the nutritional information extracted from a food label, I am here to offer a thorough evaluation and provide insights that will assist you in making well-informed decisions regarding your health and nutrition.

**Your Health Overview:**
- **Age:** {person_data['Age']} years
- **Gender:** {person_data['Gender']}
- **Medical History:** {person_data['Medical History']}
- **Current Health Conditions:** {person_data['Current Health Conditions']}
- **Allergies:** {person_data['Allergies']}
- **Current Medications:** {person_data['Current Medications']}
- **Emergency Contact:** {person_data['Emergency Contact']}
- **Primary Care Physician:** {person_data['Primary Care Physician']}
- **Blood Type:** {person_data['Blood Type']}
- **Height:** {person_data['Height']} cm
- **Weight:** {person_data['Weight']} kg
- **Blood Pressure:** {person_data['Blood Pressure']}
- **Chronic Diseases:** {person_data['Chronic Diseases']}
- **Surgical History:** {person_data['Surgical History']}
- **Immunization History:** {person_data['Immunization History']}
- **Family Medical History:** {person_data['Family Medical History']}
- **Social History:** {person_data['Social History']}

**Nutritional Information from Food Label:**
{extracted_text}

Given your current health status and the nutritional composition of the food product in question, please provide a comprehensive analysis on the following:

1. **Nutrient-Health Interaction:** How do the ingredients in this food correlate with your health conditions or medications? What potential benefits or concerns should you be aware of?
   
2. **Food Suitability:** In light of your health conditions, are there any ingredients in this food that may exacerbate existing issues such as allergies or chronic diseases?

3. **Portion Recommendations:** Given your unique health profile, what portion sizes of this food would be suitable for consumption, ensuring your health remains well-managed?

4. **Potential Risks:** If certain ingredients or nutritional elements pose risks due to your medical conditions, what are the specific risks, and how can they be mitigated?

Your health is of utmost importance, and the insights provided here will be based on the thorough analysis of your health data and the food’s nutritional profile. This is a detailed evaluation meant to support your decisions regarding food consumption, ensuring they are aligned with your well-being goals.

"""

# Specify the model stored in your .ollama folder
model_name = "llama3.2"  # Use your actual model name here

# Use the chat method to pass the prompt to your local Llama model
try:
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    # Extract and print the refined answer
    refined_text = response.get('message', {}).get('content', 'No refined output found')
    print("Personalized Health Insights:")
    print(refined_text)

except Exception as e:
    print("An error occurred:", e)
