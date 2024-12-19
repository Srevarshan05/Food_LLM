from flask import Flask, render_template, request, redirect, url_for
from paddleocr import PaddleOCR
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/images/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the CSV file containing user data
CSV_PATH = "data.csv"

# Hugging Face biomedical model (e.g., BioBERT)
LLM_MODEL = "dmis-lab/biobert-base-cased-v1.1"

# Initialize Hugging Face tokenizer and model for biomedical data
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Perform OCR on the uploaded image
            result = ocr.ocr(file_path, cls=True)
            extracted_text = "\n".join([line[1][0] for line in result[0]])

            # Read the CSV file
            data = pd.read_csv(CSV_PATH)
            person_data = data.iloc[0].to_dict()  # Assuming the first row is the user

            # Prepare the LLM prompt
            prompt = f"""
            Dear {person_data['Name']},

            Based on your comprehensive health data and the nutritional information extracted from the food label, here is a detailed analysis of how this product aligns with your dietary needs and health profile.

            **Extracted Nutritional Information:**
            {extracted_text}

            **Health Profile Overview:**
            - Age: {person_data['Age']}
            - Gender: {person_data['Gender']}
            - Allergies: {person_data['Allergies']}
            - Current Health Conditions: {person_data['Current Health Conditions']}
            - Chronic Diseases: {person_data['Chronic Diseases']}

            **Questions for Consideration**:
            - Identify specific components or nutrients in this product that could potentially interact with the user’s health conditions or allergies.
            - Analyze how the nutritional composition complements or conflicts with their dietary restrictions.
            - Based on the extracted nutritional details, evaluate the suitability of this food product for consumption by the user.
            - Suggest general guidelines for portion sizes or intake frequency that align with their overall health profile.

            Please provide a professional, evidence-based evaluation considering these factors.
            """

            # Tokenize and prepare input for the model
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)

            # Generate a response from the model
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=1024, num_beams=5, early_stopping=True)

            # Decode the model's output
            llm_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Display the results
            return render_template(
                "index.html",
                uploaded_image=file.filename,
                extracted_text=extracted_text,
                llm_output=llm_output,
                user_name=person_data["Name"]
            )

    return render_template("index.html", uploaded_image=None, extracted_text=None, llm_output=None)

if __name__ == "__main__":
    app.run(debug=True)
