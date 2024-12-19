Nutritional Facts Extraction Using PaddleOCR and NVIDIA BioBERT
This project leverages advanced Optical Character Recognition (OCR) and Natural Language Processing (NLP) technologies to extract nutritional information from product images with high accuracy. By combining PaddleOCR for text recognition and NVIDIA's BioBERT for semantic analysis, this system accurately identifies and processes nutritional facts from a variety of product labels, regardless of layout complexity.

Features
Text Extraction with PaddleOCR: Accurately extracts text from images, even from complex or non-standard layouts.
Regex-Based Filtering: Identifies nutritional information (e.g., calories, protein, fat) using a fine-tuned regular expression engine.
NVIDIA BioBERT Integration: Enhances text processing and classification for improved semantic understanding of nutritional facts.
Dockerized Workflow: Runs seamlessly on any environment with GPU acceleration via NVIDIA Docker.
Extensive Image Support: Works across a wide range of product labels, including milk cartons, cereal boxes, and packaged goods.
Technologies Used
PaddleOCR: For high-accuracy text detection and recognition.
Regex Patterns: To filter and extract relevant nutritional facts.
NVIDIA BioBERT: Deployed via NVIDIA NGC for text classification and semantic analysis.
Docker: Ensures portability and GPU-accelerated execution.
Python: Core programming language for processing and orchestration.
How It Works
Image Preprocessing: Images are preprocessed to enhance OCR accuracy.
Text Extraction: PaddleOCR extracts all text from the input image.
Regex Filtering: Extracted text is filtered using regular expressions to identify nutritional details.
NLP with BioBERT: Text is analyzed using BioBERT for enhanced understanding and categorization.
Output: The final nutritional facts are displayed or saved in structured formats (e.g., CSV or JSON).
Setup
Clone this repository:
bash
Copy code
git clone https://github.com/your-repo/nutritional-facts-extraction.git
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the project locally:
bash
Copy code
python app.py
Access the application at http://localhost:5000.
Prerequisites
Docker: Ensure NVIDIA Docker is set up for GPU acceleration.
PaddleOCR: Installed and configured as per PaddleOCR Documentation.
NVIDIA BioBERT: Available through NVIDIA NGC for deployment.
Use Cases
Nutritional data extraction for health and fitness tracking.
Automation of nutritional labeling for food packaging.
E-commerce applications for product cataloging.
License
This project is licensed under the MIT License. See the LICENSE file for more details
