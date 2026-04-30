import os
import fitz  # PyMuPDF
import hashlib

# Function to calculate the hash of the file
def calculate_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Directory paths
pdf_directory = "pdfs"
output_directory = "cleaned_texts"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process each PDF
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        output_filename = filename.replace(".pdf", ".txt")
        output_path = os.path.join(output_directory, output_filename)

        # Calculate hash of the PDF
        pdf_hash = calculate_hash(pdf_path)
        hash_path = os.path.join(output_directory, f"{pdf_hash}.hash")

        # Skip processing if hash file exists (indicating the PDF has already been processed)
        if os.path.exists(hash_path):
            print(f"Skipping {filename}, already processed.")
            continue

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)

        # Save the extracted text to the output directory
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        # Save the hash to indicate the PDF has been processed
        with open(hash_path, "w") as hash_file:
            hash_file.write("Processed")

        print(f"Processed and saved text from {filename}.")
