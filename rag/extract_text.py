import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of the PDF"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

if __name__ == "__main__":
    # Get the parent directory of the current script, i.e., the project root directory sciqa-rag/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    paper_folder = os.path.join(base_dir, "data", "papers")
    output_folder = os.path.join(base_dir, "data", "texts")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(paper_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(paper_folder, filename)
            print(f"Extracting {filename}...")
            text = extract_text_from_pdf(pdf_path)

            output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

    print("Extraction completed")
