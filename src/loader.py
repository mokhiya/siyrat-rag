# src/loader.py
# Load raw PDF files and extract their text content
import os
from pypdf import PdfReader


def load_all_pdfs(folder_path):
    """Read every PDF in folder_path and return a list of (text, filename) tuples."""
    results = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                results.append((text, file))
    return results
