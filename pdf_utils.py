# pdf_utils.py
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF document.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text, or an empty string if extraction fails.
    """
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

if __name__ == '__main__':
    # Example usage:
    # Make sure to have a PDF file named 'sample_document.pdf' in your data/ directory
    # for this example to work.
    sample_pdf_path = os.path.join("data", "sample_document.pdf")
    if os.path.exists(sample_pdf_path):
        extracted_content = extract_text_from_pdf(sample_pdf_path)
        if extracted_content:
            print(f"Extracted {len(extracted_content)} characters from {sample_pdf_path[:50]}...")
            # print(extracted_content[:500]) # Print first 500 characters
        else:
            print(f"Failed to extract text from {sample_pdf_path}.")
    else:
        print(f"Sample PDF not found at {sample_pdf_path}. Please place a PDF there to test.")