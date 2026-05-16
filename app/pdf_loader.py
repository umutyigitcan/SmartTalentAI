import os
from dataclasses import dataclass
from typing import List

import fitz


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")


@dataclass
class CVDocument:
    """
    Represents a parsed CV document loaded from a PDF file.
    """

    filename: str
    text: str


def list_pdf_files(pdf_dir: str = PDF_DIR) -> List[str]:
    """
    Return all PDF filenames inside the configured PDF directory.
    """
    if not os.path.exists(pdf_dir):
        return []

    return sorted(
        filename
        for filename in os.listdir(pdf_dir)
        if filename.lower().endswith(".pdf")
    )


def extract_pdf_text(path: str) -> str:
    """
    Extract plain text from a PDF file using PyMuPDF.
    """
    document = fitz.open(path)
    text = ""

    try:
        for page in document:
            text += page.get_text()
    finally:
        document.close()

    return text.strip()


def load_cv_documents(pdf_dir: str = PDF_DIR) -> List[CVDocument]:
    """
    Load all CV PDF files from disk and convert them into text documents.
    """
    documents: List[CVDocument] = []

    for filename in list_pdf_files(pdf_dir):
        path = os.path.join(pdf_dir, filename)
        text = extract_pdf_text(path)

        if not text:
            continue

        documents.append(
            CVDocument(
                filename=filename,
                text=text,
            )
        )

    return documents
