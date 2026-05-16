import os

import faiss
import numpy as np

from app.embedding_service import create_normalized_embedding_matrix
from app.pdf_loader import DATA_DIR, load_cv_documents


INDEX_PATH = os.path.join(DATA_DIR, "rag_index.faiss")


def build_faiss_index(embedding_matrix: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index from normalized embeddings.
    """
    if embedding_matrix.size == 0:
        raise ValueError("Embedding matrix cannot be empty.")

    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    return index


def save_faiss_index(index: faiss.IndexFlatIP, index_path: str = INDEX_PATH) -> None:
    """
    Save a FAISS index to disk.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


def build_and_save_index(index_path: str = INDEX_PATH) -> int:
    """
    Load CV documents, create embeddings, build a FAISS index, and save it to disk.

    Returns the number of indexed CV documents.
    """
    documents = load_cv_documents()

    if not documents:
        raise RuntimeError("No CV PDF files found. Add sample PDFs under data/pdfs/.")

    texts = [document.text for document in documents]
    embedding_matrix = create_normalized_embedding_matrix(texts)

    index = build_faiss_index(embedding_matrix)
    save_faiss_index(index, index_path)

    return len(documents)


if __name__ == "__main__":
    count = build_and_save_index()
    print(f"FAISS index created successfully with {count} CV documents.")
