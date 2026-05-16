from typing import List

import faiss
import numpy as np
from openai import OpenAI

from app.config import settings


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the configured API key.
    """
    settings.validate_openai_settings()

    return OpenAI(api_key=settings.openai_api_key)


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embedding vectors for a list of CV texts.
    """
    if not texts:
        return []

    client = get_openai_client()

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )

    return [item.embedding for item in response.data]


def embeddings_to_numpy(embeddings: List[List[float]]) -> np.ndarray:
    """
    Convert embedding vectors into a FAISS-compatible float32 NumPy array.
    """
    if not embeddings:
        raise ValueError("Embedding list cannot be empty.")

    return np.array(embeddings, dtype="float32")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings for cosine-similarity style FAISS search.
    """
    faiss.normalize_L2(embeddings)

    return embeddings


def create_normalized_embedding_matrix(texts: List[str]) -> np.ndarray:
    """
    Create a normalized embedding matrix from raw CV texts.
    """
    embeddings = create_embeddings(texts)
    matrix = embeddings_to_numpy(embeddings)

    return normalize_embeddings(matrix)
