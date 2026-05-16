from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from app.embedding_service import create_embeddings


def create_query_vector(query: str) -> np.ndarray:
    """
    Create and normalize an embedding vector for a recruiter query.
    """
    embeddings = create_embeddings([query])

    if not embeddings:
        raise ValueError("Could not create query embedding.")

    query_vector = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(query_vector)

    return query_vector


def search_candidates(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int = 1,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Search the FAISS index and return the best matching candidate records.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    query_vector = create_query_vector(cleaned_query)
    distances, indices = index.search(query_vector, top_k)

    matches: List[Tuple[int, float, Dict[str, Any]]] = []

    for raw_index, raw_score in zip(indices[0], distances[0]):
        candidate_index = int(raw_index)
        similarity_score = float(raw_score)

        if candidate_index < 0 or candidate_index >= len(metadata):
            continue

        matches.append(
            (
                candidate_index,
                similarity_score,
                metadata[candidate_index],
            )
        )

    return matches


def get_best_candidate_match(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Return the single best candidate match for a recruiter query.
    """
    matches = search_candidates(
        query=query,
        index=index,
        metadata=metadata,
        top_k=1,
    )

    if not matches:
        raise LookupError("No matching candidate found.")

    return matches[0]
