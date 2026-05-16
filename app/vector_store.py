import os
from typing import Any, Dict, List, Tuple

import faiss

from app.index_builder import INDEX_PATH
from app.metadata_store import METADATA_PATH, load_metadata


def load_faiss_index(index_path: str = INDEX_PATH) -> faiss.Index:
    """
    Load a FAISS index from disk.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    return faiss.read_index(index_path)


def load_vector_store() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load the FAISS index and its matching metadata records.
    """
    index = load_faiss_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Vector store mismatch: index has {index.ntotal} vectors, "
            f"but metadata has {len(metadata)} records."
        )

    return index, metadata


def get_vector_store_stats() -> Dict[str, int]:
    """
    Return basic statistics about the local vector store.
    """
    index, metadata = load_vector_store()

    return {
        "index_vectors": index.ntotal,
        "metadata_count": len(metadata),
    }
