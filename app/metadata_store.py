import json
import os
from typing import Any, Dict, List

from app.pdf_loader import CVDocument, DATA_DIR


METADATA_PATH = os.path.join(DATA_DIR, "metadata_list.json")


def build_metadata(documents: List[CVDocument]) -> List[Dict[str, Any]]:
    """
    Build metadata records for indexed CV documents.
    """
    metadata: List[Dict[str, Any]] = []

    for index, document in enumerate(documents):
        metadata.append(
            {
                "id": index,
                "filename": document.filename,
                "preview": document.text,
            }
        )

    return metadata


def save_metadata(
    metadata: List[Dict[str, Any]],
    metadata_path: str = METADATA_PATH,
) -> None:
    """
    Save CV metadata records as a JSON file.
    """
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_metadata(metadata_path: str = METADATA_PATH) -> List[Dict[str, Any]]:
    """
    Load CV metadata records from disk.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)
