import json
from typing import List, Dict, Any

from src.summarize import summarize


def build_rag(chunks: List[str], source: str = "sample") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        summary = summarize(chunk)
        records.append(
            {
                "id": f"{source}_chunk_{i}",
                "content": summary,
                "metadata": {"source": source, "chunk_index": i},
            }
        )

    return records
