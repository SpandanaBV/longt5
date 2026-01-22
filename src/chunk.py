from transformers import AutoTokenizer
from typing import List

# Initialize tokenizer once
TOKENIZER_NAME = "google/long-t5-local-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def chunk_text(text: str, chunk_size: int = 2048, overlap: int = 200) -> List[str]:
    """
    Token-based chunking for long inputs.

    Args:
        text: Raw input string.
        chunk_size: Max tokens per chunk.
        overlap: Overlap tokens between consecutive chunks.
    Returns:
        List of decoded text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenizer.encode(text)
    chunks: List[str] = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks
