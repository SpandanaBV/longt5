# Long-T5 RAG Summarization Pipeline

## Overview
This project evaluates Long-T5 for summarizing long unstructured web data into RAG-ready chunks.

## Pipeline
1. Raw web text
2. Token-based chunking (2k tokens, overlap)
3. Abstractive summarization using Long-T5
4. Structured RAG dataset (JSONL)

## Why Long-T5?
- Handles long context (up to 16k tokens)
- Abstractive summaries
- Works well with chunking for RAG

## Setup
```
pip install -r requirements.txt
```

## Run
```
python run_pipeline.py
```

## Output
```
data/rag/rag_dataset.jsonl
```
