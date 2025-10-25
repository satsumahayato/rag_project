"""
Embed corpus chunks into a vector store (ChromaDB or LanceDB-ready structure).

Usage:
  python -m src.embeddings.embed_corpus \
    --in data/processed/corpus_chunks.jsonl \
    --db .chromadb \
    --model openai
"""

import argparse, json, sys, os, time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from chromadb import PersistentClient
from chromadb.config import Settings

# --- Optional embeddings backends ---
from tqdm import tqdm

# Try OpenAI
try:
    from openai import OpenAI
    openai_client = OpenAI()
except ImportError:
    openai_client = None

# Try HuggingFace
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# --------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# --------------------------------------------------
# Embedding backends
# --------------------------------------------------

def embed_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Uses OpenAI API (requires OPENAI_API_KEY in environment).
    """
    if not openai_client:
        raise RuntimeError("OpenAI client not available. pip install openai")
    # batching
    vectors = []
    for i in tqdm(range(0, len(texts), 100), desc="OpenAI embedding"):
        batch = texts[i : i + 100]
        resp = openai_client.embeddings.create(input=batch, model=model)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def embed_hf(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Uses local Hugging Face embedding model (no API key needed).
    """
    if not SentenceTransformer:
        raise RuntimeError("Install sentence-transformers for Hugging Face backend.")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()

# --------------------------------------------------
# Vector store
# --------------------------------------------------

def store_chroma(records, embeddings, db_path):
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="corpus")

    ids = [r["chunk_id"] for r in records]
    texts = [r["text"] for r in records]

    # sanitize metadata — replace None with ""
    def sanitize(meta):
        clean = {}
        for k, v in meta.items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    metadatas = [
        sanitize({
            "doc_id": r.get("parent_doc_id"),
            "section_id": r.get("parent_section_id"),
            "title": r.get("title"),
            "source_path": r.get("source_path"),
            "page_start": r.get("page_start"),
            "page_end": r.get("page_end"),
            "lang": r.get("lang"),
        })
        for r in records
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[OK] Stored {len(records)} embeddings in persistent DB at {db_path}/ (collection: corpus)")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Embed text chunks and store in vector DB.")
    ap.add_argument("--in", dest="inp", required=True, help="Input chunks JSONL")
    ap.add_argument("--db", dest="db", default=".chromadb", help="Path to ChromaDB folder")
    ap.add_argument("--model", choices=["openai", "hf"], default="openai", help="Embedding backend")
    ap.add_argument("--hf_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--openai_model", default="text-embedding-3-small")
    args = ap.parse_args()

    chunks = read_jsonl(Path(args.inp))
    if not chunks:
        print(f"[WARN] No chunks found in {args.inp}", file=sys.stderr)
        sys.exit(1)

    texts = [c["text"] for c in chunks]

    if args.model == "openai":
        embeddings = embed_openai(texts, model=args.openai_model)
    else:
        embeddings = embed_hf(texts, model_name=args.hf_model_name)

    store_chroma(chunks, embeddings, Path(args.db))

if __name__ == "__main__":
    main()