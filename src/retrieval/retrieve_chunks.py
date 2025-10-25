"""
Retrieve top-k relevant chunks from the local Chroma vector store.

Usage:
  python -m src.retrieval.retrieve_chunks \
    --query "What is the PTO accrual rate for full-time employees?" \
    --db .chromadb --k 3 --model openai
"""

import argparse, os, sys
from typing import List, Dict, Any
from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings

# --- Embedding backends ---
try:
    from openai import OpenAI
    # openai_client = OpenAI()
except ImportError:
    openai_client = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# --- Embedding backends (OpenAI v2.x compatible) ---
from openai import OpenAI

def embed_openai_query(query: str, model: str = "text-embedding-3-small") -> list[float]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=model, input=query)
    return response.data[0].embedding


def embed_hf_query(query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    if not SentenceTransformer:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    return model.encode([query])[0].tolist()


# ------------------------------------------------------------
# Retrieval function
# ------------------------------------------------------------

def retrieve_top_k(
    query: str,
    k: int = 3,
    db_path: Path = Path(".chromadb"),
    model_backend: str = "openai",
    openai_model: str = "text-embedding-3-small",
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = "corpus"
) -> list[dict]:
    """
    Given a text query, returns top-k relevant chunks with metadata.
    """
    from chromadb import PersistentClient

    if not db_path.exists():
        raise FileNotFoundError(f"Vector DB not found at {db_path}")

    # --- Embed the query ---
    if model_backend == "openai":
        q_emb = embed_openai_query(query, model=openai_model)
    else:
        q_emb = embed_hf_query(query, model_name=hf_model_name)

    # --- Connect to Chroma persistent client ---
    client = PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(collection_name)

    # --- Perform similarity search ---
    res = collection.query(query_embeddings=[q_emb], n_results=k)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        results.append({
            "distance": dist,
            "text": doc,
            **meta
        })
    return results

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Query Chroma vector DB for top-k chunks.")
    ap.add_argument("--query", required=True, help="User query string")
    ap.add_argument("--db", default=".chromadb", help="Vector DB directory")
    ap.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    ap.add_argument("--model", choices=["openai", "hf"], default="openai", help="Embedding backend")
    ap.add_argument("--openai_model", default="text-embedding-3-small")
    ap.add_argument("--hf_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    results = retrieve_top_k(
        query=args.query,
        k=args.k,
        db_path=Path(args.db),
        model_backend=args.model,
        openai_model=args.openai_model,
        hf_model_name=args.hf_model_name
    )

    print(f"\nTop {args.k} results for query: {args.query}\n")
    for i, r in enumerate(results, start=1):
        title = r.get("title") or "(untitled)"
        page = f"p{r['page_start']}" if r.get("page_start") else ""
        print(f"#{i}: {title} {page}  dist={r['distance']:.3f}")
        print(f"→ {r['text'][:250]}…\n")

if __name__ == "__main__":
    main()