"""FAISS vector store helpers: build, persist locally, load, and similarity search."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INDEX_DIR = Path(__file__).resolve().parent.parent / "models" / "faiss_index"
MANIFEST = "manifest.json"


def build_store(docs: list[Document], embeddings) -> FAISS:
    """Embed `docs` and put them in a new FAISS index (vectors L2-normalised)."""
    return FAISS.from_documents(docs, embeddings, normalize_L2=True)


def save_store(store: FAISS, manifest: dict, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    (index_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(index_dir: Path = INDEX_DIR) -> dict:
    path = index_dir / MANIFEST
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_store(embeddings, index_dir: Path = INDEX_DIR) -> FAISS | None:
    """Load a previously saved index, or None if there is none (or it is unreadable)."""
    if not (index_dir / "index.faiss").exists():
        return None
    try:
        # The pickle part of the index is written by this app on this machine only.
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True, normalize_L2=True
        )
    except Exception:
        return None


def similarity_search(store: FAISS, query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Return (document, relevance 0-1) pairs, best first.

    FAISS returns squared L2 distance d between unit vectors, so cosine = 1 - d/2.
    """
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, max(0.0, min(1.0, 1.0 - float(d) / 2.0))) for doc, d in results]
