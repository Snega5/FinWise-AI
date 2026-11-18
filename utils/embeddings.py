"""Local HuggingFace sentence-transformer embeddings (free, runs on CPU, no API key)."""
from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_cache: dict = {}


class EmbeddingError(Exception):
    """The embedding model could not be loaded (message is user-friendly)."""


def get_embeddings(model_name: str | None = None):
    """Load (once) and return a LangChain-compatible embeddings object.

    The first run downloads the model (~90 MB) into ./models; later runs work offline.
    Vectors are L2-normalised so similarity search behaves like cosine similarity.
    """
    name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    if name in _cache:
        return _cache[name]
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        MODELS_DIR.mkdir(exist_ok=True)
        emb = HuggingFaceEmbeddings(
            model_name=name,
            cache_folder=str(MODELS_DIR),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        raise EmbeddingError(
            "The local embedding model could not be loaded. The first run needs an internet "
            "connection to download it (about 90 MB); after that it works offline. "
            f"Details: {type(e).__name__}"
        ) from e
    _cache[name] = emb
    return emb
