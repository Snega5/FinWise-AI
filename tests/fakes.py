"""Offline stand-ins used ONLY by the tests (the real app uses HuggingFace + Groq).

FakeEmbeddings: deterministic hashed bag-of-words vectors, so FAISS retrieval can be tested
without downloading a model. FakeLLM: returns a canned answer (or raises) like ChatGroq would.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on", "my", "i", "what",
        "how", "do", "does", "it", "that", "this", "with", "as", "be", "at", "by", "me", "you", "your"}


class FakeEmbeddings(Embeddings):
    model_name = "fake-hash-embeddings"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w not in STOP:
                v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def fake_llm(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


class BrokenLLM(FakeListChatModel):
    """Raises on every call, like a network failure."""

    def _call(self, *a, **k):
        raise ConnectionError("connection timed out")
