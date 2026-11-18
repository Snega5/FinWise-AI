"""RAG pipeline: load -> split -> embed -> FAISS -> retrieve (-> Groq answer in utils/llm.py)."""
from __future__ import annotations

import io
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from utils import vector_store as vs
from utils.embeddings import EmbeddingError, get_embeddings  # noqa: F401  (re-exported)
from utils.llm import get_llm

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
META_KEY = "__meta__"  # manifest entry that records which embedding model built the index


class DocumentError(Exception):
    """A document could not be read or indexed (message is user-friendly)."""


# ---------- 1. load ----------

def load_pdf(data: bytes, name: str) -> list[Document]:
    """Extract text page by page with PyPDF."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
    except Exception as e:
        raise DocumentError(f"'{name}' is not a readable PDF (it may be corrupted or password-protected).") from e

    docs: list[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": i, "kind": "pdf"}))
    if not docs:
        raise DocumentError(f"No text could be extracted from '{name}'. Scanned PDFs need OCR first.")
    return docs


def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> list[Document]:
    """Load the bundled financial knowledge documents (.md / .txt)."""
    docs: list[Document] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path.stem, "kind": "knowledge"}))
    return docs


# ---------- 2. split ----------

def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def format_context(hits: list[dict]) -> str:
    """Number the retrieved passages so the LLM can cite them as [1], [2]..."""
    return "\n\n".join(f"[{i}] ({cite(h)}) {h['text']}" for i, h in enumerate(hits, start=1))


def cite(hit: dict) -> str:
    return f"{hit['source']}, p.{hit['page']}" if hit.get("page") else str(hit["source"])


# ---------- 3. embed + store + retrieve ----------

class RAGPipeline:
    """Holds the FAISS index. Embeddings load lazily so the app starts instantly."""

    def __init__(self, embeddings=None, llm=None, index_dir: Path = vs.INDEX_DIR,
                 knowledge_dir: Path = KNOWLEDGE_DIR):
        self._embeddings = embeddings
        self._llm = llm
        self.index_dir = index_dir
        self.knowledge_dir = knowledge_dir
        self.store = None
        self.manifest: dict = {}

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def llm(self):
        return self._llm if self._llm is not None else get_llm()

    def _model_id(self) -> str:
        emb = self.embeddings
        return str(getattr(emb, "model_name", None) or type(emb).__name__)

    def _save(self) -> None:
        self.manifest[META_KEY] = {"embedding_model": self._model_id()}
        vs.save_store(self.store, self.manifest, self.index_dir)

    def indexed_sources(self) -> dict:
        """Which documents are indexed (read from disk, no model needed)."""
        manifest = self.manifest or vs.load_manifest(self.index_dir)
        return {k: v for k, v in manifest.items() if k != META_KEY}

    def ensure_index(self) -> None:
        if self.store is not None:
            return
        store = vs.load_store(self.embeddings, self.index_dir)
        manifest = vs.load_manifest(self.index_dir)
        # An index built with a different embedding model has incompatible vectors: rebuild it.
        if store is None or manifest.get(META_KEY, {}).get("embedding_model") != self._model_id():
            self.rebuild_knowledge()
        else:
            self.store = store
            self.manifest = manifest

    def rebuild_knowledge(self) -> int:
        """(Re)build the index from the bundled knowledge documents only (drops uploaded PDFs)."""
        chunks = split_documents(load_knowledge(self.knowledge_dir))
        if not chunks:
            raise DocumentError("No knowledge documents found in data/knowledge.")
        self.store = vs.build_store(chunks, self.embeddings)
        self.manifest = {}
        for c in chunks:
            entry = self.manifest.setdefault(c.metadata["source"], {"kind": "knowledge", "chunks": 0})
            entry["chunks"] += 1
        self._save()
        return len(chunks)

    def add_pdf(self, data: bytes, name: str) -> int:
        """Index an uploaded PDF. Returns number of chunks added (0 if already indexed)."""
        self.ensure_index()
        if name in self.manifest:
            return 0
        chunks = split_documents(load_pdf(data, name))
        self.store.add_documents(chunks)
        self.manifest[name] = {"kind": "pdf", "chunks": len(chunks)}
        self._save()
        return len(chunks)

    def retrieve(self, query: str, k: int = 4, min_relevance: float = 0.2) -> list[dict]:
        """Semantic similarity search. Returns hits above `min_relevance`, best first."""
        self.ensure_index()
        hits = []
        for doc, score in vs.similarity_search(self.store, query, k=k):
            if score >= min_relevance:
                hits.append({
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "kind": doc.metadata.get("kind"),
                    "score": round(score, 3),
                })
        return hits


# One shared pipeline per app process (the embedding model and index load only once).
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def set_pipeline(pipeline: RAGPipeline | None) -> None:
    """Replace the shared pipeline (used by tests)."""
    global _pipeline
    _pipeline = pipeline
