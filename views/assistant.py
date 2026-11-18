"""AI Financial Assistant page: RAG chat with PDF upload, citations and saved history."""
import streamlit as st

from utils import chat_store, ui
from utils.chatbot import answer
from utils.llm import llm_status
from utils.rag_pipeline import DocumentError, EmbeddingError, get_pipeline

SUGGESTIONS = [
    "Where did I spend the most money?",
    "How much did I spend on food this month?",
    "Compare my expenses across months",
    "How much can I save next month?",
    "Give me a budget based on my spending",
    "What is the 50/30/20 rule?",
]


def _ask(q: str) -> None:
    st.session_state["pending_q"] = q


def _index_pdfs(files) -> None:
    """Index newly uploaded PDFs (PyPDF -> chunks -> embeddings -> FAISS)."""
    done = st.session_state.setdefault("indexed_pdf_keys", set())
    for f in files:
        key = (f.name, f.size)
        if key in done:
            continue
        try:
            with st.spinner(f"Indexing {f.name} (first time can take a minute while the embedding model loads)..."):
                n = get_pipeline().add_pdf(f.getvalue(), f.name)
        except DocumentError as e:
            st.error(str(e))
            continue
        except EmbeddingError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Could not index {f.name}: {type(e).__name__}")
            continue
        done.add(key)
        st.toast(f"Indexed {f.name}" + (f" ({n} chunks)" if n else " (already indexed)"), icon="📄")


def _documents_panel() -> None:
    with st.expander("📄 Documents the assistant can search"):
        files = st.file_uploader("Upload PDF guides or policies", type=["pdf"],
                                 accept_multiple_files=True, key="pdf_upload")
        if files:
            _index_pdfs(files)
        sources = get_pipeline().indexed_sources()
        pdfs = {k: v for k, v in sources.items() if v.get("kind") == "pdf"}
        kb = {k: v for k, v in sources.items() if v.get("kind") == "knowledge"}
        if kb:
            st.markdown(f"📚 **Knowledge base:** {len(kb)} financial guides indexed "
                        f"({sum(v['chunks'] for v in kb.values())} chunks)")
        else:
            st.caption("The built-in knowledge base is indexed on your first document question "
                       "(or from Settings). The embedding model downloads once, then works offline.")
        for name, v in pdfs.items():
            st.markdown(f"📄 **{name}**: {v['chunks']} chunks")
        st.caption("PDFs are read and indexed on your computer. Nothing is uploaded except the short "
                   "retrieved passages sent to Groq when you ask a question.")


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg["role"] != "assistant":
            return
        if msg.get("source"):
            st.caption(f"Source: {msg['source']}")
        cites = msg.get("citations") or []
        if cites:
            with st.expander(f"Retrieved passages ({len(cites)})"):
                for i, c in enumerate(cites, 1):
                    st.markdown(f"**[{i}] {c['label']}** · relevance {c['score']:.2f}")
                    st.caption(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))


def render() -> None:
    ui.page_header("AI Financial Assistant", "Ask about your spending, budget, or your uploaded documents")
    cur = st.session_state.get("currency", "₹")
    df = st.session_state.get("df")

    ready, status = llm_status()
    st.caption(("🟢 " if ready else "🟡 ") + status)

    _documents_panel()

    st.caption("Try one of these:")
    for row in (SUGGESTIONS[:3], SUGGESTIONS[3:]):
        cols = st.columns(3)
        for col, q in zip(cols, row):
            col.button(q, key=f"sugg_{q}", on_click=_ask, args=(q,), **ui.stretch())

    typed = st.chat_input("Ask about your finances or your documents...")
    question = typed or st.session_state.pop("pending_q", None)
    if question:
        st.session_state.chat.append({"role": "user", "text": question})
        chat_store.add_message("user", question)
        with st.spinner("Thinking..."):
            res = answer(question, df, cur)
        msg = {"role": "assistant", "text": res["text"], "source": res["source"],
               "citations": res.get("citations", [])}
        st.session_state.chat.append(msg)
        chat_store.add_message("assistant", msg["text"], msg["source"], msg["citations"])

    if not st.session_state.chat:
        with st.chat_message("assistant"):
            st.markdown("Hi! I'm your FinWise assistant. Every figure about your money is calculated from "
                        "your actual transactions, so nothing is made up. I can also explain money concepts "
                        "and answer questions about PDFs you upload. Pick a question above or type your own.")
    for msg in st.session_state.chat:
        _render_message(msg)

    c1, _ = st.columns([1, 4])
    if st.session_state.chat and c1.button("Clear chat"):
        st.session_state.chat = []
        chat_store.clear()
        st.rerun()
    ui.disclaimer()
