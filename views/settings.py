"""Settings page: preferences, Groq connection, embedding model and index, data controls."""
import os

import streamlit as st

from utils import chat_store, ui
from utils.data_processor import load_sample
from utils.embeddings import DEFAULT_MODEL, MODELS_DIR
from utils.llm import llm_status, test_connection
from utils.rag_pipeline import DocumentError, EmbeddingError, get_pipeline


def _use_sample() -> None:
    st.session_state.df = load_sample()
    st.session_state.data_source = "Sample data"
    st.session_state.pop("loaded_file_id", None)
    st.session_state.pop("upload_notes", None)


def _clear_data() -> None:
    import pandas as pd

    st.session_state.df = pd.DataFrame(columns=["date", "description", "amount", "type", "category", "month"])
    st.session_state.data_source = "No data"


def _clear_history() -> None:
    st.session_state.chat = []
    chat_store.clear()


def _model_downloaded(name: str) -> bool:
    return (MODELS_DIR / ("models--" + name.replace("/", "--"))).exists()


def render() -> None:
    ui.page_header("Settings", "Preferences, AI connections and data controls")

    with ui.card("prefs"):
        ui.section_title("Preferences")
        c1, c2 = st.columns(2)
        st.session_state.user_name = c1.text_input("Your name", st.session_state.get("user_name", ""))
        options = ["₹", "$", "€", "£"]
        current = st.session_state.get("currency", "₹")
        st.session_state.currency = c2.selectbox("Currency symbol", options, index=options.index(current))

    st.write("")
    with ui.card("api"):
        ui.section_title("Groq API (language model)")
        ready, status = llm_status()
        (st.success if ready else st.warning)(status)
        if not ready:
            st.caption("Copy `.env.example` to `.env`, paste your free key from console.groq.com/keys, "
                       "then restart the app.")
        if st.button("Test Groq connection", disabled=not ready):
            with st.spinner("Contacting Groq..."):
                ok, msg = test_connection()
            (st.success if ok else st.error)(msg)
        st.caption("The key is read from your environment and is never displayed here.")

    st.write("")
    with ui.card("rag"):
        ui.section_title("Embeddings and knowledge index (RAG)")
        name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        if _model_downloaded(name):
            st.success(f"Embedding model `{name}` is downloaded and runs locally.")
        else:
            st.info(f"Embedding model `{name}` (about 90 MB) downloads automatically the first time you "
                    "build the index or ask a document question. After that it works offline.")
        sources = get_pipeline().indexed_sources()
        if sources:
            kb = [k for k, v in sources.items() if v.get("kind") == "knowledge"]
            pdfs = [k for k, v in sources.items() if v.get("kind") == "pdf"]
            st.markdown(f"**Indexed:** {len(kb)} knowledge guides" + (f" and {len(pdfs)} uploaded PDF(s)" if pdfs else ""))
        else:
            st.caption("No index built yet.")
        if st.button("Build / rebuild knowledge index"):
            with st.spinner("Embedding documents..."):
                try:
                    n = get_pipeline().rebuild_knowledge()
                    st.success(f"Index built: {n} chunks. (Uploaded PDFs must be re-uploaded after a rebuild.)")
                except (EmbeddingError, DocumentError) as e:
                    st.error(str(e))
        st.caption("Vectors are stored locally in `models/faiss_index`.")

    st.write("")
    with ui.card("data"):
        ui.section_title("Data")
        st.caption(f"Current data: {st.session_state.get('data_source', 'n/a')}")
        b1, b2, b3 = st.columns(3)
        b1.button("Load sample data", on_click=_use_sample)
        b2.button("Clear all data", on_click=_clear_data)
        b3.button("Clear chat history", on_click=_clear_history)

    st.write("")
    with ui.card("privacy"):
        ui.section_title("Privacy and safety")
        st.markdown(
            "- Files stay on your computer and are processed locally.\n"
            "- FinWise never asks for bank passwords, OTPs or login access.\n"
            "- All totals are calculated with Pandas, not by the AI.\n"
            "- Only aggregated figures (and short retrieved document passages) are sent to Groq, "
            "never raw transaction rows or merchant names.\n"
            "- AI answers containing numbers that are not in your data are blocked.\n"
            "- AI output is a **suggestion, not professional financial advice**."
        )
    st.caption("FinWise AI v1.0")
