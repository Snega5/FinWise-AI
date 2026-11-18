"""FinWise AI: entry point. Run with: streamlit run app.py"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="FinWise AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import chat_store, ui  # noqa: E402
from utils.data_processor import load_sample  # noqa: E402
from views import assistant, budget, dashboard, insights, settings, transactions  # noqa: E402

PAGES = {
    "📊 Dashboard": dashboard.render,
    "🧾 Transactions": transactions.render,
    "💡 AI Insights": insights.render,
    "🎯 Budget Planner": budget.render,
    "🤖 AI Assistant": assistant.render,
    "⚙️ Settings": settings.render,
}


def init_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = load_sample()
        st.session_state.data_source = "Sample data"
    st.session_state.setdefault("currency", "₹")
    st.session_state.setdefault("user_name", "")
    if "chat" not in st.session_state:
        st.session_state.chat = chat_store.load_messages()  # SQLite history from earlier sessions
    st.session_state.setdefault("docs", [])
    st.session_state.setdefault("nav", "📊 Dashboard")


init_state()
ui.inject_css()

with st.sidebar:
    st.markdown(
        '<div class="fw-brand"><div class="fw-logo">💸</div>'
        "<div><b>FinWise AI</b><small>Your personal finance advisor</small></div></div>",
        unsafe_allow_html=True,
    )
    choice = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
    st.markdown("---")
    df = st.session_state.df
    st.caption(f"Data: {st.session_state.data_source} · {len(df):,} transactions")
    ui.disclaimer()

PAGES[choice]()
