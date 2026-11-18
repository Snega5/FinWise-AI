"""Shared UI helpers: theme CSS, cards, chart styling and version-safe Streamlit helpers."""
from __future__ import annotations

import streamlit as st

GREEN = "#10B981"
BLUE = "#3B82F6"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E5EAF2"

CATEGORY_COLORS = {
    "Food": "#10B981",
    "Shopping": "#3B82F6",
    "Transport": "#06B6D4",
    "Bills": "#6366F1",
    "Entertainment": "#F59E0B",
    "Healthcare": "#EF4444",
    "Other": "#94A3B8",
    "Income": "#22C55E",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { font-family: 'Inter', sans-serif; background: linear-gradient(180deg, #F6F9FC 0%, #EEF4F8 100%); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5EAF2; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 14px; border-radius: 12px; width: 100%; transition: background .15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #ECFDF5; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, #D1FAE5, #DBEAFE); font-weight: 600;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }

.fw-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.fw-logo {
    width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #10B981, #3B82F6); font-size: 22px;
}
.fw-brand b { display: block; font-size: 18px; color: #0F172A; line-height: 1.1; }
.fw-brand small { color: #64748B; font-size: 12px; }

/* Page header */
.fw-header h1 { font-size: 30px; font-weight: 700; color: #0F172A; margin: 0 0 4px 0; padding: 0; }
.fw-header p { color: #64748B; margin: 0 0 18px 0; font-size: 15px; }

/* Cards */
.fw-card {
    background: #FFFFFF; border: 1px solid #E5EAF2; border-radius: 18px; padding: 18px 20px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
}
div[class*="st-key-card"] {
    background: #FFFFFF; border: 1px solid #E5EAF2; border-radius: 18px; padding: 18px 20px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
}
.fw-section { font-size: 16px; font-weight: 600; color: #0F172A; margin-bottom: 6px; }

.fw-metric .fw-icon {
    width: 38px; height: 38px; border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; margin-bottom: 10px;
}
.fw-green { background: #D1FAE5; } .fw-blue { background: #DBEAFE; }
.fw-amber { background: #FEF3C7; } .fw-indigo { background: #E0E7FF; } .fw-red { background: #FEE2E2; }
.fw-metric .fw-label { color: #64748B; font-size: 13px; font-weight: 500; }
.fw-metric .fw-value { color: #0F172A; font-size: 26px; font-weight: 700; margin: 2px 0; }
.fw-metric .fw-sub { font-size: 12.5px; }
.fw-sub.fw-muted { color: #64748B; } .fw-sub.fw-good { color: #059669; } .fw-sub.fw-bad { color: #DC2626; }

/* Insight cards */
.fw-insight { border-left: 5px solid #3B82F6; height: 100%; }
.fw-insight.alert { border-left-color: #F59E0B; }
.fw-insight.saving { border-left-color: #10B981; }
.fw-insight.summary { border-left-color: #3B82F6; }
.fw-insight.budget { border-left-color: #6366F1; }
.fw-insight .fw-tag { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #64748B; }
.fw-insight h4 { margin: 4px 0 6px 0; font-size: 16px; color: #0F172A; }
.fw-insight p { margin: 0; color: #334155; font-size: 14px; line-height: 1.5; }

/* Buttons and inputs */
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; }
div[data-testid="stFileUploader"] section { border-radius: 14px; }
/* Visible input borders: inputs sit on white cards, so give them a tint and an outline */
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stTextAreaRootElement"], [data-testid="stSelectbox"] div[role="group"],
[data-testid="stMultiSelect"] div[role="group"] {
    background: #F8FAFC !important; border: 1px solid #D5DDE9 !important; border-radius: 10px !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within,
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within {
    border-color: #10B981 !important;
}

.fw-disclaimer { color: #64748B; font-size: 12px; line-height: 1.4; }
"""


def _version() -> tuple[int, int]:
    try:
        parts = st.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (1, 0)


def stretch() -> dict:
    """Return the right 'full width' kwarg for this Streamlit version."""
    if _version() >= (1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


def money(x: float, cur: str | None = None) -> str:
    cur = cur or st.session_state.get("currency", "₹")
    sign = "-" if x < 0 else ""
    return f"{sign}{cur}{abs(x):,.0f}"


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="fw-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="fw-section">{text}</div>', unsafe_allow_html=True)


def card(key: str):
    """A styled container. Keys must be unique per page."""
    return st.container(key=f"card_{key}")


def metric_card(label: str, value: str, sub: str = "", accent: str = "green",
                icon: str = "💰", tone: str = "muted") -> None:
    st.markdown(
        f'<div class="fw-card fw-metric"><div class="fw-icon fw-{accent}">{icon}</div>'
        f'<div class="fw-label">{label}</div><div class="fw-value">{value}</div>'
        f'<div class="fw-sub fw-{tone}">{sub}</div></div>',
        unsafe_allow_html=True,
    )


_INSIGHT_META = {
    "alert": ("⚠️", "Spending alert"),
    "saving": ("🌱", "Savings opportunity"),
    "summary": ("📈", "Financial summary"),
    "budget": ("🎯", "Budget recommendation"),
}


def insight_card(kind: str, title: str, body: str) -> None:
    icon, tag = _INSIGHT_META.get(kind, ("💡", "Insight"))
    st.markdown(
        f'<div class="fw-card fw-insight {kind}"><div class="fw-tag">{icon} {tag}</div>'
        f"<h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 340):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F7")
    return fig


def show_chart(fig, **kw) -> None:
    st.plotly_chart(fig, **stretch(), **kw)


def show_df(df, **kw) -> None:
    st.dataframe(df, hide_index=True, **stretch(), **kw)


def disclaimer() -> None:
    st.markdown(
        '<div class="fw-disclaimer">AI-generated recommendations are suggestions, '
        "not professional financial advice.</div>",
        unsafe_allow_html=True,
    )
