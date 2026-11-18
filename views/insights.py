"""AI Insights page (numbers come from Pandas; Groq narrative is added in the AI step)."""
import plotly.graph_objects as go
import streamlit as st

from utils import financial_analyzer as fa
from utils import ui
from utils.budget_advisor import suggest_budget
from utils.facts import financial_facts
from utils.llm import LLMError, get_llm, narrate
from utils.ui import BLUE, GREEN


def build_insights(df, cur: str) -> list[tuple[str, str, str]]:
    cards: list[tuple[str, str, str]] = []
    months = sorted(df["month"].unique())
    last = months[-1]
    ms = fa.summary(df[df["month"] == last])
    lbl = fa.pretty_month(last)

    cards.append((
        "summary", f"{lbl} at a glance",
        f"You earned {ui.money(ms['income'], cur)} and spent {ui.money(ms['expenses'], cur)}, "
        f"leaving {ui.money(ms['balance'], cur)}. That is a savings rate of {ms['savings_pct']:.1f}%.",
    ))
    if ms["savings_pct"] < 20:
        cards.append((
            "alert", "Savings rate below 20%",
            f"Your savings rate in {lbl} was {ms['savings_pct']:.1f}%. A common guideline is to save at least 20% of income.",
        ))

    cmp = fa.mom_comparison(df)
    if cmp is not None:
        t = cmp["table"]
        ups = t[(t["change_pct"] > 20) & (t["change"] >= 500)]
        if not ups.empty:
            r = ups.iloc[0]
            cards.append((
                "alert", f"{r['category']} spending up {r['change_pct']:.0f}%",
                f"{r['category']} rose from {ui.money(r['previous'], cur)} in {fa.pretty_month(cmp['prev'])} "
                f"to {ui.money(r['current'], cur)} in {fa.pretty_month(cmp['curr'])} "
                f"(+{ui.money(r['change'], cur)}).",
            ))

    for r in fa.unusual_transactions(df).head(2).itertuples():
        cards.append((
            "alert", f"Unusual {r.category} transaction",
            f"{r.description} on {r.date:%d %b %Y} cost {ui.money(r.amount, cur)}, "
            f"well above your typical {r.category} spend of about {ui.money(r.typical, cur)}.",
        ))

    plan = suggest_budget(df)
    if not plan.empty:
        top = plan.sort_values("potential_saving", ascending=False).iloc[0]
        if top["potential_saving"] > 0:
            cards.append((
                "saving", f"Trim {top['category']} to save more",
                f"Cutting {top['category']} from about {ui.money(top['avg_spend'], cur)} to "
                f"{ui.money(top['suggested_budget'], cur)} a month could free up "
                f"{ui.money(top['potential_saving'], cur)}. {top['tip']}",
            ))
        total_save = float(plan["potential_saving"].sum())
        cards.append((
            "saving", "Total monthly savings potential",
            f"Following the suggested budget across all categories could save roughly "
            f"{ui.money(total_save, cur)} per month ({ui.money(total_save * 12, cur)} a year).",
        ))
        top3 = plan.head(3)
        parts = ", ".join(f"{r.category} {ui.money(r.suggested_budget, cur)}" for r in top3.itertuples())
        cards.append((
            "budget", "Suggested budget for your top categories",
            f"Based on your averages: {parts}. Open the Budget Planner to adjust these.",
        ))
    return cards


def render() -> None:
    ui.page_header("AI Financial Insights", "Spending alerts, savings opportunities and summaries")
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("Load transactions first to see insights.")
        return
    cur = st.session_state.get("currency", "₹")

    cards = build_insights(df, cur)
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, (kind, title, body) in zip(cols, cards[i:i + 2]):
            with col:
                ui.insight_card(kind, title, body)
        st.write("")

    cmp = fa.mom_comparison(df)
    if cmp is not None:
        with ui.card("compare"):
            ui.section_title(
                f"Month-to-month comparison: {fa.pretty_month(cmp['prev'])} vs {fa.pretty_month(cmp['curr'])}"
            )
            t = cmp["table"].sort_values("current", ascending=False)
            fig = go.Figure()
            fig.add_bar(x=t["category"], y=t["previous"], name=fa.pretty_month(cmp["prev"]), marker_color=BLUE)
            fig.add_bar(x=t["category"], y=t["current"], name=fa.pretty_month(cmp["curr"]), marker_color=GREEN)
            fig.update_layout(barmode="group")
            ui.show_chart(ui.style_fig(fig, 320))

    st.write("")
    with ui.card("narrative"):
        ui.section_title("AI narrative (Groq)")
        st.caption(
            "Groq turns the Pandas-calculated figures into a plain-language summary. Only aggregated "
            "numbers are sent, never merchant names or individual transactions. Any figure that is "
            "not in your data is blocked."
        )
        llm = get_llm()
        key = (st.session_state.get("data_source"), len(df), cur)
        if llm is None:
            st.info("Add your free Groq API key to `.env` (see Settings) to enable this.")
        if st.button("Generate AI explanation", disabled=llm is None, type="primary"):
            with st.spinner("Asking Groq..."):
                try:
                    st.session_state.narrative = (key, narrate(llm, financial_facts(df, cur)))
                except LLMError as e:
                    st.session_state.pop("narrative", None)
                    st.warning(str(e))
        saved = st.session_state.get("narrative")
        if saved and saved[0] == key:
            st.markdown(saved[1])
    st.write("")
    ui.disclaimer()
