"""Dashboard page."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import financial_analyzer as fa
from utils import ui
from utils.ui import BLUE, CATEGORY_COLORS, GREEN, INK


def _go_transactions():
    st.session_state["nav"] = "🧾 Transactions"


def _delta(curr: float, prev: float | None, up_is_good: bool, cur: str):
    if prev is None:
        return "", "muted"
    diff = curr - prev
    if abs(diff) < 0.5:
        return "No change vs previous month", "muted"
    arrow = "▲" if diff > 0 else "▼"
    good = (diff > 0) == up_is_good
    return f"{arrow} {ui.money(abs(diff), cur)} vs previous month", ("good" if good else "bad")


def render() -> None:
    ui.page_header("Dashboard", "Your financial snapshot at a glance")
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("No transactions loaded yet. Upload a CSV on the Transactions page.")
        st.button("Go to Transactions", on_click=_go_transactions)
        return

    cur = st.session_state.get("currency", "₹")
    months = sorted(df["month"].unique())
    labels = {fa.pretty_month(m): m for m in months}
    options = ["All time"] + list(reversed(list(labels)))
    c1, _ = st.columns([1, 3])
    choice = c1.selectbox("Period", options, index=1 if len(options) > 1 else 0)

    period = df if choice == "All time" else df[df["month"] == labels[choice]]
    s = fa.summary(period)
    prev = None
    if choice != "All time":
        idx = months.index(labels[choice])
        if idx > 0:
            prev = fa.summary(df[df["month"] == months[idx - 1]])

    inc_sub, inc_tone = _delta(s["income"], prev and prev["income"], True, cur)
    exp_sub, exp_tone = _delta(s["expenses"], prev and prev["expenses"], False, cur)
    bal_sub, bal_tone = _delta(s["balance"], prev and prev["balance"], True, cur)
    healthy = s["savings_pct"] >= 20

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ui.metric_card("Total income", ui.money(s["income"], cur), inc_sub or f"{choice}", "green", "💵", inc_tone)
    with m2:
        ui.metric_card("Total expenses", ui.money(s["expenses"], cur), exp_sub or f"{choice}", "blue", "🛍️", exp_tone)
    with m3:
        ui.metric_card("Remaining balance", ui.money(s["balance"], cur), bal_sub or f"{choice}", "indigo", "🏦", bal_tone)
    with m4:
        ui.metric_card(
            "Savings rate", f"{s['savings_pct']:.1f}%",
            "Healthy (20%+ target)" if healthy else "Below the 20% target",
            "amber", "🌱", "good" if healthy else "bad",
        )

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        with ui.card("trend"):
            ui.section_title("Monthly income vs expenses")
            t = fa.monthly_trend(df)
            fig = go.Figure()
            fig.add_bar(x=t["label"], y=t["income"], name="Income", marker_color=GREEN)
            fig.add_bar(x=t["label"], y=t["expenses"], name="Expenses", marker_color=BLUE)
            fig.add_scatter(x=t["label"], y=t["savings"], name="Net savings", mode="lines+markers",
                            line=dict(color=INK, dash="dot"))
            fig.update_layout(barmode="group")
            ui.show_chart(ui.style_fig(fig, 340))
    with right:
        with ui.card("donut"):
            ui.section_title("Expense categories")
            b = fa.category_breakdown(period)
            if b.empty:
                st.caption("No expenses in this period.")
            else:
                fig = px.pie(b, names="category", values="amount", hole=0.62,
                             color="category", color_discrete_map=CATEGORY_COLORS)
                fig.update_traces(textinfo="percent", sort=False)
                fig.update_layout(showlegend=True)
                ui.show_chart(ui.style_fig(fig, 340))

    st.write("")
    a, b2 = st.columns(2)
    with a:
        with ui.card("topcats"):
            ui.section_title("Top spending categories")
            bd = fa.category_breakdown(period)
            if bd.empty:
                st.caption("No expenses in this period.")
            else:
                bd = bd.sort_values("amount")
                fig = px.bar(bd, x="amount", y="category", orientation="h",
                             color="category", color_discrete_map=CATEGORY_COLORS)
                fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
                ui.show_chart(ui.style_fig(fig, 300))
    with b2:
        with ui.card("daily"):
            ui.section_title("Daily spending")
            daily = (period[period["type"] == "Expense"].groupby("date", as_index=False)["amount"].sum())
            if daily.empty:
                st.caption("No expenses in this period.")
            else:
                fig = px.area(daily, x="date", y="amount")
                fig.update_traces(line_color=BLUE, fillcolor="rgba(59,130,246,0.15)")
                fig.update_layout(xaxis_title=None, yaxis_title=None)
                ui.show_chart(ui.style_fig(fig, 300))

    st.write("")
    with ui.card("recent"):
        ui.section_title("Recent transactions")
        recent = period.sort_values("date", ascending=False).head(10)[
            ["date", "description", "category", "type", "amount"]
        ]
        ui.show_df(
            recent,
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "description": "Description",
                "category": "Category",
                "type": "Type",
                "amount": st.column_config.NumberColumn("Amount", format=f"{cur}%.2f"),
            },
        )
        st.button("View all transactions", on_click=_go_transactions)
