"""Budget Planner page: suggested budgets, budget vs actual, savings goal."""
import plotly.graph_objects as go
import streamlit as st

from utils import financial_analyzer as fa
from utils import ui
from utils.budget_advisor import avg_monthly_income, months_to_goal, suggest_budget
from utils.ui import BLUE, GREEN


def render() -> None:
    ui.page_header("Budget Planner", "Budgets based on your history, and progress toward your goals")
    df = st.session_state.get("df")
    if df is None or df.empty or not (df["type"] == "Expense").any():
        st.info("Load transactions with some expenses to build a budget.")
        return
    cur = st.session_state.get("currency", "₹")

    plan = suggest_budget(df)
    avg_inc = avg_monthly_income(df)
    total_avg = float(plan["avg_spend"].sum())
    total_budget = float(plan["suggested_budget"].sum())
    potential = max(total_avg - total_budget, 0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ui.metric_card("Avg monthly income", ui.money(avg_inc, cur), "", "green", "💵")
    with m2:
        ui.metric_card("Avg monthly spending", ui.money(total_avg, cur), "", "blue", "🛍️")
    with m3:
        ui.metric_card("Suggested budget", ui.money(total_budget, cur), "per month", "indigo", "🎯")
    with m4:
        ui.metric_card("Potential savings", ui.money(potential, cur), "per month", "amber", "🌱", "good")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Suggested budget", "Budget vs actual", "Savings goal"])

    with tab1:
        with ui.card("suggested"):
            ui.section_title("Average spend vs suggested budget")
            fig = go.Figure()
            fig.add_bar(x=plan["category"], y=plan["avg_spend"], name="Average spend", marker_color=BLUE)
            fig.add_bar(x=plan["category"], y=plan["suggested_budget"], name="Suggested budget", marker_color=GREEN)
            fig.update_layout(barmode="group")
            ui.show_chart(ui.style_fig(fig, 320))

            st.caption("Adjust **Your budget** to set your own limits.")
            table = plan[["category", "avg_spend", "suggested_budget"]].copy()
            saved = st.session_state.get("budget", {})
            table["your_budget"] = [float(saved.get(c, s)) for c, s in zip(table["category"], table["suggested_budget"])]
            edited = st.data_editor(
                table,
                key=f"budget_editor_{abs(hash(tuple(plan['category'])))}",
                hide_index=True,
                disabled=["category", "avg_spend", "suggested_budget"],
                column_config={
                    "category": "Category",
                    "avg_spend": st.column_config.NumberColumn("Avg spend", format=f"{cur}%.0f"),
                    "suggested_budget": st.column_config.NumberColumn("Suggested", format=f"{cur}%.0f"),
                    "your_budget": st.column_config.NumberColumn("Your budget", format=f"{cur}%.0f",
                                                                  min_value=0, step=100),
                },
                **ui.stretch(),
            )
            st.session_state.budget = dict(zip(edited["category"], edited["your_budget"]))

        st.write("")
        with ui.card("reduce"):
            ui.section_title("Where you can cut back")
            reducible = plan[plan["potential_saving"] > 0].sort_values("potential_saving", ascending=False)
            if reducible.empty:
                st.caption("No obvious savings found.")
            for r in reducible.itertuples():
                st.markdown(f"**{r.category}**: save about **{ui.money(r.potential_saving, cur)}**/month. {r.tip}")

    with tab2:
        months = sorted(df["month"].unique())
        labels = {fa.pretty_month(m): m for m in months}
        pick = st.selectbox("Month", list(reversed(list(labels))))
        spent = (df[(df["type"] == "Expense") & (df["month"] == labels[pick])]
                 .groupby("category")["amount"].sum())
        budgets = st.session_state.get("budget", {})
        with ui.card("vsactual"):
            ui.section_title(f"{pick}: budget vs actual")
            for r in plan.itertuples():
                limit = float(budgets.get(r.category, r.suggested_budget))
                actual = float(spent.get(r.category, 0.0))
                ratio = actual / limit if limit > 0 else (1.0 if actual > 0 else 0.0)
                status = "🔴 Over budget" if ratio > 1 else ("🟡 Close to limit" if ratio > 0.85 else "🟢 On track")
                st.markdown(f"**{r.category}** &nbsp; {ui.money(actual, cur)} of {ui.money(limit, cur)} &nbsp; {status}")
                st.progress(min(ratio, 1.0))

    with tab3:
        goal = st.session_state.get("goal", {"name": "Emergency fund", "target": 100000.0, "saved": 10000.0})
        with ui.card("goal"):
            ui.section_title("Savings goal")
            with st.form("goal_form"):
                name = st.text_input("Goal name", goal["name"])
                g1, g2 = st.columns(2)
                target = g1.number_input(f"Target amount ({cur})", min_value=1.0, value=float(goal["target"]), step=1000.0)
                saved = g2.number_input(f"Already saved ({cur})", min_value=0.0, value=float(goal["saved"]), step=1000.0)
                if st.form_submit_button("Save goal", type="primary"):
                    st.session_state.goal = {"name": name, "target": target, "saved": saved}
                    st.rerun()

            progress = min(goal["saved"] / goal["target"], 1.0)
            st.progress(progress, text=f"{goal['name']}: {ui.money(goal['saved'], cur)} of "
                                       f"{ui.money(goal['target'], cur)} ({progress * 100:.0f}%)")

            now_monthly = max(avg_inc - total_avg, 0)
            plan_monthly = max(avg_inc - total_budget, 0)
            n_now = months_to_goal(goal["target"], goal["saved"], now_monthly)
            n_plan = months_to_goal(goal["target"], goal["saved"], plan_monthly)

            def fmt(n):
                return "Not reachable at this rate" if n is None else ("Goal reached 🎉" if n == 0 else f"{n} month(s)")

            c1, c2 = st.columns(2)
            c1.metric("At your current habits", fmt(n_now), f"saving ~{ui.money(now_monthly, cur)}/month")
            c2.metric("With the suggested budget", fmt(n_plan), f"saving ~{ui.money(plan_monthly, cur)}/month")
            st.caption("Estimates based on your historical averages. Real results will vary.")
    st.write("")
    ui.disclaimer()
