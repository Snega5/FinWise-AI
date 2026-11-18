"""Aggregated, privacy-safe facts about the user's finances (the only data sent to the LLM).

No merchant names, descriptions or individual transactions are ever included.
"""
from __future__ import annotations

import pandas as pd

from utils import financial_analyzer as fa
from utils.budget_advisor import avg_monthly_income, suggest_budget


def _m(x: float, cur: str) -> str:
    return f"{'-' if x < 0 else ''}{cur}{abs(x):,.0f}"


def financial_facts(df: pd.DataFrame, cur: str = "₹") -> str:
    months = sorted(df["month"].unique())
    n = len(months)
    last = months[-1]
    inc = avg_monthly_income(df)
    exp = float(df.loc[df["type"] == "Expense", "amount"].sum()) / max(n, 1)
    ms = fa.summary(df[df["month"] == last])

    lines = [
        f"Data covers {fa.pretty_month(months[0])} to {fa.pretty_month(last)} ({n} month(s)).",
        f"Average monthly income: {_m(inc, cur)}. Average monthly spending: {_m(exp, cur)}. "
        f"Average monthly savings: {_m(inc - exp, cur)}.",
        f"Latest month ({fa.pretty_month(last)}): income {_m(ms['income'], cur)}, "
        f"expenses {_m(ms['expenses'], cur)}, savings rate {ms['savings_pct']:.1f}%.",
    ]

    latest = fa.category_breakdown(df[df["month"] == last])
    if not latest.empty:
        parts = ", ".join(f"{r.category} {_m(r.amount, cur)} ({r.share:.1f}%)" for r in latest.itertuples())
        lines.append(f"Latest month spending by category: {parts}.")

    cmp = fa.mom_comparison(df)
    if cmp is not None:
        t = cmp["table"]
        moves = t[t["change"].abs() >= 1].head(3)
        if not moves.empty:
            parts = ", ".join(
                f"{r.category} {'+' if r.change >= 0 else '-'}{_m(abs(r.change), cur)}" for r in moves.itertuples()
            )
            lines.append(f"Biggest increases vs {fa.pretty_month(cmp['prev'])}: {parts}.")

    plan = suggest_budget(df)
    if not plan.empty:
        parts = ", ".join(
            f"{r.category} {_m(r.suggested_budget, cur)} (could save {_m(r.potential_saving, cur)})"
            for r in plan.itertuples() if r.potential_saving > 0
        )
        if parts:
            lines.append(f"Suggested monthly budget: {parts}.")
        lines.append(f"Total potential monthly saving: {_m(float(plan['potential_saving'].sum()), cur)}.")
    return "\n".join(lines)
