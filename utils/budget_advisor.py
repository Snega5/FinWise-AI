"""Rule-based budget suggestions computed from historical spending."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from utils.financial_analyzer import n_months

# Share of average spend we suggest trimming per category (0 = treat as fixed).
REDUCE_RATE = {
    "Food": 0.10, "Shopping": 0.20, "Entertainment": 0.25, "Other": 0.15,
    "Transport": 0.05, "Bills": 0.0, "Healthcare": 0.0,
}
TIPS = {
    "Food": "Plan weekly meals and cut back on delivery orders.",
    "Shopping": "Add a 48-hour wait before non-essential purchases.",
    "Entertainment": "Review subscriptions and pick free or shared options.",
    "Other": "Review uncategorised spends; many can be reclassified or trimmed.",
    "Transport": "Combine trips and use public transport where possible.",
    "Bills": "Mostly fixed. Compare plans yearly for a better rate.",
    "Healthcare": "Essential. Keep as is.",
}


def avg_monthly_income(df: pd.DataFrame) -> float:
    return float(df.loc[df["type"] == "Income", "amount"].sum()) / n_months(df)


def suggest_budget(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df["type"] == "Expense"]
    avg = (e.groupby("category")["amount"].sum() / n_months(df)).sort_values(ascending=False)
    out = avg.rename("avg_spend").reset_index()
    rate = out["category"].map(REDUCE_RATE).fillna(0.1)
    out["suggested_budget"] = (np.round(out["avg_spend"] * (1 - rate) / 10) * 10).clip(lower=0)
    out["potential_saving"] = (out["avg_spend"] - out["suggested_budget"]).clip(lower=0)
    out["tip"] = out["category"].map(TIPS).fillna("")
    return out


def months_to_goal(target: float, saved: float, monthly: float) -> int | None:
    remaining = max(target - saved, 0)
    if remaining == 0:
        return 0
    if monthly <= 0:
        return None
    return math.ceil(remaining / monthly)
