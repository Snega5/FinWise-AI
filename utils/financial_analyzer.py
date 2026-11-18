"""All financial calculations live here (Pandas/NumPy). The LLM never does arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pretty_month(m: str) -> str:
    return pd.Period(m, "M").strftime("%b %Y")


def summary(df: pd.DataFrame) -> dict:
    income = float(df.loc[df["type"] == "Income", "amount"].sum())
    expenses = float(df.loc[df["type"] == "Expense", "amount"].sum())
    balance = income - expenses
    pct = (balance / income * 100) if income > 0 else 0.0
    return {"income": income, "expenses": expenses, "balance": balance, "savings_pct": pct}


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    g = df.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
    for c in ("Income", "Expense"):
        if c not in g.columns:
            g[c] = 0.0
    out = g.rename(columns={"Income": "income", "Expense": "expenses"})[["income", "expenses"]]
    out = out.reset_index().sort_values("month")
    out["savings"] = out["income"] - out["expenses"]
    out["label"] = out["month"].map(pretty_month)
    return out


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df["type"] == "Expense"]
    out = e.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    total = out["amount"].sum()
    out["share"] = (out["amount"] / total * 100) if total else 0.0
    return out.reset_index(drop=True)


def mom_comparison(df: pd.DataFrame) -> dict | None:
    """Compare expenses by category between the two most recent months."""
    months = sorted(df["month"].unique())
    if len(months) < 2:
        return None
    prev, curr = months[-2], months[-1]
    e = df[df["type"] == "Expense"]
    a = e[e["month"] == prev].groupby("category")["amount"].sum().rename("previous")
    b = e[e["month"] == curr].groupby("category")["amount"].sum().rename("current")
    table = pd.concat([a, b], axis=1).fillna(0).reset_index()
    table.columns = ["category", "previous", "current"]
    table["change"] = table["current"] - table["previous"]
    table["change_pct"] = table["change"] / table["previous"].replace(0, np.nan) * 100
    return {"prev": prev, "curr": curr, "table": table.sort_values("change", ascending=False)}


def unusual_transactions(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Flag expenses far above the rest of their category (leave-one-out mean + 3 std)."""
    e = df[df["type"] == "Expense"]
    frames = []
    for _, g in e.groupby("category"):
        n = len(g)
        if n < min_count:
            continue
        x = g["amount"].to_numpy(dtype=float)
        s, ss = x.sum(), (x ** 2).sum()
        mean_o = (s - x) / (n - 1)
        var_o = ((ss - x ** 2) - (n - 1) * mean_o ** 2) / (n - 2)
        std_o = np.sqrt(np.clip(var_o, 0, None))
        flag = (x > mean_o + 3 * std_o) & (x > 1.5 * mean_o)
        if flag.any():
            frames.append(g[flag].assign(typical=mean_o[flag]))
    cols = ["date", "description", "amount", "category", "typical"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames)[cols].sort_values("amount", ascending=False).reset_index(drop=True)


def n_months(df: pd.DataFrame) -> int:
    return max(int(df["month"].nunique()), 1)
