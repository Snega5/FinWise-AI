"""Loading, validating and cleaning bank-statement CSV files."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils.categorizer import ALL_CATEGORIES, categorize

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"

COLUMN_ALIASES = {
    "date": ["date", "transaction date", "txn date", "value date", "posted date"],
    "description": ["description", "narration", "details", "particulars", "merchant", "remarks"],
    "amount": ["amount", "amt", "transaction amount"],
    "type": ["type", "transaction type", "dr/cr", "cr/dr", "txn type"],
}
DEBIT_COLS = ("debit", "withdrawal", "withdrawals", "debit amount", "withdrawal amt.")
CREDIT_COLS = ("credit", "deposit", "deposits", "credit amount", "deposit amt.")

INCOME_WORDS = {"income", "credit", "cr", "c", "deposit", "salary"}


class ValidationError(Exception):
    """Raised when an uploaded file cannot be used."""


def _to_number(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace(r"(?i)rs\.?|inr", "", regex=True)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _norm_type(v) -> str:
    return "Income" if str(v).strip().lower() in INCOME_WORDS else "Expense"


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalise a raw statement.

    Returns (clean_df, warnings). Raises ValidationError for unusable files.
    Output columns: date, description, amount (>0), type (Income/Expense), category, month.
    """
    if raw is None or raw.empty:
        raise ValidationError("The file is empty.")

    notes: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    mapping: dict[str, str] = {}
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns and a not in mapping:
                mapping[a] = std
                break
    df = df.rename(columns=mapping)

    # Separate debit / credit columns -> one signed amount column.
    if "amount" not in df.columns:
        debit = next((c for c in DEBIT_COLS if c in df.columns), None)
        credit = next((c for c in CREDIT_COLS if c in df.columns), None)
        if debit or credit:
            d = _to_number(df[debit]).fillna(0) if debit else 0
            c = _to_number(df[credit]).fillna(0) if credit else 0
            df["amount"] = c - d
            notes.append("Used separate debit/credit columns to work out income and expenses.")

    missing = [c for c in ("date", "description", "amount") if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Your CSV needs: date, description, amount (and ideally type)."
        )

    # Dates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().mean() > 0.3:
            alt = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            if alt.isna().sum() < dates.isna().sum():
                dates = alt
    df["date"] = dates

    # Amounts
    df["amount"] = _to_number(df["amount"])

    bad = df["date"].isna() | df["amount"].isna()
    if bad.any():
        notes.append(f"Skipped {int(bad.sum())} row(s) with an invalid date or amount.")
        df = df[~bad].copy()
    if df.empty:
        raise ValidationError("No valid rows found. Check the date and amount columns.")

    # Type
    if "type" in df.columns:
        df["type"] = df["type"].map(_norm_type)
    else:
        if (df["amount"] >= 0).all():
            raise ValidationError(
                "No 'type' column found and all amounts are positive, so income and "
                "expenses can't be told apart. Add a 'type' column (Income/Expense) "
                "or use negative amounts for expenses."
            )
        df["type"] = np.where(df["amount"] >= 0, "Income", "Expense")
        notes.append("No 'type' column: treated negative amounts as expenses.")
    df["amount"] = df["amount"].abs().round(2)

    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # Category
    auto = [categorize(d, t) for d, t in zip(df["description"], df["type"])]
    if "category" in df.columns:
        given = df["category"].astype(str).str.strip().str.title()
        valid = given.isin(ALL_CATEGORIES)
        df["category"] = np.where(valid, given, auto)
    else:
        df["category"] = auto

    df["month"] = df["date"].dt.strftime("%Y-%m")
    df = df[["date", "description", "amount", "type", "category", "month"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df, notes


def load_csv(file_or_path) -> tuple[pd.DataFrame, list[str]]:
    return clean_transactions(pd.read_csv(file_or_path))


def load_sample() -> pd.DataFrame:
    df, _ = load_csv(SAMPLE_PATH)
    return df
