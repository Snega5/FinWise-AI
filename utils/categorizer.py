"""Rule-based transaction categorizer (keyword matching, no external services)."""
from __future__ import annotations

import re

CATEGORIES = ["Food", "Shopping", "Transport", "Bills", "Entertainment", "Healthcare", "Other"]
ALL_CATEGORIES = CATEGORIES + ["Income"]

# Order matters: the first category whose keyword matches wins.
KEYWORDS: dict[str, list[str]] = {
    "Food": [
        "swiggy", "zomato", "uber eats", "bigbasket", "dmart", "blinkit", "zepto", "grocer",
        "restaurant", "cafe", "starbucks", "domino", "pizza", "kfc", "mcdonald", "biryani",
        "bakery", "dunzo", "food", "dining", "chai",
    ],
    "Healthcare": [
        "pharmacy", "apollo", "hospital", "clinic", "doctor", "dr ", "medic", "dental",
        "diagnostic", "lab ", "netmeds", "pharmeasy", "1mg", "health",
    ],
    "Bills": [
        "electricity", "tneb", "bescom", "broadband", "airtel", "jio", "vodafone", "recharge",
        "rent", "insurance", "emi", "water bill", "gas bill", "lpg", "bill", "wifi", "internet",
    ],
    "Entertainment": [
        "netflix", "spotify", "prime video", "hotstar", "bookmyshow", "pvr", "inox", "movie",
        "steam", "youtube premium", "gaming", "concert",
    ],
    "Transport": [
        "uber", "ola", "rapido", "metro", "petrol", "fuel", "irctc", "redbus", "bus ",
        "taxi", "parking", "toll", "fastag", "indigo", "airlines",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho", "mall", "store",
        "decathlon", "ikea", "croma", "reliance digital", "shopping",
    ],
}

_PATTERNS = {
    cat: re.compile(r"\b(?:" + "|".join(re.escape(k.strip()) for k in kws) + r")", re.IGNORECASE)
    for cat, kws in KEYWORDS.items()
}


def categorize(description: str, txn_type: str = "Expense") -> str:
    """Return a category for a transaction description."""
    if txn_type == "Income":
        return "Income"
    text = str(description)
    for cat, pattern in _PATTERNS.items():
        if pattern.search(text):
            return cat
    return "Other"
