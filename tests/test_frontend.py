"""Headless smoke tests. Run from the project root: python tests/test_frontend.py"""
import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

_TMP = Path(tempfile.mkdtemp(prefix="finwise_fe_"))
os.environ["FINWISE_DB"] = str(_TMP / "chat.db")      # keep test chats out of the real history
os.environ.pop("GROQ_API_KEY", None)                   # never call the live API from tests

import pandas as pd
from fakes import FakeEmbeddings
from utils.rag_pipeline import RAGPipeline, set_pipeline

set_pipeline(RAGPipeline(embeddings=FakeEmbeddings(), index_dir=_TMP / "idx"))  # no model download
from streamlit.testing.v1 import AppTest

from utils import financial_analyzer as fa
from utils.chatbot import answer
from utils.data_processor import ValidationError, clean_transactions, load_sample

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# ---- Data layer ----
df = load_sample()
s = fa.summary(df)
check("sample loads", len(df) > 100)
check("income/expense split", s["income"] > 0 and s["expenses"] > 0)
check("unusual transactions flagged", len(fa.unusual_transactions(df)) >= 1,
      str(fa.unusual_transactions(df)))
check("food categorised", (df[df.description.str.contains("Swiggy")].category == "Food").all())
check("rent is Bills", (df[df.description == "House Rent"].category == "Bills").all())

# ---- CSV validation ----
signed = pd.DataFrame({"Date": ["01/10/2026", "02/10/2026"], "Narration": ["Salary", "Swiggy"],
                       "Amount": [50000, -300]})
d, notes = clean_transactions(signed)
check("signed amounts", list(d["type"]) == ["Income", "Expense"])
dc = pd.DataFrame({"Date": ["2026-10-01", "2026-10-02"], "Details": ["Salary", "Uber Ride"],
                   "Debit": [None, 250], "Credit": [40000, None]})
d, _ = clean_transactions(dc)
check("debit/credit columns", list(d["type"]) == ["Income", "Expense"])
for name, bad in {
    "missing column": pd.DataFrame({"foo": [1], "bar": [2]}),
    "empty": pd.DataFrame(),
    "ambiguous type": pd.DataFrame({"date": ["2026-01-01"], "description": ["x"], "amount": [100]}),
}.items():
    try:
        clean_transactions(bad)
        check(f"rejects {name}", False)
    except ValidationError:
        check(f"rejects {name}", True)

# ---- Chatbot ----
for q in ["Where did I spend the most money?", "How much did I spend on food this month?",
          "Compare my expenses across months", "How much can I save next month?",
          "Give me a budget based on my spending", "Explain the policy in my uploaded document"]:
    r = answer(q, df, use_llm=False)
    check(f"chat: {q}", bool(r["text"]) and "nan" not in r["text"].lower(), r["text"])

# ---- Streamlit pages ----
NAV = ["📊 Dashboard", "🧾 Transactions", "💡 AI Insights", "🎯 Budget Planner",
       "🤖 AI Assistant", "⚙️ Settings"]
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
check("app boots", not at.exception, str(at.exception))
for page in NAV:
    at.radio(key="nav").set_value(page).run()
    check(f"page: {page}", not at.exception, str([e.value for e in at.exception]))

at.radio(key="nav").set_value("🤖 AI Assistant").run()
at.chat_input[0].set_value("How much did I spend on food this month?").run()
check("assistant chat flow", not at.exception and len(at.session_state["chat"]) == 2)

print("\n" + ("ALL TESTS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
