"""Chat router: decides how each question is answered.

Three routes (the key rule: numbers about the user's money always come from Pandas):

* transaction  -> Pandas calculates the answer; Groq (optional) only explains the result.
* document     -> RAG: FAISS retrieves passages from the knowledge base / uploaded PDFs and
                  Groq answers from that context only (with [n] citations).
* advice       -> mixed: Pandas facts + retrieved guidance -> Groq writes suggestions.

Without a Groq key everything still works: transaction answers are exact, and document
questions return the most relevant retrieved passages.
"""
from __future__ import annotations

import calendar
import re

import pandas as pd

from utils import financial_analyzer as fa
from utils.budget_advisor import avg_monthly_income, suggest_budget
from utils.facts import financial_facts
from utils.llm import LLMError, advise, answer_from_context, explain_facts, get_llm
from utils.rag_pipeline import DocumentError, EmbeddingError, cite, format_context, get_pipeline

MIN_RELEVANCE = 0.2
TOP_K = 4

CATEGORY_WORDS = {
    "Food": ["food", "eat", "dining", "restaurant", "grocer", "swiggy", "zomato"],
    "Shopping": ["shopping", "shop", "amazon", "flipkart"],
    "Transport": ["transport", "travel", "commute", "uber", "fuel", "petrol", "cab"],
    "Bills": ["bill", "rent", "utilit", "recharge"],
    "Entertainment": ["entertain", "movie", "subscription", "netflix"],
    "Healthcare": ["health", "medical", "doctor", "pharmacy", "medicine"],
    "Other": ["other", "misc"],
}
MONTH_NUMBERS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}

# ---------- question classification ----------

DOC_WORDS = re.compile(r"\b(documents?|policy|policies|pdfs?|uploaded|upload|attachment|guides?)\b")
PERSONAL = re.compile(r"\b(my|i|me|mine|we|our)\b|\b(this|last|previous|current|latest) month\b")
ADVICE = re.compile(
    r"\bhow (can|do|could|should) i\b|\bhow to\b|\btips?\b|\badvice\b|\bsuggest|\brecommend|"
    r"\breduce\b|\bcut (down|back)\b|\bimprove\b|\bbudget\b|\bshould i\b"
)
KNOWLEDGE = re.compile(
    r"\bwhat (is|are|does)\b|what's|\bexplain\b|\bdefine\b|\bmeaning of\b|\bhow does\b|\bwhy\b|\brule\b|"
    r"emergency fund|compound interest|credit score|credit card|\bdebt\b|\bloan\b|insurance|invest|"
    r"mutual fund|inflation|net worth|\btax"
)
COMPARE = re.compile(r"compar|\bversus\b|\bvs\b")
SPEND = re.compile(r"\b(spend|spent|spending|much|cost|costs|expenses?|total)\b")
TOP = re.compile(r"\b(most|highest|biggest|top|largest)\b")
SAVE = re.compile(r"\bsave\b|\bsaving|\bsavings\b|\bnext month\b|\bafford\b")
OVERVIEW = re.compile(r"\b(total|income|earn|earned|balance|expenses?|spending|spent|spend)\b")


def _word(w: str, q: str) -> bool:
    return re.search(r"\b" + re.escape(w), q) is not None


def _find_category(q: str) -> str | None:
    for cat, words in CATEGORY_WORDS.items():
        if any(_word(w, q) for w in words):
            return cat
    return None


def classify(question: str) -> str:
    """Return 'document', 'advice', 'transaction' or 'help'."""
    q = question.lower().strip()
    personal = PERSONAL.search(q) is not None
    if DOC_WORDS.search(q):
        return "document"
    if not personal and KNOWLEDGE.search(q):
        return "document"
    if ADVICE.search(q):
        return "advice"
    if (COMPARE.search(q) or SAVE.search(q) or TOP.search(q) or OVERVIEW.search(q)
            or _find_category(q)):
        return "transaction"
    if KNOWLEDGE.search(q):
        return "document"
    return "help"


# ---------- helpers ----------

def _money(x: float, cur: str) -> str:
    return f"{cur}{x:,.0f}"


def _out(text: str, source: str, mode: str, citations: list | None = None) -> dict:
    return {"text": text, "source": source, "mode": mode, "citations": citations or []}


def _resolve_month(q: str, months: list[str]) -> tuple[str | None, str]:
    if re.search(r"last month|previous month", q):
        m = months[-2] if len(months) > 1 else months[-1]
        return m, fa.pretty_month(m)
    if re.search(r"this month|current month|latest month", q):
        return months[-1], f"{fa.pretty_month(months[-1])} (latest month in your data)"
    for name, num in MONTH_NUMBERS.items():
        if re.search(rf"\b{name}\b", q):
            for m in reversed(months):
                if int(m[5:7]) == num:
                    return m, fa.pretty_month(m)
    return None, "across all months in your data"


def _scope(df: pd.DataFrame, month: str | None) -> pd.DataFrame:
    return df if month is None else df[df["month"] == month]


# ---------- transaction handlers: each returns {"text", "facts"} ----------
# "text" is what the user sees (exact Pandas figures). "facts" is what Groq may see:
# aggregated numbers only, with no merchant names or descriptions.

def _category_spend(df, cat, month, label, cur):
    e = _scope(df, month)
    e = e[e["type"] == "Expense"]
    total = float(e["amount"].sum())
    c = e[e["category"] == cat]
    spent = float(c["amount"].sum())
    if c.empty:
        msg = f"I found no **{cat}** expenses {label}."
        return {"text": msg, "facts": msg.replace("**", "")}
    share = spent / total * 100 if total else 0
    big = c.sort_values("amount", ascending=False).iloc[0]
    text = (f"You spent **{_money(spent, cur)}** on **{cat}** {label}: {len(c)} transactions, "
            f"{share:.1f}% of your total spending. Largest: *{big['description']}* "
            f"({_money(float(big['amount']), cur)}).")
    facts = (f"Category: {cat}. Period: {label}. Spent {_money(spent, cur)} across {len(c)} transactions, "
             f"which is {share:.1f}% of total spending of {_money(total, cur)}. "
             f"The largest single {cat} transaction was {_money(float(big['amount']), cur)}.")
    return {"text": text, "facts": facts}


def _top_categories(df, month, label, cur):
    b = fa.category_breakdown(_scope(df, month))
    if b.empty:
        msg = "There are no expenses to analyse for that period."
        return {"text": msg, "facts": msg}
    rows = list(b.head(3).itertuples())
    lines = [f"{i}. **{r.category}**: {_money(r.amount, cur)} ({r.share:.1f}%)" for i, r in enumerate(rows, 1)]
    facts = f"Highest spending categories {label}: " + "; ".join(
        f"{r.category} {_money(r.amount, cur)} ({r.share:.1f}% of spending)" for r in rows) + "."
    return {"text": f"Your highest spending categories {label}:\n\n" + "\n".join(lines), "facts": facts}


def _compare(df, cur):
    cmp = fa.mom_comparison(df)
    if cmp is None:
        msg = "I need at least two months of data to compare."
        return {"text": msg, "facts": msg}
    t = cmp["table"]
    prev, curr = fa.pretty_month(cmp["prev"]), fa.pretty_month(cmp["curr"])
    rows = [f"| Category | {prev} | {curr} | Change |", "|---|---:|---:|---:|"]
    facts = [f"Expense comparison, {prev} versus {curr}."]
    for r in t.itertuples():
        sign = "+" if r.change >= 0 else "-"
        rows.append(f"| {r.category} | {_money(r.previous, cur)} | {_money(r.current, cur)} | "
                    f"{sign}{_money(abs(r.change), cur)} |")
        facts.append(f"{r.category}: {prev} {_money(r.previous, cur)}, {curr} {_money(r.current, cur)}, "
                     f"change {sign}{_money(abs(r.change), cur)}.")
    tot_p, tot_c = float(t["previous"].sum()), float(t["current"].sum())
    sign = "+" if tot_c >= tot_p else "-"
    rows.append(f"| **Total** | **{_money(tot_p, cur)}** | **{_money(tot_c, cur)}** | "
                f"**{sign}{_money(abs(tot_c - tot_p), cur)}** |")
    facts.append(f"Total: {prev} {_money(tot_p, cur)}, {curr} {_money(tot_c, cur)}, "
                 f"change {sign}{_money(abs(tot_c - tot_p), cur)}.")
    return {"text": "Expense comparison between your two most recent months:\n\n" + "\n".join(rows),
            "facts": "\n".join(facts)}


def _savings(df, cur):
    recent = df[df["month"].isin(sorted(df["month"].unique())[-3:])]
    n = fa.n_months(recent)
    inc = avg_monthly_income(recent)
    exp = float(recent.loc[recent["type"] == "Expense", "amount"].sum()) / n
    plan = suggest_budget(recent)
    keep = max(inc - exp, 0)
    planned = max(inc - float(plan["suggested_budget"].sum()), 0)
    text = (f"Based on your last {n} month(s): average income **{_money(inc, cur)}**, average spending "
            f"**{_money(exp, cur)}**, so you typically keep about **{_money(keep, cur)}** per month. "
            f"Following the suggested budget could raise that to roughly **{_money(planned, cur)}**. "
            "This is an estimate from past data, not a guarantee.")
    facts = (f"Based on the last {n} month(s): average monthly income {_money(inc, cur)}; average monthly "
             f"spending {_money(exp, cur)}; typical monthly saving {_money(keep, cur)}; estimated monthly "
             f"saving if the suggested budget is followed {_money(planned, cur)}. This is an estimate.")
    return {"text": text, "facts": facts}


def _budget(df, cur):
    plan = suggest_budget(df)
    rows = ["| Category | Avg monthly spend | Suggested budget |", "|---|---:|---:|"]
    facts = ["Suggested monthly budget by category (average spend -> suggested budget):"]
    for r in plan.itertuples():
        rows.append(f"| {r.category} | {_money(r.avg_spend, cur)} | {_money(r.suggested_budget, cur)} |")
        facts.append(f"{r.category}: {_money(r.avg_spend, cur)} -> {_money(r.suggested_budget, cur)}.")
    tot_a, tot_b = float(plan["avg_spend"].sum()), float(plan["suggested_budget"].sum())
    rows.append(f"| **Total** | **{_money(tot_a, cur)}** | **{_money(tot_b, cur)}** |")
    facts.append(f"Total: {_money(tot_a, cur)} -> {_money(tot_b, cur)}.")
    return {"text": "A suggested monthly budget based on your history:\n\n" + "\n".join(rows),
            "facts": "\n".join(facts)}


def _overview(df, month, label, cur):
    s = fa.summary(_scope(df, month))
    text = (f"For {label}: income **{_money(s['income'], cur)}**, expenses **{_money(s['expenses'], cur)}**, "
            f"remaining balance **{_money(s['balance'], cur)}** (savings rate {s['savings_pct']:.1f}%).")
    facts = (f"Period: {label}. Income {_money(s['income'], cur)}. Expenses {_money(s['expenses'], cur)}. "
             f"Remaining balance {_money(s['balance'], cur)}. Savings rate {s['savings_pct']:.1f}%.")
    return {"text": text, "facts": facts}


def _transaction(q, df, cur):
    months = sorted(df["month"].unique())
    cat = _find_category(q)
    month, label = _resolve_month(q, months)
    if COMPARE.search(q):
        return _compare(df, cur)
    if cat and SPEND.search(q):
        return _category_spend(df, cat, month, label, cur)
    if TOP.search(q):
        return _top_categories(df, month, label, cur)
    if SAVE.search(q):
        return _savings(df, cur)
    if cat:
        return _category_spend(df, cat, month, label, cur)
    return _overview(df, month, label, cur)


# ---------- retrieval helpers ----------

def _retrieve(question: str, pipeline, k: int = TOP_K) -> tuple[list[dict], str | None]:
    """Search the vector store. Returns (hits, error_message). Never raises."""
    try:
        return pipeline.retrieve(question, k=k, min_relevance=MIN_RELEVANCE), None
    except (EmbeddingError, DocumentError) as e:
        return [], str(e)
    except Exception as e:
        return [], f"Document search failed ({type(e).__name__})."


def _citations(hits: list[dict]) -> list[dict]:
    return [{"label": cite(h), "score": h["score"], "text": h["text"]} for h in hits]


def _snippet(text: str, n: int = 350) -> str:
    s = " ".join(text.split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def _extractive(hits: list[dict]) -> str:
    return "\n\n".join(f"**[{i}] {cite(h)}**: {_snippet(h['text'])}" for i, h in enumerate(hits, 1))


def _doc_source(hits: list[dict]) -> str:
    return "Uploaded PDF" if any(h.get("kind") == "pdf" for h in hits) else "Knowledge base"


# ---------- routes ----------

def _document(question, pipeline, llm) -> dict:
    hits, err = _retrieve(question, pipeline)
    if err:
        return _out(f"I couldn't search the documents. {err}", "System", "document")
    if not hits:
        return _out("I couldn't find anything relevant in the knowledge base or your uploaded documents. "
                    "Try rephrasing, or upload a PDF about this topic.", _doc_source([]), "document")
    cites = _citations(hits)
    src = _doc_source(hits)
    if llm is None:
        return _out("Here are the most relevant passages I found. Add a free Groq API key to get a "
                    "written answer built from them.\n\n" + _extractive(hits), f"{src} (retrieval only)",
                    "document", cites)
    try:
        text = answer_from_context(llm, question, format_context(hits))
        return _out(text, f"{src} + Groq", "document", cites)
    except LLMError as e:
        return _out(f"*{e}* Here are the retrieved passages instead:\n\n" + _extractive(hits),
                    f"{src} (retrieval only)", "document", cites)


def _advice(q, question, df, cur, pipeline, llm) -> dict:
    parts: list[str] = []
    if re.search(r"\bbudget\b", q):
        parts.append(_budget(df, cur)["text"])
    hits, _ = _retrieve(question, pipeline, k=3)
    cites = _citations(hits)
    source = "Transactions (Pandas)" + (" + knowledge base" if hits else "")
    if llm is not None:
        try:
            tips = advise(llm, question, financial_facts(df, cur), format_context(hits))
            parts.append("**💬 AI suggestions**\n\n" + tips)
            return _out("\n\n".join(parts), source + " + Groq", "advice", cites)
        except LLMError as e:
            parts.append(f"*{e}*")
    if hits:
        parts.append("**Relevant guidance**\n\n" + _extractive(hits[:2]))
    if not parts or (len(parts) == 1 and parts[0].startswith("*")):
        parts.insert(0, _savings(df, cur)["text"])
    return _out("\n\n".join(parts), source, "advice", cites)


def _transaction_route(q, question, df, cur, llm) -> dict:
    res = _transaction(q, df, cur)
    text, source = res["text"], "Transactions (Pandas)"
    if llm is not None:
        try:
            text += "\n\n**💬 AI explanation**\n\n" + explain_facts(llm, question, res["facts"])
            source += " + Groq explanation"
        except LLMError as e:
            text += f"\n\n*{e}*"
    return _out(text, source, "transaction")


HELP_TEXT = (
    "I can help with questions like *where did I spend the most*, *how much did I spend on food this "
    "month*, *compare my expenses across months*, *how much can I save next month*, *give me a budget*, "
    "or general money questions such as *what is the 50/30/20 rule?* and questions about your uploaded PDFs."
)


def answer(question: str, df: pd.DataFrame | None, currency: str = "₹", pipeline=None, llm=None,
           use_llm: bool = True) -> dict:
    """Answer one question. Returns {'text', 'source', 'mode', 'citations'}.

    `pipeline` / `llm` can be injected (tests); otherwise the shared pipeline and the Groq client
    from the environment are used. With no Groq key the answer is built without the LLM.
    """
    q = question.lower().strip()
    if not q:
        return _out(HELP_TEXT, "System", "help")
    llm = (llm if llm is not None else get_llm()) if use_llm else None
    mode = classify(q)

    if mode == "document":
        return _document(question, pipeline or get_pipeline(), llm)
    if mode == "help":
        return _out(HELP_TEXT, "System", "help")
    if df is None or df.empty:
        return _out("Please load some transactions first (Transactions page).", "System", mode)
    if mode == "advice":
        return _advice(q, question, df, currency, pipeline or get_pipeline(), llm)
    return _transaction_route(q, question, df, currency, llm)
