"""Groq LLM integration: client setup, LangChain chains, and the number-grounding guard."""
from __future__ import annotations

import os
import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser

from utils.prompts import ADVICE_PROMPT, DOC_QA_PROMPT, EXPLAIN_PROMPT, NARRATIVE_PROMPT

DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMError(Exception):
    """Any problem calling or trusting the language model (message is user-friendly)."""


class UngroundedOutput(LLMError):
    """The model produced figures that are not in the data it was given."""


# ---------- client ----------

def _api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return "" if (not key or key.startswith("your_")) else key


@lru_cache(maxsize=4)
def _make_llm(key: str, model: str):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, api_key=key, temperature=0.2, max_retries=2, timeout=30)


def get_llm():
    """Return a ChatGroq client, or None when no usable key is configured."""
    key = _api_key()
    if not key:
        return None
    return _make_llm(key, os.getenv("GROQ_MODEL", DEFAULT_MODEL))


def llm_status() -> tuple[bool, str]:
    if _api_key():
        return True, f"Groq ready (model: {os.getenv('GROQ_MODEL', DEFAULT_MODEL)})"
    return False, "No Groq API key found. Answers use Pandas and retrieved passages only."


def test_connection() -> tuple[bool, str]:
    """Make one tiny call to Groq so the Settings page can confirm the key really works."""
    llm = get_llm()
    if llm is None:
        return False, "No Groq API key found. Add GROQ_API_KEY to your .env file."
    try:
        llm.invoke("Reply with the single word OK.")
    except Exception as e:
        return False, _friendly(e)
    return True, "Connected to Groq successfully."


# ---------- running chains safely ----------

def _friendly(e: Exception) -> str:
    name, msg = type(e).__name__, str(e).lower()
    if "401" in msg or "invalid api key" in msg or "authentication" in name.lower():
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
    if "notfound" in name.lower() or "404" in msg:
        return ("Groq could not find that model. Set GROQ_MODEL in .env to a current model "
                "(see console.groq.com/docs/models).")
    if "429" in msg or "rate" in msg:
        return "Groq's free-tier rate limit was reached. Wait a minute and try again."
    if "connect" in msg or "timeout" in msg or "timed out" in msg:
        return "Could not reach Groq. Check your internet connection."
    return f"The language model call failed ({name})."


def _run(prompt, llm, inputs: dict) -> str:
    chain = prompt | llm | StrOutputParser()  # LangChain runnable pipeline
    try:
        return chain.invoke(inputs).strip()
    except Exception as e:  # network, auth, rate limit, etc.
        raise LLMError(_friendly(e)) from e


# ---------- number grounding guard ----------

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_float(tok: str) -> float | None:
    try:
        return float(tok.replace(",", "").rstrip("."))
    except ValueError:
        return None


def numbers_grounded(output: str, source: str) -> tuple[bool, list[str]]:
    """Check that every meaningful number in `output` also appears in `source`.

    Ignores small whole numbers (counts, days, list markers) and years. Allows
    rounding to the nearest whole unit for larger numbers.
    """
    allowed = [v for v in (_to_float(t) for t in _NUM.findall(source)) if v is not None]
    bad: list[str] = []
    for tok in _NUM.findall(output):
        v = _to_float(tok)
        if v is None:
            continue
        if v == int(v) and (v <= 31 or 2000 <= v <= 2100):
            continue
        tol = 0.5 if v >= 100 else 0.051
        if not any(abs(v - a) <= tol for a in allowed):
            bad.append(tok)
    return (not bad), bad


def _guard(text: str, *sources: str) -> str:
    ok, _bad = numbers_grounded(text, "\n".join(sources))
    if not ok:
        # Deliberately do not echo the rejected figures: the user should never see invented numbers.
        raise UngroundedOutput(
            "The AI explanation was withheld because it contained figures that are not in your data. "
            "The calculated figures shown are accurate."
        )
    return text


# ---------- chains ----------

def answer_from_context(llm, question: str, context: str) -> str:
    text = _run(DOC_QA_PROMPT, llm, {"question": question, "context": context})
    return _guard(text, context)


def explain_facts(llm, question: str, facts: str) -> str:
    text = _run(EXPLAIN_PROMPT, llm, {"question": question, "facts": facts})
    return _guard(text, facts)


def advise(llm, question: str, facts: str, context: str) -> str:
    text = _run(ADVICE_PROMPT, llm, {"question": question, "facts": facts, "context": context or "(none)"})
    return _guard(text, facts, context)


def narrate(llm, facts: str) -> str:
    text = _run(NARRATIVE_PROMPT, llm, {"facts": facts})
    return _guard(text, facts)