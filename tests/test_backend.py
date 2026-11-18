"""Backend tests: RAG, router, LLM guard, SQLite, and the Streamlit pages that use them.

Run from the project root:  python tests/test_backend.py

Everything runs offline: HuggingFace embeddings and Groq are replaced by the stand-ins in
tests/fakes.py. The two checks that need the real services (real embedding model, live Groq)
run only when they are reachable and are reported as SKIP otherwise.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "data"))

TMP = Path(tempfile.mkdtemp(prefix="finwise_test_"))
os.environ["FINWISE_DB"] = str(TMP / "chat.db")
os.environ.pop("GROQ_API_KEY", None)  # never call the live API from the offline tests

from fakes import BrokenLLM, FakeEmbeddings, fake_llm  # noqa: E402
from generate_sample_pdf import POLICY_PAGES, make_pdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from utils import chat_store, chatbot, llm  # noqa: E402
from utils import financial_analyzer as fa  # noqa: E402
from utils.data_processor import load_sample  # noqa: E402
from utils.facts import financial_facts  # noqa: E402
from utils.prompts import EXPLAIN_PROMPT  # noqa: E402
from utils.rag_pipeline import (DocumentError, RAGPipeline, load_knowledge, set_pipeline,  # noqa: E402
                                split_documents)

failures, skipped = [], []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def skip(name, why):
    print(f"SKIP {name}  ({why})")
    skipped.append(name)


def new_pipeline(sub="idx", emb=None):
    return RAGPipeline(embeddings=emb or FakeEmbeddings(), index_dir=TMP / sub)


chatbot.MIN_RELEVANCE = 0.0  # the stand-in embeddings are cruder than MiniLM
df = load_sample()
CUR = "₹"
PDF = make_pdf(POLICY_PAGES)

# ---------------- chunking ----------------
docs = load_knowledge()
chunks = split_documents(docs)
check("knowledge docs loaded", len(docs) >= 7, f"{len(docs)} docs")
check("chunks created and bounded", len(chunks) > len(docs) and max(len(c.page_content) for c in chunks) <= 900)

# ---------------- FAISS + retrieval ----------------
p = new_pipeline()
n = p.rebuild_knowledge()
check("index built", n == len(chunks) and (TMP / "idx" / "index.faiss").exists())
hit = p.retrieve("how many months of expenses should an emergency fund cover", k=3, min_relevance=0)[0]
check("semantic retrieval: emergency fund", hit["source"] == "emergency_fund_and_savings", str(hit["source"]))
hit = p.retrieve("what is the 50/30/20 rule needs wants savings", k=3, min_relevance=0)[0]
check("semantic retrieval: 50/30/20", hit["source"] == "budgeting_basics", str(hit["source"]))

p2 = new_pipeline()  # new object, same folder -> must load from disk, not rebuild
p2.ensure_index()
check("index persists and reloads", p2.store is not None and "budgeting_basics" in p2.indexed_sources())
check("manifest hides internal meta key", "__meta__" not in p2.indexed_sources())

p3 = new_pipeline(emb=FakeEmbeddings(dim=128))  # different vector size -> rebuild, no crash
p3.embeddings.model_name = "other-model"
r = p3.retrieve("emergency fund", k=2, min_relevance=0)
check("index rebuilt when embedding model changes", len(r) == 2)

# ---------------- PDF ingestion ----------------
p = new_pipeline("idx_pdf")
added = p.add_pdf(PDF, "policy.pdf")
check("pdf indexed", added >= 1 and p.indexed_sources()["policy.pdf"]["kind"] == "pdf")
check("pdf not re-indexed twice", p.add_pdf(PDF, "policy.pdf") == 0)
hit = p.retrieve("waiting period for large non-essential purchases above 15,000", k=2, min_relevance=0)[0]
check("pdf retrieval with page number", hit["source"] == "policy.pdf" and hit["page"] == 1, str(hit))
for label, data in {"corrupt pdf": b"not a pdf at all", "empty pdf": make_pdf([[""]])}.items():
    try:
        p.add_pdf(data, label + ".pdf")
        check(f"rejects {label}", False)
    except DocumentError:
        check(f"rejects {label}", True)

# ---------------- classification ----------------
EXPECT = {
    "Where did I spend the most money?": "transaction",
    "How much did I spend on food this month?": "transaction",
    "Compare my expenses across months": "transaction",
    "How much can I save next month?": "transaction",
    "Give me a budget based on my spending": "advice",
    "Explain the financial policy in my uploaded document": "document",
    "What is the 50/30/20 rule?": "document",
    "What is an emergency fund?": "document",
    "How can I reduce my shopping expenses?": "advice",
    "What is the difference between a credit card and a debit card?": "document",
    "hello there": "help",
}
for q, mode in EXPECT.items():
    check(f"classify: {q}", chatbot.classify(q) == mode, chatbot.classify(q))

# ---------------- number-grounding guard ----------------
ok, bad = llm.numbers_grounded("You spent ₹12,345 (23.4%).", "Spent ₹12,345 which is 23.4% of total.")
check("guard accepts grounded numbers", ok, str(bad))
ok, bad = llm.numbers_grounded("You spent ₹99,999.", "Spent ₹12,345.")
check("guard blocks invented numbers", (not ok) and "99,999" in bad)
ok, _ = llm.numbers_grounded("In 2026 save for 3 months on day 15.", "")
check("guard ignores years and small counts", ok)

# ---------------- privacy of what is sent to the LLM ----------------
facts = financial_facts(df, CUR)
leak = [w for w in ("Swiggy", "Zomato", "Netflix", "Amazon", "Salary", "Acme", "Uber") if w.lower() in facts.lower()]
check("LLM facts contain no merchants/descriptions", not leak, str(leak))

# ---------------- router: transaction (exact Pandas numbers) ----------------
last = sorted(df["month"].unique())[-1]
food = float(df[(df.month == last) & (df.type == "Expense") & (df.category == "Food")].amount.sum())
r = chatbot.answer("How much did I spend on food this month?", df, CUR, use_llm=False)
check("food spend equals Pandas", f"{CUR}{food:,.0f}" in r["text"] and r["mode"] == "transaction", r["text"])
top = fa.category_breakdown(df).iloc[0]
r = chatbot.answer("Where did I spend the most money?", df, CUR, use_llm=False)
check("top category equals Pandas", top.category in r["text"] and f"{CUR}{top.amount:,.0f}" in r["text"])
r = chatbot.answer("Compare my expenses across months", df, CUR, use_llm=False)
check("comparison table", "| Category |" in r["text"] and "nan" not in r["text"].lower())
r = chatbot.answer("How much can I save next month?", df, CUR, use_llm=False)
check("savings estimate labelled as estimate", "estimate" in r["text"].lower())

# with an LLM: grounded explanation is added; invented figures are blocked; failures degrade
good = fake_llm(f"You spent {CUR}{food:,.0f} on food. Try planning meals to trim it a little.")
r = chatbot.answer("How much did I spend on food this month?", df, CUR, llm=good)
check("grounded LLM explanation appended", "AI explanation" in r["text"] and "Groq" in r["source"], r["text"])
bad_llm = fake_llm("You spent ₹98,765 on food, which is a lot.")
r = chatbot.answer("How much did I spend on food this month?", df, CUR, llm=bad_llm)
check("invented figure is withheld", "98,765" not in r["text"] and "withheld" in r["text"] and f"{CUR}{food:,.0f}" in r["text"], r["text"])
r = chatbot.answer("How much did I spend on food this month?", df, CUR, llm=BrokenLLM(responses=["x"]))
check("LLM failure keeps exact answer", f"{CUR}{food:,.0f}" in r["text"] and "reach Groq" in r["text"], r["text"])

# ---------------- router: document (RAG) ----------------
pipe = new_pipeline("idx_chat")
pipe.add_pdf(PDF, "policy.pdf")
r = chatbot.answer("Explain the financial policy in my uploaded document", df, CUR, pipeline=pipe, use_llm=False)
check("doc route without LLM: passages + citations", r["mode"] == "document" and r["citations"] and "[1]" in r["text"], r["text"][:200])
doc_llm = fake_llm("Purchases above Rs. 15,000 need a 7-day waiting period and both partners' agreement [1].")
r = chatbot.answer("What is the waiting period for large purchases in my uploaded policy?", df, CUR, pipeline=pipe, llm=doc_llm)
check("doc route with LLM answers with citation", "waiting period" in r["text"] and "[1]" in r["text"] and "Groq" in r["source"], r["text"])
r = chatbot.answer("What is the waiting period for large purchases in my uploaded policy?", df, CUR, pipeline=pipe,
                   llm=fake_llm("The limit is Rs. 77,000 [1]."))
check("invented figure in doc answer is withheld", "77,000" not in r["text"] and "withheld" in r["text"] and r["citations"], r["text"][:300])
r = chatbot.answer("What is an emergency fund?", df, CUR, pipeline=pipe, use_llm=False)
check("knowledge question routed to RAG", r["mode"] == "document" and r["citations"])

# ---------------- router: advice (mixed) ----------------
r = chatbot.answer("Give me a budget based on my spending", df, CUR, pipeline=pipe, use_llm=False)
check("advice without LLM shows budget table", "Suggested budget" in r["text"] and r["mode"] == "advice", r["text"][:200])
plan_total = fa.category_breakdown(df)  # sanity that facts exist
tips = fake_llm("- Trim shopping a little.\n- Review subscriptions.\nThese are suggestions, not professional financial advice.")
r = chatbot.answer("How can I reduce my shopping expenses?", df, CUR, pipeline=pipe, llm=tips)
check("advice with LLM", "AI suggestions" in r["text"] and "Groq" in r["source"], r["text"][:200])

# ---------------- LLM client + prompts (real langchain-groq, no network) ----------------
msgs = EXPLAIN_PROMPT.format_messages(question="q?", facts="f")
check("prompt template renders", len(msgs) == 2 and "FACTS" in msgs[1].content)
check("no key -> no client", llm.get_llm() is None and llm.llm_status()[0] is False)
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
check("placeholder key ignored", llm.get_llm() is None)
os.environ["GROQ_API_KEY"] = "gsk_dummy_for_test"
client = llm.get_llm()
check("ChatGroq client constructed from env key", type(client).__name__ == "ChatGroq")
os.environ.pop("GROQ_API_KEY")

# ---------------- SQLite chat history ----------------
chat_store.clear()
chat_store.add_message("user", "hi")
chat_store.add_message("assistant", "hello", "Knowledge base", [{"label": "x", "score": 0.5, "text": "t"}])
hist = chat_store.load_messages()
check("chat history saved and ordered", [m["role"] for m in hist] == ["user", "assistant"] and hist[1]["citations"][0]["label"] == "x")
chat_store.clear()
check("chat history cleared", chat_store.load_messages() == [])

# ---------------- Streamlit pages ----------------
set_pipeline(new_pipeline("idx_ui"))
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
check("app boots", not at.exception, str(at.exception))
for page in ["💡 AI Insights", "🤖 AI Assistant", "⚙️ Settings"]:
    at.radio(key="nav").set_value(page).run()
    check(f"page renders: {page}", not at.exception, str([e.value for e in at.exception]))

at.radio(key="nav").set_value("🤖 AI Assistant").run()
at.button(key="sugg_What is the 50/30/20 rule?").click().run()
msgs_ = at.session_state["chat"]
check("suggestion -> RAG answer with citations", not at.exception and len(msgs_) == 2 and msgs_[1]["citations"], str(at.exception))
at.chat_input[0].set_value("How much did I spend on food this month?").run()
check("typed transaction question answered", not at.exception and len(at.session_state["chat"]) == 4)
check("chat persisted to SQLite", len(chat_store.load_messages()) == 4)

import views.insights as vi  # noqa: E402
vi.get_llm = lambda: fake_llm("March was steady. Consider trimming discretionary spending. "
                              "These are suggestions, not professional financial advice.")
at.radio(key="nav").set_value("💡 AI Insights").run()
btn = [b for b in at.button if b.label == "Generate AI explanation"][0]
check("narrative button enabled with LLM", not btn.disabled)
btn.click().run()
check("AI narrative generated", not at.exception and "narrative" in at.session_state, str(at.exception))

# ---------------- real services (only when reachable) ----------------
try:
    from utils.embeddings import get_embeddings
    vec = get_embeddings().embed_query("how much should I save?")
    check("REAL HuggingFace embeddings (384-d)", len(vec) == 384)
except Exception as e:
    skip("real HuggingFace embedding model", f"not reachable here: {type(e).__name__}")
if os.getenv("GROQ_API_KEY"):
    ok, msg = llm.test_connection()
    check("LIVE Groq connection", ok, msg)
else:
    skip("live Groq call", "no GROQ_API_KEY set")

print(f"\n{len(failures)} failed, {len(skipped)} skipped" if failures or skipped else "\nALL TESTS PASSED")
if not failures:
    print("ALL EXECUTED TESTS PASSED")
sys.exit(1 if failures else 0)
