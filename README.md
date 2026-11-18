# FinWise AI: Your Personal AI Financial Advisor

Upload a bank-statement CSV, see where your money goes, and chat with an AI assistant that
answers from **your real numbers** (calculated with Pandas) and from **financial guides and PDFs**
(RAG). Built only with free, open-source tools.

| Layer | Tool |
|---|---|
| UI | Streamlit + Plotly |
| Calculations | Pandas / NumPy (the LLM never does arithmetic) |
| LLM | Groq free tier through LangChain (`ChatGroq`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2`, run locally |
| Vector DB | FAISS, stored locally in `models/faiss_index` |
| PDFs / chunking | PyPDF + LangChain `RecursiveCharacterTextSplitter` |
| Chat history | SQLite (`data/chat_history.db`) |

## 1. Install

Use Python 3.10 or 3.11.

```bash
cd FinWise-AI
python -m venv venv
# Windows:   venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The install downloads PyTorch (about 1-2 GB), so allow some time and disk space.

## 2. Get your free Groq API key

1. Go to https://console.groq.com/keys and sign up (free).
2. Click **Create API Key** and copy it.
3. Copy `.env.example` to `.env` and paste the key:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
4. Never share or commit `.env`.

The app also works **without** a key: numbers are still exact and document questions return the
most relevant passages. The key adds written explanations and suggestions.

## 3. Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501 with synthetic sample data loaded. The first document question (or
**Settings > Build knowledge index**) downloads the embedding model once (about 90 MB, internet
needed), then everything runs offline except the Groq call.

Check **Settings > Test Groq connection** to confirm the key works.

## 4. How the AI part works

```
question -> router (utils/chatbot.py)
   |-- transaction question -> Pandas computes the figures -> Groq only explains them
   |-- document question    -> RAG: FAISS retrieves passages -> Groq answers from them [with citations]
   '-- advice question      -> Pandas facts + retrieved guidance -> Groq writes suggestions
```

RAG pipeline (`utils/rag_pipeline.py`): PyPDF extracts text -> LangChain splits it into chunks (800
characters, 120 overlap) -> HuggingFace embeds them locally -> FAISS stores them -> similarity
search retrieves the top passages -> a LangChain pipeline (`prompt | ChatGroq | parser`) answers
using only that context.

Safety built in:
- **No invented numbers.** Every figure is computed in Python. If the LLM output contains a number
  that is not in the data it was given, the output is blocked and the exact answer is shown.
- **Privacy.** Only aggregated figures and short retrieved passages go to Groq, never raw
  transactions or merchant names. Files stay on your computer.
- **Failure handling.** Missing key, bad key, rate limit, no internet, unreadable PDF and model
  download problems all produce a clear message instead of a crash.
- AI output is labelled as suggestions, not professional financial advice.

## 5. Folder structure

```
FinWise-AI/
  app.py                     entry point, sidebar navigation
  requirements.txt  .env.example  .gitignore
  .streamlit/config.toml     light theme
  data/
    sample_transactions.csv  synthetic transactions
    knowledge/*.md           7 financial guides (the knowledge base)
    sample_documents/        household_money_policy.pdf for testing PDF Q&A
    generate_sample_data.py  generate_sample_pdf.py
  models/                    embedding model cache + FAISS index (created on first run)
  utils/
    data_processor.py        CSV validation and cleaning
    categorizer.py           keyword categorizer
    financial_analyzer.py    Pandas analytics
    budget_advisor.py        budget and goal maths
    facts.py                 privacy-safe aggregate facts for the LLM
    prompts.py               prompt templates
    llm.py                   Groq client, chains, number-grounding guard
    embeddings.py            local HuggingFace embeddings
    vector_store.py          FAISS build / save / load / search
    rag_pipeline.py          PDF load, chunking, indexing, retrieval
    chatbot.py               question router
    chat_store.py            SQLite chat history
    ui.py                    theme and shared components
  views/                     the six pages (dashboard, transactions, insights, budget, assistant, settings)
  tests/                     test_frontend.py, test_backend.py, fakes.py
```

(`views/` is used instead of `pages/` because Streamlit auto-creates its own menu from a `pages/`
folder, which would clash with the custom sidebar.)

## 6. Test it

Automated (works offline; uses stand-ins for the embedding model and Groq):

```bash
python tests/test_frontend.py
python tests/test_backend.py
```

`test_backend.py` also runs two real-service checks (HuggingFace download, live Groq call) when
they are available, and reports `SKIP` otherwise.

Manual checklist:

1. **Dashboard**: pick a month; cards and charts update.
2. **Transactions**: upload your own CSV (`date, description, amount, type`), change a category,
   click **Save category changes**.
3. **AI Insights**: click **Generate AI explanation** (needs the Groq key).
4. **Budget Planner**: edit budgets, set a savings goal.
5. **AI Assistant**: upload `data/sample_documents/household_money_policy.pdf`, then ask the
   questions below.

Example questions:

| Type | Question |
|---|---|
| Transactions | Where did I spend the most money? |
| Transactions | How much did I spend on food this month? |
| Transactions | Compare my expenses across months |
| Transactions | How much can I save next month? |
| Budget | Give me a budget based on my spending |
| Advice | How can I reduce my shopping expenses? |
| Knowledge | What is the 50/30/20 rule? / What is an emergency fund? |
| Your PDF | Explain the financial policy in my uploaded document |
| Your PDF | What is the waiting period for large purchases? |

## 7. Troubleshooting

- **"embedding model could not be loaded"**: the first run needs internet to download it once.
- **"Groq rejected the API key"**: re-copy the key into `.env` and restart the app.
- **Rate limit message**: the free tier has limits; wait a minute.
- **Model not found**: set `GROQ_MODEL` in `.env` to a current model from the Groq console.
- **Install problems on Python 3.12+**: use Python 3.10 or 3.11.
- **Scanned PDFs** have no extractable text and need OCR first.

*Suggestions only, not professional financial advice. Never enter real bank credentials anywhere.*
