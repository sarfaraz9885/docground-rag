# DocGround — Document-Grounded RAG with Faithfulness Evaluation

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sarfaraz9885-docground-rag-app-jyqlth.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**🔗 Live demo:** https://sarfaraz9885-docground-rag-app-jyqlth.streamlit.app/

A retrieval-augmented question-answering system that answers questions **only** from a
provided document set, returns **citations** for every answer, and **scores its own
faithfulness** to flag hallucinations.

- **Ingest** PDFs / Markdown / text → chunk → embed → store in a **Pinecone** vector index.
- **Retrieve** top-k semantically relevant chunks (with `source` + `page` metadata).
- **Generate** answers strictly from retrieved context, with inline `[source, page]` citations,
  refusing out-of-scope questions ("Not found in the provided documents.").
- **Evaluate** every answer for faithfulness, answer-relevance, retrieval hit, and a
  hallucination flag via LLM-as-judge.
- **Serve** via a Streamlit UI scorecard, plus an optional FastAPI service (`/query`, `/evaluate`).

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | **Python 3.12** |
| Orchestration | LangChain |
| Vector store | **Pinecone** (serverless, managed) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | OpenAI or Anthropic Claude (configurable) |
| API | FastAPI + Uvicorn (optional) |
| UI | Streamlit (runs the pipeline in-process) |

> **Python version:** this stack is tested on **3.12**. A `runtime.txt` is included, but
> Streamlit Community Cloud currently ignores it ([known issue](https://github.com/streamlit/streamlit/issues/15326))
> — so you must pick **Python 3.12** in the deploy dialog's *Advanced settings* (see below).
> Don't deploy on 3.14: some compiled dependencies may lack 3.14 wheels, and you gain nothing.

## Setup

```bash
# 1. Create a 3.12 virtual environment and activate it
python3.12 -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell on Windows
# source .venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env             # then edit .env and add your real keys
```

You need an **OpenAI** key (embeddings + LLM) and a **Pinecone** key
([app.pinecone.io](https://app.pinecone.io)). The Pinecone index
(`PINECONE_INDEX`, default `docground-rag`) is **created automatically** on the
first ingest as a serverless index with dimension 1536 / cosine.

`.env` is git-ignored — never commit secrets.

## Usage

```bash
# 1. Ingest documents in data/ into Pinecone (creates the index if needed)
python -m src.ingest                 # ingest everything in data/
python -m src.ingest --reset         # wipe the index's vectors first, then ingest

# 2. Run the Streamlit UI (runs retrieve + generate + evaluate in-process)
streamlit run app.py

# 3. (Optional) Run the FastAPI service
uvicorn src.api:app --reload         # docs at http://localhost:8000/docs

# 4. Run the batch evaluation harness -> console scorecard + eval/results.md
python eval/run_eval.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (the index is in Pinecone's cloud, so there's no
   local store to upload — only code).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app.py` on your branch.
3. Open **Advanced settings** and **select Python 3.12** (the version can't be
   changed after deploy — you'd have to delete and redeploy). `runtime.txt` is
   currently ignored by Cloud, so this dropdown is the reliable control.
4. Still in **Advanced settings → Secrets**, paste your keys in TOML form:

   ```toml
   OPENAI_API_KEY = "sk-..."
   PINECONE_API_KEY = "pcsk_..."
   PINECONE_INDEX = "docground-rag"
   EMBEDDING_MODEL = "text-embedding-3-small"
   LLM_PROVIDER = "openai"
   OPENAI_CHAT_MODEL = "gpt-4o-mini"
   ```

   `app.py` copies these into the environment at startup, so `src/config.py`
   reads them exactly as it reads `.env` locally.
5. Make sure the Pinecone index has already been populated (run `python -m src.ingest`
   locally once against the same index).

## Repo structure

```
DOCS_RAG/
├── requirements.txt
├── runtime.txt            # python-3.12 (pins Streamlit Cloud)
├── .env.example           # API key NAMES only
├── .gitignore
├── app.py                 # Streamlit UI (in-process RAG + scorecard)
├── data/                  # source documents (PDF/MD/txt)
├── src/
│   ├── config.py          # env loading (python-dotenv)
│   ├── llm.py             # chat-model factory (OpenAI / Anthropic)
│   ├── ingest.py          # load → chunk → embed → Pinecone
│   ├── retrieve.py        # query → top-k chunks
│   ├── generate.py        # context + question → grounded answer + citations
│   ├── evaluate.py        # faithfulness / relevance / hit / hallucination
│   └── api.py             # FastAPI app (optional)
└── eval/
    ├── questions.json     # 15 test questions (10 answerable, 5 unanswerable)
    ├── run_eval.py        # batch eval → scorecard table + results.md
    └── results.md         # generated scorecard (committed as proof)
```
