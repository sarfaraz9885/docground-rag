# DocGround — Document-Grounded RAG with Faithfulness Evaluation

A retrieval-augmented question-answering system that answers questions **only** from a
provided document set, returns **citations** for every answer, and **scores its own
faithfulness** to flag hallucinations. Built to demonstrate RAG, retrieval pipelines,
embeddings, vector search, and LLM evaluation end to end.

> **Demo domain:** AI/LLM engineering documentation (LangChain docs, RAG & hallucination
> papers from arXiv, public prompting guides). Swap in any document set via the `data/` folder.

---

## Environment setup (read this first)

**Python: use 3.12, not 3.14.** Chroma and parts of the LangChain stack do not yet have
working Python 3.14 support (onnxruntime has no 3.14 wheels, hnswlib fails to compile, and
pydantic v1 paths warn/break on 3.14). Python 3.12 is the known-good version for this whole
stack. Install 3.12 from python.org alongside whatever else you have.

**PyCharm + virtual environment:**
1. Install Python 3.12.
2. In PyCharm: `File → Settings → Project → Python Interpreter → Add Interpreter →
   Add Local Interpreter → Virtualenv → New environment`, and set **Base interpreter** to
   your Python 3.12. PyCharm creates `.venv/` and activates it automatically in its terminal.
3. Verify in PyCharm's terminal: `python --version` → should print `3.12.x`.
4. Install everything inside this venv (`pip install -r requirements.txt`).

**Streamlit Cloud must match.** Add a file named `runtime.txt` in the repo root with one line:
```
python-3.12
```
This pins the deployed environment to 3.12 so local and cloud behave identically.

### Vector DB choice: Chroma (not Pinecone)

Chroma is the right call here: it runs **locally with zero infrastructure**, persists to disk,
needs no extra account or API key, and installs cleanly on Python 3.12. Pinecone is a hosted
cloud vector DB — fine, but it adds an account, an API key, and network dependency for no
benefit at this corpus size. Keep Chroma. In an interview, "I used a local vector store and
can explain when I'd migrate to a hosted DB like Pinecone at scale" is a stronger answer than
defaulting to a cloud service you didn't need.

---

## Why this project exists

Most RAG demos stop at "retrieve + generate." This one adds the part real teams care about:
**did the model actually stay grounded in the source, or did it make something up?** Every
answer is scored for faithfulness and unsupported claims are flagged.

---

## Features

- **Ingestion pipeline** — load PDFs / markdown / text, chunk with overlap, embed, store in a vector DB.
- **Retrieval** — semantic search over embeddings; returns top-k chunks with source metadata.
- **Grounded generation** — answers built strictly from retrieved context, with inline citations.
- **Evaluation layer** — for each answer, compute:
  - **Faithfulness** (LLM-as-judge): is every claim supported by the retrieved context? (0–1)
  - **Answer relevance**: does the answer address the question? (0–1)
  - **Retrieval hit**: were relevant chunks actually retrieved? (precision-style signal)
  - **Hallucination flag**: boolean — any claim not grounded in context.
- **API** — FastAPI endpoints for `/query` and `/evaluate`.
- **UI** — lightweight Streamlit front end (matches your existing deployment style).

---

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  Documents  │──▶│  Chunk+Embed │──▶│  Vector Store │   │   FastAPI    │
│ (PDF/MD/txt)│   │ (ingest.py)  │   │   (Chroma)    │   │   /query     │
└─────────────┘   └──────────────┘   └───────┬───────┘   │   /evaluate  │
                                             │           └──────┬───────┘
                          question ──────────┼─────▶ retrieve   │
                                             ▼                  ▼
                                      top-k chunks ──▶ LLM ──▶ grounded answer
                                                                  │
                                                                  ▼
                                                        Eval layer (eval.py)
                                                  faithfulness | relevance | hit
                                                                  │
                                                                  ▼
                                                       Streamlit UI shows
                                                  answer + citations + scores
```

---

## Tech stack

| Layer | Choice | Note |
|-------|--------|------|
| Language | Python 3.11 | |
| Orchestration | LangChain | text splitters, retrievers, prompt templates |
| Vector store | Chroma | local, zero infra; persists to disk |
| Embeddings | OpenAI `text-embedding-3-small` | cheap; swap for local if needed |
| LLM | Anthropic Claude or OpenAI | you have access to both |
| API | FastAPI + Uvicorn | |
| UI | Streamlit | deploy free on Streamlit Cloud |
| Eval | LLM-as-judge + simple metrics | no heavy framework needed |

---

## Suggested repo structure

```
docground/
├── README.md
├── requirements.txt
├── runtime.txt           # contains: python-3.12  (pins Streamlit Cloud)
├── .env.example          # API keys — never commit the real .env
├── .gitignore            # must include .env, chroma_db/, __pycache__/, .venv/
├── data/                 # source documents (a few PDFs/MD files)
├── src/
│   ├── ingest.py         # load → chunk → embed → store
│   ├── retrieve.py       # query → top-k chunks
│   ├── generate.py       # context + question → grounded answer + citations
│   ├── evaluate.py       # faithfulness / relevance / hit / hallucination flag
│   └── api.py            # FastAPI app
├── app.py                # Streamlit UI
└── eval/
    ├── questions.json     # ~15 test questions (some answerable, some NOT)
    └── run_eval.py        # batch-run eval, print a scorecard table
```

---

## Build order (each step is independently testable)

**1. Ingestion (`ingest.py`)**
- Load docs from `data/` (use LangChain's `PyPDFLoader`, `TextLoader`).
- Split with `RecursiveCharacterTextSplitter` (chunk_size ~800, overlap ~120).
- Embed and persist to Chroma. Store `source` + `page` in each chunk's metadata — you need this for citations.
- ✅ Test: run it, confirm Chroma has N chunks.

**2. Retrieval (`retrieve.py`)**
- Given a question, return top-k (k=4) chunks with their metadata.
- ✅ Test: ask a question, eyeball that retrieved chunks are on-topic.

**3. Generation (`generate.py`)**
- Prompt template: *"Answer ONLY using the context below. Cite sources as [source, page]. If the context doesn't contain the answer, say 'Not found in the provided documents.'"*
- This refusal instruction is what makes hallucination measurable — include questions the docs *can't* answer.
- ✅ Test: answerable Q gets a cited answer; unanswerable Q gets the refusal.

**4. Evaluation (`evaluate.py`)** — the differentiator
- **Faithfulness (LLM-as-judge):** send the answer + retrieved context to the LLM, ask it to rate 0–1 whether every claim is supported, and list any unsupported claims. Return score + the flagged claims.
- **Answer relevance:** LLM rates 0–1 whether the answer addresses the question.
- **Retrieval hit:** for your test questions you know the expected source — check if it appears in retrieved chunks.
- **Hallucination flag:** `True` if faithfulness < 0.7 or any unsupported claim found.
- ✅ Test: deliberately feed a wrong answer and confirm faithfulness drops.

**5. API (`api.py`)**
- `POST /query` → `{answer, citations, scores}`
- `POST /evaluate` → run eval on a single Q/A pair.

**6. UI (`app.py`)**
- Text box for the question; show the answer, the citations, and a small scorecard
  (faithfulness, relevance, hallucination flag). Color the flag red/green.

**7. Eval harness (`eval/run_eval.py`)**
- Run all ~15 test questions, print a table: question | faithfulness | relevance | hit | flag.
- Save results to `eval/results.md` — **screenshot this for your portfolio and resume.** A scorecard is concrete proof you can evaluate models, which is exactly what the JD asks for.

---

## What to put on your resume once it's built

Replace the placeholder block with bullets like:

- Built a document-grounded RAG system (LangChain + Chroma) that answers questions strictly from a
  source corpus with inline citations and refuses out-of-scope questions.
- Implemented an evaluation layer scoring **faithfulness, answer relevance, and retrieval hit rate**,
  with automatic hallucination flagging via LLM-as-judge; reported results across a 15-question test set.
- Served the pipeline via FastAPI and deployed an interactive Streamlit demo.

…plus the GitHub + live demo links.

---

## Guardrails (don't skip these)

- `.gitignore` **must** contain `.env`, `chroma_db/`, `__pycache__/`. Never commit API keys.
- Provide `.env.example` with key *names* only.
- Keep `data/` to public docs (arXiv papers, open documentation). No copyrighted books/articles.
- Write a real README on the repo (you can adapt this file). A clean README is half the impression.

---

## Stretch goals (only if time allows — not required to apply)

- Add a second retrieval strategy (e.g. hybrid keyword + semantic) and compare eval scores between them.
  *This single addition demonstrates "compare models/approaches using structured methods" from the JD.*
- Dockerize the API (`Dockerfile`) — covers the Docker preferred-qual.
- Swap embeddings to a local model and report the quality/cost tradeoff.
