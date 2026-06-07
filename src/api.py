"""FastAPI service for DocGround.

Endpoints
---------
- GET  /            -> health/info
- POST /query       -> retrieve + grounded answer + citations + (optional) scores
- POST /evaluate    -> score a single question/answer pair against retrieved context

Run from the repo root:

    uvicorn src.api:app --reload
    # interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .retrieve import DEFAULT_K
from .generate import generate_answer
from .evaluate import evaluate

app = FastAPI(
    title="DocGround RAG API",
    description="Document-grounded question answering with faithfulness evaluation.",
    version="1.0.0",
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    k: int = Field(DEFAULT_K, ge=1, le=20, description="Number of chunks to retrieve.")
    expected_source: str | None = Field(
        None, description="Optional expected source file name, for the retrieval-hit check."
    )
    run_eval: bool = Field(
        True, description="If true, also run the evaluation layer and return scores."
    )


class CitationContext(BaseModel):
    text: str
    source: str
    page: int | str
    score: float | None
    citation: str


class Scores(BaseModel):
    faithfulness: float
    relevance: float
    unsupported_claims: list[str]
    hallucination: bool
    retrieval_hit: bool | None
    notes: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[str]
    context: list[CitationContext]
    scores: Scores | None


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1, description="The answer to score.")
    k: int = Field(DEFAULT_K, ge=1, le=20, description="Chunks to retrieve for context.")
    expected_source: str | None = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict:
    """Health/info endpoint."""
    return {
        "service": "DocGround RAG API",
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "vector_store": "pinecone",
        "pinecone_index": settings.pinecone_index,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a question from the document store, with citations and scores."""
    try:
        result = generate_answer(req.question, k=req.k)
    except Exception as exc:  # surface config/store errors as clean HTTP errors
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    scores: Scores | None = None
    if req.run_eval:
        try:
            ev = evaluate(
                question=req.question,
                answer=result.answer,
                chunks=result.chunks,
                expected_source=req.expected_source,
            )
            scores = Scores(**ev.to_dict())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        citations=result.citations,
        context=[CitationContext(**c.to_dict()) for c in result.chunks],
        scores=scores,
    )


@app.post("/evaluate", response_model=Scores)
def evaluate_pair(req: EvaluateRequest) -> Scores:
    """Score a single (question, answer) pair against freshly retrieved context."""
    from .retrieve import retrieve

    try:
        chunks = retrieve(req.question, k=req.k)
        ev = evaluate(
            question=req.question,
            answer=req.answer,
            chunks=chunks,
            expected_source=req.expected_source,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return Scores(**ev.to_dict())
