"""Evaluation layer: faithfulness, answer-relevance, retrieval-hit, hallucination.

This is the project's differentiator. For a given (question, answer, context):

- **Faithfulness** (LLM-as-judge): is every claim supported by the context? 0-1,
  plus the list of unsupported claims.
- **Answer relevance** (LLM-as-judge): does the answer address the question? 0-1.
- **Retrieval hit**: if we know the expected source for a question, did it appear
  in the retrieved chunks? (precision-style signal; None when no expectation given)
- **Hallucination flag**: True if faithfulness < 0.7 OR any unsupported claim.

Quick manual check from the repo root (feeds a deliberately wrong answer):

    python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .llm import get_chat_model
from .generate import GenerationResult, REFUSAL, generate_answer
from .retrieve import RetrievedChunk, format_context

# Faithfulness at or above this threshold is considered grounded.
FAITHFULNESS_THRESHOLD = 0.7


# --------------------------------------------------------------------------- #
# Structured-output schemas the judge must fill in.
# --------------------------------------------------------------------------- #
class FaithfulnessVerdict(BaseModel):
    """Judge output for faithfulness."""

    score: float = Field(
        description="0.0-1.0. 1.0 = every claim fully supported by the context; "
        "0.0 = nothing supported.",
        ge=0.0,
        le=1.0,
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer NOT supported by the context. Empty if all supported.",
    )
    reasoning: str = Field(default="", description="One or two sentences of justification.")


class RelevanceVerdict(BaseModel):
    """Judge output for answer relevance."""

    score: float = Field(
        description="0.0-1.0. 1.0 = directly and fully addresses the question; "
        "0.0 = irrelevant.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(default="", description="One or two sentences of justification.")


# --------------------------------------------------------------------------- #
# Final, JSON-serializable evaluation record.
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    faithfulness: float
    relevance: float
    unsupported_claims: list[str]
    hallucination: bool
    retrieval_hit: bool | None  # None when no expected source was provided
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "unsupported_claims": self.unsupported_claims,
            "hallucination": self.hallucination,
            "retrieval_hit": self.retrieval_hit,
            "notes": self.notes,
        }


_FAITHFULNESS_SYSTEM = (
    "You are a strict faithfulness evaluator for a retrieval-augmented system.\n"
    "You are given a CONTEXT and an ANSWER. Decide whether EVERY factual claim in "
    "the ANSWER is directly supported by the CONTEXT.\n"
    "Judge ONLY against the context — never use outside knowledge.\n"
    "Return a score from 0.0 (no claims supported) to 1.0 (all claims supported) "
    "and list every claim that is not supported by the context."
)

_RELEVANCE_SYSTEM = (
    "You are an answer-relevance evaluator.\n"
    "Given a QUESTION and an ANSWER, rate from 0.0 (irrelevant) to 1.0 (directly and "
    "fully addresses the question) how well the answer responds to the question.\n"
    "Judge only relevance to the question, not factual correctness."
)


def judge_faithfulness(answer: str, context: str) -> FaithfulnessVerdict:
    """LLM-as-judge: is the answer grounded in the context?"""
    llm = get_chat_model(temperature=0.0).with_structured_output(FaithfulnessVerdict)
    return llm.invoke(
        [
            ("system", _FAITHFULNESS_SYSTEM),
            ("human", f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ]
    )


def judge_relevance(question: str, answer: str) -> RelevanceVerdict:
    """LLM-as-judge: does the answer address the question?"""
    llm = get_chat_model(temperature=0.0).with_structured_output(RelevanceVerdict)
    return llm.invoke(
        [
            ("system", _RELEVANCE_SYSTEM),
            ("human", f"QUESTION:\n{question}\n\nANSWER:\n{answer}"),
        ]
    )


def retrieval_hit(chunks: list[RetrievedChunk], expected_source: str | None) -> bool | None:
    """Did a chunk from ``expected_source`` appear in retrieval?

    Returns None if no expectation was provided. Matching is case-insensitive and
    substring-based so "paper.pdf" matches "paper.pdf" regardless of path noise.
    """
    if not expected_source:
        return None
    target = expected_source.strip().lower()
    return any(target in (c.source or "").lower() for c in chunks)


def evaluate(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    expected_source: str | None = None,
) -> EvaluationResult:
    """Score one (question, answer, context) triple.

    Refusals are handled specially: declining when the context lacks the answer is
    correct behavior, so faithfulness=1.0, no unsupported claims, and not a
    hallucination. Relevance is still judged (a refusal that should have answered
    will score low).
    """
    hit = retrieval_hit(chunks, expected_source)

    is_refusal = answer.strip().rstrip(".").lower() == REFUSAL.rstrip(".").lower()
    if is_refusal:
        return EvaluationResult(
            faithfulness=1.0,
            relevance=1.0,
            unsupported_claims=[],
            hallucination=False,
            retrieval_hit=hit,
            notes="Model refused (no claims to verify); treated as faithful.",
        )

    context = format_context(chunks)
    faith = judge_faithfulness(answer, context)
    rel = judge_relevance(question, answer)

    hallucination = faith.score < FAITHFULNESS_THRESHOLD or bool(faith.unsupported_claims)

    return EvaluationResult(
        faithfulness=round(faith.score, 3),
        relevance=round(rel.score, 3),
        unsupported_claims=faith.unsupported_claims,
        hallucination=hallucination,
        retrieval_hit=hit,
        notes=faith.reasoning,
    )


def evaluate_generation(
    result: GenerationResult, expected_source: str | None = None
) -> EvaluationResult:
    """Convenience wrapper to evaluate a :class:`GenerationResult` directly."""
    return evaluate(
        question=result.question,
        answer=result.answer,
        chunks=result.chunks,
        expected_source=expected_source,
    )


def _demo() -> None:
    """Sanity check: a real grounded answer should score high; a fabricated
    answer fed against the same context should score low and flag hallucination.
    """
    question = "What chunk size and overlap does the guidance suggest?"
    good = generate_answer(question)
    print("=== Grounded answer ===")
    print(good.answer)
    good_eval = evaluate_generation(good, expected_source="sample_rag_notes.md")
    print(good_eval.to_dict(), "\n")

    print("=== Deliberately WRONG answer (same context) ===")
    wrong = "The guidance recommends a chunk size of 5000 words with no overlap, and says embeddings are unnecessary."
    print(wrong)
    wrong_eval = evaluate(question, wrong, good.chunks, expected_source="sample_rag_notes.md")
    print(wrong_eval.to_dict())


if __name__ == "__main__":
    _demo()
