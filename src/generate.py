"""Grounded generation: answer strictly from retrieved context, with citations.

Run a quick manual check from the repo root:

    python -m src.generate "what chunk size should I use?"
    python -m src.generate "who won the 2022 world cup?"   # -> refusal

Contract enforced by the prompt:
- Answer ONLY from the retrieved context.
- Cite every claim inline as [source, page].
- If the context lacks the answer, reply EXACTLY:
  "Not found in the provided documents."
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import get_chat_model
from .retrieve import RetrievedChunk, retrieve, format_context, DEFAULT_K

# The exact refusal string the system must emit when the context is insufficient.
REFUSAL = "Not found in the provided documents."

# Matches inline citations like [source.pdf, 3] or [notes.md, 12].
_CITATION_RE = re.compile(r"\[[^\[\]]+?,\s*[^\[\]]+?\]")

SYSTEM_PROMPT = (
    "You are DocGround, a meticulous question-answering assistant.\n"
    "Answer the user's question using ONLY the information in the provided "
    "context. Do not use any outside or prior knowledge.\n"
    "Cite your sources inline after each claim using the exact tags shown in "
    "the context, in the form [source, page].\n"
    f"If the context does not contain enough information to answer, reply with "
    f'EXACTLY this sentence and nothing else: "{REFUSAL}"'
)

USER_PROMPT = (
    "Context:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer (grounded in the context, with [source, page] citations):"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)


@dataclass
class GenerationResult:
    """Everything produced for one question, ready for the API/UI/eval layers."""

    question: str
    answer: str
    refused: bool
    citations: list[str]                       # unique [source, page] tags in the answer
    chunks: list[RetrievedChunk] = field(default_factory=list)  # context used

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "citations": self.citations,
            "context": [c.to_dict() for c in self.chunks],
        }


def _extract_citations(answer: str) -> list[str]:
    """Pull unique [source, page] tags out of the answer, preserving order."""
    seen: dict[str, None] = {}
    for match in _CITATION_RE.findall(answer):
        seen.setdefault(match.strip(), None)
    return list(seen.keys())


def _is_refusal(answer: str) -> bool:
    """True if the model declined to answer (matches the refusal sentence)."""
    normalized = answer.strip().rstrip(".").lower()
    return normalized == REFUSAL.rstrip(".").lower()


def generate_answer(question: str, k: int = DEFAULT_K) -> GenerationResult:
    """Retrieve context for ``question`` and produce a grounded, cited answer."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    chunks = retrieve(question, k=k)

    # Edge case: nothing retrieved at all -> refuse without calling the LLM.
    if not chunks:
        return GenerationResult(
            question=question, answer=REFUSAL, refused=True, citations=[], chunks=[]
        )

    context = format_context(chunks)
    chain = _PROMPT | get_chat_model(temperature=0.0) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}).strip()

    refused = _is_refusal(answer)
    citations = [] if refused else _extract_citations(answer)
    return GenerationResult(
        question=question,
        answer=answer,
        refused=refused,
        citations=citations,
        chunks=chunks,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a grounded answer.")
    parser.add_argument("question", type=str, help="The question to answer.")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"Chunks to retrieve (default {DEFAULT_K})."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = generate_answer(args.question, k=args.k)
    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print(f"Refused: {result.refused}")
    print(f"Citations: {result.citations or '(none)'}")
    print(f"Context chunks used: {len(result.chunks)}")
