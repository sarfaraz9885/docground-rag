"""Retrieval: turn a question into the top-k most relevant chunks.

Run a quick manual check from the repo root:

    python -m src.retrieve "what chunk size should I use?"
    python -m src.retrieve "who won the world cup?" --k 4

Each result carries its ``source`` and ``page`` metadata so the generation and
evaluation layers can build ``[source, page]`` citations and a retrieval-hit
signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from langchain_core.documents import Document

from .config import settings
from .ingest import get_vectorstore

# Default number of chunks to retrieve (per spec).
DEFAULT_K = 4


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk with everything the rest of the app needs."""

    text: str
    source: str
    page: int | str
    score: float | None  # cosine similarity (higher = closer); None if unavailable

    @property
    def citation(self) -> str:
        """Human-readable citation tag, e.g. ``[paper.pdf, 3]``."""
        return f"[{self.source}, {self.page}]"

    def to_dict(self) -> dict:
        """JSON-serializable form for API responses."""
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "score": self.score,
            "citation": self.citation,
        }


def _normalize_page(page: object) -> int | str:
    """Render a page number cleanly.

    Pinecone stores numeric metadata as floats, so a page of ``1`` round-trips
    as ``1.0``. Collapse whole-number floats back to ints so citations read
    ``[doc.md, 1]`` rather than ``[doc.md, 1.0]``.
    """
    if isinstance(page, float) and page.is_integer():
        return int(page)
    return page  # type: ignore[return-value]


def _to_chunk(doc: Document, score: float | None) -> RetrievedChunk:
    """Map a LangChain Document (+ optional score) to a RetrievedChunk."""
    return RetrievedChunk(
        text=doc.page_content,
        source=doc.metadata.get("source", "unknown"),
        page=_normalize_page(doc.metadata.get("page", "?")),
        score=score,
    )


def retrieve(question: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
    """Return the top-``k`` chunks most semantically similar to ``question``.

    Uses similarity-with-score when available so we can surface a similarity for
    debugging/eval; falls back gracefully if the backend lacks scores.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    vectorstore = get_vectorstore()
    try:
        pairs = vectorstore.similarity_search_with_score(question, k=k)
        return [_to_chunk(doc, float(score)) for doc, score in pairs]
    except Exception:
        # Some store configs don't return scores; fall back to plain search.
        docs = vectorstore.similarity_search(question, k=k)
        return [_to_chunk(doc, None) for doc in docs]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered context block for prompting.

    The generator is instructed to cite using the ``[source, page]`` tags that
    prefix each block.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {c.citation}\n{c.text}")
    return "\n\n".join(blocks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a question.")
    parser.add_argument("question", type=str, help="The question to search for.")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"Chunks to retrieve (default {DEFAULT_K})."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = retrieve(args.question, k=args.k)
    print(f"Top {len(results)} chunk(s) for: {args.question!r}\n")
    for rank, c in enumerate(results, start=1):
        sim = f"{c.score:.4f}" if c.score is not None else "n/a"
        print(f"#{rank}  {c.citation}  (similarity={sim})")
        snippet = c.text.strip().replace("\n", " ")
        print(f"    {snippet[:160]}{'...' if len(snippet) > 160 else ''}\n")
