"""DocGround — Streamlit UI.

A single-process Streamlit app that runs the full RAG + evaluation pipeline
in-process (no separate API server needed), so it deploys to Streamlit
Community Cloud as-is.

Layout (per spec):
- a question box,
- the grounded answer,
- the citations used,
- a small scorecard: faithfulness, answer-relevance, and a hallucination flag
  colored red (hallucinated) / green (faithful).

Run locally from the repo root:

    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

# set_page_config must be the first Streamlit command.
st.set_page_config(page_title="DocGround RAG", page_icon="📑", layout="centered")

# --------------------------------------------------------------------------- #
# Secrets bridge: on Streamlit Community Cloud there is no .env file — secrets
# live in st.secrets. Copy them into the environment BEFORE importing src.* so
# config.py (which reads os.environ) picks them up. Locally this is a no-op and
# the .env file is used instead.
# --------------------------------------------------------------------------- #
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ.setdefault(_key, _val)
except Exception:
    # No secrets.toml present (typical local dev) — fall back to .env.
    pass

# Import after the secrets bridge so configuration resolves correctly.
from src.generate import generate_answer  # noqa: E402
from src.evaluate import evaluate_generation, FAITHFULNESS_THRESHOLD  # noqa: E402
from src.retrieve import DEFAULT_K  # noqa: E402


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("⚙️ Configuration")
    st.caption("Document-grounded answers with a self-check for hallucinations.")
    st.markdown("---")

    k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=DEFAULT_K,
        help="How many context chunks to pull from Pinecone for each question.",
    )
    expected_source = st.text_input(
        "Expected source (optional)",
        value="",
        placeholder="e.g. sample_rag_notes.md",
        help="If set, the scorecard reports whether this file was actually retrieved.",
    ).strip() or None

    run_eval = st.toggle(
        "Run faithfulness evaluation",
        value=True,
        help="Score the answer for faithfulness, relevance, and hallucination.",
    )
    st.markdown("---")
    st.info("Vectors are stored in **Pinecone**. Answers come only from the ingested documents.")


# --------------------------------------------------------------------------- #
# Main — question box
# --------------------------------------------------------------------------- #
st.title("📑 DocGround")
st.write("Ask a question and get an answer grounded **only** in your documents — with citations and a faithfulness scorecard.")

with st.form("ask"):
    question = st.text_area(
        "Your question",
        placeholder="What chunk size and overlap does the guidance recommend?",
        height=90,
    )
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)


def _score_color(score: float, good: float = FAITHFULNESS_THRESHOLD) -> str:
    """Pick an emoji delta-color hint for a 0-1 score (green good / red poor)."""
    return "normal" if score >= good else "inverse"


if submitted:
    if not question or not question.strip():
        st.warning("Please type a question first.")
        st.stop()

    # 1) Retrieve + generate a grounded answer.
    with st.spinner("Retrieving context and generating a grounded answer..."):
        try:
            result = generate_answer(question.strip(), k=k)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

    # 2) Answer.
    st.subheader("Answer")
    if result.refused:
        st.warning(result.answer)  # the refusal sentence
    else:
        st.markdown(result.answer)

    # 3) Citations.
    st.subheader("Citations")
    if result.citations:
        st.markdown("  ".join(f"`{c}`" for c in result.citations))
    elif result.refused:
        st.caption("No citations — the answer was not found in the documents.")
    else:
        st.caption("The answer did not include any inline citations.")

    # Retrieved context, collapsed by default.
    with st.expander(f"Retrieved context ({len(result.chunks)} chunk(s))"):
        for i, c in enumerate(result.chunks, start=1):
            sim = f"{c.score:.3f}" if c.score is not None else "n/a"
            st.markdown(f"**[{i}] {c.citation}** · similarity={sim}")
            st.caption(c.text.strip()[:500] + ("…" if len(c.text) > 500 else ""))

    # 4) Scorecard.
    if run_eval:
        with st.spinner("Scoring faithfulness and relevance..."):
            try:
                ev = evaluate_generation(result, expected_source=expected_source)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()

        st.subheader("Scorecard")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Faithfulness",
            f"{ev.faithfulness:.2f}",
            delta="grounded" if ev.faithfulness >= FAITHFULNESS_THRESHOLD else "low",
            delta_color=_score_color(ev.faithfulness),
        )
        col2.metric(
            "Relevance",
            f"{ev.relevance:.2f}",
            delta="on-topic" if ev.relevance >= 0.7 else "off-topic",
            delta_color=_score_color(ev.relevance, good=0.7),
        )
        # Retrieval hit: green check / red cross / neutral when no expectation.
        if ev.retrieval_hit is None:
            col3.metric("Retrieval hit", "—", help="Set an expected source to enable.")
        else:
            col3.metric(
                "Retrieval hit",
                "yes" if ev.retrieval_hit else "no",
                delta="found" if ev.retrieval_hit else "missed",
                delta_color="normal" if ev.retrieval_hit else "inverse",
            )

        # Hallucination flag — explicitly colored red / green.
        if ev.hallucination:
            st.error("🔴 Hallucination flag: **TRUE** — the answer contains unsupported claims.")
        else:
            st.success("🟢 Hallucination flag: **FALSE** — the answer is grounded in the context.")

        if ev.unsupported_claims:
            with st.expander(f"Unsupported claims ({len(ev.unsupported_claims)})"):
                for claim in ev.unsupported_claims:
                    st.markdown(f"- {claim}")

        if ev.notes:
            st.caption(f"Judge notes: {ev.notes}")
