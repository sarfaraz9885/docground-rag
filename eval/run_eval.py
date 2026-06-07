"""Batch evaluation harness for DocGround.

Runs every question in ``eval/questions.json`` through the full pipeline
(retrieve -> grounded answer -> LLM-as-judge evaluation), prints a scorecard
table to the console, and writes a Markdown report to ``eval/results.md``.

For each question we record:
- behavior: did the system ANSWER or REFUSE?
- behavior_ok: was that the *correct* behavior? (answerable -> answer,
  unanswerable -> refuse)
- faithfulness, relevance (0-1, LLM-as-judge),
- hallucination flag,
- retrieval hit (only meaningful when an expected_source is given).

Run from the repo root (requires data ingested into Pinecone first):

    python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src`` importable when this script is run as ``python eval/run_eval.py``
# (its own directory, not the repo root, is on sys.path by default).
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8 so the
# scorecard (and any unicode in answers) prints cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tabulate import tabulate  # noqa: E402

from src.generate import generate_answer  # noqa: E402
from src.evaluate import evaluate_generation  # noqa: E402

QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.md"


def _truncate(text: str, width: int = 48) -> str:
    """Shorten a string for table display."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def run() -> list[dict]:
    """Evaluate every question and return a list of per-question records."""
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    records: list[dict] = []

    for i, q in enumerate(questions, start=1):
        qid = q.get("id", f"q{i:02d}")
        question = q["question"]
        answerable = bool(q.get("answerable", True))
        expected_source = q.get("expected_source")

        print(f"[{i}/{len(questions)}] {qid}: {question}")

        result = generate_answer(question)
        ev = evaluate_generation(result, expected_source=expected_source)

        behavior = "refuse" if result.refused else "answer"
        # Correct behavior: answerable -> should answer; unanswerable -> should refuse.
        behavior_ok = (result.refused != answerable)

        records.append(
            {
                "id": qid,
                "question": question,
                "answerable": answerable,
                "behavior": behavior,
                "behavior_ok": behavior_ok,
                "faithfulness": ev.faithfulness,
                "relevance": ev.relevance,
                "hallucination": ev.hallucination,
                "retrieval_hit": ev.retrieval_hit,
                "unsupported_claims": ev.unsupported_claims,
                "answer": result.answer,
            }
        )

    return records


def _summary(records: list[dict]) -> dict:
    """Aggregate metrics across all records."""
    n = len(records)
    behavior_ok = sum(r["behavior_ok"] for r in records)
    hallucinations = sum(r["hallucination"] for r in records)
    mean_faith = sum(r["faithfulness"] for r in records) / n if n else 0.0
    mean_rel = sum(r["relevance"] for r in records) / n if n else 0.0

    # Retrieval hit rate only over questions that declared an expected source.
    with_expected = [r for r in records if r["retrieval_hit"] is not None]
    hits = sum(1 for r in with_expected if r["retrieval_hit"])

    return {
        "n": n,
        "behavior_ok": behavior_ok,
        "behavior_acc": behavior_ok / n if n else 0.0,
        "hallucinations": hallucinations,
        "halluc_rate": hallucinations / n if n else 0.0,
        "mean_faithfulness": mean_faith,
        "mean_relevance": mean_rel,
        "retrieval_checked": len(with_expected),
        "retrieval_hits": hits,
        "retrieval_hit_rate": (hits / len(with_expected)) if with_expected else None,
    }


def _table_rows(records: list[dict]) -> list[list]:
    """Build display rows for the scorecard table."""
    rows = []
    for r in records:
        hit = "-" if r["retrieval_hit"] is None else ("yes" if r["retrieval_hit"] else "no")
        rows.append(
            [
                r["id"],
                _truncate(r["question"]),
                "yes" if r["answerable"] else "no",
                r["behavior"],
                "PASS" if r["behavior_ok"] else "FAIL",
                f"{r['faithfulness']:.2f}",
                f"{r['relevance']:.2f}",
                "YES" if r["hallucination"] else "no",
                hit,
            ]
        )
    return rows


HEADERS = [
    "ID", "Question", "Answerable", "Behavior", "OK",
    "Faith", "Relev", "Halluc", "Hit",
]


def write_results_md(records: list[dict], summary: dict) -> None:
    """Write a Markdown scorecard report to eval/results.md."""
    table = tabulate(_table_rows(records), headers=HEADERS, tablefmt="github")

    hit_line = (
        f"{summary['retrieval_hits']}/{summary['retrieval_checked']} "
        f"({summary['retrieval_hit_rate']:.0%})"
        if summary["retrieval_hit_rate"] is not None
        else "n/a (no expected sources declared)"
    )

    md = f"""# DocGround — Evaluation Results

Automated scorecard produced by `eval/run_eval.py` over `eval/questions.json`.

## Summary

| Metric | Value |
|--------|-------|
| Questions evaluated | {summary['n']} |
| Correct behavior (answer vs. refuse) | {summary['behavior_ok']}/{summary['n']} ({summary['behavior_acc']:.0%}) |
| Mean faithfulness | {summary['mean_faithfulness']:.2f} |
| Mean answer-relevance | {summary['mean_relevance']:.2f} |
| Hallucinations flagged | {summary['hallucinations']}/{summary['n']} ({summary['halluc_rate']:.0%}) |
| Retrieval hit rate | {hit_line} |

- **Correct behavior** = the system answered questions it could support and
  refused the ones it could not (the unanswerable controls).
- **Faithfulness / relevance** are LLM-as-judge scores in [0, 1].
- A **hallucination** is flagged when faithfulness < 0.7 or any unsupported
  claim is detected.

## Per-question scorecard

{table}

## Unsupported claims detected

"""
    flagged = [r for r in records if r["unsupported_claims"]]
    if flagged:
        for r in flagged:
            md += f"\n**{r['id']} — {r['question']}**\n\n"
            for claim in r["unsupported_claims"]:
                md += f"- {claim}\n"
    else:
        md += "_None — no unsupported claims were detected._\n"

    RESULTS_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    print(f"Loading questions from {QUESTIONS_PATH} ...\n")
    records = run()
    summary = _summary(records)

    print("\n" + "=" * 80)
    print("SCORECARD")
    print("=" * 80)
    print(tabulate(_table_rows(records), headers=HEADERS, tablefmt="github"))

    print("\nSummary")
    print("-------")
    print(f"  Questions evaluated : {summary['n']}")
    print(f"  Correct behavior    : {summary['behavior_ok']}/{summary['n']} "
          f"({summary['behavior_acc']:.0%})")
    print(f"  Mean faithfulness   : {summary['mean_faithfulness']:.2f}")
    print(f"  Mean relevance      : {summary['mean_relevance']:.2f}")
    print(f"  Hallucinations      : {summary['hallucinations']}/{summary['n']} "
          f"({summary['halluc_rate']:.0%})")
    if summary["retrieval_hit_rate"] is not None:
        print(f"  Retrieval hit rate  : {summary['retrieval_hits']}/"
              f"{summary['retrieval_checked']} ({summary['retrieval_hit_rate']:.0%})")

    write_results_md(records, summary)
    print(f"\nWrote report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
