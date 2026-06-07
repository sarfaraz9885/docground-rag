# DocGround — Evaluation Results

Automated scorecard produced by `eval/run_eval.py` over `eval/questions.json`.

## Summary

| Metric | Value |
|--------|-------|
| Questions evaluated | 15 |
| Correct behavior (answer vs. refuse) | 15/15 (100%) |
| Mean faithfulness | 1.00 |
| Mean answer-relevance | 0.99 |
| Hallucinations flagged | 0/15 (0%) |
| Retrieval hit rate | 10/10 (100%) |

- **Correct behavior** = the system answered questions it could support and
  refused the ones it could not (the unanswerable controls).
- **Faithfulness / relevance** are LLM-as-judge scores in [0, 1].
- A **hallucination** is flagged when faithfulness < 0.7 or any unsupported
  claim is detected.

## Per-question scorecard

| ID   | Question                                         | Answerable   | Behavior   | OK   |   Faith |   Relev | Halluc   | Hit   |
|------|--------------------------------------------------|--------------|------------|------|---------|---------|----------|-------|
| q01  | What is Retrieval-Augmented Generation (RAG)?    | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q02  | What chunk size and overlap does the guidance... | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q03  | Why is overlap used when chunking documents?     | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q04  | What is semantic search and why is it called ... | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q05  | How is each chunk turned into something that ... | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q06  | What does it mean for an answer to be faithful?  | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q07  | What is a hallucination according to these no... | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q08  | When should a document-grounded system refuse... | yes          | answer     | PASS |       1 |     0.9 | no       | yes   |
| q09  | What form do citations take in DocGround?        | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q10  | What two components does RAG combine?            | yes          | answer     | PASS |       1 |     1   | no       | yes   |
| q11  | Who won the 2022 FIFA World Cup?                 | no           | refuse     | PASS |       1 |     1   | no       | -     |
| q12  | What is the capital of Australia?                | no           | refuse     | PASS |       1 |     1   | no       | -     |
| q13  | What was OpenAI's total revenue in 2024?         | no           | refuse     | PASS |       1 |     1   | no       | -     |
| q14  | How do I fine-tune a Llama 3 model with LoRA?    | no           | refuse     | PASS |       1 |     1   | no       | -     |
| q15  | What are Pinecone's paid pricing tiers?          | no           | refuse     | PASS |       1 |     1   | no       | -     |

## Unsupported claims detected

_None — no unsupported claims were detected._
