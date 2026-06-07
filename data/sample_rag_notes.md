# Retrieval-Augmented Generation: Field Notes

These are original notes for testing the DocGround pipeline. Replace this file
with your real corpus (LangChain docs, arXiv RAG/hallucination papers, prompting
guides) when you are ready.

## What RAG is

Retrieval-Augmented Generation (RAG) combines a retriever with a generator. The
retriever finds relevant passages from an external document store, and the
language model conditions its answer on those passages. This grounds answers in
source material instead of relying only on the model's parametric memory.

## Chunking

Long documents are split into smaller chunks before embedding. A common starting
point is a chunk size of around 800 characters with an overlap of about 120
characters. Overlap keeps sentences that straddle a chunk boundary from losing
their context, which improves retrieval quality.

## Embeddings and vector search

Each chunk is converted into a dense vector by an embedding model. At query time
the question is embedded the same way, and the store returns the chunks whose
vectors are closest to the question vector. This is called semantic search
because it matches meaning rather than exact keywords.

## Faithfulness and hallucination

An answer is faithful when every claim it makes is supported by the retrieved
context. A hallucination is any claim that is not supported by the provided
sources. Measuring faithfulness requires checking each claim against the context,
which is why a document-grounded system should refuse to answer when the context
does not contain the information.

## Citations

A grounded system cites its sources so a reader can verify each claim. In
DocGround, citations take the form [source, page], pointing back to the exact
document and page a statement came from.
