"""Ingestion pipeline: load -> chunk -> embed -> persist to Chroma.

Run as a module from the repo root:

    python -m src.ingest                 # ingest everything in data/
    python -m src.ingest --reset         # wipe the store first, then ingest
    python -m src.ingest --data ./data   # ingest a specific folder

What it does
------------
1. Loads every PDF / Markdown / text file under ``data/``.
2. Splits documents with ``RecursiveCharacterTextSplitter`` (chunk_size=800,
   overlap=120) so adjacent chunks share context.
3. Embeds chunks with OpenAI ``text-embedding-3-small``.
4. Persists vectors + metadata to a Pinecone serverless index (auto-created on
   first run).

Every chunk keeps ``source`` (file name) and ``page`` in its metadata so the
generation step can produce ``[source, page]`` citations.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import settings, REPO_ROOT

# --- Chunking parameters (per spec) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Extensions we know how to load.
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTS = PDF_EXTS | TEXT_EXTS


def _load_file(path: Path) -> list[Document]:
    """Load a single file into one or more LangChain ``Document`` objects.

    PDFs yield one Document per page (so ``page`` metadata is real). Text and
    Markdown files yield a single Document; we set ``page`` to 1 for a uniform
    citation schema downstream.
    """
    ext = path.suffix.lower()

    if ext in PDF_EXTS:
        # PyPDFLoader sets metadata["page"] (0-indexed) per page automatically.
        docs = PyPDFLoader(str(path)).load()
    elif ext in TEXT_EXTS:
        # encoding="utf-8" avoids platform-default decode errors on Windows.
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        return []

    # Normalize metadata: store a clean file name in `source`, ensure `page` exists.
    for d in docs:
        d.metadata["source"] = path.name
        # PyPDF uses 0-indexed pages; present them 1-indexed for humans.
        if ext in PDF_EXTS:
            d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
        else:
            d.metadata.setdefault("page", 1)
    return docs


def load_documents(data_dir: Path) -> list[Document]:
    """Recursively load every supported document under ``data_dir``."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: list[Document] = []
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No supported documents (.pdf/.md/.txt) found in {data_dir}. "
            "Drop some files in data/ and re-run."
        )

    for path in files:
        loaded = _load_file(path)
        print(f"  loaded {len(loaded):>3} doc(s) from {path.name}")
        docs.extend(loaded)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split on natural boundaries first, falling back to characters.
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # records char offset in metadata["start_index"]
    )
    chunks = splitter.split_documents(docs)
    return chunks


def get_embeddings() -> OpenAIEmbeddings:
    """Build the OpenAI embeddings client (key validated up front)."""
    settings.require_openai()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_pinecone_client() -> Pinecone:
    """Build the Pinecone client (key validated up front)."""
    settings.require_pinecone()
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(pc: Pinecone | None = None) -> Pinecone:
    """Create the serverless index if it doesn't exist, then wait until ready.

    The index dimension is derived from the configured embedding model
    (text-embedding-3-small -> 1536) and the metric is cosine. Returns the
    Pinecone client so callers can reuse it.
    """
    pc = pc or get_pinecone_client()
    name = settings.pinecone_index

    if not pc.has_index(name):
        print(
            f"Creating Pinecone index '{name}' "
            f"(dim={settings.embedding_dimension}, metric=cosine, "
            f"{settings.pinecone_cloud}/{settings.pinecone_region}) ..."
        )
        pc.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Serverless indexes come up in a few seconds; poll until ready.
        while not pc.describe_index(name).status.get("ready", False):
            time.sleep(1)
        print("Index is ready.")
    else:
        # Index exists — make sure its dimension matches our embedding model,
        # otherwise upserts fail later with a cryptic 400.
        existing_dim = pc.describe_index(name).dimension
        if existing_dim != settings.embedding_dimension:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {existing_dim}, but "
                f"embedding model '{settings.embedding_model}' produces "
                f"{settings.embedding_dimension}-dim vectors.\n"
                f"Delete and recreate the index at the right size, e.g.:\n"
                f"    python -c \"from src.ingest import get_pinecone_client; "
                f"get_pinecone_client().delete_index('{name}')\"\n"
                f"then re-run ingestion (it will recreate the index)."
            )
    return pc


def get_vectorstore(embeddings: OpenAIEmbeddings | None = None) -> PineconeVectorStore:
    """Open (creating if needed) the Pinecone-backed vector store.

    Shared by ingest/retrieve so the index name + embeddings stay in sync.
    """
    pc = ensure_index()
    return PineconeVectorStore(
        index=pc.Index(settings.pinecone_index),
        embedding=embeddings or get_embeddings(),
    )


def ingest(data_dir: Path, reset: bool = False) -> int:
    """Run the full pipeline. Returns the number of chunks stored."""
    pc = ensure_index()

    if reset:
        # Wipe all existing vectors so re-ingestion starts clean (keeps the index).
        print(f"Resetting: deleting all vectors in index '{settings.pinecone_index}' ...")
        try:
            pc.Index(settings.pinecone_index).delete(delete_all=True)
        except Exception as exc:  # empty index raises 404 — safe to ignore
            print(f"  (nothing to delete: {exc})")

    print(f"Loading documents from {data_dir} ...")
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} document section(s).")

    print(f"Splitting (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = split_documents(docs)
    print(f"Produced {len(chunks)} chunk(s).")

    print(f"Embedding with '{settings.embedding_model}' and upserting to Pinecone ...")
    vectorstore = get_vectorstore()
    # add_documents embeds each chunk and upserts vectors + metadata.
    vectorstore.add_documents(chunks)

    total = pc.Index(settings.pinecone_index).describe_index_stats().get(
        "total_vector_count", len(chunks)
    )
    print(f"Done. Pinecone index '{settings.pinecone_index}' now holds {total} vector(s).")
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data",
        help="Folder of documents to ingest (default: ./data).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma store before ingesting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(data_dir=args.data, reset=args.reset)
