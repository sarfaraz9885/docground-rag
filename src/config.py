"""Central configuration for DocGround.

Loads environment variables from a local ``.env`` file (via python-dotenv) and
exposes them as a single typed ``Settings`` object so the rest of the codebase
never reads ``os.environ`` directly. No secrets are hardcoded here — only the
*names* of the variables, with sensible non-secret defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

# Load .env from the repo root (one level above this file's src/ directory).
# Existing real environment variables always win over .env values.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)


# Output dimension of each supported OpenAI embedding model. Used to create the
# Pinecone index with the right vector size.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all runtime configuration."""

    # --- Provider keys (secrets — read from environment, never committed) ---
    openai_api_key: str
    anthropic_api_key: str
    pinecone_api_key: str

    # --- Model / provider selection ---
    llm_provider: str          # "openai" or "anthropic"
    openai_chat_model: str
    anthropic_chat_model: str
    embedding_model: str

    # --- Vector store (Pinecone serverless) ---
    pinecone_index: str        # index name (created on first ingest if missing)
    pinecone_cloud: str        # serverless cloud, e.g. "aws"
    pinecone_region: str       # serverless region, e.g. "us-east-1"

    # --- Service wiring ---
    api_base_url: str

    @property
    def embedding_dimension(self) -> int:
        """Vector dimension for the configured embedding model.

        Falls back to 1536 (the text-embedding-3-small size) for unknown models;
        override explicitly via EMBEDDING_DIMENSIONS if you wire in a new model.
        """
        return EMBEDDING_DIMENSIONS.get(self.embedding_model, 1536)

    def require_openai(self) -> None:
        """Raise a clear error if the OpenAI key (needed for embeddings) is missing."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

    def require_pinecone(self) -> None:
        """Raise a clear error if the Pinecone key (needed for the vector store) is missing."""
        if not self.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

    def require_llm(self) -> None:
        """Raise a clear error if the selected chat provider lacks its key."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set."
            )


def load_settings() -> Settings:
    """Build a :class:`Settings` from the current environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        anthropic_chat_model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        pinecone_index=os.getenv("PINECONE_INDEX", "docground-rag"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
    )


# A module-level singleton is convenient for the small scripts in this project.
settings: Settings = load_settings()
