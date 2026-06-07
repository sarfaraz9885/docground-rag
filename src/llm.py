"""Chat-model factory shared by generation and evaluation.

Centralizes provider selection (OpenAI vs Anthropic) so the rest of the code
asks for "a chat model" without caring which backend is configured.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings

# Model families that reject a custom `temperature` (only the default is allowed).
# GPT-5 / o-series reasoning models fall in this bucket.
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5-nano", "o1", "o3", "o4")


def _openai_rejects_temperature(model: str) -> bool:
    name = model.lower()
    return any(name.startswith(p) for p in _FIXED_TEMPERATURE_PREFIXES)


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Return a configured chat model for the active provider.

    ``temperature`` defaults to 0 for deterministic, grounded answers. It is
    silently dropped for OpenAI reasoning models that only allow the default.
    """
    settings.require_llm()

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_chat_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=1024,
        )

    # Default: OpenAI.
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key,
    }
    if not _openai_rejects_temperature(settings.openai_chat_model):
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)
