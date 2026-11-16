"""Backend adapter registry and factory (`create_backend`)."""

from __future__ import annotations

from typing import Any

from apart_epistemics.backends.base import (
    BackendConfigError,
    BudgetExceededError,
    TransientBackendError,
    TransportFn,
)
from apart_epistemics.backends.providers import (
    AdaptionBackend,
    AnthropicBackend,
    GeminiBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    OpenRouterBackend,
)


def create_backend(name: str, **kwargs: Any) -> Any:
    """Return the configured backend adapter for `name`."""
    norm = name.strip().lower()
    if norm == "ollama":
        return OllamaBackend(
            model=str(kwargs.get("model", "llama3.1:8b")),
            base_url=kwargs.get("base_url"),
            transport=kwargs.get("transport"),
        )
    if norm == "openrouter":
        return OpenRouterBackend(
            model=str(kwargs.get("model", "openai/gpt-4o-mini")),
            pin_provider=kwargs.get("pin_provider"),
            reasoning_effort=str(kwargs.get("reasoning_effort", "low")),
            api_key=kwargs.get("api_key"),
            transport=kwargs.get("transport"),
        )
    if norm == "openai":
        return OpenAICompatibleBackend(
            model=str(kwargs.get("model", "gpt-4o-mini")),
            api_key_env="OPENAI_API_KEY",
            base_url=str(kwargs.get("base_url", "https://api.openai.com/v1")),
            provider_name="openai",
            transport=kwargs.get("transport"),
        )
    if norm == "anthropic":
        return AnthropicBackend(
            model=str(kwargs.get("model", "claude-3-7-sonnet-latest")),
            api_key=kwargs.get("api_key"),
            transport=kwargs.get("transport"),
        )
    if norm == "gemini":
        return GeminiBackend(
            model=str(kwargs.get("model", "gemini-2.5-flash")),
            api_key=kwargs.get("api_key"),
            transport=kwargs.get("transport"),
        )
    if norm == "adaption":
        adaption_kwargs: dict[str, Any] = {
            "model": str(kwargs.get("model", "deepseek-v4-flash")),
            "api_key": kwargs.get("api_key"),
            "min_output_tokens": int(kwargs.get("min_output_tokens", 8192)),
            "transport": kwargs.get("transport"),
        }
        if kwargs.get("base_url"):
            adaption_kwargs["base_url"] = str(kwargs["base_url"])
        return AdaptionBackend(**adaption_kwargs)
    raise BackendConfigError(f"Unknown backend '{name}'.")


__all__ = [
    "AdaptionBackend",
    "AnthropicBackend",
    "BackendConfigError",
    "BudgetExceededError",
    "GeminiBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "OpenRouterBackend",
    "TransientBackendError",
    "TransportFn",
    "create_backend",
]
