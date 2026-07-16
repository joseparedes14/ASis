"""
LLM provider factory.

Abstracts model creation so the rest of the application is decoupled
from any specific LLM provider. Switching providers is a config change.
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.config.logging_config import get_logger
from app.config.settings import Settings

logger = get_logger(__name__)


class LLMProviderError(Exception):
    """Raised when the LLM provider cannot be initialized."""


def create_llm(settings: Settings) -> BaseChatModel:
    """Create and return a chat model based on application settings.

    Supports multiple providers through a factory pattern. The provider
    and model are determined by settings (environment variables).

    Args:
        settings: Application settings containing LLM configuration.

    Returns:
        Configured BaseChatModel instance.

    Raises:
        LLMProviderError: If the provider is unsupported or initialization fails.
    """
    provider = settings.llm_provider.lower()
    logger.info("Initializing LLM — provider=%s, model=%s", provider, settings.llm_model)

    try:
        if provider == "ollama":
            return _create_ollama(settings)
        elif provider == "openai":
            return _create_openai(settings)
        elif provider == "anthropic":
            return _create_anthropic(settings)
        else:
            raise LLMProviderError(f"Unsupported LLM provider: '{provider}'")
    except LLMProviderError:
        raise
    except Exception as e:
        raise LLMProviderError(
            f"Failed to initialize {provider} LLM: {e}"
        ) from e


def _create_ollama(settings: Settings) -> BaseChatModel:
    """Create an Ollama-backed chat model.

    Args:
        settings: Application settings.

    Returns:
        ChatOllama instance configured for the specified model.
    """
    from langchain_ollama import ChatOllama

    logger.debug("Connecting to Ollama at %s", settings.ollama_base_url)
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=4096,
    )


def _create_openai(settings: Settings) -> BaseChatModel:
    """Create an OpenAI-backed chat model.

    Args:
        settings: Application settings with OpenAI API key.

    Returns:
        ChatOpenAI instance.

    Raises:
        LLMProviderError: If API key is not configured.
    """
    if not settings.openai_api_key:
        raise LLMProviderError(
            "OpenAI API key is required. Set OPENAI_API_KEY in your .env file."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )


def _create_anthropic(settings: Settings) -> BaseChatModel:
    """Create an Anthropic-backed chat model.

    Args:
        settings: Application settings with Anthropic API key.

    Returns:
        ChatAnthropic instance.

    Raises:
        LLMProviderError: If API key is not configured.
    """
    if not settings.anthropic_api_key:
        raise LLMProviderError(
            "Anthropic API key is required. Set ANTHROPIC_API_KEY in your .env file."
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
    )


def get_available_providers() -> list[str]:
    """Return list of supported LLM providers.

    Returns:
        List of provider name strings.
    """
    return ["ollama", "openai", "anthropic"]


def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible.

    Args:
        base_url: Ollama API base URL.

    Returns:
        True if Ollama is reachable, False otherwise.
    """
    import urllib.request
    import urllib.error

    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False
