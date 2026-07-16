"""Tests for LLM provider factory and configuration."""

import pytest

from app.config.settings import Settings
from app.models.llm import LLMProviderError, get_available_providers


class TestLLMFactory:
    """Tests for the LLM provider factory."""

    def test_available_providers(self):
        """Should list all supported providers."""
        providers = get_available_providers()
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_openai_requires_api_key(self):
        """OpenAI provider should fail without API key."""
        from app.models.llm import create_llm

        settings = Settings(llm_provider="openai", openai_api_key=None)
        with pytest.raises(LLMProviderError, match="API key"):
            create_llm(settings)

    def test_anthropic_requires_api_key(self):
        """Anthropic provider should fail without API key."""
        from app.models.llm import create_llm

        settings = Settings(llm_provider="anthropic", anthropic_api_key=None)
        with pytest.raises(LLMProviderError, match="API key"):
            create_llm(settings)

    def test_unsupported_provider_raises(self):
        """Unsupported provider should raise a validation error."""
        with pytest.raises(Exception):
            Settings(llm_provider="unsupported_provider")


class TestSettings:
    """Tests for application settings validation."""

    def test_default_settings(self):
        """Default settings should be valid."""
        settings = Settings()
        assert settings.llm_provider == "ollama"
        assert settings.llm_model  # Model is set via .env or default
        assert settings.log_level == "INFO"
        assert settings.require_confirmation is True

    def test_invalid_log_level(self):
        """Invalid log level should raise validation error."""
        with pytest.raises(Exception):
            Settings(log_level="INVALID")

    def test_sensitive_tools_default(self):
        """Default sensitive tools list should include key operations."""
        settings = Settings()
        assert "download_attachment" in settings.sensitive_tools
        assert "save_file" in settings.sensitive_tools
