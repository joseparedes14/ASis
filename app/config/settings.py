"""
Application settings loaded from environment variables.

Uses pydantic-settings to provide validated, typed configuration
with automatic .env file loading.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the ASis application.

    All settings can be overridden via environment variables or a .env file.
    The .env file is loaded automatically from the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Configuration ───────────────────────────────────────────────
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama', 'openai', 'anthropic'",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Model name (provider-specific)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (required if provider is 'openai')",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key (required if provider is 'anthropic')",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation",
    )

    # ── Email Configuration ─────────────────────────────────────────────
    email_imap_server: Optional[str] = Field(
        default=None,
        description="IMAP server address (e.g., imap.gmail.com)",
    )
    email_imap_port: int = Field(
        default=993,
        description="IMAP server port",
    )
    email_address: Optional[str] = Field(
        default=None,
        description="Email address for authentication",
    )
    email_password: Optional[str] = Field(
        default=None,
        description="Email password or app-specific password",
    )
    email_use_ssl: bool = Field(
        default=True,
        description="Use SSL for IMAP connection",
    )
    email_oauth_client_id: Optional[str] = Field(default=None)
    email_oauth_client_secret: Optional[str] = Field(default=None)
    email_oauth_token_file: Optional[str] = Field(default=None)

    # ── SMTP Configuration (for sending emails) ─────────────────────────
    email_smtp_server: str = Field(
        default="smtp.gmail.com",
        description="SMTP server address for sending emails",
    )
    email_smtp_port: int = Field(
        default=465,
        description="SMTP server port (465 for SSL, 587 for TLS)",
    )
    email_oauth_credentials_file: Path = Field(
        default=Path("./credentials.json"),
        description="Path to Google OAuth2 credentials JSON file",
    )

    # ── Email Monitor Configuration ────────────────────────────────────
    monitored_senders_file: Path = Field(
        default=Path("./config/monitored_senders.json"),
        description="JSON file with monitored sender email addresses",
    )
    email_monitor_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Email monitor polling interval in minutes",
    )

    # ── Storage Configuration ───────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for application data",
    )
    downloads_dir: Path = Field(
        default=Path("./data/downloads"),
        description="Directory for downloaded files",
    )
    attachments_dir: Path = Field(
        default=Path("./data/attachments"),
        description="Directory for email attachments",
    )

    # ── ASIORGA Configuration ──────────────────────────────────────────
    asiorga_root: Path = Field(
        default=Path.home() / "Desktop" / "ASIorga",
        description="Root directory for organized documents (ASIorga)",
    )
    monitored_folders_file: Path = Field(
        default=Path("./config/monitored_folders.json"),
        description="JSON file with monitored folder paths",
    )
    destination_folders_file: Path = Field(
        default=Path("./config/destination_folders.json"),
        description="JSON file with destination folder configurations",
    )

    # ── Application Settings ────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_file: Optional[Path] = Field(
        default=Path("./data/logs/asis.log"),
        description="Log file path (None for console only)",
    )
    max_history_length: int = Field(
        default=50,
        description="Maximum number of messages in conversation history",
    )

    # ── Security ────────────────────────────────────────────────────────
    require_confirmation: bool = Field(
        default=True,
        description="Require user confirmation for sensitive actions",
    )
    sensitive_tools: list[str] = Field(
        default_factory=lambda: [
            "download_attachment",
            "save_file",
            "delete_email",
            "organize_documents",
            "send_email",
            "add_monitored_folder",
            "create_destination_folder",
            "delete_destination_folder",
        ],
        description="Tools that require user confirmation before execution",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {valid_levels}")
        return upper

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Ensure LLM provider is supported."""
        supported = {"ollama", "openai", "anthropic"}
        lower = v.lower()
        if lower not in supported:
            raise ValueError(f"Unsupported LLM provider '{v}'. Must be one of: {supported}")
        return lower

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.data_dir, self.downloads_dir, self.attachments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def get_monitored_senders(self) -> list[str]:
        """Read the list of monitored sender email addresses from JSON.

        Returns:
            List of enabled sender email addresses.
        """
        if not self.monitored_senders_file.exists():
            return []
        try:
            data = json.loads(self.monitored_senders_file.read_text(encoding="utf-8"))
            return [
                s["email"]
                for s in data.get("senders", [])
                if s.get("enabled", True)
            ]
        except (json.JSONDecodeError, KeyError):
            return []


def get_settings() -> Settings:
    """Factory function to create and return application settings.

    Returns:
        Validated Settings instance loaded from environment.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
