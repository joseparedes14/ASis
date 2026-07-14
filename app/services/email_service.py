"""
Email service interface.

Defines the abstract interface for email operations. Concrete implementations
will be added later (IMAP, OAuth, etc.) without modifying existing code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EmailAttachment:
    """Represents an email attachment.

    Attributes:
        filename: Original filename of the attachment.
        content_type: MIME type (e.g., 'application/pdf').
        size_bytes: Size in bytes.
        content: Raw bytes of the attachment (loaded on demand).
    """

    filename: str
    content_type: str
    size_bytes: int
    content: Optional[bytes] = field(default=None, repr=False)


@dataclass
class EmailMessage:
    """Represents a parsed email message.

    Attributes:
        message_id: Unique message identifier.
        subject: Email subject line.
        sender: Sender email address.
        recipients: List of recipient addresses.
        date: Date the email was sent.
        body_text: Plain text body.
        body_html: HTML body (if available).
        attachments: List of attachments.
        folder: Mailbox folder (e.g., 'INBOX').
        is_read: Whether the email has been read.
    """

    message_id: str
    subject: str
    sender: str
    recipients: list[str]
    date: datetime
    body_text: str = ""
    body_html: Optional[str] = None
    attachments: list[EmailAttachment] = field(default_factory=list)
    folder: str = "INBOX"
    is_read: bool = False

    @property
    def has_attachments(self) -> bool:
        """Check if email has any attachments."""
        return len(self.attachments) > 0

    @property
    def pdf_attachments(self) -> list[EmailAttachment]:
        """Return only PDF attachments."""
        return [a for a in self.attachments if a.content_type == "application/pdf"]


class EmailServiceBase(ABC):
    """Abstract base class for email service implementations.

    Implement this interface to add support for different email
    protocols or providers (IMAP, Gmail API, Outlook Graph, etc.).
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the email server.

        Raises:
            ConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the email server."""
        ...

    @abstractmethod
    async def search_emails(
        self,
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> list[EmailMessage]:
        """Search for emails matching the given criteria.

        Args:
            sender: Filter by sender address (partial match).
            subject: Filter by subject (partial match).
            date_from: Filter by minimum date.
            date_to: Filter by maximum date.
            folder: Mailbox folder to search.
            limit: Maximum number of results.

        Returns:
            List of matching EmailMessage objects.
        """
        ...

    @abstractmethod
    async def get_email(self, message_id: str) -> Optional[EmailMessage]:
        """Retrieve a single email by its message ID.

        Args:
            message_id: Unique message identifier.

        Returns:
            EmailMessage if found, None otherwise.
        """
        ...

    @abstractmethod
    async def download_attachment(
        self, message_id: str, attachment_filename: str
    ) -> Optional[bytes]:
        """Download a specific attachment from an email.

        Args:
            message_id: Email message identifier.
            attachment_filename: Name of the attachment to download.

        Returns:
            Raw bytes of the attachment, or None if not found.
        """
        ...


class EmailServiceNotConfigured(EmailServiceBase):
    """Placeholder service that raises errors when email is not configured.

    Used as the default until a real email service is configured.
    """

    async def connect(self) -> None:
        raise NotImplementedError(
            "Email service is not configured. "
            "Set EMAIL_IMAP_SERVER and credentials in your .env file."
        )

    async def disconnect(self) -> None:
        pass

    async def search_emails(self, **kwargs) -> list[EmailMessage]:  # type: ignore[override]
        raise NotImplementedError("Email service is not configured.")

    async def get_email(self, message_id: str) -> Optional[EmailMessage]:
        raise NotImplementedError("Email service is not configured.")

    async def download_attachment(
        self, message_id: str, attachment_filename: str
    ) -> Optional[bytes]:
        raise NotImplementedError("Email service is not configured.")
