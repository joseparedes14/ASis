"""
Email tools for the AI agent.

LangChain-compatible tools for email operations. These are placeholder
implementations that will be connected to the email service layer
when the service is fully implemented.
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.services.smtp_service import GmailSMTPService
from app.tools.base import ToolRiskLevel, tool_metadata

logger = get_logger(__name__)


# ── Argument Schemas ────────────────────────────────────────────────────


class SearchEmailsInput(BaseModel):
    """Input schema for the search_emails tool."""

    sender: Optional[str] = Field(
        default=None,
        description="Filter by sender email address or name (partial match).",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Filter by email subject (partial match).",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Start date filter in YYYY-MM-DD format.",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )


class GetEmailContentInput(BaseModel):
    """Input schema for the get_email_content tool."""

    message_id: str = Field(
        description="Unique identifier of the email to retrieve.",
    )


class DownloadAttachmentInput(BaseModel):
    """Input schema for the download_attachment tool."""

    message_id: str = Field(
        description="Email message ID containing the attachment.",
    )
    filename: str = Field(
        description="Exact filename of the attachment to download.",
    )


class SendEmailInput(BaseModel):
    """Input schema for the send_email tool."""

    to: str = Field(
        description="Recipient email address.",
    )
    subject: str = Field(
        description="Email subject line.",
    )
    body: str = Field(
        description="Body content of the email. The agent writes the complete message here.",
    )


# ── Tool Implementations ───────────────────────────────────────────────


@tool(args_schema=SearchEmailsInput)
def search_emails(
    sender: Optional[str] = None,
    subject: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search for emails matching specified criteria.

    Use this tool to find emails by sender, subject, date range, or any
    combination of these filters. Returns a summary of matching emails.
    """
    logger.info(
        "search_emails called — sender=%s, subject=%s, date_from=%s, date_to=%s, limit=%d",
        sender,
        subject,
        date_from,
        date_to,
        limit,
    )
    # TODO: Connect to EmailService when implemented
    return (
        "[Email search not yet implemented] "
        f"Would search for emails with: sender={sender}, subject={subject}, "
        f"date_from={date_from}, date_to={date_to}, limit={limit}"
    )


@tool(args_schema=GetEmailContentInput)
def get_email_content(message_id: str) -> str:
    """Retrieve the full content of a specific email by its ID.

    Use this tool after search_emails to read the complete body and
    metadata of a specific email message.
    """
    logger.info("get_email_content called — message_id=%s", message_id)
    # TODO: Connect to EmailService when implemented
    return f"[Email retrieval not yet implemented] Would retrieve email: {message_id}"


@tool(args_schema=DownloadAttachmentInput)
def download_attachment(message_id: str, filename: str) -> str:
    """Download a specific attachment from an email.

    Use this tool to save an email attachment to the local filesystem.
    This action requires user confirmation before execution.
    """
    logger.info(
        "download_attachment called — message_id=%s, filename=%s",
        message_id,
        filename,
    )
    # TODO: Connect to EmailService + StorageService when implemented
    return (
        f"[Attachment download not yet implemented] "
        f"Would download '{filename}' from email {message_id}"
    )


@tool(args_schema=SendEmailInput)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Use this tool to send an email message. You (the agent) write the complete
    subject and body of the email. The user will be asked for confirmation
    before the email is actually sent.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Complete body content of the email.
    """
    import asyncio

    logger.info(
        "send_email called — to=%s, subject=%s",
        to,
        subject,
    )

    try:
        settings = get_settings()
        service = GmailSMTPService(settings)

        async def _send() -> bool:
            await service.connect()
            try:
                return await service.send_email(to=to, subject=subject, body=body)
            finally:
                await service.disconnect()

        send_result = asyncio.run(_send())

        if send_result:
            return f"Email sent successfully to {to} with subject: '{subject}'"
        else:
            return f"Failed to send email to {to}. Please check the logs for details."

    except FileNotFoundError as e:
        logger.error("OAuth2 credentials not found: %s", e)
        return (
            f"Error: {e}\n"
            "Please run 'python scripts/authorize_gmail.py' first to authorize Gmail access."
        )
    except ConnectionError as e:
        logger.error("SMTP connection error: %s", e)
        return f"Error connecting to Gmail SMTP: {e}"
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return f"Configuration error: {e}"
    except Exception as e:
        logger.error("Unexpected error sending email: %s", e)
        return f"Error sending email: {e}"


# ── Tool List ───────────────────────────────────────────────────────────

# Set metadata
search_emails.metadata = tool_metadata(risk_level=ToolRiskLevel.LOW, category="email")
get_email_content.metadata = tool_metadata(risk_level=ToolRiskLevel.LOW, category="email")
download_attachment.metadata = tool_metadata(
    risk_level=ToolRiskLevel.HIGH,
    requires_confirmation=True,
    category="email",
)
send_email.metadata = tool_metadata(
    risk_level=ToolRiskLevel.HIGH,
    requires_confirmation=True,
    category="email",
)

EMAIL_TOOLS = [search_emails, get_email_content, download_attachment, send_email]
