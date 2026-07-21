"""
Email tools for the AI agent.

LangChain-compatible tools for email operations including searching,
reading, downloading attachments, and sending emails via Gmail.
"""

import asyncio
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.services.imap_service import ImapEmailService
from app.services.smtp_service import GmailSMTPService
from app.services.storage_service import StorageService
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
    unread_only: bool = Field(
        default=False,
        description="If true, only return unread emails.",
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


class CheckAndDownloadInput(BaseModel):
    """Input schema for the check_and_download_documents tool."""

    sender_email: str = Field(
        description="Email address of the sender to check for documents.",
    )


# ── Tool Implementations ───────────────────────────────────────────────


@tool(args_schema=SearchEmailsInput)
def search_emails(
    sender: Optional[str] = None,
    subject: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 10,
    unread_only: bool = False,
) -> str:
    """Search for emails matching specified criteria.

    Use this tool to find emails by sender, subject, date range, or any
    combination of these filters. Returns a summary of matching emails.
    """
    logger.info(
        "search_emails called — sender=%s, subject=%s, unread_only=%s",
        sender,
        subject,
        unread_only,
    )

    try:
        settings = get_settings()
        service = ImapEmailService(settings)

        async def _search() -> list:
            await service.connect()
            try:
                parsed_from = None
                if date_from:
                    parsed_from = datetime.strptime(date_from, "%Y-%m-%d")
                parsed_to = None
                if date_to:
                    parsed_to = datetime.strptime(date_to, "%Y-%m-%d")

                return await service.search_emails(
                    sender=sender,
                    subject=subject,
                    date_from=parsed_from,
                    date_to=parsed_to,
                    limit=limit,
                    unread_only=unread_only,
                )
            finally:
                await service.disconnect()

        emails = asyncio.run(_search())

        if not emails:
            return "No se encontraron emails con los criterios especificados."

        results = []
        for i, msg in enumerate(emails, 1):
            attachments_info = ""
            if msg.has_attachments:
                att_names = [a.filename for a in msg.attachments]
                attachments_info = f" | Adjuntos: {', '.join(att_names)}"

            results.append(
                f"{i}. [{msg.message_id}] De: {msg.sender}\n"
                f"   Asunto: {msg.subject}\n"
                f"   Fecha: {msg.date.strftime('%Y-%m-%d %H:%M')}{attachments_info}"
            )

        return f"Se encontraron {len(emails)} emails:\n\n" + "\n\n".join(results)

    except FileNotFoundError as e:
        return (
            f"Error: {e}\n"
            "Ejecuta 'python scripts/authorize_gmail.py' primero para autorizar Gmail."
        )
    except ConnectionError as e:
        return f"Error de conexión IMAP: {e}"
    except Exception as e:
        logger.error("Error searching emails: %s", e)
        return f"Error buscando emails: {e}"


@tool(args_schema=GetEmailContentInput)
def get_email_content(message_id: str) -> str:
    """Retrieve the full content of a specific email by its ID.

    Use this tool after search_emails to read the complete body and
    metadata of a specific email message.
    """
    logger.info("get_email_content called — message_id=%s", message_id)

    try:
        settings = get_settings()
        service = ImapEmailService(settings)

        async def _get():
            await service.connect()
            try:
                return await service.get_email(message_id)
            finally:
                await service.disconnect()

        email_msg = asyncio.run(_get())

        if not email_msg:
            return f"No se encontró el email con ID: {message_id}"

        parts = [
            f"De: {email_msg.sender}",
            f"Para: {', '.join(email_msg.recipients)}",
            f"Asunto: {email_msg.subject}",
            f"Fecha: {email_msg.date.strftime('%Y-%m-%d %H:%M')}",
            f"Carpeta: {email_msg.folder}",
        ]

        if email_msg.has_attachments:
            att_info = []
            for att in email_msg.attachments:
                att_info.append(f"  - {att.filename} ({att.content_type}, {att.size_bytes} bytes)")
            parts.append("Adjuntos:\n" + "\n".join(att_info))

        parts.append(f"\n--- Cuerpo ---\n{email_msg.body_text}")

        if email_msg.body_html:
            parts.append(f"\n--- HTML ---\n{email_msg.body_html[:500]}...")

        return "\n".join(parts)

    except FileNotFoundError as e:
        return (
            f"Error: {e}\n"
            "Ejecuta 'python scripts/authorize_gmail.py' primero para autorizar Gmail."
        )
    except ConnectionError as e:
        return f"Error de conexión IMAP: {e}"
    except Exception as e:
        logger.error("Error getting email content: %s", e)
        return f"Error obteniendo contenido del email: {e}"


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

    try:
        settings = get_settings()
        service = ImapEmailService(settings)
        storage = StorageService(settings)

        async def _download():
            await service.connect()
            try:
                email_msg = await service.get_email(message_id)
                sender = email_msg.sender if email_msg else None

                content = await service.download_attachment(message_id, filename)
                if content:
                    path = storage.save_attachment(content, filename, sender=sender)
                    return f"Archivo descargado: {path}"
                return f"No se encontró el adjunto '{filename}' en el email {message_id}"
            finally:
                await service.disconnect()

        return asyncio.run(_download())

    except FileNotFoundError as e:
        return (
            f"Error: {e}\n"
            "Ejecuta 'python scripts/authorize_gmail.py' primero para autorizar Gmail."
        )
    except ConnectionError as e:
        return f"Error de conexión IMAP: {e}"
    except Exception as e:
        logger.error("Error downloading attachment: %s", e)
        return f"Error descargando adjunto: {e}"


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


@tool(args_schema=CheckAndDownloadInput)
def check_and_download_documents(sender_email: str) -> str:
    """Check for the most recent email from a sender and download all attachments.

    Automatically searches INBOX for the latest email from the given sender,
    downloads all attachments, and saves them to data/attachments/<sender>/.
    Use this tool to manually trigger a check for documents from a specific sender.
    """
    logger.info("check_and_download_documents called — sender_email=%s", sender_email)

    try:
        settings = get_settings()
        service = ImapEmailService(settings)
        storage = StorageService(settings)

        async def _check():
            await service.connect()
            try:
                emails = await service.search_emails(
                    sender=sender_email, limit=1, unread_only=True
                )

                if not emails:
                    return f"No se encontraron emails de {sender_email}."

                email_msg = emails[0]

                if not email_msg.has_attachments:
                    return (
                        f"El email más reciente de {sender_email} no tiene adjuntos.\n"
                        f"Asunto: {email_msg.subject}"
                    )

                downloaded_files = []
                for att in email_msg.attachments:
                    content = await service.download_attachment(
                        email_msg.message_id, att.filename
                    )
                    if content:
                        path = storage.save_attachment(
                            content, att.filename, sender=sender_email
                        )
                        downloaded_files.append(path.name)

                if downloaded_files:
                    return (
                        f"Se descargaron {len(downloaded_files)} archivos de {sender_email}:\n"
                        f"Asunto: {email_msg.subject}\n"
                        f"Archivos: {', '.join(downloaded_files)}\n"
                        f"Ubicación: data/attachments/{sender_email}/"
                    )
                return "No se pudieron descargar los adjuntos."

            finally:
                await service.disconnect()

        return asyncio.run(_check())

    except FileNotFoundError as e:
        return (
            f"Error: {e}\n"
            "Ejecuta 'python scripts/authorize_gmail.py' primero para autorizar Gmail."
        )
    except ConnectionError as e:
        return f"Error de conexión IMAP: {e}"
    except Exception as e:
        logger.error("Error checking documents: %s", e)
        return f"Error al verificar documentos: {e}"


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
check_and_download_documents.metadata = tool_metadata(
    risk_level=ToolRiskLevel.MEDIUM,
    requires_confirmation=False,
    category="email",
)

EMAIL_TOOLS = [
    search_emails,
    get_email_content,
    download_attachment,
    send_email,
    check_and_download_documents,
]
