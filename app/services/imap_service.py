"""
IMAP email service with OAuth2 authentication.

Provides email reading capability (search, fetch, download attachments)
using Gmail's IMAP server with OAuth2 token-based authentication.
"""

import email
from datetime import datetime
from email.header import decode_header
from typing import Optional

from imapclient import IMAPClient

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.services.email_service import EmailAttachment, EmailMessage, EmailServiceBase
from app.services.oauth_helper import load_oauth_credentials

logger = get_logger(__name__)


class ImapEmailService(EmailServiceBase):
    """IMAP email service with OAuth2 token management.

    Handles connection, authentication, and email operations
    via Gmail's IMAP server using OAuth2 credentials.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._imap: Optional[IMAPClient] = None

    async def connect(self) -> None:
        """Establish connection to Gmail IMAP server with OAuth2 auth."""
        if not self._settings.email_address:
            raise ValueError(
                "Email address not configured. Set EMAIL_ADDRESS in your .env file."
            )

        creds = load_oauth_credentials(self._settings)

        logger.info(
            "Connecting to %s:%d...",
            self._settings.email_imap_server,
            self._settings.email_imap_port,
        )

        try:
            self._imap = IMAPClient(
                self._settings.email_imap_server,
                port=self._settings.email_imap_port,
                ssl=True,
            )
            self._imap.oauthbearer_login(self._settings.email_address, creds.token)
            logger.info("Connected and authenticated to Gmail IMAP")

        except Exception as e:
            logger.error("Failed to connect to IMAP server: %s", e)
            raise ConnectionError(f"IMAP connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close the IMAP connection."""
        if self._imap:
            try:
                self._imap.logout()
                logger.info("Disconnected from Gmail IMAP")
            except Exception as e:
                logger.warning("Error disconnecting from IMAP: %s", e)
            finally:
                self._imap = None

    async def search_emails(
        self,
        sender: Optional[str] = None,
        subject: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
    ) -> list[EmailMessage]:
        """Search for emails matching the given criteria."""
        if not self._imap:
            raise ConnectionError("Not connected. Call connect() first.")

        self._imap.select_folder(folder, readonly=True)

        criteria = []
        if unread_only:
            criteria.append("UNSEEN")
        if sender:
            criteria.append(f'FROM "{sender}"')
        if subject:
            criteria.append(f'SUBJECT "{subject}"')
        if date_from:
            criteria.append(f'SINCE {date_from.strftime("%d-%b-%Y")}')
        if date_to:
            criteria.append(f'BEFORE {date_to.strftime("%d-%b-%Y")}')

        search_criteria = " ".join(criteria) if criteria else "ALL"
        logger.info("IMAP search: %s", search_criteria)

        message_ids = self._imap.search(search_criteria)

        if not message_ids:
            return []

        # Reverse to get newest first, then apply limit
        message_ids = list(reversed(message_ids))[:limit]

        fetched = self._imap.fetch(message_ids, ["RFC822"])

        emails = []
        for uid, data in fetched.items():
            raw_email = data[b"RFC822"]
            parsed = self._parse_email(raw_email, folder, uid=uid)
            if parsed:
                emails.append(parsed)

        logger.info("Found %d emails", len(emails))
        return emails

    async def get_email(self, message_id: str) -> Optional[EmailMessage]:
        """Retrieve a single email by its IMAP UID."""
        if not self._imap:
            raise ConnectionError("Not connected. Call connect() first.")

        self._imap.select_folder("INBOX", readonly=True)

        try:
            uid = int(message_id)
        except ValueError:
            return None

        fetched = self._imap.fetch([uid], ["RFC822"])
        if uid not in fetched:
            return None

        raw_email = fetched[uid][b"RFC822"]
        return self._parse_email(raw_email, "INBOX", uid=uid)

    async def download_attachment(
        self, message_id: str, attachment_filename: str
    ) -> Optional[bytes]:
        """Download a specific attachment from an email by IMAP UID."""
        if not self._imap:
            raise ConnectionError("Not connected. Call connect() first.")

        self._imap.select_folder("INBOX", readonly=True)

        try:
            uid = int(message_id)
        except ValueError:
            return None

        fetched = self._imap.fetch([uid], ["RFC822"])
        if uid not in fetched:
            return None

        raw_email = fetched[uid][b"RFC822"]
        msg = email.message_from_bytes(raw_email)

        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in content_disposition:
                filename = self._decode_filename(part.get_filename() or "")
                if filename and filename == attachment_filename:
                    return part.get_payload(decode=True)

        return None

    async def download_all_attachments(
        self, message_id: str
    ) -> list[tuple[str, bytes]]:
        """Download all attachments from an email by IMAP UID."""
        if not self._imap:
            raise ConnectionError("Not connected. Call connect() first.")

        self._imap.select_folder("INBOX", readonly=True)

        try:
            uid = int(message_id)
        except ValueError:
            return []

        fetched = self._imap.fetch([uid], ["RFC822"])
        if uid not in fetched:
            return []

        raw_email = fetched[uid][b"RFC822"]
        msg = email.message_from_bytes(raw_email)

        attachments = []
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in content_disposition:
                filename = self._decode_filename(part.get_filename() or "")
                content = part.get_payload(decode=True)
                if filename and content:
                    attachments.append((filename, content))

        return attachments

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark an email as read (set \\Seen flag).

        Args:
            message_id: IMAP UID of the email.

        Returns:
            True if successful, False otherwise.
        """
        if not self._imap:
            raise ConnectionError("Not connected. Call connect() first.")

        try:
            uid = int(message_id)
        except ValueError:
            return False

        try:
            self._imap.select_folder("INBOX")
            self._imap.set_flags(uid, [b"\\Seen"])
            logger.info("Marked email %s as read", message_id)
            return True
        except Exception as e:
            logger.error("Failed to mark email as read: %s", e)
            return False

    def _parse_email(
        self,
        raw_bytes: bytes,
        folder: str = "INBOX",
        uid: Optional[int] = None,
    ) -> Optional[EmailMessage]:
        """Parse raw email bytes into an EmailMessage dataclass.

        Args:
            raw_bytes: Raw email content from IMAP.
            folder: Mailbox folder the email was fetched from.
            uid: IMAP UID — used as message_id for subsequent operations.

        Returns:
            Parsed EmailMessage, or None on parse failure.
        """
        try:
            msg = email.message_from_bytes(raw_bytes)

            subject = self._decode_header(msg.get("Subject", ""))
            sender = self._decode_header(msg.get("From", ""))
            message_id = str(uid) if uid is not None else msg.get("Message-ID", "")

            date_str = msg.get("Date", "")
            try:
                date_tuple = email.utils.parsedate_to_datetime(date_str)
            except Exception:
                date_tuple = datetime.now()

            recipients = []
            to_header = msg.get("To", "")
            if to_header:
                recipients = [addr.strip() for addr in to_header.split(",")]

            body_text = ""
            body_html = None
            attachments = []

            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    filename = self._decode_filename(part.get_filename() or "")
                    content = part.get_payload(decode=True)
                    if filename and content:
                        attachments.append(EmailAttachment(
                            filename=filename,
                            content_type=content_type,
                            size_bytes=len(content),
                            content=content,
                        ))
                elif content_type == "text/plain" and not body_text:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_text = payload.decode(charset, errors="replace")
                elif content_type == "text/html" and not body_html:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_html = payload.decode(charset, errors="replace")

            return EmailMessage(
                message_id=message_id,
                subject=subject,
                sender=sender,
                recipients=recipients,
                date=date_tuple,
                body_text=body_text,
                body_html=body_html,
                attachments=attachments,
                folder=folder,
            )

        except Exception as e:
            logger.error("Failed to parse email: %s", e)
            return None

    def _decode_header(self, header_value: str) -> str:
        """Decode an email header value that may contain encoded words."""
        if not header_value:
            return ""

        decoded_parts = decode_header(header_value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                result.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                result.append(part)
        return " ".join(result)

    @staticmethod
    def _decode_filename(filename: str) -> str:
        """Decode an encoded MIME filename (e.g. =?UTF-8?Q?...?=)."""
        if not filename:
            return ""
        decoded_parts = decode_header(filename)
        name_parts = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                name_parts.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                name_parts.append(part)
        return "".join(name_parts)
