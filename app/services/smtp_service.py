"""
Gmail SMTP service with OAuth2 authentication.

Provides email sending capability using Gmail's SMTP server
with OAuth2 token-based authentication. Tokens are automatically
refreshed when expired, requiring user authorization only once.
"""

import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from google.oauth2.credentials import Credentials

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.services.oauth_helper import load_oauth_credentials

logger = get_logger(__name__)


class GmailSMTPService:
    """Gmail SMTP service with OAuth2 token management.

    Handles authentication, token refresh, and email sending
    via Gmail's SMTP server using OAuth2 credentials.

    Usage:
        settings = get_settings()
        service = GmailSMTPService(settings)
        await service.connect()
        await service.send_email(
            to="recipient@example.com",
            subject="Hello",
            body="Message content"
        )
        await service.disconnect()
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the SMTP service with application settings.

        Args:
            settings: Application settings containing OAuth2 configuration.
        """
        self._settings = settings
        self._creds: Optional[Credentials] = None
        self._smtp: Optional[smtplib.SMTP_SSL] = None

    def _load_credentials(self) -> Credentials:
        """Load OAuth2 credentials from disk or perform authorization flow.

        Returns:
            Valid OAuth2 credentials.

        Raises:
            FileNotFoundError: If credentials.json doesn't exist.
            ValueError: If authorization fails.
        """
        return load_oauth_credentials(self._settings)

    def _build_xoauth2_string(self) -> str:
        """Build the XOAUTH2 authentication string for SMTP.

        Returns:
            Base64-encoded XOAUTH2 string for SMTP authentication.
        """
        auth_string = (
            f"user={self._settings.email_address}\x01"
            f"auth=Bearer {self._creds.token}\x01"
            f"\x01"
        )
        return base64.b64encode(auth_string.encode()).decode()

    async def connect(self) -> None:
        """Establish connection to Gmail SMTP server.

        Loads credentials and establishes SSL connection to smtp.gmail.com.

        Raises:
            ConnectionError: If connection fails.
            ValueError: If not configured.
        """
        if not self._settings.email_address:
            raise ValueError(
                "Email address not configured. Set EMAIL_ADDRESS in your .env file."
            )

        # Load/refresh credentials
        self._creds = self._load_credentials()

        logger.info(
            "Connecting to %s:%d...",
            self._settings.email_smtp_server,
            self._settings.email_smtp_port,
        )

        try:
            self._smtp = smtplib.SMTP_SSL(
                self._settings.email_smtp_server,
                self._settings.email_smtp_port,
            )
            self._smtp.ehlo()

            # Authenticate with XOAUTH2 (SMTP uses AUTH, not AUTHENTICATE)
            auth_string = self._build_xoauth2_string()
            self._smtp.docmd("AUTH", f"XOAUTH2 {auth_string}")

            logger.info("Connected and authenticated to Gmail SMTP")

        except Exception as e:
            logger.error("Failed to connect to SMTP server: %s", e)
            raise ConnectionError(f"SMTP connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close the SMTP connection."""
        if self._smtp:
            try:
                self._smtp.quit()
                logger.info("Disconnected from Gmail SMTP")
            except Exception as e:
                logger.warning("Error disconnecting from SMTP: %s", e)
            finally:
                self._smtp = None

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        body_html: Optional[str] = None,
    ) -> bool:
        """Send an email via Gmail SMTP.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Plain text body of the email.
            body_html: Optional HTML body (if provided, email will be multipart).

        Returns:
            True if email was sent successfully, False otherwise.

        Raises:
            ConnectionError: If not connected to SMTP server.
            ValueError: If recipient is invalid.
        """
        if not self._smtp:
            raise ConnectionError("Not connected to SMTP server. Call connect() first.")

        if not to or "@" not in to:
            raise ValueError(f"Invalid recipient email address: {to}")

        # Build the MIME message
        msg = MIMEMultipart("alternative")
        msg["From"] = self._settings.email_address
        msg["To"] = to
        msg["Subject"] = subject

        # Add plain text part
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Add HTML part if provided
        if body_html:
            msg.attach(MIMEText(body_html, "html", "utf-8"))

        try:
            logger.info(
                "Sending email from %s to %s: '%s'",
                self._settings.email_address,
                to,
                subject,
            )

            self._smtp.sendmail(
                self._settings.email_address,
                to,
                msg.as_string(),
            )

            logger.info("Email sent successfully")
            return True

        except smtplib.SMTPRecipientsRefused as e:
            logger.error("Recipient refused: %s", e)
            return False
        except smtplib.SMTPException as e:
            logger.error("SMTP error: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error sending email: %s", e)
            return False
