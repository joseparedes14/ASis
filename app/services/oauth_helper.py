"""
Shared OAuth2 credential management.

Provides a single source of truth for loading, refreshing, and saving
OAuth2 credentials used by both SMTP (sending) and IMAP (reading) services.
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.config.logging_config import get_logger
from app.config.settings import Settings

logger = get_logger(__name__)

SCOPES = ["https://mail.google.com/"]


def load_oauth_credentials(settings: Settings) -> Credentials:
    """Load OAuth2 credentials from disk, refresh if expired, or authorize.

    This is the single entry point for obtaining valid OAuth2 credentials.
    Both GmailSMTPService and ImapEmailService use this function.

    Args:
        settings: Application settings with OAuth2 file paths.

    Returns:
        Valid OAuth2 credentials.

    Raises:
        FileNotFoundError: If credentials.json doesn't exist and no token exists.
        ValueError: If authorization fails.
    """
    token_path = Path(settings.email_oauth_token_file)
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        logger.debug("Loaded existing token from %s", token_path)

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired access token...")
        creds.refresh(Request())
        _save_token(creds, token_path)
        logger.info("Access token refreshed successfully")

    if not creds or not creds.valid:
        creds = _perform_authorization_flow(settings, token_path)

    return creds


def _perform_authorization_flow(settings: Settings, token_path: Path) -> Credentials:
    """Execute the browser-based OAuth2 authorization flow.

    Opens the browser for user consent and exchanges the authorization
    code for tokens.

    Args:
        settings: Application settings.
        token_path: Where to save the obtained token.

    Returns:
        Valid OAuth2 credentials.

    Raises:
        FileNotFoundError: If credentials.json doesn't exist.
        ValueError: If authorization fails.
    """
    credentials_file = Path(settings.email_oauth_credentials_file)

    if not credentials_file.exists():
        raise FileNotFoundError(
            f"OAuth2 credentials file not found: {credentials_file}\n"
            "Please download it from Google Cloud Console and place it in the project root."
        )

    logger.info("Starting OAuth2 authorization flow...")
    logger.info("A browser window will open for authorization.")

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
    creds = flow.run_local_server(port=0)

    _save_token(creds, token_path)
    logger.info("Authorization completed. Token saved to %s", token_path)
    return creds


def _save_token(creds: Credentials, token_path: Path) -> None:
    """Save OAuth2 token to disk.

    Args:
        creds: Credentials to save.
        token_path: Path where to save the token.
    """
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w") as f:
        f.write(creds.to_json())
    logger.debug("Token saved to %s", token_path)
