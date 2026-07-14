"""
Gmail OAuth2 Authorization Script.

Run this script ONCE to authorize your application to send emails
via Gmail SMTP. After authorization, a refresh token will be saved
to data/oauth_token.json and used automatically for future requests.

Usage:
    python scripts/authorize_gmail.py

Requirements:
    1. A Google Cloud project with Gmail API enabled
    2. OAuth2 credentials (credentials.json) in project root
    3. EMAIL_ADDRESS configured in .env file
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config.settings import get_settings
from app.services.smtp_service import GmailSMTPService


def main() -> None:
    """Main entry point for authorization script."""
    print("=" * 60)
    print("  Gmail OAuth2 Authorization")
    print("=" * 60)
    print()

    try:
        settings = get_settings()

        # Check if email is configured
        if not settings.email_address:
            print("ERROR: EMAIL_ADDRESS not configured.")
            print()
            print("Please add your Gmail address to .env file:")
            print("  EMAIL_ADDRESS=your-email@gmail.com")
            sys.exit(1)

        # Check if credentials file exists
        credentials_file = Path(settings.email_oauth_credentials_file)
        if not credentials_file.exists():
            print(f"ERROR: OAuth2 credentials file not found: {credentials_file}")
            print()
            print("Please download credentials.json from Google Cloud Console:")
            print("  1. Go to https://console.cloud.google.com/")
            print("  2. Select your project (or create one)")
            print("  3. Go to APIs & Services > Credentials")
            print("  4. Create OAuth 2.0 Client ID (Desktop app)")
            print("  5. Download the JSON file and save as credentials.json")
            sys.exit(1)

        print(f"Email address: {settings.email_address}")
        print(f"Credentials file: {credentials_file}")
        print(f"Token file: {settings.email_oauth_token_file}")
        print()

        # Create service and perform authorization
        service = GmailSMTPService(settings)

        print("Starting authorization flow...")
        print("A browser window will open. Please:")
        print("  1. Sign in with your Google account")
        print("  2. Review the permissions")
        print("  3. Click 'Allow'")
        print()

        # This will open the browser and perform the OAuth flow
        creds = service._load_credentials()

        print()
        print("=" * 60)
        print("  Authorization completed successfully!")
        print("=" * 60)
        print()
        print(f"Token saved to: {settings.email_oauth_token_file}")
        print()
        print("You can now use the send_email tool in your agent.")
        print("The token will be refreshed automatically when needed.")
        print()

        # Test connection
        print("Testing SMTP connection...")
        import asyncio

        async def test_connection():
            await service.connect()
            await service.disconnect()
            return True

        asyncio.run(test_connection())

        print()
        print("SUCCESS: SMTP connection test passed!")
        print()

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("If you're seeing 'access_not_configured' or similar,")
        print("make sure Gmail API is enabled in your Google Cloud project.")
        sys.exit(1)


if __name__ == "__main__":
    main()
