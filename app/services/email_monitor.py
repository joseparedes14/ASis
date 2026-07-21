"""
Background email monitor service.

Periodically checks IMAP for new emails from monitored senders,
downloads attachments, and stores notifications for the main thread.
"""

import asyncio
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.services.storage_service import StorageService

logger = get_logger(__name__)


@dataclass
class DownloadNotification:
    """Notification of downloaded email attachments."""

    sender: str
    subject: str
    filenames: list[str]
    timestamp: datetime
    message_id: str = ""


class EmailMonitor:
    """Background email monitor that polls IMAP for new messages.

    Runs in a daemon thread and checks for new emails from monitored
    senders at a configurable interval. Downloads attachments and stores
    notifications that the main thread can pick up.

    Usage:
        settings = get_settings()
        monitor = EmailMonitor(settings)
        monitor.start()

        # In main loop:
        notifications = monitor.get_notifications()

        # On shutdown:
        monitor.stop()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._notifications: queue.Queue[DownloadNotification] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._downloaded_file = settings.data_dir / ".downloaded_emails.txt"

    def start(self) -> None:
        """Start the background monitor thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Email monitor is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="email-monitor",
            daemon=True,
        )
        self._thread.start()

        monitored = self._settings.get_monitored_senders()
        interval = self._settings.email_monitor_interval
        logger.info(
            "Email monitor started — interval=%dm, senders=%s",
            interval,
            monitored,
        )

    def stop(self) -> None:
        """Stop the background monitor thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Email monitor stopped")

    def get_notifications(self) -> list[DownloadNotification]:
        """Extract all pending notifications (non-blocking).

        Returns:
            List of notifications since last check.
        """
        notifs = []
        while not self._notifications.empty():
            try:
                notifs.append(self._notifications.get_nowait())
            except queue.Empty:
                break
        return notifs

    def _poll_loop(self) -> None:
        """Main polling loop running in background thread."""
        interval_seconds = self._settings.email_monitor_interval * 60

        while not self._stop_event.is_set():
            try:
                self._check_and_download()
            except Exception as e:
                logger.error("Monitor poll error: %s", e, exc_info=True)

            self._stop_event.wait(interval_seconds)

    def _check_and_download(self) -> None:
        """Connect to IMAP, search for new emails, download attachments."""
        monitored = self._settings.get_monitored_senders()
        if not monitored:
            return

        from app.services.imap_service import ImapEmailService

        service = ImapEmailService(self._settings)
        storage = StorageService(self._settings)

        asyncio.run(self._do_check(service, storage, monitored))

    async def _do_check(
        self,
        service: "ImapEmailService",  # noqa: F821
        storage: StorageService,
        senders: list[str],
    ) -> None:
        """Perform the actual IMAP check and download."""
        try:
            await service.connect()
        except Exception as e:
            logger.error("Monitor: IMAP connection failed: %s", e)
            return

        try:
            for sender_email in senders:
                emails = await service.search_emails(
                    sender=sender_email,
                    limit=1,
                    unread_only=True,
                )

                for email_msg in emails:
                    if not email_msg.has_attachments:
                        logger.debug(
                            "Monitor: Email %s has no attachments, skipping",
                            email_msg.message_id,
                        )
                        continue

                    if self._already_downloaded(email_msg.message_id):
                        logger.debug(
                            "Monitor: Skipping already-downloaded email %s",
                            email_msg.message_id,
                        )
                        continue

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
                            logger.info(
                                "Monitor: Downloaded %s (%d bytes) from %s",
                                att.filename,
                                len(content),
                                sender_email,
                            )

                    if downloaded_files:
                        self._notifications.put(
                            DownloadNotification(
                                sender=sender_email,
                                subject=email_msg.subject,
                                filenames=downloaded_files,
                                timestamp=datetime.now(),
                                message_id=email_msg.message_id,
                            )
                        )
                        self._mark_downloaded(email_msg.message_id)

                    # Mark as read regardless of whether attachments were downloaded
                    await service.mark_as_read(email_msg.message_id)
                    logger.info(
                        "Monitor: Marked email %s as read", email_msg.message_id
                    )

        finally:
            await service.disconnect()

    def _already_downloaded(self, message_id: str) -> bool:
        """Check if an email has already been processed."""
        if not self._downloaded_file.exists():
            return False
        try:
            content = self._downloaded_file.read_text(encoding="utf-8")
            return message_id in content.splitlines()
        except Exception:
            return False

    def _mark_downloaded(self, message_id: str) -> None:
        """Mark an email as processed to avoid re-downloading."""
        try:
            self._downloaded_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._downloaded_file, "a", encoding="utf-8") as f:
                f.write(message_id + "\n")
        except Exception as e:
            logger.error("Failed to mark email as downloaded: %s", e)
