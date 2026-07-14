"""
Local storage service.

Handles file operations: saving downloads, organizing attachments,
and managing the local data directory structure.
"""

from pathlib import Path
from typing import Optional

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.utils.helpers import sanitize_filename

logger = get_logger(__name__)


class StorageService:
    """Manages local file storage for downloaded and processed files.

    Provides organized directory structures and safe file operations
    for attachments, downloads, and other user data.

    Args:
        settings: Application settings with storage paths.
    """

    def __init__(self, settings: Settings) -> None:
        self._data_dir = settings.data_dir
        self._downloads_dir = settings.downloads_dir
        self._attachments_dir = settings.attachments_dir
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required storage directories."""
        for dir_path in [self._data_dir, self._downloads_dir, self._attachments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Storage directories verified")

    def save_file(
        self,
        content: bytes,
        filename: str,
        subdirectory: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> Path:
        """Save binary content to a file.

        Args:
            content: File content as bytes.
            filename: Target filename (will be sanitized).
            subdirectory: Optional subdirectory within the base directory.
            base_dir: Base directory. Defaults to downloads_dir.

        Returns:
            Path to the saved file.

        Raises:
            OSError: If the file cannot be written.
        """
        base = base_dir or self._downloads_dir
        safe_name = sanitize_filename(filename)

        if subdirectory:
            target_dir = base / sanitize_filename(subdirectory)
        else:
            target_dir = base

        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / safe_name

        # Handle name collisions
        file_path = self._resolve_collision(file_path)

        file_path.write_bytes(content)
        logger.info("Saved file: %s (%d bytes)", file_path, len(content))
        return file_path

    def save_attachment(
        self,
        content: bytes,
        filename: str,
        sender: Optional[str] = None,
    ) -> Path:
        """Save an email attachment with optional sender-based organization.

        Args:
            content: Attachment content as bytes.
            filename: Original attachment filename.
            sender: Email sender (used as subdirectory name).

        Returns:
            Path to the saved attachment.
        """
        subdirectory = sanitize_filename(sender) if sender else None
        return self.save_file(
            content=content,
            filename=filename,
            subdirectory=subdirectory,
            base_dir=self._attachments_dir,
        )

    def list_files(
        self, directory: Optional[Path] = None, pattern: str = "*"
    ) -> list[Path]:
        """List files in a directory matching a glob pattern.

        Args:
            directory: Directory to list. Defaults to downloads_dir.
            pattern: Glob pattern to filter files.

        Returns:
            List of matching file paths.
        """
        target = directory or self._downloads_dir
        if not target.exists():
            return []
        return sorted(target.glob(pattern))

    def _resolve_collision(self, path: Path) -> Path:
        """Generate a unique filename if the path already exists.

        Appends a numeric suffix (e.g., file_1.pdf, file_2.pdf).

        Args:
            path: Original file path.

        Returns:
            Unique file path that doesn't exist yet.
        """
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1

        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    @property
    def downloads_dir(self) -> Path:
        """Return the downloads directory path."""
        return self._downloads_dir

    @property
    def attachments_dir(self) -> Path:
        """Return the attachments directory path."""
        return self._attachments_dir
