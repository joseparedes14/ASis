"""
Background folder monitor service.

Uses watchdog to detect new files in monitored folders, extracts
their content, classifies them using the LLM, and moves them to
the appropriate ASIORGA destination folder.
"""

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Supported file extensions for processing
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".jpg", ".jpeg", ".png",
    ".csv", ".xlsx", ".xls",
}


@dataclass
class FileNotification:
    """Notification when a file is processed."""
    filename: str
    source_folder: str
    destination_folder: str
    timestamp: datetime
    success: bool
    message: str


class _DebouncedHandler(FileSystemEventHandler):
    """Debounced file event handler to avoid processing duplicates."""

    def __init__(self, monitor: "FolderMonitor") -> None:
        self._monitor = monitor
        self._timers: dict[str, threading.Timer] = {}
        self._debounce_seconds = 3.0

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events with debounce."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # Debounce: wait for file to finish writing
        key = str(file_path)
        if key in self._timers:
            self._timers[key].cancel()

        self._timers[key] = threading.Timer(
            self._debounce_seconds,
            self._monitor._process_file,
            args=(file_path,),
        )
        self._timers[key].start()

    def on_moved(self, event: FileMovedEvent) -> None:
        """Handle file move/rename events (e.g. browser downloads finishing)."""
        if event.is_directory:
            return

        file_path = Path(event.dest_path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # Debounce: wait for file to finish writing
        key = str(file_path)
        if key in self._timers:
            self._timers[key].cancel()

        self._timers[key] = threading.Timer(
            self._debounce_seconds,
            self._monitor._process_file,
            args=(file_path,),
        )
        self._timers[key].start()


# Module-level singleton instance
_instance: Optional["FolderMonitor"] = None
_instance_lock = threading.Lock()


def get_folder_monitor(settings=None) -> "FolderMonitor":
    """Get or create the global FolderMonitor singleton.

    Ensures all components (CLI, widget, tools) share the same instance.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FolderMonitor(settings)
    return _instance


class FolderMonitor:
    """Background folder monitor using watchdog.

    Monitors specified folders for new files, extracts content,
    classifies documents, and moves them to ASIORGA.

    Usage:
        monitor = FolderMonitor(settings)
        monitor.start()

        # In main loop:
        notifications = monitor.get_notifications()

        # On shutdown:
        monitor.stop()
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._config_path = Path("./config/monitored_folders.json")
        self._notifications: queue.Queue[FileNotification] = queue.Queue()
        self._observer: Optional[Observer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Lazy-loaded services
        self._folder_manager = None
        self._extractor = None
        self._classifier = None
        self._image_classifier = None

    def _get_folder_manager(self):
        """Lazy-load FolderManager."""
        if self._folder_manager is None:
            from app.services.folder_manager import FolderManager
            self._folder_manager = FolderManager()
        return self._folder_manager

    def _get_extractor(self):
        """Lazy-load DocumentExtractor."""
        if self._extractor is None:
            from app.services.document_extractor import DocumentExtractor
            self._extractor = DocumentExtractor()
        return self._extractor

    def _get_classifier(self, llm=None):
        """Lazy-load DocumentClassifier."""
        if self._classifier is None:
            from app.services.document_classifier import DocumentClassifier
            self._classifier = DocumentClassifier(llm)
        elif llm is not None:
            self._classifier.set_llm(llm)
        return self._classifier

    def _get_image_classifier(self):
        """Lazy-load ImageClassifier."""
        if self._image_classifier is None:
            from app.services.image_classifier import ImageClassifier
            self._image_classifier = ImageClassifier()
        return self._image_classifier

    def set_llm(self, llm) -> None:
        """Set the LLM for document classification."""
        self._get_classifier(llm)

    def start(self) -> None:
        """Start monitoring all configured folders."""
        if self._observer and self._observer.is_alive():
            logger.warning("Folder monitor is already running")
            return

        folders = self._load_monitored_folders()
        if not folders:
            logger.info("No folders configured for monitoring")
            return

        self._observer = Observer()
        handler = _DebouncedHandler(self)

        for folder_path in folders:
            path = Path(folder_path)
            if path.is_dir():
                self._observer.schedule(handler, str(path), recursive=False)
                logger.info("Monitoring folder: %s", path)

        self._observer.start()
        logger.info("Folder monitor started — %d folders", len(folders))

    def stop(self) -> None:
        """Stop all folder monitoring."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        logger.info("Folder monitor stopped")

    def add_folder(self, path: str) -> str:
        """Add a folder to monitoring.

        Args:
            path: Folder path or known name (e.g., "Descargas").

        Returns:
            Success or error message.
        """
        fm = self._get_folder_manager()
        resolved = fm.resolve_monitoring_path(path)

        if resolved is None:
            return f"No se pudo encontrar la carpeta: {path}"

        # Add to config
        config = self._load_config()
        str_path = str(resolved)
        if str_path not in config["folders"]:
            config["folders"].append(str_path)
            self._save_config(config)

        # Ensure observer is running, then schedule the new folder
        if not (self._observer and self._observer.is_alive()):
            self._observer = Observer()

        handler = _DebouncedHandler(self)
        self._observer.schedule(handler, str_path, recursive=False)
        if not self._observer.is_alive():
            self._observer.start()
        logger.info("Added folder to monitoring: %s", resolved)

        return f"Carpeta '{resolved.name}' añadida al monitoreo. Se detectarán archivos nuevos automáticamente."

    def remove_folder(self, path: str) -> str:
        """Remove a folder from monitoring.

        Note: watchdog doesn't support unscheduling individual folders,
        so removing a folder clears the config but the full observer
        will be reset on next start().

        Args:
            path: Folder path or known name.

        Returns:
            Success or error message.
        """
        fm = self._get_folder_manager()
        resolved = fm.resolve_monitoring_path(path)

        if resolved is None:
            return f"No se pudo encontrar la carpeta: {path}"

        config = self._load_config()
        str_path = str(resolved)

        if str_path in config["folders"]:
            config["folders"].remove(str_path)
            self._save_config(config)

            # If no more folders, stop the observer
            if not config["folders"] and self._observer and self._observer.is_alive():
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None

            logger.info("Removed folder from monitoring: %s", resolved)
            return f"Carpeta '{resolved.name}' eliminada del monitoreo."

        return f"La carpeta '{resolved.name}' no estaba en el monitoreo."

    def list_folders(self) -> list[str]:
        """List all monitored folders.

        Returns:
            List of folder path strings.
        """
        return self._load_monitored_folders()

    def get_notifications(self) -> list[FileNotification]:
        """Extract all pending notifications (non-blocking)."""
        notifs = []
        while not self._notifications.empty():
            try:
                notifs.append(self._notifications.get_nowait())
            except queue.Empty:
                break
        return notifs

    def _load_config(self) -> dict:
        """Load monitoring configuration."""
        if not self._config_path.exists():
            return {"folders": []}
        try:
            return json.loads(self._config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            return {"folders": []}

    def _save_config(self, config: dict) -> None:
        """Save monitoring configuration."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_monitored_folders(self) -> list[str]:
        """Load list of monitored folder paths."""
        config = self._load_config()
        return config.get("folders", [])

    def _process_file(self, file_path: Path) -> None:
        """Process a newly detected file.

        Extracts content, classifies, and moves to destination folder.
        Runs in a background thread from the debounce timer.
        """
        logger.info("Processing new file: %s", file_path)

        try:
            # Wait a bit for file to finish writing
            time.sleep(1)

            if not file_path.exists():
                logger.warning("File disappeared before processing: %s", file_path)
                return

            # Check if it's an image that should go directly to Fotos
            suffix = file_path.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png"}:
                img_class = self._get_image_classifier()
                classification = img_class.classify(file_path)

                if classification == "photo":
                    # Move directly to Fotos
                    fm = self._get_folder_manager()
                    result = fm.move_file(file_path, "Fotos")
                    self._notifications.put(FileNotification(
                        filename=file_path.name,
                        source_folder=str(file_path.parent),
                        destination_folder="Fotos",
                        timestamp=datetime.now(),
                        success=True,
                        message=result,
                    ))
                    logger.info("Photo classified → Fotos: %s", file_path.name)
                    return

            # Extract content
            extractor = self._get_extractor()
            content = extractor.extract(file_path)

            if content is None:
                content = ""

            # Get folder descriptions for classification
            fm = self._get_folder_manager()
            folder_descriptions = fm.get_destination_descriptions()

            # Classify using LLM (with timeout)
            classifier = self._get_classifier()
            destination = classifier.classify(
                content=content,
                filename=file_path.name,
                file_type=suffix,
                folder_descriptions=folder_descriptions,
                file_size=f"{file_path.stat().st_size / 1024:.1f} KB",
            )

            if destination is None:
                destination = "Documentos"

            # Move file to destination
            result = fm.move_file(file_path, destination)

            self._notifications.put(FileNotification(
                filename=file_path.name,
                source_folder=str(file_path.parent),
                destination_folder=destination,
                timestamp=datetime.now(),
                success=True,
                message=result,
            ))

            logger.info("File processed: %s → %s", file_path.name, destination)

        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, e, exc_info=True)
            self._notifications.put(FileNotification(
                filename=file_path.name,
                source_folder=str(file_path.parent),
                destination_folder="ERROR",
                timestamp=datetime.now(),
                success=False,
                message=str(e),
            ))
