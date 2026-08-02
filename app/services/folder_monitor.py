"""
Background folder monitor service.

Uses watchdog to detect new files in monitored folders, extracts
their content, classifies them using the LLM, and moves them to
the appropriate ASIORGA destination folder.
"""

import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Número de correcciones tras el cual se reconstruyen centroides desde disco
_REBUILD_AFTER_CORRECTIONS = 20

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
    confidence: float = 0.0
    method: str = "unknown"


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
        self._correction_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._corrections_since_rebuild = 0

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
        """Lazy-load DocumentClassifier and pre-embed folder descriptions."""
        if self._classifier is None:
            from app.services.document_classifier import DocumentClassifier
            self._classifier = DocumentClassifier(llm)
            # Pre-embed destination folder descriptions for fast classification
            fm = self._get_folder_manager()
            folders = fm.list_destinations()
            if folders:
                self._classifier.load_folders(folders)
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

    def stop(self) -> None:
        """Stop all folder monitoring."""
        self._stop_event.set()
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
        self._ensure_correction_loop()
        logger.info("Added folder to monitoring: %s", resolved)

        return (
            f"Carpeta '{resolved.name}' anadida al monitoreo. "
            "Se detectaran archivos nuevos automaticamente."
        )

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

    def _correction_check_loop(self) -> None:
        """Periodically scan for manual corrections (every 60s)."""
        while not self._stop_event.is_set():
            self.check_corrections()
            self._stop_event.wait(timeout=60)

    def _ensure_correction_loop(self) -> None:
        """Ensure the periodic correction-scan thread is running.

        The correction thread must always be alive so that manual moves
        between ASIORGA folders are detected even when the monitor was
        started before any folder was configured.
        """
        if self._correction_thread is not None and self._correction_thread.is_alive():
            return
        self._stop_event.clear()
        self._correction_thread = threading.Thread(
            target=self._correction_check_loop, daemon=True
        )
        self._correction_thread.start()

    def check_corrections(self) -> int:
        """Detect manual corrections by comparing classification log
        with actual file locations in ASIORGA folders.

        When a correction is detected, also updates the destination
        folder's seed_texts and updates centroids incrementally.
        Also detects deleted files and cleans them up.

        Returns:
            Number of new corrections detected.
        """
        corrections = 0
        deletions = 0
        try:
            from app.services.document_classifier import ClassificationLog
            log = ClassificationLog()

            fm = self._get_folder_manager()
            folders = fm.list_destinations()

            active_basenames: set[str] = set()

            for folder in folders:
                folder_name = folder["name"]
                folder_path = fm._asiorga_root / folder_name
                if not folder_path.exists():
                    continue

                for file_path in folder_path.iterdir():
                    if not file_path.is_file():
                        continue

                    file_path_str = str(file_path)
                    active_basenames.add(file_path.name)

                    result = log.record_correction(file_path_str, folder_name)
                    if result is not None:
                        corrections += 1
                        predicted_folder, summary = result
                        self._notifications.put(FileNotification(
                            filename=file_path.name,
                            source_folder=predicted_folder,
                            destination_folder=folder_name,
                            timestamp=datetime.now(),
                            success=True,
                            message=(
                                f"Correccion manual detectada: movido de "
                                f"'{predicted_folder}' a '{folder_name}'"
                            ),
                            method="correccion",
                        ))
                        self._update_seed_texts_from_file(
                            file_path, folder_name, summary=summary,
                        )
                        if predicted_folder != folder_name:
                            self._remove_seed_text(
                                predicted_folder, summary, source_name="correccion",
                            )

                        classifier = self._get_classifier()
                        extractor = self._get_extractor()
                        content = extractor.extract(file_path) or ""
                        emb = None
                        if content.strip():
                            emb = classifier.get_document_embedding(content, summary=summary)
                            if emb is not None:
                                classifier.add_document_embedding(
                                    folder_name, emb, names=[file_path.name]
                                )
                        if predicted_folder != folder_name:
                            removed_by_name = (
                                emb is not None
                                and classifier.remove_document_by_name(
                                    file_path.name, predicted_folder
                                )
                            )
                            if not removed_by_name and emb is not None:
                                classifier.remove_document_embedding(
                                    predicted_folder, emb
                                )

            # --- Deletion detection ---
            orphaned = log.get_orphaned_entries(active_basenames)
            for file_path_str, predicted_folder, summary, filename in orphaned:
                deletions += 1
                logger.info("[MONITOR] Archivo eliminado detectado: %s", filename)
                self._notifications.put(FileNotification(
                    filename=filename,
                    source_folder=predicted_folder,
                    destination_folder="ELIMINADO",
                    timestamp=datetime.now(),
                    success=True,
                    message=f"Archivo eliminado: {filename}",
                    method="deleted",
                ))

                # Remove from RAG index
                try:
                    from app.services.rag_service import get_rag_indexer
                    get_rag_indexer().remove_file(file_path_str)
                except Exception as e:
                    logger.warning("RAG removal failed for %s: %s", filename, e)

                if summary:
                    self._remove_seed_text(predicted_folder, summary, source_name="eliminacion")
                    classifier = self._get_classifier()
                    if not classifier.remove_document_by_name(filename, predicted_folder):
                        emb = classifier.get_document_embedding(summary, summary=summary)
                        if emb is not None:
                            classifier.remove_document_embedding(predicted_folder, emb)

            total = corrections + deletions
            if total:
                logger.info(
                    "[MONITOR] %d correcciones, %d eliminaciones — centroides actualizados",
                    corrections, deletions,
                )
                self._corrections_since_rebuild += total
                if self._corrections_since_rebuild >= _REBUILD_AFTER_CORRECTIONS:
                    self._corrections_since_rebuild = 0
                    thread = threading.Thread(
                        target=self._rebuild_centroids_worker, daemon=True,
                    )
                    thread.start()
        except Exception as e:
            logger.error("[MONITOR] Error checking corrections: %s", e, exc_info=True)
        return corrections

    def _extract_snippet(self, file_path: Path) -> str:
        """Extract a short text snippet from a file for use as seed text."""
        extractor = self._get_extractor()
        content = extractor.extract(file_path) or ""
        if not content.strip():
            return ""
        clean = re.sub(r"\[Página\s+\d+\]", "", content)
        clean = clean.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.strip() for ln in clean.split("\n")]
        for line in lines:
            line = line.strip()
            if len(line) < 20:
                continue
            if re.search(
                r"\b(import |def |class |from |return |http|www\.|github\.com|git@)",
                line, re.I,
            ):
                continue
            if re.search(r"^(Participantes|Repositorio|Nombre|Alumno)", line, re.I):
                continue
            return re.sub(r"\s+", " ", line).strip()[:150]
        clean_flat = re.sub(r"\s+", " ", clean).strip()
        return clean_flat[:150]

    def _update_seed_texts_from_file(
        self, file_path: Path, folder_name: str, summary: str = "",
    ) -> None:
        """Add a text to a folder's seed_texts (summary if available, else raw snippet)."""
        try:
            text = summary or self._extract_snippet(file_path)
            if not text:
                return
            fm = self._get_folder_manager()
            folders = fm.list_destinations()
            for f in folders:
                if f["name"] == folder_name:
                    existing = f.get("seed_texts") or []
                    if text not in existing:
                        fm.update_seed_texts(folder_name, existing + [text])
                        logger.info(
                            "[MONITOR] Seed text anadido a '%s' desde %s",
                            folder_name, file_path.name,
                        )
                    break
        except Exception as e:
            logger.warning("[MONITOR] Error actualizando seed_texts: %s", e)

    def _remove_seed_text(self, folder_name: str, text: str, source_name: str = "") -> None:
        """Remove a specific text from a folder's seed_texts."""
        try:
            if not text:
                return
            fm = self._get_folder_manager()
            folders = fm.list_destinations()
            for f in folders:
                if f["name"] == folder_name:
                    existing = f.get("seed_texts") or []
                    if text in existing:
                        new_texts = [t for t in existing if t != text]
                        fm.update_seed_texts(folder_name, new_texts)
                        logger.info(
                            "[MONITOR] Seed text removido de '%s'%s",
                            folder_name,
                            f" desde {source_name}" if source_name else "",
                        )
                    break
        except Exception as e:
            logger.warning("[MONITOR] Error eliminando seed_texts: %s", e)

    def _extract_file_text(self, file_path: Path) -> str:
        """Extract text from a file for centroid rebuild."""
        try:
            extractor = self._get_extractor()
            return extractor.extract(file_path) or ""
        except Exception:
            return ""

    def _rebuild_centroids_worker(self) -> None:
        """Background worker: recalculates centroids from seed_texts + actual files."""
        try:
            classifier = self._get_classifier()
            fm = self._get_folder_manager()
            folders = fm.list_destinations()
            if folders:
                classifier.rebuild_centroids(folders, extract_fn=self._extract_file_text)
        except Exception as e:
            logger.error("[MONITOR] Error en rebuild de centroides: %s", e, exc_info=True)

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

        # Start periodic correction scan
        self._ensure_correction_loop()

        logger.info("Folder monitor started — %d folders", len(folders))

    def _process_file(self, file_path: Path) -> None:
        """Process a newly detected file.

        Extracts content, classifies, and moves to destination folder.
        Runs in a background thread from the debounce timer.
        """
        logger.info("[MONITOR] Procesando archivo detectado: %s", file_path)

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
                    fm = self._get_folder_manager()
                    result, dest_path = fm.move_file(file_path, "Fotos")
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
            logger.info(
                "[MONITOR] Contenido extraído de %s: %d caracteres",
                file_path.name,
                len(content) if content else 0,
            )

            if content is None:
                content = ""

            # Get folder descriptions for classification
            fm = self._get_folder_manager()
            folder_descriptions = fm.get_destination_descriptions()

            # Classify using embeddings
            classifier = self._get_classifier()
            classify_result = classifier.classify(
                content=content,
                filename=file_path.name,
                file_type=suffix,
                folder_descriptions=folder_descriptions,
                file_size=f"{file_path.stat().st_size / 1024:.1f} KB",
                source_path=str(file_path),
            )

            destination = classify_result.folder
            suggested_name = classify_result.suggested_name
            confidence = classify_result.confidence
            method = classify_result.method

            if destination is None:
                destination = "Documentos"
                suggested_name = file_path.name

            logger.info(
                "[MONITOR] Clasificación final: %s → %s | nombre=%s | "
                "confianza=%.2f | metodo=%s",
                file_path.name, destination, suggested_name, confidence, method,
            )

            # Move file to destination
            result, dest_path = fm.move_file(file_path, destination, suggested_name=suggested_name)

            # Save summary as seed text for future classification
            if classify_result.summary and destination != "Documentos":
                self._update_seed_texts_from_file(
                    file_path, destination, summary=classify_result.summary,
                )

            # Auto-index into RAG
            try:
                from app.services.rag_service import get_rag_indexer
                get_rag_indexer().index_new(dest_path)
            except Exception as e:
                logger.warning("RAG indexing failed for %s: %s", dest_path.name, e)

            self._notifications.put(FileNotification(
                filename=suggested_name or file_path.name,
                source_folder=str(file_path.parent),
                destination_folder=destination,
                timestamp=datetime.now(),
                success=True,
                message=result,
                confidence=confidence,
                method=method,
            ))

            logger.info("File processed: %s → %s (%.2f)", file_path.name, destination, confidence)

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
