"""
Folder management service for ASIORGA.

Manages the creation, deletion, and listing of destination folders
within the ASIORGA directory structure. Persists folder configuration
to JSON.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Default root folder for organized documents
ASIORGA_ROOT = Path.home() / "Desktop" / "ASIorga"

# Windows known folder name mapping (Spanish → English → Path)
KNOWN_FOLDERS: dict[str, Path] = {
    "descargas": Path.home() / "Downloads",
    "downloads": Path.home() / "Downloads",
    "documentos": Path.home() / "Documents",
    "documents": Path.home() / "Documents",
    "escritorio": Path.home() / "Desktop",
    "desktop": Path.home() / "Desktop",
    "imágenes": Path.home() / "Pictures",
    "pictures": Path.home() / "Pictures",
    "música": Path.home() / "Music",
    "music": Path.home() / "Music",
    "vídeos": Path.home() / "Videos",
    "videos": Path.home() / "Videos",
}

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".jpg", ".jpeg", ".png",
    ".csv", ".xlsx", ".xls",
}


class FolderManager:
    """Manages ASIORGA folder structure and destination folder configuration.

    Handles:
    - Creating and deleting destination folders
    - Persisting folder metadata (name, description) to JSON
    - Resolving folder paths from user input
    - Moving files to their classified destination
    - Listing folder contents
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or Path("./config/destination_folders.json")
        self._asiorga_root = ASIORGA_ROOT
        self._ensure_root()
        self._sync_physical_folders()

    def _ensure_root(self) -> None:
        """Create ASIORGA root directory if it doesn't exist."""
        self._asiorga_root.mkdir(parents=True, exist_ok=True)

    def _sync_physical_folders(self) -> None:
        """Create physical folders for all entries in the config.

        Ensures that every folder defined in destination_folders.json
        actually exists on disk inside ASIORGA.
        """
        config = self._load_config()
        for folder in config.get("folders", []):
            folder_path = self._asiorga_root / folder["path"]
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info("Created missing ASIORGA folder: %s", folder_path)

    def _load_config(self) -> dict:
        """Load destination folders configuration from JSON."""
        if not self._config_path.exists():
            return {"folders": []}
        try:
            return json.loads(self._config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            return {"folders": []}

    def _save_config(self, config: dict) -> None:
        """Save destination folders configuration to JSON."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def resolve_monitoring_path(self, name_or_path: str) -> Optional[Path]:
        """Resolve a folder name or path to an absolute path for monitoring.

        Supports:
        - Absolute paths (e.g., C:\\Users\\josem\\Downloads)
        - Common Windows folder names (e.g., "Descargas", "Documentos")

        Args:
            name_or_path: Folder name or absolute path.

        Returns:
            Resolved Path if valid, None otherwise.
        """
        # 1. If it's an absolute path and exists
        if Path(name_or_path).is_absolute() and Path(name_or_path).is_dir():
            return Path(name_or_path)

        # 2. If it's a known folder name
        normalized = name_or_path.lower().strip()
        if normalized in KNOWN_FOLDERS:
            resolved = KNOWN_FOLDERS[normalized]
            if resolved.is_dir():
                return resolved

        # 3. Check if it exists relative to home
        home_path = Path.home() / name_or_path
        if home_path.is_dir():
            return home_path

        return None

    def create_destination(self, name: str, description: str) -> str:
        """Create a new destination folder in ASIORGA.

        Args:
            name: Name of the folder.
            description: Description of what this folder contains.

        Returns:
            Success or error message.
        """
        config = self._load_config()

        # Check if folder already exists in config
        for folder in config["folders"]:
            if folder["name"].lower() == name.lower():
                return f"Ya existe una carpeta destino llamada '{name}'."

        # Create the physical folder
        folder_path = self._asiorga_root / name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Add to config
        config["folders"].append({
            "name": name,
            "description": description,
            "path": name,
        })
        self._save_config(config)

        logger.info("Created destination folder: %s — %s", name, description)
        return f"Carpeta '{name}' creada en {folder_path}. Descripción: {description}"

    def delete_destination(self, name: str) -> str:
        """Delete a destination folder from ASIORGA.

        Only removes the config entry. The physical folder is preserved
        to avoid data loss.

        Args:
            name: Name of the folder to remove.

        Returns:
            Success or error message.
        """
        config = self._load_config()

        for i, folder in enumerate(config["folders"]):
            if folder["name"].lower() == name.lower():
                removed = config["folders"].pop(i)
                self._save_config(config)
                logger.info("Removed destination folder config: %s", name)
                return (
                    f"Carpeta '{name}' eliminada de la configuración. "
                    f"La carpeta física en {self._asiorga_root / name} se conserva."
                )

        return f"No se encontró una carpeta destino llamada '{name}'."

    def list_destinations(self) -> list[dict]:
        """List all configured destination folders.

        Returns:
            List of folder dicts with name, description, and path.
        """
        config = self._load_config()
        return config.get("folders", [])

    def get_destination_descriptions(self) -> str:
        """Get a formatted string of all destination folders and their descriptions.

        This is used by the document classifier to know where to file documents.

        Returns:
            Formatted string with folder names and descriptions.
        """
        folders = self.list_destinations()
        if not folders:
            return "No hay carpetas destino configuradas."

        lines = []
        for f in folders:
            folder_path = self._asiorga_root / f["path"]
            exists = "✓" if folder_path.exists() else "✗"
            lines.append(f"- {f['name']}: {f['description']} [{exists}]")
        return "\n".join(lines)

    def move_file(self, source: Path, destination_name: str) -> str:
        """Move a file to a destination folder within ASIORGA.

        Args:
            source: Path of the file to move.
            destination_name: Name of the destination folder.

        Returns:
            Success or error message.
        """
        if not source.exists():
            return f"El archivo {source} no existe."

        dest_folder = self._asiorga_root / destination_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_path = dest_folder / source.name

        # Handle duplicate filenames
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            shutil.move(str(source), str(dest_path))
            logger.info("Moved %s → %s", source.name, dest_path)
            return f"Archivo movido a {dest_path}"
        except Exception as e:
            logger.error("Failed to move %s: %s", source, e)
            return f"Error al mover el archivo: {e}"

    def list_folder_contents(self, folder_name: Optional[str] = None) -> str:
        """List contents of a folder within ASIORGA.

        Args:
            folder_name: Name of the subfolder, or None for root.

        Returns:
            Formatted string with folder contents.
        """
        if folder_name:
            target = self._asiorga_root / folder_name
        else:
            target = self._asiorga_root

        if not target.exists():
            return f"La carpeta {target} no existe."

        entries = sorted(target.iterdir())
        if not entries:
            return f"La carpeta {target.name} está vacía."

        lines = [f"Contenido de {target.name}:"]
        for entry in entries:
            if entry.is_dir():
                count = len(list(entry.iterdir()))
                lines.append(f"  📁 {entry.name}/ ({count} archivos)")
            else:
                size_kb = entry.stat().st_size / 1024
                lines.append(f"  📄 {entry.name} ({size_kb:.1f} KB)")

        return "\n".join(lines)
