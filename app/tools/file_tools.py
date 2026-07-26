"""
File management tools for the AI agent.

LangChain-compatible tools for local file operations such as
saving, listing, and organizing files.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.services.storage_service import StorageService
from app.tools.base import ToolRiskLevel, tool_metadata

logger = get_logger(__name__)

_settings = Settings()
_storage = StorageService(_settings)


# ── Argument Schemas ────────────────────────────────────────────────────


class SaveFileInput(BaseModel):
    """Input schema for the save_file tool."""

    filename: str = Field(
        description="Name for the saved file.",
    )
    content: str = Field(
        description="Text content to save to the file.",
    )
    subdirectory: Optional[str] = Field(
        default=None,
        description="Optional subdirectory within the data folder.",
    )


class ListFilesInput(BaseModel):
    """Input schema for the list_files tool."""

    directory: Optional[str] = Field(
        default=None,
        description="Directory to list. Defaults to the downloads folder.",
    )
    pattern: str = Field(
        default="*",
        description="Glob pattern to filter files (e.g., '*.pdf').",
    )


class OrganizeDocumentsInput(BaseModel):
    """Input schema for the organize_documents tool."""

    source_directory: Optional[str] = Field(
        default=None,
        description="Source directory to organize. Defaults to downloads.",
    )
    organize_by: str = Field(
        default="extension",
        description="Organization strategy: 'extension', 'date', or 'name'.",
    )


# ── Tool Implementations ───────────────────────────────────────────────


@tool(args_schema=SaveFileInput)
def save_file(
    filename: str,
    content: str,
    subdirectory: Optional[str] = None,
) -> str:
    """Save text content to a file in the local data directory.

    Use this tool to save text, notes, summaries, or processed data
    to a local file. Files are saved in the configured data directory.
    This action requires user confirmation.
    """
    logger.info(
        "save_file called: filename=%s, sub=%s, len=%d",
        filename, subdirectory, len(content),
    )
    try:
        file_path = _storage.save_file(
            content=content.encode("utf-8"),
            filename=filename,
            subdirectory=subdirectory,
        )
        return f"Archivo guardado: {file_path}"
    except Exception as e:
        logger.error("Error saving file: %s", e)
        return f"Error al guardar archivo: {e}"


@tool(args_schema=ListFilesInput)
def list_files(
    directory: Optional[str] = None,
    pattern: str = "*",
) -> str:
    """List files in a local directory matching a pattern.

    Use this tool to see what files are available in the data directory
    or any subdirectory. Supports glob patterns like '*.pdf'.
    """
    logger.info("list_files called: directory=%s, pattern=%s", directory, pattern)
    try:
        target = Path(directory) if directory else _storage.downloads_dir
        if not target.exists():
            return f"El directorio no existe: {target}"
        files = sorted(target.glob(pattern))
        if not files:
            return f"No se encontraron archivos en '{target}' con patron '{pattern}'"
        lines = [f"Archivos en {target} ({len(files)} resultados):"]
        for f in files[:50]:
            size = f.stat().st_size if f.is_file() else 0
            kind = "dir" if f.is_dir() else f"{size / 1024:.1f}KB"
            lines.append(f"  {f.name} [{kind}]")
        if len(files) > 50:
            lines.append(f"  ... y {len(files) - 50} mas")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Error listing files: %s", e)
        return f"Error al listar archivos: {e}"


@tool(args_schema=OrganizeDocumentsInput)
def organize_documents(
    source_directory: Optional[str] = None,
    organize_by: str = "extension",
) -> str:
    """Organize files in a directory by extension, date, or name.

    Use this tool to automatically sort and organize downloaded files
    into a structured directory layout. This action requires user confirmation.
    """
    logger.info(
        "organize_documents called: source=%s, by=%s",
        source_directory, organize_by,
    )
    try:
        source = Path(source_directory) if source_directory else _storage.downloads_dir
        if not source.exists():
            return f"El directorio no existe: {source}"
        files = [f for f in source.iterdir() if f.is_file()]
        if not files:
            return f"No hay archivos en '{source}' para organizar"

        moved = 0
        for f in files:
            if organize_by == "extension":
                ext = f.suffix.lstrip(".").lower() or "sin_extension"
                dest_dir = source / ext
            elif organize_by == "date":
                from datetime import datetime
                ts = datetime.fromtimestamp(f.stat().st_mtime)
                dest_dir = source / ts.strftime("%Y-%m")
            elif organize_by == "name":
                letter = f.stem[0].upper() if f.stem else "otros"
                dest_dir = source / letter
            else:
                return (
                    f"Estrategia no reconocida: '{organize_by}'. "
                    "Usa 'extension', 'date' o 'name'."
                )

            dest_dir.mkdir(exist_ok=True)
            target = dest_dir / f.name
            if target.exists():
                stem = f.stem
                suffix = f.suffix
                counter = 1
                while target.exists():
                    target = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            f.rename(target)
            moved += 1

        return (
            f"Organizados {moved} archivos en '{source}' "
            f"por {organize_by}"
        )
    except Exception as e:
        logger.error("Error organizing documents: %s", e)
        return f"Error al organizar documentos: {e}"


# ── Tool List ───────────────────────────────────────────────────────────

# Set metadata
save_file.metadata = tool_metadata(
    risk_level=ToolRiskLevel.MEDIUM,
    requires_confirmation=True,
    category="file",
)
list_files.metadata = tool_metadata(risk_level=ToolRiskLevel.LOW, category="file")
organize_documents.metadata = tool_metadata(
    risk_level=ToolRiskLevel.MEDIUM,
    requires_confirmation=True,
    category="file",
)

FILE_TOOLS = [save_file, list_files, organize_documents]
