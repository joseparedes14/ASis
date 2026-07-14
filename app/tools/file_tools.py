"""
File management tools for the AI agent.

LangChain-compatible tools for local file operations such as
saving, listing, and organizing files.
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config.logging_config import get_logger
from app.tools.base import ToolRiskLevel, tool_metadata

logger = get_logger(__name__)


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
        "save_file called — filename=%s, subdirectory=%s, content_length=%d",
        filename,
        subdirectory,
        len(content),
    )
    # TODO: Connect to StorageService when fully wired
    return (
        f"[File save not yet implemented] "
        f"Would save '{filename}' ({len(content)} chars) to {subdirectory or 'downloads'}"
    )


@tool(args_schema=ListFilesInput)
def list_files(
    directory: Optional[str] = None,
    pattern: str = "*",
) -> str:
    """List files in a local directory matching a pattern.

    Use this tool to see what files are available in the data directory
    or any subdirectory. Supports glob patterns like '*.pdf'.
    """
    logger.info(
        "list_files called — directory=%s, pattern=%s",
        directory,
        pattern,
    )
    # TODO: Connect to StorageService when fully wired
    return (
        f"[File listing not yet implemented] "
        f"Would list files in '{directory or 'downloads'}' matching '{pattern}'"
    )


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
        "organize_documents called — source=%s, organize_by=%s",
        source_directory,
        organize_by,
    )
    # TODO: Implement organization logic
    return (
        f"[Document organization not yet implemented] "
        f"Would organize '{source_directory or 'downloads'}' by {organize_by}"
    )


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
