"""
Utility helper functions for the ASis application.

General-purpose utilities shared across modules.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_json_dumps(data: Any, **kwargs: Any) -> str:
    """Serialize data to JSON string with safe defaults.

    Handles common non-serializable types like datetime and Path.

    Args:
        data: Data to serialize.
        **kwargs: Additional arguments passed to json.dumps.

    Returns:
        JSON string representation.
    """

    class SafeEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    return json.dumps(data, cls=SafeEncoder, ensure_ascii=False, **kwargs)


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length with an ellipsis suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum allowed length.
        suffix: String to append when truncated.

    Returns:
        Truncated text or original if within limit.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are unsafe for filenames.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename safe for all operating systems.
    """
    # Characters not allowed in Windows filenames
    unsafe_chars = '<>:"/\\|?*'
    sanitized = filename
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, "_")
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    return sanitized or "unnamed"


def format_timestamp(dt: datetime | None = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime object as a string.

    Args:
        dt: Datetime to format. Uses current time if None.
        fmt: strftime format string.

    Returns:
        Formatted datetime string.
    """
    dt = dt or datetime.now()
    return dt.strftime(fmt)


def ensure_path(path: str | Path) -> Path:
    """Convert a string to Path and ensure parent directories exist.

    Args:
        path: File or directory path.

    Returns:
        Path object with parent directories created.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
