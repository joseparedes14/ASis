"""
Base utilities for the tool system.

Provides metadata helpers and permission checking for tools.
"""

from enum import Enum
from typing import Any


class ToolRiskLevel(str, Enum):
    """Risk classification for tools.

    Used to determine whether a tool requires user confirmation
    before execution.
    """

    LOW = "low"  # Read-only operations (search, list)
    MEDIUM = "medium"  # Write operations (save file)
    HIGH = "high"  # Destructive or sensitive operations (delete, send)


def tool_metadata(
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    requires_confirmation: bool = False,
    category: str = "general",
) -> dict[str, Any]:
    """Create standard metadata for a tool.

    Attach this to a tool's metadata dict to enable the permission
    system and UI categorization.

    Args:
        risk_level: How risky this tool operation is.
        requires_confirmation: Whether the user must confirm before execution.
        category: Logical grouping (e.g., 'email', 'file', 'general').

    Returns:
        Metadata dictionary to attach to a LangChain tool.

    Example:
        @tool(metadata=tool_metadata(risk_level=ToolRiskLevel.HIGH, requires_confirmation=True))
        def delete_email(message_id: str) -> str:
            ...
    """
    return {
        "risk_level": risk_level.value,
        "requires_confirmation": requires_confirmation,
        "category": category,
    }


def requires_user_confirmation(tool_metadata: dict[str, Any]) -> bool:
    """Check if a tool's metadata indicates it requires user confirmation.

    Args:
        tool_metadata: The tool's metadata dictionary.

    Returns:
        True if user confirmation is needed.
    """
    return tool_metadata.get("requires_confirmation", False)
