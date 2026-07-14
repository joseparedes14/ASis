"""Tests for the tool system — registry, metadata, and tool definitions."""

from app.tools.base import ToolRiskLevel, tool_metadata, requires_user_confirmation
from app.tools.registry import ToolRegistry


class TestToolMetadata:
    """Tests for tool metadata helpers."""

    def test_tool_metadata_defaults(self):
        """Default metadata should be low risk, no confirmation."""
        meta = tool_metadata()
        assert meta["risk_level"] == "low"
        assert meta["requires_confirmation"] is False
        assert meta["category"] == "general"

    def test_tool_metadata_high_risk(self):
        """High-risk tools should be marked accordingly."""
        meta = tool_metadata(
            risk_level=ToolRiskLevel.HIGH,
            requires_confirmation=True,
            category="email",
        )
        assert meta["risk_level"] == "high"
        assert meta["requires_confirmation"] is True
        assert meta["category"] == "email"

    def test_requires_user_confirmation_true(self):
        """Should detect confirmation requirement from metadata."""
        meta = {"requires_confirmation": True}
        assert requires_user_confirmation(meta) is True

    def test_requires_user_confirmation_false(self):
        """Should return False when no confirmation is needed."""
        meta = {"requires_confirmation": False}
        assert requires_user_confirmation(meta) is False

    def test_requires_user_confirmation_missing(self):
        """Should default to False when key is missing."""
        assert requires_user_confirmation({}) is False


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_empty_registry(self):
        """New registry should start empty."""
        registry = ToolRegistry()
        assert registry.count == 0
        assert registry.get_all() == []

    def test_discover_tools_registers_all(self):
        """Discovery should find tools from all modules."""
        registry = ToolRegistry()
        registry.discover_tools()

        # Should have tools from both email and file modules
        assert registry.count > 0
        names = registry.get_tool_names()
        assert "search_emails" in names
        assert "save_file" in names
        assert "list_files" in names

    def test_get_by_category(self):
        """Should filter tools by category."""
        registry = ToolRegistry()
        registry.discover_tools()

        email_tools = registry.get_by_category("email")
        file_tools = registry.get_by_category("file")

        assert len(email_tools) > 0
        assert len(file_tools) > 0
        assert all(
            t.metadata.get("category") == "email" for t in email_tools
        )

    def test_get_nonexistent_tool(self):
        """Getting a non-existent tool should return None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None
