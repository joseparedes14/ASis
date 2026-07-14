"""
Tool registry with auto-discovery.

Centralized registry that collects all tools from tool modules
and provides them to the agent. Adding new tools only requires
creating functions in the tools/ directory — no graph changes needed.
"""

from langchain_core.tools import BaseTool

from app.config.logging_config import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Central registry for all agent tools.

    Provides auto-discovery of tools from registered modules and
    a unified interface for the agent to access them.

    Usage:
        registry = ToolRegistry()
        registry.discover_tools()
        tools = registry.get_all()
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a single tool.

        Args:
            tool: LangChain-compatible tool to register.
        """
        if tool.name in self._tools:
            logger.warning("Tool '%s' is being overwritten in registry", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_many(self, tools: list[BaseTool]) -> None:
        """Register multiple tools at once.

        Args:
            tools: List of tools to register.
        """
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """Return all registered tools.

        Returns:
            List of all registered tools.
        """
        return list(self._tools.values())

    def get_by_category(self, category: str) -> list[BaseTool]:
        """Return tools filtered by category.

        Args:
            category: Category name (e.g., 'email', 'file').

        Returns:
            List of tools in the specified category.
        """
        return [
            t
            for t in self._tools.values()
            if t.metadata and t.metadata.get("category") == category
        ]

    def get_tool_names(self) -> list[str]:
        """Return names of all registered tools.

        Returns:
            Sorted list of tool name strings.
        """
        return sorted(self._tools.keys())

    def discover_tools(self) -> None:
        """Auto-discover and register tools from all tool modules.

        Imports tool modules and registers their exported tool lists.
        To add new tools, create a module in app/tools/ that exports
        a list named *_TOOLS (e.g., EMAIL_TOOLS, FILE_TOOLS).
        """
        logger.info("Discovering tools...")

        # Import tool modules — add new modules here
        from app.tools.email_tools import EMAIL_TOOLS
        from app.tools.file_tools import FILE_TOOLS

        self.register_many(EMAIL_TOOLS)
        self.register_many(FILE_TOOLS)

        logger.info(
            "Tool discovery complete — %d tools registered: %s",
            len(self._tools),
            ", ".join(self.get_tool_names()),
        )

    @property
    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.get_tool_names()})"


def create_tool_registry() -> ToolRegistry:
    """Factory function to create and populate a tool registry.

    Returns:
        ToolRegistry with all discovered tools registered.
    """
    registry = ToolRegistry()
    registry.discover_tools()
    return registry
