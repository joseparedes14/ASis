"""
Agent state definition for the LangGraph state machine.

Defines the shared state that flows through all nodes in the agent graph.
Uses TypedDict with LangGraph's annotation system for proper message
accumulation.
"""

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the ASis agent graph.

    This state is passed between all nodes in the graph. Each field
    serves a specific purpose in the agent's execution flow.

    Attributes:
        messages: Conversation history with automatic accumulation.
            Uses LangGraph's add_messages reducer to append new messages.
        pending_tool_calls: List of tool calls the agent wants to execute.
        tool_results: Results from the most recent tool execution cycle.
        requires_confirmation: Flag indicating the current action needs
            user approval before proceeding.
        user_confirmed: Whether the user has confirmed a pending action.
        error: Error message from the most recent operation, if any.
        metadata: Flexible dictionary for passing additional context
            between nodes (e.g., processing flags, counters).
        iteration_count: Number of reasoning iterations in the current
            request, used to prevent infinite loops.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    requires_confirmation: bool
    user_confirmed: Optional[bool]
    error: Optional[str]
    metadata: dict[str, Any]
    iteration_count: int


def create_initial_state() -> AgentState:
    """Create a fresh initial state for a new conversation turn.

    Returns:
        AgentState with all fields initialized to sensible defaults.
    """
    return AgentState(
        messages=[],
        pending_tool_calls=[],
        tool_results=[],
        requires_confirmation=False,
        user_confirmed=None,
        error=None,
        metadata={},
        iteration_count=0,
    )


# Maximum allowed iterations to prevent infinite loops
MAX_ITERATIONS = 10
