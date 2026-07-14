"""
Graph node functions for the ASis agent.

Each function is a node in the LangGraph StateGraph. Nodes receive
the current AgentState and return a partial state update dictionary.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.state import MAX_ITERATIONS, AgentState
from app.config.logging_config import get_logger
from app.config.prompts import get_error_prompt, get_system_prompt
from app.tools.base import requires_user_confirmation

logger = get_logger(__name__)


def process_input(state: AgentState) -> dict[str, Any]:
    """Process and validate user input.

    First node in the graph. Ensures the conversation has a system
    message and resets per-turn state fields.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with initialized fields.
    """
    logger.debug("Node: process_input")

    messages = list(state["messages"])

    # Ensure system prompt is present as the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=get_system_prompt())
        messages = [system_msg] + messages
        logger.debug("Added system prompt to conversation")

    return {
        "messages": messages,
        "pending_tool_calls": [],
        "tool_results": [],
        "requires_confirmation": False,
        "user_confirmed": None,
        "error": None,
        "iteration_count": 0,
    }


def agent_reasoning(state: AgentState, llm_with_tools: Any) -> dict[str, Any]:
    """LLM reasoning node — decides what to do next.

    Invokes the LLM with the current conversation and available tools.
    The LLM either responds directly or requests tool calls.

    Args:
        state: Current agent state.
        llm_with_tools: LLM instance with tools bound.

    Returns:
        Partial state update with LLM response and any tool calls.
    """
    iteration = state.get("iteration_count", 0) + 1
    logger.debug("Node: agent_reasoning (iteration %d/%d)", iteration, MAX_ITERATIONS)

    # Safety: prevent infinite reasoning loops
    if iteration > MAX_ITERATIONS:
        logger.warning("Max iterations reached (%d), forcing response", MAX_ITERATIONS)
        return {
            "messages": [
                AIMessage(
                    content="He alcanzado el límite de iteraciones. "
                    "Aquí tienes lo que he encontrado hasta ahora."
                )
            ],
            "pending_tool_calls": [],
            "iteration_count": iteration,
        }

    try:
        response = llm_with_tools.invoke(state["messages"])
        logger.debug(
            "LLM response — has_tool_calls=%s, content_length=%d",
            bool(response.tool_calls),
            len(response.content) if response.content else 0,
        )

        pending_calls = []
        if response.tool_calls:
            pending_calls = [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                }
                for tc in response.tool_calls
            ]

        return {
            "messages": [response],
            "pending_tool_calls": pending_calls,
            "iteration_count": iteration,
        }

    except Exception as e:
        logger.error("LLM invocation failed: %s", e, exc_info=True)
        error_msg = get_error_prompt(str(e))
        return {
            "messages": [AIMessage(content=error_msg)],
            "pending_tool_calls": [],
            "error": str(e),
            "iteration_count": iteration,
        }


def check_confirmation(
    state: AgentState, tools_by_name: dict[str, Any], settings: Any
) -> dict[str, Any]:
    """Check if any pending tool calls require user confirmation.

    Inspects tool metadata and application settings to determine
    if the user needs to approve the pending actions.

    Args:
        state: Current agent state.
        tools_by_name: Dictionary mapping tool names to tool objects.
        settings: Application settings.

    Returns:
        Partial state update with confirmation requirement flag.
    """
    logger.debug("Node: check_confirmation")

    if not settings.require_confirmation:
        return {"requires_confirmation": False}

    for call in state.get("pending_tool_calls", []):
        tool_name = call["name"]
        tool_obj = tools_by_name.get(tool_name)

        if tool_obj and tool_obj.metadata:
            if requires_user_confirmation(tool_obj.metadata):
                logger.info(
                    "Tool '%s' requires user confirmation", tool_name
                )
                return {"requires_confirmation": True}

        # Also check against the settings-level sensitive tools list
        if tool_name in settings.sensitive_tools:
            logger.info(
                "Tool '%s' is in sensitive tools list — confirmation required",
                tool_name,
            )
            return {"requires_confirmation": True}

    return {"requires_confirmation": False}


def tool_execution(
    state: AgentState, tools_by_name: dict[str, Any]
) -> dict[str, Any]:
    """Execute pending tool calls and collect results.

    Runs each tool with the provided arguments and wraps results
    as ToolMessage objects for the conversation history.

    Args:
        state: Current agent state.
        tools_by_name: Dictionary mapping tool names to tool objects.

    Returns:
        Partial state update with tool results and messages.
    """
    logger.debug("Node: tool_execution")

    pending = state.get("pending_tool_calls", [])
    if not pending:
        logger.warning("tool_execution called with no pending tool calls")
        return {"tool_results": [], "pending_tool_calls": []}

    results: list[dict[str, Any]] = []
    tool_messages: list[ToolMessage] = []

    for call in pending:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_id = call["id"]

        logger.info("Executing tool: %s(%s)", tool_name, tool_args)

        tool_obj = tools_by_name.get(tool_name)
        if not tool_obj:
            error_result = f"Error: Tool '{tool_name}' not found in registry."
            logger.error(error_result)
            results.append({"tool": tool_name, "result": error_result, "success": False})
            tool_messages.append(
                ToolMessage(content=error_result, tool_call_id=tool_id)
            )
            continue

        try:
            result = tool_obj.invoke(tool_args)
            logger.info("Tool '%s' completed successfully", tool_name)
            results.append({"tool": tool_name, "result": result, "success": True})
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )
        except Exception as e:
            error_result = f"Error executing '{tool_name}': {e}"
            logger.error(error_result, exc_info=True)
            results.append({"tool": tool_name, "result": error_result, "success": False})
            tool_messages.append(
                ToolMessage(content=error_result, tool_call_id=tool_id)
            )

    return {
        "messages": tool_messages,
        "tool_results": results,
        "pending_tool_calls": [],
    }


def generate_response(state: AgentState) -> dict[str, Any]:
    """Generate the final response node.

    This is a pass-through node — the actual response was already
    generated by agent_reasoning. This node exists as a clear
    termination point in the graph and for future post-processing.

    Args:
        state: Current agent state.

    Returns:
        Empty partial state update.
    """
    logger.debug("Node: generate_response")
    return {}


# ── Routing Functions ───────────────────────────────────────────────────


def should_execute_tools(state: AgentState) -> str:
    """Determine the next node after agent reasoning.

    Routes to tool execution if the LLM requested tool calls,
    otherwise routes to response generation.

    Args:
        state: Current agent state.

    Returns:
        Next node name: 'check_confirmation' or 'generate_response'.
    """
    if state.get("pending_tool_calls"):
        logger.debug("Routing → check_confirmation (tools requested)")
        return "check_confirmation"
    logger.debug("Routing → generate_response (no tools)")
    return "generate_response"


def should_confirm_or_execute(state: AgentState) -> str:
    """Determine whether to ask for confirmation or execute directly.

    Args:
        state: Current agent state.

    Returns:
        Next node name: 'user_confirmation' or 'tool_execution'.
    """
    if state.get("requires_confirmation"):
        logger.debug("Routing → user_confirmation (confirmation required)")
        return "user_confirmation"
    logger.debug("Routing → tool_execution (no confirmation needed)")
    return "tool_execution"


def after_confirmation(state: AgentState) -> str:
    """Determine next node after user confirmation response.

    Args:
        state: Current agent state.

    Returns:
        Next node name: 'tool_execution' or 'generate_response'.
    """
    if state.get("user_confirmed"):
        logger.debug("Routing → tool_execution (user confirmed)")
        return "tool_execution"
    logger.debug("Routing → generate_response (user rejected)")
    return "generate_response"
