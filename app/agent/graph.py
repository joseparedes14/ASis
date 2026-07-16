"""
LangGraph state graph definition for the ASis agent.

Assembles nodes, edges, and conditional routing into a compiled
StateGraph that orchestrates the agent's execution flow.
"""

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    after_confirmation,
    agent_reasoning,
    check_confirmation,
    generate_response,
    process_input,
    route_after_input,
    should_confirm_or_execute,
    should_execute_tools,
    tool_execution,
)
from app.agent.state import AgentState
from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.models.llm import create_llm
from app.tools.registry import create_tool_registry

logger = get_logger(__name__)


def _user_confirmation_node(state: AgentState) -> dict[str, Any]:
    """Placeholder node for user confirmation.

    In CLI mode, confirmation is handled in the main loop before
    resuming the graph. This node is a pass-through that signals
    the graph should interrupt for user input.

    Args:
        state: Current agent state.

    Returns:
        State indicating confirmation is pending.
    """
    logger.debug("Node: user_confirmation (interrupt point)")
    # The CLI loop handles actual user prompting via interrupt_before
    return {"requires_confirmation": True}


def build_graph(settings: Settings) -> StateGraph:
    """Build and compile the agent StateGraph.

    Creates the full execution graph with:
    - Input processing
    - LLM reasoning with tool binding
    - Confirmation checking
    - Tool execution
    - Response generation

    The graph supports conditional routing and interrupts for
    user confirmation of sensitive operations.

    Args:
        settings: Application settings for LLM and tool configuration.

    Returns:
        Compiled StateGraph ready for invocation.

    Architecture:
        ┌──────────────┐
        │ process_input │
        └──────┬───────┘
               │
        ┌──────▼────────┐
        │agent_reasoning │◄──────────────┐
        └──────┬────────┘               │
               │                        │
        ┌──────▼──────────────┐         │
        │ should_execute_tools │         │
        └──┬──────────┬──────┘         │
           │          │                 │
    (no tools)   (has tools)            │
           │          │                 │
           │   ┌──────▼──────────────┐  │
           │   │ check_confirmation  │  │
           │   └──┬──────────┬──────┘  │
           │      │          │          │
           │ (no confirm) (confirm)     │
           │      │          │          │
           │      │  ┌───────▼────────┐ │
           │      │  │user_confirmation│ │
           │      │  └──┬────────┬───┘ │
           │      │   (yes)    (no)     │
           │      │     │        │      │
           │   ┌──▼─────▼──┐    │      │
           │   │tool_execution│──┘      │
           │   └────────────┘───────────┘
           │
        ┌──▼──────────────┐
        │generate_response │
        └──────┬──────────┘
               │
            ┌──▼──┐
            │ END │
            └─────┘
    """
    logger.info("Building agent graph...")

    # ── Initialize components ───────────────────────────────────────
    llm = create_llm(settings)
    registry = create_tool_registry()
    tools = registry.get_all()
    tools_by_name = {t.name: t for t in tools}

    # Bind tools to LLM for function calling
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    logger.info("LLM configured with %d tools", len(tools))

    # ── Create partial node functions with dependencies ─────────────
    reasoning_node = partial(agent_reasoning, llm_with_tools=llm_with_tools)
    confirmation_check_node = partial(
        check_confirmation, tools_by_name=tools_by_name, settings=settings
    )
    execution_node = partial(tool_execution, tools_by_name=tools_by_name)

    # ── Build graph ─────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("process_input", process_input)
    graph.add_node("agent_reasoning", reasoning_node)
    graph.add_node("check_confirmation", confirmation_check_node)
    graph.add_node("user_confirmation", _user_confirmation_node)
    graph.add_node("tool_execution", execution_node)
    graph.add_node("generate_response", generate_response)

    # Set entry point
    graph.set_entry_point("process_input")

    # Add edges
    graph.add_conditional_edges(
        "process_input",
        route_after_input,
        {
            "agent_reasoning": "agent_reasoning",
            "tool_execution": "tool_execution",
            "generate_response": "generate_response",
        },
    )

    # Conditional: after reasoning, check if tools are needed
    graph.add_conditional_edges(
        "agent_reasoning",
        should_execute_tools,
        {
            "check_confirmation": "check_confirmation",
            "generate_response": "generate_response",
        },
    )

    # Conditional: after confirmation check, ask user or execute
    graph.add_conditional_edges(
        "check_confirmation",
        should_confirm_or_execute,
        {
            "user_confirmation": "user_confirmation",
            "tool_execution": "tool_execution",
        },
    )

    # Conditional: after user confirmation, execute or abort
    graph.add_conditional_edges(
        "user_confirmation",
        after_confirmation,
        {
            "tool_execution": "tool_execution",
            "generate_response": "generate_response",
        },
    )

    # After tool execution, loop back to reasoning
    graph.add_edge("tool_execution", "agent_reasoning")

    # Response generation is terminal
    graph.add_edge("generate_response", END)

    logger.info("Agent graph built successfully")

    # Compile with interrupt support for confirmation
    compiled = graph.compile(interrupt_before=["user_confirmation"])
    return compiled
