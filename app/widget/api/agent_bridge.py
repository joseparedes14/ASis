"""Bridge between the widget and the existing LangGraph agent.

Initializes the agent graph from app.agent, routes messages through it,
handles tool confirmations via a callback, and returns responses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph
from app.agent.state import AgentState, create_initial_state
from app.config.settings import get_settings
from app.models.llm import check_ollama_connection
from app.services.folder_monitor import get_folder_monitor
from app.widget.api.llm_client import AgentStatus

logger = logging.getLogger("asis.widget.agent_bridge")


class AgentBridge:
    """Synchronous wrapper around the LangGraph agent for the widget.

    Args:
        confirmation_handler: Callable that receives a list of pending
            tool call dicts and returns True to confirm or False to abort.
            If ``None``, all tools are auto-confirmed.
    """

    def __init__(
        self,
        confirmation_handler: Callable[[list[dict]], bool] | None = None,
    ) -> None:
        self._settings = get_settings()
        self._graph = build_graph(self._settings)
        self._state: AgentState = create_initial_state()
        self._config = {"configurable": {"thread_id": "widget-session-1"}}
        self._confirm_handler = confirmation_handler

        # Start folder monitor (singleton — same instance used by tools)
        self._folder_monitor = get_folder_monitor(self._settings)
        from app.models.llm import create_llm
        llm = create_llm(self._settings)
        self._folder_monitor.set_llm(llm)
        self._folder_monitor.start()

        logger.info(
            "AgentBridge initialized — provider=%s, model=%s",
            self._settings.llm_provider,
            self._settings.llm_model,
        )

    # ── Public API ───────────────────────────────────────────────────

    def send_message(self, text: str) -> str:
        """Send a user message through the full agent graph."""
        self._state["messages"].append(HumanMessage(content=text))

        try:
            final_state = self._run_graph(self._state)
        except Exception as e:
            logger.error("Graph execution failed: %s", e, exc_info=True)
            return f"[Error del agente] {e}"

        if final_state:
            self._state = final_state

        return self._extract_response()

    def summarize_clipboard(self, text: str) -> str:
        """Send clipboard text to the agent for summarization."""
        prompt = (
            "Resume el siguiente texto del portapapeles de forma "
            "concisa y clara. Preserva los puntos clave:\n\n"
            f"{text}"
        )
        return self.send_message(prompt)

    def analyze_document(self, filename: str, content: str) -> str:
        """Send a document to the agent for analysis."""
        prompt = (
            f"Analiza el siguiente documento ('{filename}') y proporciona:\n"
            "1. Un resumen breve\n"
            "2. Puntos clave\n"
            "3. Categorias o temas relevantes\n\n"
            f"Contenido:\n{content[:8000]}"
        )
        return self.send_message(prompt)

    def check_status(self) -> AgentStatus:
        """Check if the LLM backend is reachable."""
        try:
            if self._settings.llm_provider == "ollama":
                ok = check_ollama_connection(self._settings.ollama_base_url)
                return AgentStatus.ONLINE if ok else AgentStatus.OFFLINE
            return AgentStatus.ONLINE
        except Exception:
            return AgentStatus.OFFLINE

    def get_folder_notifications(self) -> list:
        """Get pending folder monitoring notifications."""
        return self._folder_monitor.get_notifications()

    def stop_folder_monitor(self) -> None:
        """Stop the folder monitor on shutdown."""
        self._folder_monitor.stop()

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    @property
    def settings(self):
        return self._settings

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._state = create_initial_state()
        logger.info("Conversation history cleared")

    # ── Private ──────────────────────────────────────────────────────

    def _run_graph(self, state: AgentState) -> AgentState | None:
        """Stream through the graph, asking the user for confirmation
        when sensitive tools are invoked."""
        final_state: AgentState | None = None

        for event in self._graph.stream(
            state, config=self._config, stream_mode="values"
        ):
            final_state = event

        if final_state is None:
            return None

        # Handle tool confirmation interrupt
        if final_state.get("requires_confirmation"):
            pending = final_state.get("pending_tool_calls", [])
            logger.info(
                "Confirmation needed for %d tool(s): %s",
                len(pending),
                [c["name"] for c in pending],
            )

            if self._confirm_handler is not None:
                confirmed = self._confirm_handler(pending)
            else:
                confirmed = True

            final_state["user_confirmed"] = confirmed
            final_state["metadata"] = {
                **final_state.get("metadata", {}),
                "resume": True,
            }

            for event in self._graph.stream(
                final_state, config=self._config, stream_mode="values"
            ):
                final_state = event

        return final_state

    def _extract_response(self) -> str:
        """Extract the last AI message content from the state."""
        for msg in reversed(self._state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "[El agente no genero una respuesta de texto]"
