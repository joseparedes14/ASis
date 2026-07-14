"""Tests for the agent graph construction and state management."""

from app.agent.state import AgentState, MAX_ITERATIONS, create_initial_state


class TestAgentState:
    """Tests for AgentState creation and defaults."""

    def test_create_initial_state_returns_valid_state(self):
        """Initial state should have all required fields with defaults."""
        state = create_initial_state()

        assert state["messages"] == []
        assert state["pending_tool_calls"] == []
        assert state["tool_results"] == []
        assert state["requires_confirmation"] is False
        assert state["user_confirmed"] is None
        assert state["error"] is None
        assert state["metadata"] == {}
        assert state["iteration_count"] == 0

    def test_max_iterations_is_reasonable(self):
        """Max iterations should be a positive number to prevent infinite loops."""
        assert MAX_ITERATIONS > 0
        assert MAX_ITERATIONS <= 50  # Sanity upper bound

    def test_state_is_typed_dict(self):
        """AgentState should be usable as a TypedDict."""
        state = create_initial_state()
        # TypedDict supports dict-like access
        assert "messages" in state
        assert "error" in state
