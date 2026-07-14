"""
Memory layer for the ASis agent.

Provides short-term memory (conversation history within the graph state)
and an interface for future persistent memory implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.messages import BaseMessage

from app.config.logging_config import get_logger

logger = get_logger(__name__)


class MemoryBackend(ABC):
    """Abstract base class for memory persistence backends.

    Implement this interface to add persistent memory storage
    (e.g., SQLite, Redis, filesystem-based).
    """

    @abstractmethod
    async def save_conversation(
        self, conversation_id: str, messages: list[BaseMessage]
    ) -> None:
        """Persist conversation messages.

        Args:
            conversation_id: Unique identifier for the conversation.
            messages: List of messages to persist.
        """
        ...

    @abstractmethod
    async def load_conversation(
        self, conversation_id: str
    ) -> Optional[list[BaseMessage]]:
        """Load persisted conversation messages.

        Args:
            conversation_id: Unique identifier for the conversation.

        Returns:
            List of messages if found, None otherwise.
        """
        ...

    @abstractmethod
    async def clear_conversation(self, conversation_id: str) -> None:
        """Clear persisted conversation data.

        Args:
            conversation_id: Unique identifier for the conversation.
        """
        ...


class InMemoryBackend(MemoryBackend):
    """Simple in-memory storage backend for development and testing.

    Messages are stored in a dictionary and lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[BaseMessage]] = {}

    async def save_conversation(
        self, conversation_id: str, messages: list[BaseMessage]
    ) -> None:
        """Save messages to in-memory store."""
        self._store[conversation_id] = list(messages)
        logger.debug(
            "Saved %d messages for conversation %s", len(messages), conversation_id
        )

    async def load_conversation(
        self, conversation_id: str
    ) -> Optional[list[BaseMessage]]:
        """Load messages from in-memory store."""
        messages = self._store.get(conversation_id)
        if messages:
            logger.debug(
                "Loaded %d messages for conversation %s",
                len(messages),
                conversation_id,
            )
        return messages

    async def clear_conversation(self, conversation_id: str) -> None:
        """Clear messages from in-memory store."""
        self._store.pop(conversation_id, None)
        logger.debug("Cleared conversation %s", conversation_id)


class ConversationMemory:
    """Manages conversation memory with configurable backend.

    Provides a high-level interface for memory operations, abstracting
    the underlying storage mechanism. Includes message trimming to
    keep conversation history within configured limits.

    Args:
        backend: Storage backend to use. Defaults to InMemoryBackend.
        max_messages: Maximum number of messages to retain.
    """

    def __init__(
        self,
        backend: Optional[MemoryBackend] = None,
        max_messages: int = 50,
    ) -> None:
        self._backend = backend or InMemoryBackend()
        self._max_messages = max_messages

    def trim_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Trim message history to stay within the configured limit.

        Always keeps the system message (first message) if present,
        and trims older messages from the middle.

        Args:
            messages: Full list of conversation messages.

        Returns:
            Trimmed list of messages within the limit.
        """
        if len(messages) <= self._max_messages:
            return messages

        # Keep system message + most recent messages
        system_messages = [m for m in messages[:1] if m.type == "system"]
        recent_messages = messages[-(self._max_messages - len(system_messages)) :]

        trimmed = system_messages + recent_messages
        logger.info(
            "Trimmed conversation from %d to %d messages",
            len(messages),
            len(trimmed),
        )
        return trimmed

    async def save(
        self, conversation_id: str, messages: list[BaseMessage]
    ) -> None:
        """Save conversation with automatic trimming.

        Args:
            conversation_id: Unique conversation identifier.
            messages: Messages to save.
        """
        trimmed = self.trim_messages(messages)
        await self._backend.save_conversation(conversation_id, trimmed)

    async def load(
        self, conversation_id: str
    ) -> list[BaseMessage]:
        """Load conversation history.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            List of messages, empty list if no history found.
        """
        messages = await self._backend.load_conversation(conversation_id)
        return messages or []

    async def clear(self, conversation_id: str) -> None:
        """Clear conversation history.

        Args:
            conversation_id: Unique conversation identifier.
        """
        await self._backend.clear_conversation(conversation_id)

    @property
    def backend_type(self) -> str:
        """Return the type name of the current backend."""
        return type(self._backend).__name__
