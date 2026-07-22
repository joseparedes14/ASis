"""Response panel component — scrollable chat message display."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.widget.styles import (
    COLORS,
    CORNER_RADIUS_XS,
    FONT_FAMILY,
    PADDING_SM,
)


class ChatBubble(QWidget):
    """A single chat message bubble."""

    def __init__(
        self,
        text: str,
        is_user: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_user = is_user
        self._setup_ui(text)

    def _setup_ui(self, text: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        label.setFont(QFont(FONT_FAMILY, 11))

        if self._is_user:
            label.setStyleSheet(f"""
                background-color: {COLORS.accent_dim};
                border: 1px solid rgba(137, 180, 250, 0.2);
                border-radius: {CORNER_RADIUS_XS}px;
                padding: 8px 12px;
                color: {COLORS.text_primary};
                font-family: {FONT_FAMILY};
                font-size: 12px;
                margin-left: 40px;
            """)
        else:
            label.setStyleSheet(f"""
                background-color: {COLORS.bg_surface};
                border: 1px solid {COLORS.border_subtle};
                border-radius: {CORNER_RADIUS_XS}px;
                padding: 8px 12px;
                color: {COLORS.text_primary};
                font-family: {FONT_FAMILY};
                font-size: 12px;
                margin-right: 40px;
            """)

        layout.addWidget(label)

        sender = "Tú" if self._is_user else "ASis"
        sender_color = COLORS.accent if self._is_user else COLORS.text_muted
        sender_label = QLabel(sender)
        sender_label.setStyleSheet(f"""
            font-size: 10px;
            color: {sender_color};
            font-family: {FONT_FAMILY};
            font-weight: 600;
            padding-left: 4px;
            padding-bottom: 2px;
        """)
        layout.insertWidget(0, sender_label)


class ConfirmationBar(QWidget):
    """Inline confirmation prompt with Confirm / Cancel buttons."""

    confirmed = pyqtSignal()
    rejected = pyqtSignal()

    def __init__(
        self, tool_calls: list[dict], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._setup_ui(tool_calls)

    def _setup_ui(self, tool_calls: list[dict]) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        header = QLabel("\u26a0\ufe0f  El agente quiere ejecutar:")
        header.setStyleSheet(f"""
            color: {COLORS.warning};
            font-size: 11px;
            font-weight: 600;
            font-family: {FONT_FAMILY};
        """)
        layout.addWidget(header)

        for call in tool_calls:
            args_str = ", ".join(
                f"{k}={v!r:.60}" for k, v in call.get("args", {}).items()
            )
            detail = QLabel(f"  \u2022 {call['name']}({args_str})")
            detail.setWordWrap(True)
            detail.setStyleSheet(f"""
                color: {COLORS.text_secondary};
                font-size: 11px;
                font-family: {FONT_FAMILY};
                padding: 2px 4px;
            """)
            layout.addWidget(detail)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        confirm_btn = QPushButton("Confirmar")
        confirm_btn.setFixedHeight(28)
        confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS.success_dim};
                border: 1px solid {COLORS.success};
                border-radius: 6px;
                padding: 4px 16px;
                color: {COLORS.success};
                font-family: {FONT_FAMILY};
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: rgba(166,227,161,0.25); }}
        """)
        confirm_btn.clicked.connect(self.confirmed.emit)

        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setFixedHeight(28)
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS.error_dim};
                border: 1px solid {COLORS.error};
                border-radius: 6px;
                padding: 4px 16px;
                color: {COLORS.error};
                font-family: {FONT_FAMILY};
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: rgba(243,139,168,0.25); }}
        """)
        cancel_btn.clicked.connect(self.rejected.emit)

        btn_row.addWidget(confirm_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)


class ResponsePanel(QWidget):
    """Scrollable panel that displays chat messages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(PADDING_SM, 4, PADDING_SM, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 5px;
                margin: 4px 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS.border_strong};
                border-radius: 2px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS.text_muted};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        self._container = QWidget()
        self._container.setStyleSheet("background-color: transparent;")
        self._messages_layout = QVBoxLayout(self._container)
        self._messages_layout.setContentsMargins(4, 8, 4, 8)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch()

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

        self._placeholder = QLabel("Envia un mensaje para comenzar...")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"""
            color: {COLORS.text_disabled};
            font-size: 12px;
            font-family: {FONT_FAMILY};
            padding: 40px 20px;
        """)
        self._messages_layout.insertWidget(0, self._placeholder)

    def add_user_message(self, text: str) -> None:
        if self._placeholder.isVisible():
            self._placeholder.hide()
        bubble = ChatBubble(text, is_user=True)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def add_agent_message(self, text: str) -> None:
        if self._placeholder.isVisible():
            self._placeholder.hide()
        bubble = ChatBubble(text, is_user=False)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def add_system_message(self, text: str) -> None:
        if self._placeholder.isVisible():
            self._placeholder.hide()
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            color: {COLORS.text_muted};
            font-size: 11px;
            font-family: {FONT_FAMILY};
            padding: 4px 12px;
        """)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, label)
        self._scroll_to_bottom()

    def show_thinking(self) -> None:
        self._thinking_label = QLabel("Pensando...")
        self._thinking_label.setStyleSheet(f"""
            color: {COLORS.warning};
            font-size: 11px;
            font-family: {FONT_FAMILY};
            padding: 4px 12px;
            font-style: italic;
        """)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, self._thinking_label
        )
        self._scroll_to_bottom()

    def hide_thinking(self) -> None:
        if hasattr(self, "_thinking_label") and self._thinking_label is not None:
            self._thinking_label.hide()
            self._thinking_label.deleteLater()
            self._thinking_label = None

    def clear_messages(self) -> None:
        self.remove_confirmation()
        while self._messages_layout.count() > 1:
            item = self._messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._placeholder.show()
        self._messages_layout.insertWidget(0, self._placeholder)

    def show_confirmation(
        self, tool_calls: list[dict]
    ) -> ConfirmationBar:
        """Insert a confirmation bar and return it for signal connection."""
        if self._placeholder.isVisible():
            self._placeholder.hide()
        bar = ConfirmationBar(tool_calls)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, bar
        )
        self._scroll_to_bottom()
        return bar

    def remove_confirmation(self) -> None:
        """Remove any active confirmation bar."""
        for i in range(self._messages_layout.count()):
            item = self._messages_layout.itemAt(i)
            if item and item.widget() and isinstance(
                item.widget(), ConfirmationBar
            ):
                item.widget().deleteLater()
                break

    def _scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
