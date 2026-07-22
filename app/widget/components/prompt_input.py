"""Prompt input component — quick text input bar for the agent."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

from app.widget.styles import (
    COLORS,
    CORNER_RADIUS_SM,
    FONT_FAMILY,
    PADDING_SM,
)


class PromptInput(QWidget):
    """Minimalist text input with send button."""

    message_sent = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(PADDING_SM, 4, PADDING_SM, PADDING_SM)
        layout.setSpacing(6)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Escribe un mensaje...")
        self._input.returnPressed.connect(self._on_send)
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS.bg_input};
                border: 1px solid {COLORS.border_default};
                border-radius: {CORNER_RADIUS_SM}px;
                padding: 10px 14px;
                color: {COLORS.text_primary};
                font-family: {FONT_FAMILY};
                font-size: 13px;
                selection-background-color: {COLORS.accent_dim};
            }}
            QLineEdit:focus {{
                border: 1px solid {COLORS.accent};
            }}
            QLineEdit::placeholder {{
                color: {COLORS.text_disabled};
            }}
        """)

        self._send_btn = QPushButton("\u23CE")
        self._send_btn.setFixedSize(38, 38)
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.clicked.connect(self._on_send)
        self._send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS.accent};
                border: none;
                border-radius: {CORNER_RADIUS_SM}px;
                color: #1e1e2c;
                font-size: 16px;
                font-weight: 700;
                font-family: {FONT_FAMILY};
            }}
            QPushButton:hover {{
                background-color: #a0c4ff;
            }}
            QPushButton:pressed {{
                background-color: #7aa2f0;
            }}
            QPushButton:disabled {{
                background-color: {COLORS.bg_surface};
                color: {COLORS.text_disabled};
            }}
        """)

        layout.addWidget(self._input)
        layout.addWidget(self._send_btn)

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if text:
            self._input.clear()
            self.message_sent.emit(text)

    def set_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)

    def focus_input(self) -> None:
        self._input.setFocus()
