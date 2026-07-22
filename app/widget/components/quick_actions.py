"""Quick actions component — action buttons grid and role/temperature selectors."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widget.api.llm_client import ROLE_PRESETS, TEMPERATURE_PRESETS
from app.widget.styles import COLORS, FONT_FAMILY, GAP, PADDING_SM


class QuickActions(QWidget):
    """Grid of quick action buttons, role selector, and temperature selector."""

    action_clicked = pyqtSignal(str)
    role_changed = pyqtSignal(str)
    temperature_changed = pyqtSignal(float)
    file_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active_role = "General"
        self._active_temp = "Normal"
        self._role_buttons: dict[str, QPushButton] = {}
        self._temp_buttons: dict[str, QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PADDING_SM, 4, PADDING_SM, PADDING_SM)
        layout.setSpacing(GAP)

        # ── Section label: Acciones ───────────────────────────────
        actions_label = QLabel("Acciones Rapidas")
        actions_label.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 600;
            color: {COLORS.text_muted};
            font-family: {FONT_FAMILY};
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(actions_label)

        # ── Action buttons row ────────────────────────────────────
        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)

        action_defs = [
            ("\U0001F4CB Resumir", "clipboard"),
            ("\U0001F4AC Chat", "chat_full"),
            ("\U0001F4CE Documentos", "documents"),
        ]

        for label, action in action_defs:
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(34)
            btn.clicked.connect(lambda checked, a=action: self._on_action(a))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS.bg_surface};
                    border: 1px solid {COLORS.border_subtle};
                    border-radius: {6}px;
                    padding: 6px 10px;
                    color: {COLORS.text_secondary};
                    font-family: {FONT_FAMILY};
                    font-size: 11px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {COLORS.bg_hover};
                    color: {COLORS.text_primary};
                    border: 1px solid {COLORS.border_default};
                }}
                QPushButton:pressed {{
                    background-color: {COLORS.bg_active};
                }}
            """)
            actions_row.addWidget(btn)
        layout.addLayout(actions_row)

        # ── Section label: Rol ────────────────────────────────────
        role_label = QLabel("Rol del Agente")
        role_label.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 600;
            color: {COLORS.text_muted};
            font-family: {FONT_FAMILY};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        """)
        layout.addWidget(role_label)

        # ── Role buttons row ──────────────────────────────────────
        role_row = QHBoxLayout()
        role_row.setSpacing(4)

        for role_name in ROLE_PRESETS:
            btn = QPushButton(role_name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, r=role_name: self._on_role(r))
            self._role_buttons[role_name] = btn
            role_row.addWidget(btn)
        layout.addLayout(role_row)
        self._update_role_styles()

        # ── Section label: Temperatura ────────────────────────────
        temp_label = QLabel("Temperatura")
        temp_label.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 600;
            color: {COLORS.text_muted};
            font-family: {FONT_FAMILY};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        """)
        layout.addWidget(temp_label)

        # ── Temperature buttons row ───────────────────────────────
        temp_row = QHBoxLayout()
        temp_row.setSpacing(4)

        for temp_name, temp_val in TEMPERATURE_PRESETS.items():
            btn = QPushButton(f"{temp_name}  {temp_val}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.clicked.connect(
                lambda checked, n=temp_name, v=temp_val: self._on_temp(n, v)
            )
            self._temp_buttons[temp_name] = btn
            temp_row.addWidget(btn)
        layout.addLayout(temp_row)
        self._update_temp_styles()

    def _on_action(self, action: str) -> None:
        if action == "documents":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Seleccionar documento",
                "",
                "Todos los archivos (*.*)",
            )
            if path:
                self.file_selected.emit(path)
        else:
            self.action_clicked.emit(action)

    def _on_role(self, role: str) -> None:
        self._active_role = role
        self._update_role_styles()
        self.role_changed.emit(role)

    def _on_temp(self, name: str, value: float) -> None:
        self._active_temp = name
        self._update_temp_styles()
        self.temperature_changed.emit(value)

    def _update_role_styles(self) -> None:
        for name, btn in self._role_buttons.items():
            active = name == self._active_role
            if active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.accent_dim};
                        border: 1px solid {COLORS.accent};
                        border-radius: {6}px;
                        padding: 5px 10px;
                        color: {COLORS.accent};
                        font-family: {FONT_FAMILY};
                        font-size: 11px;
                        font-weight: 600;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.bg_surface};
                        border: 1px solid {COLORS.border_subtle};
                        border-radius: {6}px;
                        padding: 5px 10px;
                        color: {COLORS.text_secondary};
                        font-family: {FONT_FAMILY};
                        font-size: 11px;
                    }}
                    QPushButton:hover {{
                        background-color: {COLORS.bg_hover};
                        color: {COLORS.text_primary};
                    }}
                """)

    def _update_temp_styles(self) -> None:
        for name, btn in self._temp_buttons.items():
            active = name == self._active_temp
            if active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.accent_dim};
                        border: 1px solid {COLORS.accent};
                        border-radius: {6}px;
                        padding: 5px 8px;
                        color: {COLORS.accent};
                        font-family: {FONT_FAMILY};
                        font-size: 11px;
                        font-weight: 600;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLORS.bg_surface};
                        border: 1px solid {COLORS.border_subtle};
                        border-radius: {6}px;
                        padding: 5px 8px;
                        color: {COLORS.text_secondary};
                        font-family: {FONT_FAMILY};
                        font-size: 11px;
                    }}
                    QPushButton:hover {{
                        background-color: {COLORS.bg_hover};
                        color: {COLORS.text_primary};
                    }}
                """)
