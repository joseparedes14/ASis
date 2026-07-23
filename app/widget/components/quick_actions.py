"""Quick actions component — action buttons and functions menu."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widget.styles import COLORS, FONT_FAMILY, GAP, PADDING_SM


class QuickActions(QWidget):
    """Grid of quick action buttons and a functions menu."""

    action_clicked = pyqtSignal(str)
    file_selected = pyqtSignal(str)
    function_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PADDING_SM, 4, PADDING_SM, PADDING_SM)
        layout.setSpacing(GAP)

        # ── Section label: Acciones ───────────────────────────────
        actions_label = QLabel("Acciones Rápidas")
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

        # ── Functions Menu Button ─────────────────────────────────
        self._menu_btn = QPushButton("⚙ Menú")
        self._menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._menu_btn.setFixedHeight(34)
        self._menu_btn.setStyleSheet(self._button_style(accent=True))
        
        # Setup the QMenu
        self._menu = QMenu(self)
        self._menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS.bg_elevated};
                border: 1px solid {COLORS.border_default};
                border-radius: 6px;
                padding: 4px;
                color: {COLORS.text_primary};
                font-family: {FONT_FAMILY};
                font-size: 12px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {COLORS.accent_dim};
                color: {COLORS.accent};
            }}
        """)
        
        self._add_function("Clasificar Documento", "Función: Clasificar Documento\nPor favor adjunta o arrastra el documento que deseas clasificar.")
        self._add_function("Resumir Reunión", "Función: Resumir Reunión\nPega aquí las notas o transcripción de la reunión para generar un resumen.")
        self._add_function("Redactar Email", "Función: Redactar Email\nDestinatario: [Nombre]\nAsunto: [Tema]\nPuntos clave:\n- \n- ")
        
        self._menu.addSeparator()
        
        action_monitor = QAction("Añadir carpeta a monitorizar", self)
        action_monitor.triggered.connect(self._on_add_monitor_folder)
        self._menu.addAction(action_monitor)
        
        self._menu_btn.setMenu(self._menu)
        actions_row.addWidget(self._menu_btn)

        layout.addLayout(actions_row)

    def _button_style(self, accent: bool = False) -> str:
        bg = COLORS.accent_dim if accent else COLORS.bg_surface
        text = COLORS.accent if accent else COLORS.text_secondary
        border = COLORS.accent if accent else COLORS.border_subtle
        
        return f"""
            QPushButton {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: {6}px;
                padding: 6px 10px;
                color: {text};
                font-family: {FONT_FAMILY};
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS.bg_hover};
                color: {COLORS.text_primary};
                border: 1px solid {COLORS.border_default};
            }}
            QPushButton::menu-indicator {{
                image: none;
            }}
        """

    def _add_function(self, name: str, prompt_template: str) -> None:
        action = QAction(name, self)
        action.triggered.connect(lambda: self.function_selected.emit(prompt_template))
        self._menu.addAction(action)

    def _on_add_monitor_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta a monitorizar",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if path:
            self.action_clicked.emit(f"add_monitor_folder:{path}")

    def _on_action(self, action: str) -> None:
        self.action_clicked.emit(action)
