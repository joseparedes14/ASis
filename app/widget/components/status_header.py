"""Status header component — agent status indicator and window controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from app.widget.api.llm_client import AgentStatus
from app.widget.styles import COLORS, COMPACT_SIZE, FONT_FAMILY, QCOLOR_ACCENT


class StatusDot(QWidget):
    """Small animated dot indicating agent status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._color = QCOLOR_ACCENT
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_status(self, status: AgentStatus) -> None:
        color_map = {
            AgentStatus.ONLINE: QColor(166, 227, 161),
            AgentStatus.THINKING: QColor(249, 226, 175),
            AgentStatus.OFFLINE: QColor(243, 139, 168),
        }
        self._color = color_map.get(status, QCOLOR_ACCENT)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(self._color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(2, 2, 6, 6)
        p.end()


def _make_passthrough(label: QLabel) -> QLabel:
    """Make a QLabel transparent to mouse events so clicks reach the parent."""
    label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    return label


class StatusHeader(QWidget):
    """Draggable header with status dot, model name, and collapse button."""

    drag_requested = pyqtSignal(int, int)
    collapse_clicked = pyqtSignal()
    compact_clicked = pyqtSignal()
    close_clicked = pyqtSignal()

    COMPACT_HEIGHT = COMPACT_SIZE
    EXPANDED_HEIGHT = 48

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded = False
        self._drag_pos = None
        self._status = AgentStatus.OFFLINE
        self._model_name = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFixedHeight(self.COMPACT_HEIGHT)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(14, 0, 14, 0)
        self._layout.setSpacing(8)

        self._dot = StatusDot()
        self._icon_label = _make_passthrough(QLabel("A"))
        self._icon_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 700;
            color: {COLORS.accent};
            font-family: {FONT_FAMILY};
        """)

        self._name_label = _make_passthrough(QLabel("ASis"))
        self._name_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {COLORS.text_primary};
            font-family: {FONT_FAMILY};
        """)

        self._status_label = _make_passthrough(QLabel("Offline"))
        self._status_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS.text_muted};
            font-family: {FONT_FAMILY};
        """)

        self._model_label = _make_passthrough(QLabel(""))
        self._model_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS.text_secondary};
            font-family: {FONT_FAMILY};
        """)

        self._close_btn = QPushButton("\u2715")
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                font-size: 13px;
                color: {COLORS.text_muted};
                border-radius: 14px;
                font-family: {FONT_FAMILY};
            }}
            QPushButton:hover {{
                background-color: {COLORS.error_dim};
                color: {COLORS.error};
            }}
        """)
        self._close_btn.clicked.connect(self.close_clicked.emit)

        self._layout.addWidget(self._icon_label)
        self._layout.addWidget(self._dot)
        self._layout.addWidget(self._name_label)
        self._layout.addStretch()
        self._layout.addWidget(self._status_label)
        self._layout.addWidget(self._model_label)
        self._layout.addWidget(self._close_btn)

        self._compact_widgets = [
            self._name_label,
            self._status_label,
            self._model_label,
            self._close_btn,
        ]

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.setFixedHeight(
            self.EXPANDED_HEIGHT if expanded else self.COMPACT_HEIGHT
        )
        for w in self._compact_widgets:
            w.setVisible(expanded)

    def update_status(self, status: AgentStatus, model_name: str = "") -> None:
        self._status = status
        self._model_name = model_name
        self._dot.set_status(status)

        status_text = {
            AgentStatus.ONLINE: "Online",
            AgentStatus.THINKING: "Pensando...",
            AgentStatus.OFFLINE: "Offline",
        }
        self._status_label.setText(status_text.get(status, "Offline"))
        self._model_label.setText(model_name)

        status_color = {
            AgentStatus.ONLINE: COLORS.success,
            AgentStatus.THINKING: COLORS.warning,
            AgentStatus.OFFLINE: COLORS.error,
        }
        self._status_label.setStyleSheet(f"""
            font-size: 11px;
            color: {status_color.get(status, COLORS.text_muted)};
            font-family: {FONT_FAMILY};
        """)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._expanded:
            self.compact_clicked.emit()
        else:
            self._drag_pos = event.globalPosition().toPoint() - self.window().pos()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None:
            self.drag_requested.emit(
                event.globalPosition().toPoint().x() - self._drag_pos.x(),
                event.globalPosition().toPoint().y() - self._drag_pos.y(),
            )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None
