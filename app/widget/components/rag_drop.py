"""Drag & Drop zone component — for sending documents to the agent's memory."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.widget.styles import COLORS, CORNER_RADIUS_XS, FONT_FAMILY, PADDING_SM


class RagDropZone(QWidget):
    """Minimalist drag & drop zone for document ingestion."""

    file_dropped = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_active = False
        self.setAcceptDrops(True)
        self.setFixedHeight(56)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PADDING_SM, 4, PADDING_SM, PADDING_SM)

        self._label = QLabel("\U0001F4CE Arrastra un documento aqui")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(self._idle_style())

    def _idle_style(self) -> str:
        return f"""
            background-color: {COLORS.bg_surface};
            border: 2px dashed {COLORS.border_default};
            border-radius: {CORNER_RADIUS_XS}px;
            padding: 8px;
            color: {COLORS.text_muted};
            font-family: {FONT_FAMILY};
            font-size: 12px;
        """

    def _active_style(self) -> str:
        return f"""
            background-color: {COLORS.accent_dim};
            border: 2px dashed {COLORS.accent};
            border-radius: {CORNER_RADIUS_XS}px;
            padding: 8px;
            color: {COLORS.accent};
            font-family: {FONT_FAMILY};
            font-size: 12px;
            font-weight: 600;
        """

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_active = True
            self._label.setText("\U0001F4E5 Suelta el documento")
            self._label.setStyleSheet(self._active_style())
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drag_active = False
        self._label.setText("\U0001F4CE Arrastra un documento aqui")
        self._label.setStyleSheet(self._idle_style())

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._drag_active = False
        self._label.setText("\U0001F4CE Arrastra un documento aqui")
        self._label.setStyleSheet(self._idle_style())

        urls = event.mimeData().urls()
        if not urls:
            return

        for url in urls:
            file_path = url.toLocalFile()
            if file_path:
                path = Path(file_path)
                if path.is_file():
                    try:
                        content = path.read_text(encoding="utf-8", errors="replace")
                        self.file_dropped.emit(path.name, content)
                    except Exception:
                        self.file_dropped.emit(path.name, f"[No se pudo leer: {path.suffix}]")
