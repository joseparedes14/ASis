"""Main glassmorphism dashboard widget for the ASis agent.

Frameless, draggable window with Windows DWM blur, compact/expanded
states, and smooth animations. Assembles all component sub-widgets.
"""

from __future__ import annotations

import logging
import sys
import threading

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from app.services.email_monitor import EmailMonitor
from app.widget.api.agent_bridge import AgentBridge
from app.widget.api.llm_client import AgentStatus
from app.widget.components.prompt_input import PromptInput
from app.widget.components.quick_actions import QuickActions
from app.widget.components.rag_drop import RagDropZone
from app.widget.components.response_panel import ResponsePanel
from app.widget.components.status_header import StatusHeader
from app.widget.components.system_metrics import SystemMetrics
from app.widget.styles import (
    COMPACT_SIZE,
    CORNER_RADIUS,
    EXPANDED_HEIGHT,
    EXPANDED_WIDTH,
    WIDGET_MARGIN,
    global_stylesheet,
)

if sys.platform == "win32":
    from app.widget.dwm import enable_blur_behind, enable_dark_mode

logger = logging.getLogger("asis.widget.dashboard")

ANIMATION_DURATION = 250


class DashboardWidget(QWidget):
    """The main ASis desktop dashboard widget.

    Features:
        - Glassmorphism with Windows DWM blur
        - Compact (small circle) / Expanded (full dashboard) states
        - Draggable when expanded
        - Keyboard shortcut: Ctrl+Shift+A to toggle
    """

    toggle_signal = pyqtSignal()
    _sig_response = pyqtSignal(str)
    _sig_system = pyqtSignal(str)
    _sig_status = pyqtSignal(object, str)
    _sig_input_enabled = pyqtSignal(bool)
    _sig_confirmation = pyqtSignal(list)
    _sig_folder_notification = pyqtSignal(str, str, str)  # filename, source, destination

    def __init__(self) -> None:
        super().__init__()
        self._expanded = False
        self._drag_pos = None
        self._confirm_event = threading.Event()
        self._confirm_result = False
        self._agent = AgentBridge(
            confirmation_handler=self._handle_confirmation
        )
        self._start_email_monitor()
        self._setup_window()
        self._setup_components()
        self._setup_layout()
        self._setup_animation()
        self._setup_timers()
        self._connect_signals()
        self.toggle_signal.connect(self.toggle)
        self._sig_response.connect(self._on_agent_response)
        self._sig_system.connect(self._on_system_message)
        self._sig_status.connect(self._on_status_update)
        self._sig_input_enabled.connect(self._on_input_enabled)
        self._sig_confirmation.connect(self._on_confirmation_needed)
        self._sig_folder_notification.connect(self._on_folder_notification)

    def _start_email_monitor(self) -> None:
        """Start the background email monitor if monitored senders are configured."""
        settings = self._agent.settings
        monitored_senders = settings.get_monitored_senders()
        if monitored_senders:
            self._email_monitor = EmailMonitor(settings)
            self._email_monitor.start()
            logger.info(
                "Email monitor active — watching: %s",
                ", ".join(monitored_senders),
            )
        else:
            self._email_monitor = None
            logger.info("No monitored senders configured — email monitor not started")

    def _setup_window(self) -> None:
        self.setWindowTitle("ASis Dashboard")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - EXPANDED_WIDTH - WIDGET_MARGIN
        y = WIDGET_MARGIN
        self.setGeometry(x, y, COMPACT_SIZE, COMPACT_SIZE)

        self.setStyleSheet(global_stylesheet())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("ASis — Click para abrir")

    def _setup_components(self) -> None:
        self._header = StatusHeader()
        self._response_panel = ResponsePanel()
        self._prompt_input = PromptInput()
        self._quick_actions = QuickActions()
        self._system_metrics = SystemMetrics(refresh_ms=3000)
        self._rag_drop = RagDropZone()

    def _setup_layout(self) -> None:
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._main_layout.addWidget(self._header)
        self._main_layout.addWidget(self._response_panel)
        self._main_layout.addWidget(self._prompt_input)
        self._main_layout.addWidget(self._quick_actions)
        self._main_layout.addWidget(self._system_metrics)
        self._main_layout.addWidget(self._rag_drop)

        self._response_panel.hide()
        self._prompt_input.hide()
        self._quick_actions.hide()
        self._system_metrics.hide()
        self._rag_drop.hide()

    def _setup_animation(self) -> None:
        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setDuration(ANIMATION_DURATION)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _setup_timers(self) -> None:
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._check_status)
        self._status_timer.start(5000)
        self._check_status()

        # Timer for folder monitoring notifications
        self._folder_notification_timer = QTimer(self)
        self._folder_notification_timer.timeout.connect(self._check_folder_notifications)
        self._folder_notification_timer.start(2000)

    def _connect_signals(self) -> None:
        self._header.collapse_clicked.connect(self.collapse)
        self._header.compact_clicked.connect(self.expand)
        self._header.drag_requested.connect(self._on_drag)
        self._header.close_clicked.connect(self.close)

        self._prompt_input.message_sent.connect(self._on_send_message)

        self._quick_actions.action_clicked.connect(self._on_action)
        self._quick_actions.file_selected.connect(self._on_file_selected)
        self._quick_actions.function_selected.connect(self._on_function_selected)

        self._rag_drop.file_dropped.connect(self._on_file_dropped)

    def closeEvent(self, event) -> None:
        """Clean up resources when the widget is closed."""
        self._folder_monitor.stop()
        logger.info("Widget closed — folder monitor stopped")
        super().closeEvent(event)

    # ── State Management ────────────────────────────────────────────

    def toggle(self) -> None:
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self) -> None:
        if self._expanded:
            return
        self._expanded = True
        self.setAcceptDrops(True)

        self._response_panel.show()
        self._prompt_input.show()
        self._quick_actions.show()
        self._system_metrics.show()
        self._rag_drop.show()
        self._header.set_expanded(True)

        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - EXPANDED_WIDTH - WIDGET_MARGIN
        y = WIDGET_MARGIN
        start_geo = self.geometry()
        end_geo = QRect(x, y, EXPANDED_WIDTH, EXPANDED_HEIGHT)

        self._anim.stop()
        self._anim.setStartValue(start_geo)
        self._anim.setEndValue(end_geo)
        self._anim.start()

    def collapse(self) -> None:
        if not self._expanded:
            return
        self._expanded = False
        self.setAcceptDrops(False)

        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - COMPACT_SIZE - WIDGET_MARGIN
        y = WIDGET_MARGIN
        start_geo = self.geometry()
        end_geo = QRect(x, y, COMPACT_SIZE, COMPACT_SIZE)

        self._anim.stop()
        self._anim.setStartValue(start_geo)
        self._anim.setEndValue(end_geo)
        self._anim.finished.connect(self._on_collapse_finished)
        self._anim.start()

    def _on_collapse_finished(self) -> None:
        self._anim.finished.disconnect(self._on_collapse_finished)
        self._response_panel.hide()
        self._prompt_input.hide()
        self._quick_actions.hide()
        self._system_metrics.hide()
        self._rag_drop.hide()
        self._header.set_expanded(False)

    # ── Event Handlers ──────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(
            float(rect.x()),
            float(rect.y()),
            float(rect.width()),
            float(rect.height()),
            CORNER_RADIUS if self._expanded else COMPACT_SIZE // 2,
            CORNER_RADIUS if self._expanded else COMPACT_SIZE // 2,
        )

        painter.setClipPath(path)
        painter.fillRect(rect, QColor(255, 255, 255, 160))
        painter.setClipping(False)

        pen = QColor(0, 0, 0, 20)
        painter.setPen(pen)
        painter.drawRoundedRect(
            rect.adjusted(0, 0, -1, -1),
            CORNER_RADIUS if self._expanded else COMPACT_SIZE // 2,
            CORNER_RADIUS if self._expanded else COMPACT_SIZE // 2,
        )

        painter.end()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if sys.platform == "win32":
            try:
                enable_blur_behind(self)
                # enable_dark_mode(self)
            except Exception as e:
                logger.debug("DWM blur failed: %s", e)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            self._expanded
            and event.button() == Qt.MouseButton.LeftButton
        ):
            pos = event.position().toPoint()
            header_rect = self._header.geometry()
            if header_rect.contains(pos):
                self._drag_pos = pos

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            self.move(new_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._expanded:
            self.collapse()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            if not self._expanded:
                self.expand()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                from pathlib import Path
                p = Path(path)
                if p.is_file():
                    try:
                        content = p.read_text(encoding="utf-8", errors="replace")
                        self._on_file_dropped(p.name, content)
                    except Exception:
                        self._on_file_dropped(p.name, f"[No se pudo leer: {p.suffix}]")

    # ── Thread-safe slots (run on main thread) ─────────────────────

    def _on_agent_response(self, text: str) -> None:
        self._response_panel.add_agent_message(text)

    def _on_system_message(self, text: str) -> None:
        self._response_panel.add_system_message(text)

    def _on_status_update(self, status: AgentStatus, model: str) -> None:
        self._header.update_status(status, model)

    def _on_input_enabled(self, enabled: bool) -> None:
        self._prompt_input.set_enabled(enabled)

    def _on_confirmation_needed(self, tool_calls: list) -> None:
        """Show confirmation bar in the response panel (main thread)."""
        bar = self._response_panel.show_confirmation(tool_calls)
        bar.confirmed.connect(self._on_confirm_approved)
        bar.rejected.connect(self._on_confirm_rejected)

    def _on_confirm_approved(self) -> None:
        self._response_panel.remove_confirmation()
        self._confirm_result = True
        self._confirm_event.set()

    def _on_confirm_rejected(self) -> None:
        self._response_panel.remove_confirmation()
        self._confirm_result = False
        self._confirm_event.set()

    def _handle_confirmation(self, pending_tools: list[dict]) -> bool:
        """Called from worker thread — signals the UI and blocks until
        the user responds."""
        self._confirm_event.clear()
        self._sig_confirmation.emit(pending_tools)
        self._confirm_event.wait()
        return self._confirm_result

    # ── Actions ─────────────────────────────────────────────────────

    def _on_drag(self, x: int, y: int) -> None:
        self.move(x, y)

    def _on_send_message(self, text: str) -> None:
        self._response_panel.add_user_message(text)
        self._header.update_status(AgentStatus.THINKING, self._agent.model_name)
        self._prompt_input.set_enabled(False)

        thread = threading.Thread(
            target=self._send_message_worker, args=(text,), daemon=True
        )
        thread.start()

    def _send_message_worker(self, text: str) -> None:
        try:
            response = self._agent.send_message(text)
        except Exception as e:
            response = f"[Error] {e}"
        self._sig_input_enabled.emit(True)
        self._sig_status.emit(AgentStatus.ONLINE, self._agent.model_name)
        self._sig_response.emit(response)

    def _on_action(self, action: str) -> None:
        if action == "clipboard":
            clipboard = QApplication.clipboard()
            text = clipboard.text()
            if text:
                self._response_panel.add_system_message(
                    "Copiando texto del portapapeles..."
                )
                self._header.update_status(
                    AgentStatus.THINKING, self._agent.model_name
                )
                thread = threading.Thread(
                    target=self._clipboard_worker, args=(text,), daemon=True
                )
                thread.start()
            else:
                self._response_panel.add_system_message(
                    "El portapapeles esta vacio."
                )
        elif action == "chat_full":
            self._response_panel.add_system_message(
                "Modo chat completo: usa la entrada para interactuar con el agente."
            )
        elif action.startswith("add_monitor_folder:"):
            path = action.split(":", 1)[1]
            from app.services.folder_monitor import get_folder_monitor
            fm = get_folder_monitor()
            result = fm.add_folder(path)
            self._response_panel.add_system_message(f"📁 {result}")

    def _clipboard_worker(self, text: str) -> None:
        try:
            response = self._agent.summarize_clipboard(text)
        except Exception as e:
            response = f"[Error al resumir] {e}"
        self._sig_status.emit(AgentStatus.ONLINE, self._agent.model_name)
        self._sig_response.emit(response)

    def _on_function_selected(self, prompt_template: str) -> None:
        self._prompt_input._input.setText(prompt_template)
        self._prompt_input._input.setFocus()
        self._response_panel.add_system_message(
            "Función seleccionada. Rellena los datos en el cuadro de texto y presiona Enviar."
        )

    def _on_file_selected(self, path: str) -> None:
        self._response_panel.add_system_message(
            f"Analizando documento: {path.split('/')[-1].split(chr(92))[-1]}"
        )
        self._header.update_status(AgentStatus.THINKING, self._agent.model_name)

        thread = threading.Thread(
            target=self._file_worker, args=(path,), daemon=True
        )
        thread.start()

    def _on_file_dropped(self, filename: str, content: str) -> None:
        self._response_panel.add_system_message(
            f"Documento recibido: {filename}"
        )
        self._header.update_status(AgentStatus.THINKING, self._agent.model_name)

        thread = threading.Thread(
            target=self._doc_worker, args=(filename, content), daemon=True
        )
        thread.start()

    def _file_worker(self, path: str) -> None:
        from pathlib import Path
        try:
            p = Path(path)
            content = p.read_text(encoding="utf-8", errors="replace")
            response = self._agent.analyze_document(p.name, content)
        except Exception as e:
            response = f"[Error al analizar] {e}"
        self._sig_status.emit(AgentStatus.ONLINE, self._agent.model_name)
        self._sig_response.emit(response)

    def _doc_worker(self, filename: str, content: str) -> None:
        try:
            response = self._agent.analyze_document(filename, content)
        except Exception as e:
            response = f"[Error al analizar] {e}"
        self._sig_status.emit(AgentStatus.ONLINE, self._agent.model_name)
        self._sig_response.emit(response)

    # ── Status Check ────────────────────────────────────────────────

    def _check_status(self) -> None:
        thread = threading.Thread(
            target=self._status_worker, daemon=True
        )
        thread.start()

    def _status_worker(self) -> None:
        try:
            status = self._agent.check_status()
            model = self._agent.model_name
        except Exception:
            status = AgentStatus.OFFLINE
            model = ""
        self._sig_status.emit(status, model)

    def _check_folder_notifications(self) -> None:
        """Check for folder monitoring notifications in background thread."""
        thread = threading.Thread(
            target=self._folder_notification_worker, daemon=True
        )
        thread.start()

    def _folder_notification_worker(self) -> None:
        """Worker thread to check folder notifications."""
        try:
            notifications = self._agent.get_folder_notifications()
            for notif in notifications:
                self._sig_folder_notification.emit(
                    notif.filename,
                    notif.source_folder,
                    notif.destination_folder,
                )
        except Exception as e:
            logger.error("Error checking folder notifications: %s", e)

    def _on_folder_notification(self, filename: str, source: str, destination: str) -> None:
        """Display a folder monitoring notification in the response panel."""
        if destination == "ERROR":
            msg = f"❌ Error al procesar **{filename}**"
        else:
            msg = f"📁 **{filename}** clasificado → **ASIorga/{destination}**"
        self._response_panel.add_system_message(msg)
