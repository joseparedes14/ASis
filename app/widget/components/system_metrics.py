"""System metrics component — minimal RAM/VRAM usage monitor."""

from __future__ import annotations

import logging
import subprocess
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.widget.styles import COLORS, FONT_FAMILY, PADDING_SM

logger = logging.getLogger("asis.widget.metrics")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MetricBar(QWidget):
    """Custom painted progress bar for a single metric."""

    def __init__(
        self,
        label: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._value = 0.0
        self._detail = ""
        self.setFixedHeight(28)

    def update_value(self, percent: float, detail: str = "") -> None:
        self._value = min(100.0, max(0.0, percent))
        self._detail = detail
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bar_h = 6
        bar_y = 2
        text_y = bar_y + bar_h + 3

        # Label
        p.setPen(QColor(COLORS.text_muted))
        p.setFont(p.font())
        p.drawText(0, text_y, w, h - text_y, Qt.AlignmentFlag.AlignLeft, self._label)

        # Percentage
        percent_text = f"{self._value:.0f}%"
        p.setPen(QColor(COLORS.text_secondary))
        p.drawText(0, text_y, w, h - text_y, Qt.AlignmentFlag.AlignRight, percent_text)

        # Detail (GB)
        if self._detail:
            p.setPen(QColor(COLORS.text_disabled))
            p.drawText(0, text_y, w - 45, h - text_y, Qt.AlignmentFlag.AlignRight, self._detail)

        # Bar background
        bar_rect_x = 0
        bar_rect_w = w - 50
        bg_color = QColor(COLORS.bg_surface)
        p.setBrush(bg_color)
        p.setPen(Qt.PenStyle.NoPen)
        bar_path = QPainterPath()
        bar_path.addRoundedRect(
            float(bar_rect_x), float(bar_y), float(bar_rect_w), float(bar_h), 3, 3
        )
        p.drawPath(bar_path)

        # Bar fill
        if self._value > 0:
            fill_w = bar_rect_w * (self._value / 100.0)
            if self._value > 85:
                fill_color = QColor(COLORS.error)
            elif self._value > 65:
                fill_color = QColor(COLORS.warning)
            else:
                fill_color = QColor(COLORS.accent)
            p.setBrush(fill_color)
            fill_path = QPainterPath()
            fill_path.addRoundedRect(
                float(bar_rect_x), float(bar_y), float(fill_w), float(bar_h), 3, 3
            )
            p.drawPath(fill_path)

        p.end()


class SystemMetrics(QWidget):
    """Minimal RAM and VRAM usage display with periodic refresh."""

    def __init__(
        self,
        refresh_ms: int = 3000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._has_nvidia = False
        self._setup_ui()
        self._check_nvidia()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(refresh_ms)
        self._refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PADDING_SM, 4, PADDING_SM, PADDING_SM)
        layout.setSpacing(6)

        section_label = QLabel("Sistema")
        section_label.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 600;
            color: {COLORS.text_muted};
            font-family: {FONT_FAMILY};
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(section_label)

        self._ram_bar = MetricBar("RAM")
        self._vram_bar = MetricBar("VRAM")
        layout.addWidget(self._ram_bar)
        layout.addWidget(self._vram_bar)

        if not HAS_PSUTIL:
            self._ram_bar.update_value(0, "psutil no instalado")

    def _check_nvidia(self) -> None:
        if sys.platform != "win32":
            self._has_nvidia = False
            return
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            self._has_nvidia = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._has_nvidia = False

        if not self._has_nvidia:
            self._vram_bar.update_value(0, "No disponible")

    def _refresh(self) -> None:
        self._refresh_ram()
        if self._has_nvidia:
            self._refresh_vram()

    def _refresh_ram(self) -> None:
        if not HAS_PSUTIL:
            return
        try:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            self._ram_bar.update_value(
                mem.percent,
                f"{used_gb:.1f}/{total_gb:.1f} GB",
            )
        except Exception as e:
            logger.debug("Failed to read RAM: %s", e)

    def _refresh_vram(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    gpu_util = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    self._vram_bar.update_value(
                        gpu_util,
                        f"{mem_used / 1024:.1f}/{mem_total / 1024:.1f} GB",
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
            logger.debug("Failed to read VRAM: %s", e)
