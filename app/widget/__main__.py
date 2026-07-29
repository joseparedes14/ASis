"""ASis Widget entry point.

Launches the glassmorphism desktop dashboard for the ASis AI agent.

Usage:
    python -m app.widget

Global keyboard shortcut:
    Ctrl+Shift+A — Toggle widget visibility
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication

from app.config.logging_config import setup_logging
from app.widget.dashboard import DashboardWidget


def main() -> None:
    # Inicializar logging antes de cualquier otra cosa
    log_file = Path("data/logs/widget.log")
    setup_logging(level="INFO", log_file=log_file)

    # Habilitar DPI awareness per-monitor antes de crear QApplication
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    app.setStyleSheet("""
        * {
            font-family: "Segoe UI Variable", "Segoe UI", "Inter", sans-serif;
        }
    """)

    dashboard = DashboardWidget()
    dashboard.show()

    shortcut = QShortcut(QKeySequence("Ctrl+Shift+A"), dashboard)
    shortcut.activated.connect(dashboard.toggle_signal.emit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
