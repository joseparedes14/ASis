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
