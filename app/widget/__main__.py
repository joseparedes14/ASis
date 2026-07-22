"""ASis Widget entry point.

Launches the glassmorphism desktop dashboard for the ASis AI agent.

Usage:
    python -m app.widget

Global keyboard shortcut:
    Ctrl+Shift+A — Toggle widget visibility
"""

from __future__ import annotations

import sys

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication

from app.widget.dashboard import DashboardWidget


def main() -> None:
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
