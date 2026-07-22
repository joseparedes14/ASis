"""Glassmorphism theme constants and QSS stylesheet generators.

Color palette inspired by Catppuccin Mocha, tuned for a muted, minimal
desktop widget aesthetic. All visual constants live here for easy
customization.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor

# ── Color Palette ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Colors:
    # Backgrounds
    bg_base: str = "rgba(18, 18, 28, 220)"
    bg_surface: str = "rgba(30, 30, 46, 180)"
    bg_elevated: str = "rgba(40, 40, 60, 200)"
    bg_input: str = "rgba(36, 36, 52, 200)"
    bg_hover: str = "rgba(50, 50, 70, 180)"
    bg_active: str = "rgba(58, 58, 82, 200)"

    # Borders
    border_subtle: str = "rgba(255, 255, 255, 0.06)"
    border_default: str = "rgba(255, 255, 255, 0.10)"
    border_strong: str = "rgba(255, 255, 255, 0.16)"

    # Text
    text_primary: str = "#cdd6f4"
    text_secondary: str = "#a6adc8"
    text_muted: str = "#6c7086"
    text_disabled: str = "#45475a"

    # Accents
    accent: str = "#89b4fa"
    accent_dim: str = "rgba(137, 180, 250, 0.15)"
    success: str = "#a6e3a1"
    success_dim: str = "rgba(166, 227, 161, 0.15)"
    warning: str = "#f9e2af"
    warning_dim: str = "rgba(249, 226, 175, 0.15)"
    error: str = "#f38ba8"
    error_dim: str = "rgba(243, 139, 168, 0.15)"

    # Shadows & overlays
    shadow: str = "rgba(0, 0, 0, 0.4)"
    overlay: str = "rgba(0, 0, 0, 0.3)"


COLORS = Colors()


# ── Qt Color Helpers ─────────────────────────────────────────────────────

QCOLOR_ACCENT = QColor(137, 180, 250)
QCOLOR_SUCCESS = QColor(166, 227, 161)
QCOLOR_WARNING = QColor(249, 226, 175)
QCOLOR_ERROR = QColor(243, 139, 168)
QCOLOR_TEXT = QColor(205, 214, 244)
QCOLOR_TEXT_SEC = QColor(166, 173, 200)
QCOLOR_BG = QColor(18, 18, 28, 220)
QCOLOR_BG_SURFACE = QColor(30, 30, 46, 180)


# ── Typography ───────────────────────────────────────────────────────────

FONT_FAMILY = "Segoe UI Variable, Segoe UI, Inter, sans-serif"


# ── Layout Constants ─────────────────────────────────────────────────────

COMPACT_SIZE = 52
EXPANDED_WIDTH = 420
EXPANDED_HEIGHT = 620
CORNER_RADIUS = 16
CORNER_RADIUS_SM = 10
CORNER_RADIUS_XS = 6
PADDING = 16
PADDING_SM = 10
PADDING_XS = 6
GAP = 10
GAP_SM = 6
WIDGET_MARGIN = 20


# ── QSS Generators ──────────────────────────────────────────────────────


def scrollbar_qss() -> str:
    return f"""
        QScrollBar:vertical {{
            background: transparent;
            width: 6px;
            margin: 4px 2px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS.border_strong};
            border-radius: 3px;
            min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLORS.text_muted};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
    """


def input_qss() -> str:
    return f"""
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
        QLineEdit:disabled {{
            color: {COLORS.text_disabled};
            border: 1px solid {COLORS.border_subtle};
        }}
    """


def action_button_qss(
    bg: str = "bg_surface",
    hover: str = "bg_hover",
    text: str = "text_primary",
    border: str = "border_subtle",
) -> str:
    bg_val = getattr(COLORS, bg)
    hover_val = getattr(COLORS, hover)
    text_val = getattr(COLORS, text)
    border_val = getattr(COLORS, border)
    return f"""
        QPushButton {{
            background-color: {bg_val};
            border: 1px solid {border_val};
            border-radius: {CORNER_RADIUS_XS}px;
            padding: 8px 12px;
            color: {text_val};
            font-family: {FONT_FAMILY};
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {hover_val};
        }}
        QPushButton:pressed {{
            background-color: {COLORS.bg_active};
        }}
        QPushButton:disabled {{
            color: {COLORS.text_disabled};
            border: 1px solid {COLORS.border_subtle};
        }}
    """


def role_button_qss(active: bool = False) -> str:
    if active:
        return f"""
            QPushButton {{
                background-color: {COLORS.accent_dim};
                border: 1px solid {COLORS.accent};
                border-radius: {CORNER_RADIUS_XS}px;
                padding: 5px 10px;
                color: {COLORS.accent};
                font-family: {FONT_FAMILY};
                font-size: 11px;
                font-weight: 600;
            }}
        """
    return f"""
        QPushButton {{
            background-color: {COLORS.bg_surface};
            border: 1px solid {COLORS.border_subtle};
            border-radius: {CORNER_RADIUS_XS}px;
            padding: 5px 10px;
            color: {COLORS.text_secondary};
            font-family: {FONT_FAMILY};
            font-size: 11px;
        }}
        QPushButton:hover {{
            background-color: {COLORS.bg_hover};
            color: {COLORS.text_primary};
        }}
    """


def global_stylesheet() -> str:
    return f"""
        * {{
            font-family: {FONT_FAMILY};
        }}
        QWidget {{
            background-color: transparent;
            color: {COLORS.text_primary};
        }}
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        QLabel {{
            background-color: transparent;
        }}
        {scrollbar_qss()}
    """
