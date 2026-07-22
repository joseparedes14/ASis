"""Windows DWM (Desktop Window Manager) API helpers for glassmorphism effects.

Uses ctypes to call undocumented/undocumented DWM functions for:
- Blur-behind-window (acrylic/frosted glass)
- Dark mode title bar
- Extended frame into client area
"""

from __future__ import annotations

import ctypes
import sys

if sys.platform != "win32":
    raise ImportError("dwm.py is only available on Windows")


dwmapi = ctypes.windll.dwmapi
user32 = ctypes.windll.user32

# ── Constants ────────────────────────────────────────────────────────────

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWA_NCRENDERING_POLICY = 2
NCRENDERING_POLICY_DISABLED = 2

DWMSBT_NONE = 0
DWMSBT_MAINWINDOW = 2
DWMSBT_TRANSIENTWINDOW = 3

ACCENT_DISABLED = 0
ACCENT_ENABLE_GRADIENT = 1
ACCENT_ENABLE_TRANSPARENTGRADIENT = 2
ACCENT_ENABLE_BLURBEHIND = 3
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4


# ── Structures ───────────────────────────────────────────────────────────


class MARGINS(ctypes.Structure):
    _fields_ = [
        ("cxLeftWidth", ctypes.c_int),
        ("cxRightWidth", ctypes.c_int),
        ("cyTopHeight", ctypes.c_int),
        ("cyBottomHeight", ctypes.c_int),
    ]


class WINDOWATTRIBUTE(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_int),
        ("AccentFlags", ctypes.c_int),
        ("GradientColor", ctypes.c_uint32),
        ("AnimationId", ctypes.c_uint32),
    ]


# ── Public API ───────────────────────────────────────────────────────────


def get_hwnd(widget) -> int | None:
    """Get the native HWND from a QWidget."""
    try:
        hwnd = int(widget.winId())
        return hwnd if hwnd else None
    except (AttributeError, TypeError, ValueError):
        return None


def enable_blur_behind(widget, acrylic: bool = True) -> bool:
    """Enable blur-behind effect on a widget's window.

    Args:
        widget: A QWidget instance.
        acrylic: If True, use acrylic blur (Windows 11). If False, use
                 simple blur (Windows 10 fallback).

    Returns:
        True if the effect was applied successfully.
    """
    hwnd = get_hwnd(widget)
    if hwnd is None:
        return False

    try:
        margins = MARGINS(-1, -1, -1, -1)
        hr = dwmapi.DwmExtendFrameIntoClientArea(
            hwnd, ctypes.byref(margins)
        )
        if hr != 0:
            return False

        accent = WINDOWATTRIBUTE()
        accent.AccentState = (
            ACCENT_ENABLE_ACRYLICBLURBEHIND if acrylic
            else ACCENT_ENABLE_BLURBEHIND
        )
        accent.GradientColor = 0x01000000  # near-opaque black tint

        hr2 = dwmapi.DwmSetWindowAttribute(
            hwnd,
            19,  # DWMWA_ACCENT_POLICY
            ctypes.byref(accent),
            ctypes.sizeof(accent),
        )
        return hr2 == 0
    except (OSError, ValueError):
        return False


def enable_dark_mode(widget, enabled: bool = True) -> bool:
    """Enable or disable dark mode title bar for a window."""
    hwnd = get_hwnd(widget)
    if hwnd is None:
        return False

    try:
        value = ctypes.c_int(1 if enabled else 0)
        hr = dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
        return hr == 0
    except (OSError, ValueError):
        return False


def disable_nc_rendering(widget) -> bool:
    """Disable non-client area rendering for custom title bar drawing."""
    hwnd = get_hwnd(widget)
    if hwnd is None:
        return False

    try:
        value = ctypes.c_int(NCRENDERING_POLICY_DISABLED)
        hr = dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_NCRENDERING_POLICY,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
        return hr == 0
    except (OSError, ValueError):
        return False


def extend_frame(widget, top: int = 0, left: int = 0, right: int = 0, bottom: int = 0) -> bool:
    """Extend the window frame into the client area with custom margins.

    Setting all margins to -1 enables blur behind the entire window.
    """
    hwnd = get_hwnd(widget)
    if hwnd is None:
        return False

    try:
        margins = MARGINS(left, right, top, bottom)
        hr = dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))
        return hr == 0
    except (OSError, ValueError):
        return False
