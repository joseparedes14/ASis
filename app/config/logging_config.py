"""
Logging configuration for the ASis application.

Provides structured, professional logging with both console
and file output support. Uses Rich for colorized console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure application-wide logging.

    Sets up a root logger with:
    - Console handler with colored output (via Rich if available).
    - Optional file handler for persistent logs.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If None, logs only to console.

    Returns:
        Configured root logger for the application.
    """
    logger = logging.getLogger("asis")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on re-initialization
    logger.handlers.clear()

    # ── Console Handler ─────────────────────────────────────────────
    console_formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ── File Handler (optional) ─────────────────────────────────────
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(funcName)s:%(lineno)d │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging initialized — level=%s, file=%s", level, log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Child logger under the 'asis' namespace.
    """
    return logging.getLogger(f"asis.{name}")
