"""Structured logging configuration for Nido."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def get_log_dir() -> Path:
    """Return XDG state directory for Nido logs."""
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        base = Path(xdg_state_home)
    else:
        base = Path.home() / ".local" / "state"
    log_dir = base / "nido"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up file and console loggers.

    Audio data is never passed to this logger.
    """
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("nido")
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler in XDG state directory
    try:
        log_file = get_log_dir() / "nido.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logger: {e}")

    return logger


def get_logger(name: str = "nido") -> logging.Logger:
    """Get named logger under the nido namespace."""
    return logging.getLogger(name)
