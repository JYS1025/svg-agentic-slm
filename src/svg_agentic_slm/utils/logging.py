"""Logging setup utilities.

Provides a centralized logging configuration so all modules
use consistent formatting and log levels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """Configure and return the root logger for the project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If provided, logs are also
                  written to this file.
        log_format: Optional custom log format string.

    Returns:
        Configured root logger.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("svg_agentic_slm")
    logger.setLevel(numeric_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the project namespace.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A logger instance.
    """
    return logging.getLogger(f"svg_agentic_slm.{name}")
