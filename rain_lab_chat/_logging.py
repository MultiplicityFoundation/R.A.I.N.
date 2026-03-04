"""Centralized logging configuration for rain_lab_chat.

All modules in the package should use ``get_logger(__name__)`` instead of
bare ``print()`` calls.  The root ``rain_lab_chat`` logger writes to stderr
by default so it never contaminates stdout pipelines.

Users can control verbosity via the standard ``logging`` API::

    import logging
    logging.getLogger("rain_lab_chat").setLevel(logging.DEBUG)

Or via the ``RAIN_LOG_LEVEL`` environment variable (DEBUG, INFO, WARNING, …).
"""

from __future__ import annotations

import logging
import os
import sys

_PACKAGE_LOGGER_NAME = "rain_lab_chat"

# One-time setup guard
_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger(_PACKAGE_LOGGER_NAME)

    # Don't add handlers if the application already configured this logger
    if root.handlers:
        return

    level_name = os.environ.get("RAIN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``rain_lab_chat`` namespace."""
    _configure_once()
    return logging.getLogger(name)
