"""Consistent logs include the thread name for pipeline demonstrations."""

import logging


def configure_logging(level="INFO"):
    """Configure console logging; level is a validated symbolic log level.

    Args:
        level: Symbolic logging level such as INFO or ERROR.

    Returns:
        None; updates or displays application state.
    """
    logging.basicConfig(
        level=level, format="%(levelname)s [%(threadName)s] %(message)s",
        force=True)
