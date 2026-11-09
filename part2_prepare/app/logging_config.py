"""Consistent logs include the thread name for pipeline demonstrations."""

import logging


def configure_logging(level="INFO"):
    """Configure console logging; level is a validated symbolic log level."""
    logging.basicConfig(
        level=level, format="%(levelname)s [%(threadName)s] %(message)s",
        force=True)
