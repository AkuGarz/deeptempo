"""Logging setup for DeepTempo.

Uses Python's standard logging module with a simple format.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Sets up basic formatting if no handlers exist yet.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
