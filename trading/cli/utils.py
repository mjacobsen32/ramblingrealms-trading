import logging
from pathlib import Path

from rich.logging import RichHandler

FORMAT = "%(message)s"


def init_file_logger(log_level: str, log_dir: str):
    # File handler for DEBUG and above
    if log_level != "NOTSET":
        logger = logging.getLogger()
        file_handler = logging.FileHandler(Path(log_dir) / "rr_trading.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(file_handler)
        logging.info("File logger initialized with level: %s", log_level)


def init_logger(log_level: str):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    console_handler = RichHandler(markup=True)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(console_handler)

    # Prevent propagation to avoid default handler
    logger.propagate = False

    logging.info("Console logger initialized with level: %s", log_level)
