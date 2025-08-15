import logging
from pathlib import Path

from rich.logging import RichHandler

FORMAT = "%(message)s"

logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("plotly").setLevel(logging.WARNING)
logging.getLogger("rich").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("pandas").setLevel(logging.WARNING)
logging.getLogger("vectorbt").setLevel(logging.WARNING)
logging.getLogger("chrome").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)


def init_file_logger(log_level: str, log_dir: str):
    # File handler for DEBUG and above
    logger = logging.getLogger()
    if log_level != "NOTSET":
        file_handler = logging.FileHandler(Path(log_dir) / "rr_trading.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(file_handler)
        logging.info("File logger initialized with level: %s", log_level)


def init_logger(log_level: str, log_file_level: str):
    logger = logging.getLogger()
    logger.handlers.clear()

    console_level = getattr(logging, log_level, logging.NOTSET)
    file_level = getattr(logging, log_file_level, logging.NOTSET)

    logger.setLevel(min(console_level, file_level))
    console_handler = RichHandler(markup=True)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(console_handler)

    # Prevent propagation to avoid default handler
    logger.propagate = False

    logging.info(
        "Root logger initialized with level: %s", logging.getLevelName(logger.level)
    )
    logging.info("Console logger initialized with level: %s", log_level)
