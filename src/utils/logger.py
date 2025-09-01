import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

LOG_FILE = "logs/app.log"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
