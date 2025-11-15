import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str, log_dir="logs", filename="pipeline.log"):

    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, filename)

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicates if logger already exists
    if logger.handlers:
        return logger

    # ---- File Handler (Writes to file) ----
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # ---- Console Handler (Prints to terminal) ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger