import logging

def setup_logger(log_file="training.log"):
    """
    Set up a logger to record training/validation information.
    
    Args:
        log_file: Path to the log file.
    
    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger("segmentation_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
