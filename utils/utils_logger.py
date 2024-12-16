import logging

# def setup_logger(log_file="training.log"):
#     """
#     Set up a logger to record training/validation information.
    
#     Args:
#         log_file: Path to the log file.
    
#     Returns:
#         logger: Configured logger instance.
#     """
#     logger = logging.getLogger("segmentation_logger")
#     logger.setLevel(logging.INFO)

#     # File handler
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setLevel(logging.INFO)

#     # Console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)

#     # Formatter
#     formatter = logging.Formatter("%(asctime)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)

#     # Add handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     return logger




def setup_logger(model_name, log_dir="logs"):
    """
    Set up a logger to record training/validation information with unique filenames.

    Args:
        model_name: Name of the model (used for unique log files).
        log_dir: Directory where logs are stored.

    Returns:
        logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
    log_file = os.path.join(log_dir, f"{model_name}_training.log")

    logger = logging.getLogger(model_name)
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

    # Add handlers
    if not logger.hasHandlers():  # Prevent duplicate handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
