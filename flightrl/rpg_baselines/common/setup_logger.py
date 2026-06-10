import os
from logging.handlers import RotatingFileHandler
import logging

def setup_logging(
    name: str = "logger1", 
    filename: str = "example.log", 
    max_bytes: int = 5*1024*1024,
    backup_count: int = 5
     ):
    """Setup + return logger to a file with rotation."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    
    if not logger.handlers:
        handler = RotatingFileHandler(
            os.path.join(filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(handler)
    return logger