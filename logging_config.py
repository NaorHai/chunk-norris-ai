import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Set up logging configuration for the application.
    Creates a logs directory if it doesn't exist and configures both file and console logging.
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    log_file = os.path.join(log_dir, 'app.log')
    
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        return None
    
    # Create a logger
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # Create and configure file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Test logging
        logger.info("Logging system initialized")
        logger.info(f"Log file: {os.path.abspath(log_file)}")
        
        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None 