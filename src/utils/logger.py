"""Logging configuration and utilities"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Global console for rich output
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format: Log message format
        use_rich: Whether to use rich formatting for console output
    """
    # Default format
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create handlers
    handlers = []
    
    # Console handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            enable_link_path=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format))
    
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.root.handlers = []
    logging.root.setLevel(getattr(logging, level.upper()))
    for handler in handlers:
        logging.root.addHandler(handler)
    
    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TimedLogger:
    """Context manager for timing operations"""
    
    def __init__(self, logger: logging.Logger, message: str, level: str = "INFO"):
        self.logger = logger
        self.message = message
        self.level = getattr(logging, level.upper())
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"{self.message} - Starting...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.log(
                self.level,
                f"{self.message} - Completed in {elapsed.total_seconds():.2f}s"
            )
        else:
            self.logger.error(
                f"{self.message} - Failed after {elapsed.total_seconds():.2f}s: {exc_val}"
            )


def log_model_info(logger: logging.Logger, model_name: str, **kwargs):
    """Log model information in a structured format"""
    logger.info(f"Model: {model_name}")
    for key, value in kwargs.items():
        logger.info(f"  {key}: {value}")


def log_performance_metrics(logger: logging.Logger, metrics: dict):
    """Log performance metrics in a structured format"""
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")