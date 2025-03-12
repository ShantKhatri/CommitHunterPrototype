"""
Logging Configuration Module

This module provides centralized logging configuration for the CommitHunter application.
"""

import logging
import logging.handlers
import os
from typing import Optional, Dict, Any
from pathlib import Path

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup application-wide logging configuration.
    
    Args:
        config: Logging configuration dictionary containing:
            - level: Logging level (INFO, DEBUG, etc.)
            - file: Log file path
            - format: Log message format
            - max_size: Maximum size of log file before rotation
            - backup_count: Number of backup log files to keep
    """
    log_file = Path(config.get('file', 'commit_hunter.log'))
    log_dir = log_file.parent
    os.makedirs(log_dir, exist_ok=True)
    
    log_level = getattr(logging, config.get('level', 'INFO').upper())
    log_format = config.get('format', 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    max_size = config.get('max_size', 10 * 1024 * 1024)  # 10 MB default
    backup_count = config.get('backup_count', 5)
    
    formatter = logging.Formatter(log_format)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    root_logger.info("Logging system initialized")
    root_logger.debug(f"Log level: {logging.getLevelName(log_level)}")
    root_logger.debug(f"Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Name of the component requesting the logger
        
    Returns:
        Logger instance configured according to application settings
    """
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds contextual information to log messages.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict] = None):
        """
        Initialize the logger adapter.
        
        Args:
            logger: The logger instance to adapt
            extra: Dictionary of extra contextual information
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict) -> tuple:
        """
        Process the log message and keyword arguments.
        
        Args:
            msg: The log message
            kwargs: Keyword arguments for the logging call
            
        Returns:
            Tuple of (modified message, modified keyword arguments)
        """
        context = []
        for key, value in self.extra.items():
            context.append(f"{key}={value}")
        
        if context:
            msg = f"[{' '.join(context)}] {msg}"
        
        return msg, kwargs

def create_context_logger(name: str, **context) -> LoggerAdapter:
    """
    Create a logger that automatically includes contextual information.
    
    Args:
        name: Name of the component requesting the logger
        **context: Keyword arguments defining the context
        
    Returns:
        LoggerAdapter instance that includes the specified context
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)

class PerformanceLogger:
    """
    Special logger for performance-related metrics and events.
    """
    
    def __init__(self, name: str):
        """
        Initialize the performance logger.
        
        Args:
            name: Name of the component using the performance logger
        """
        self.logger = get_logger(f"performance.{name}")
        self.context = {}
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", 
                  **additional_context) -> None:
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (optional)
            **additional_context: Additional contextual information
        """
        context = {**self.context, **additional_context}
        metric_str = f"{metric_name}={value}"
        if unit:
            metric_str += f"{unit}"
            
        context_str = " ".join(f"{k}={v}" for k, v in context.items())
        if context_str:
            self.logger.info(f"{metric_str} [{context_str}]")
        else:
            self.logger.info(metric_str)
    
    def set_context(self, **context) -> None:
        """
        Set persistent context for all subsequent log entries.
        
        Args:
            **context: Keyword arguments defining the context
        """
        self.context = context

def log_exception(logger: logging.Logger, exc: Exception, 
                 message: str = "An error occurred") -> None:
    """
    Helper function to log an exception with consistent formatting.
    
    Args:
        logger: Logger instance to use
        exc: Exception to log
        message: Optional message to include with the exception
    """
    logger.exception(f"{message}: {str(exc)}", exc_info=exc)