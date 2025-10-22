"""
Centralized logging module for benchmarking scripts.

This module provides a unified logging interface that:
- Logs to both file and console
- Maintains consistent log formatting
- Provides different log levels for different types of information
- Automatically creates log directories and handles log rotation
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class BenchmarkLogger:
    """
    Centralized logger for benchmarking operations.
    
    Features:
    - Dual output: file and console
    - Automatic log file naming with timestamps
    - Configurable log levels
    - Structured logging for different components
    """
    
    def __init__(self,
                 name: str = "benchmark",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG):
        """
        Initialize the benchmark logger.

        Args:
            name: Logger name (used in log file naming)
            log_dir: Directory to store log files
            log_level: Overall logging level
            console_level: Console output level
            file_level: File output level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.console_level = console_level
        self.file_level = file_level

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.logger.setLevel(log_level)

        # Prevent propagation to parent loggers to avoid duplicate logging
        self.logger.propagate = False

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        self._setup_formatters()

        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler()

        # Log initialization
        self.logger.info(f"Benchmark logger initialized: {name}")
        self.logger.info(f"Log directory: {self.log_dir.absolute()}")
        self.logger.info(f"Console level: {logging.getLevelName(console_level)}")
        self.logger.info(f"File level: {logging.getLevelName(file_level)}")
    
    def _setup_formatters(self):
        """Setup log formatters for different outputs."""
        # Console formatter - more compact
        self.console_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File formatter - more detailed
        self.file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)-30s | %(levelname)-8s | %(funcName)-20s:%(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _setup_console_handler(self):
        """Setup console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file handler with timestamped filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{self.name}_{timestamp}.log"
        log_filepath = self.log_dir / log_filename
        
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_filepath = log_filepath
        self.logger.info(f"Log file created: {log_filepath}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback."""
        self.logger.exception(message)
    
    def log_function_entry(self, func_name: str, **kwargs):
        """Log function entry with parameters."""
        params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.debug(f"ENTER {func_name}({params_str})")
    
    def log_function_exit(self, func_name: str, result=None):
        """Log function exit with result."""
        if result is not None:
            self.logger.debug(f"EXIT {func_name} -> {result}")
        else:
            self.logger.debug(f"EXIT {func_name}")
    
    def log_benchmark_start(self, model_name: str, dataset: str, config: dict):
        """Log benchmark start with configuration."""
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING BENCHMARK")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Configuration: {config}")
        self.logger.info("=" * 80)
    
    def log_benchmark_end(self, model_name: str, dataset: str, results: dict):
        """Log benchmark completion with results."""
        self.logger.info("=" * 80)
        self.logger.info(f"BENCHMARK COMPLETED")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Results: {results}")
        self.logger.info("=" * 80)
    
    def log_engine_creation(self, engine_type: str, model_id: str, **kwargs):
        """Log engine creation details."""
        self.logger.info(f"Creating {engine_type} engine for model: {model_id}")
        for key, value in kwargs.items():
            self.logger.debug(f"  {key}: {value}")
    
    def log_gpu_memory(self, stage: str, allocated_gb: float, reserved_gb: float):
        """Log GPU memory usage."""
        self.logger.info(f"GPU Memory [{stage}]: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")
    
    def log_calibration_progress(self, current: int, total: int, prefill_tokens: int, generation_tokens: int, flops: float):
        """Log calibration progress."""
        self.logger.info(f"Calibration [{current}/{total}]: P={prefill_tokens}, G={generation_tokens}, FLOPs={flops/1e12:.2f}T")
    
    def log_batch_processing(self, batch_idx: int, batch_size: int, total_examples: int):
        """Log batch processing progress."""
        self.logger.debug(f"Processing batch {batch_idx}: {batch_size} examples (total: {total_examples})")
    
    def log_metrics(self, accuracy: float, avg_gen_tokens: float, latency_ms: float, flops_info: str):
        """Log benchmark metrics."""
        self.logger.info(f"Metrics: acc={accuracy:.3f}, avg_gen_tokens={avg_gen_tokens:.2f}, latency_ms={latency_ms:.2f}, {flops_info}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log error with optional exception details."""
        self.logger.error(f"ERROR: {error_msg}")
        if exception:
            self.logger.exception(f"Exception details: {exception}")
    
    def close(self):
        """Close the logger and all handlers."""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.info("Logger closed")


# Global logger instance
_global_logger: Optional[BenchmarkLogger] = None


def get_logger(name: str = "benchmark", **kwargs) -> BenchmarkLogger:
    """
    Get or create a global logger instance.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for BenchmarkLogger
        
    Returns:
        BenchmarkLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = BenchmarkLogger(name=name, **kwargs)
    
    return _global_logger


def setup_logging(name: str = "benchmark",
                  log_dir: str = "logs",
                  log_level: int = logging.INFO,
                  console_level: int = logging.INFO,
                  file_level: int = logging.DEBUG) -> BenchmarkLogger:
    """
    Setup logging for the application.

    Args:
        name: Logger name
        log_dir: Directory to store log files
        log_level: Overall logging level
        console_level: Console output level
        file_level: File output level

    Returns:
        Configured BenchmarkLogger instance
    """
    return get_logger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        console_level=console_level,
        file_level=file_level
    )


# Convenience functions for quick logging
def log_info(message: str):
    """Log info message using global logger."""
    logger = get_logger()
    logger.info(message)


def log_debug(message: str):
    """Log debug message using global logger."""
    logger = get_logger()
    logger.debug(message)


def log_warning(message: str):
    """Log warning message using global logger."""
    logger = get_logger()
    logger.warning(message)


def log_error(message: str, exception: Optional[Exception] = None):
    """Log error message using global logger."""
    logger = get_logger()
    logger.log_error(message, exception)


def log_critical(message: str):
    """Log critical message using global logger."""
    logger = get_logger()
    logger.critical(message)
