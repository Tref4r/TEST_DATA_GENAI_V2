"""
Error handling and logging utilities for the vector database implementation.
"""
import logging
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union
import traceback

# Configure logging
def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file
        level: Logging level
        formatter: Optional custom formatter
    
    Returns:
        Configured logger instance
    """
    if formatter is None:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Base exception class for vector database operations
class VectorDBError(Exception):
    """Base exception class for vector database operations."""
    pass

class ConnectionError(VectorDBError):
    """Raised when connection to vector database fails."""
    pass

class QueryError(VectorDBError):
    """Raised when a query operation fails."""
    pass

class InsertionError(VectorDBError):
    """Raised when an insertion operation fails."""
    pass

class UpdateError(VectorDBError):
    """Raised when an update operation fails."""
    pass

class DeletionError(VectorDBError):
    """Raised when a deletion operation fails."""
    pass

# Error mapping dictionary
ERROR_MAPPING = {
    ConnectionError: "Failed to connect to vector database",
    QueryError: "Query operation failed",
    InsertionError: "Insertion operation failed",
    UpdateError: "Update operation failed",
    DeletionError: "Deletion operation failed"
}

def handle_exceptions(
    error_types: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    logger: Optional[logging.Logger] = None,
    default_message: str = "An error occurred during vector database operation"
) -> Callable:
    """
    Decorator for handling exceptions in vector database operations.
    
    Args:
        error_types: Exception type or tuple of types to catch
        logger: Optional logger instance
        default_message: Default error message
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except error_types as e:
                error_message = ERROR_MAPPING.get(
                    type(e), default_message
                )
                
                if logger:
                    logger.error(
                        f"{error_message}: {str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                
                raise VectorDBError(f"{error_message}: {str(e)}") from e
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Set up logger
    logger = setup_logger(
        "vector_db",
        "vector_db.log"
    )
    
    # Example function with error handling
    @handle_exceptions(error_types=(ConnectionError, QueryError), logger=logger)
    def example_function():
        raise ConnectionError("Failed to connect to database")
    
    try:
        example_function()
    except VectorDBError as e:
        print(f"Caught error: {e}")
