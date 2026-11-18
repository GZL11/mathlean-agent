"""
Base logger protocol for Famou 2.0.

Defines the interface that all loggers must implement.
"""

from typing import Any, Dict, Optional, Protocol

from famou.core.data import Program


class Logger(Protocol):
    """
    Protocol for loggers.

    All logger implementations must implement these methods.
    This ensures consistent logging across all Famou components.

    Example:
        >>> logger = RichLogger(name="experiment")
        >>> logger.info("Starting iteration", iteration=1, experiment_id="exp_123")
        >>> logger.program(best_program)
    """

    def info(self, message: str, **context: Any) -> None:
        """
        Log info message with optional context.

        Args:
            message: The message to log
            **context: Additional context (e.g., iteration=1, experiment_id="exp_123")
        """
        ...

    def warning(self, message: str, **context: Any) -> None:
        """
        Log warning message with optional context.

        Args:
            message: The message to log
            **context: Additional context
        """
        ...

    def error(self, message: str, **context: Any) -> None:
        """
        Log error message with optional context.

        Args:
            message: The message to log
            **context: Additional context
        """
        ...

    def debug(self, message: str, **context: Any) -> None:
        """
        Log debug message with optional context.

        Args:
            message: The message to log
            **context: Additional context
        """
        ...

    def program(self, program: Program, title: Optional[str] = None) -> None:
        """
        Pretty-print a Program object.

        Args:
            program: The program to display
            title: Optional custom title for the display
        """
        ...

    def stats(self, stats: Dict[str, Any], title: str = "Statistics") -> None:
        """
        Display statistics in a formatted way.

        Args:
            stats: Dictionary of statistics to display
            title: Title for the stats display
        """
        ...
