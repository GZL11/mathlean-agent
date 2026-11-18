"""
LocalLogger - Dual-output logging for Famou experiments.

Provides both:
1. Rich console output (colored, formatted, human-readable)
2. Plain text .log file (human-readable, same as console)
3. Structured JSONL file (machine-readable, for analysis)

Designed for local experiments with smart context-aware formatting.
"""

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

from famou.core.data import Program


class LocalLogger:
    """
    Multi-output logger for local experiments.

    Logs to:
    - Console: Rich-formatted, colored, human-readable
    - experiment.log: Plain text file (human-readable, same as console)
    - experiment.jsonl: Structured JSONL (machine-readable, for analysis)

    Features smart context-aware formatting:
    - Detects iteration/island context and formats concisely
    - Multi-line output for complex information
    - Minimal emoji usage (🚀 for start, ✅ for completion)
    - Full datetime timestamps

    Example:
        >>> logger = LocalLogger(
        ...     name="circle_packing",
        ...     experiment_id="exp_123",
        ...     log_dir="./famou_data/exp_123",
        ...     level="INFO"
        ... )
        >>> logger.info("Starting evolution", iteration=1, island_id=0, max_iterations=100)

        # Console output:
        2026-01-12 12:34:56.789 INFO  [Island 0 | Iter 1/100] Starting evolution

        # experiment.log (same as console):
        2026-01-12 12:34:56.789 INFO  [Island 0 | Iter 1/100] Starting evolution

        # experiment.jsonl (structured):
        {"timestamp": 1704720896.789, "level": "INFO", "logger": "circle_packing",
         "experiment_id": "exp_123", "message": "Starting evolution",
         "iteration": 1, "island_id": 0, "max_iterations": 100}
    """

    def __init__(
        self,
        name: str = "famou",
        experiment_id: Optional[str] = None,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        console_enabled: bool = True,
        file_enabled: bool = True,
        jsonl_enabled: bool = True,
    ):
        """
        Initialize LocalLogger with multi-output support.

        Args:
            name: Logger name (displayed in logs)
            experiment_id: Experiment ID (added to all log entries)
            log_dir: Directory to store log files (default: ./logs)
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
            console_enabled: Enable console output (default: True)
            file_enabled: Enable plain text .log file output (default: True)
            jsonl_enabled: Enable structured .jsonl file output (default: True)
        """
        self.name = name
        self.experiment_id = experiment_id
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.jsonl_enabled = jsonl_enabled

        # Rich console for formatted output
        self.console = Console() if console_enabled else None

        # Log level
        self.level = getattr(logging, level.upper(), logging.INFO)

        # Setup file logging
        self.text_log_file = None
        self.jsonl_log_file = None

        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Plain text log file (human-readable)
            if file_enabled:
                self.text_log_file = log_path / "experiment.log"
                self._write_text_header()

            # JSONL log file (machine-readable)
            if jsonl_enabled:
                self.jsonl_log_file = log_path / "experiment.jsonl"
                self._write_jsonl_header()

    def _write_text_header(self) -> None:
        """Write header to plain text log file."""
        if not self.text_log_file:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        header_lines = [
            "=" * 80,
            f"Famou Evolution Log",
            f"Logger: {self.name}",
            f"Experiment ID: {self.experiment_id}",
            f"Log Level: {logging.getLevelName(self.level)}",
            f"Started: {timestamp}",
            "=" * 80,
            "",
        ]

        with open(self.text_log_file, "a") as f:
            f.write("\n".join(header_lines) + "\n")

    def _write_jsonl_header(self) -> None:
        """Write header to JSONL log file."""
        if not self.jsonl_log_file:
            return

        header = {
            "event": "log_start",
            "timestamp": time.time(),
            "logger": self.name,
            "experiment_id": self.experiment_id,
            "level": logging.getLevelName(self.level),
        }

        with open(self.jsonl_log_file, "a") as f:
            f.write(json.dumps(header) + "\n")

    def info(self, message: str, **context: Any) -> None:
        """
        Log info-level message.

        Args:
            message: The message to log
            **context: Additional context (e.g., iteration=1, score=0.85)
        """
        if self.level <= logging.INFO:
            self._log("INFO", message, context, "blue")

    def warning(self, message: str, **context: Any) -> None:
        """Log warning-level message."""
        if self.level <= logging.WARNING:
            self._log("WARNING", message, context, "yellow")

    def error(self, message: str, **context: Any) -> None:
        """Log error-level message."""
        if self.level <= logging.ERROR:
            self._log("ERROR", message, context, "red")

    def debug(self, message: str, **context: Any) -> None:
        """Log debug-level message."""
        if self.level <= logging.DEBUG:
            self._log("DEBUG", message, context, "dim")

    def _log(self, level: str, message: str, context: Dict[str, Any], color: str) -> None:
        """
        Internal logging method.

        Logs to console (Rich formatted), plain text file, and JSONL file.
        """
        timestamp = time.time()

        # Format message (same for console and text file)
        formatted_message = self._format_console_message(message, context, color, timestamp)

        # Console output (Rich formatted)
        if self.console_enabled and self.console:
            self.console.print(formatted_message)

        # Plain text file output (same as console but without Rich markup)
        if self.file_enabled and self.text_log_file:
            self._write_text_log(formatted_message)

        # JSONL file output (structured JSON)
        if self.jsonl_enabled and self.jsonl_log_file:
            self._write_jsonl_log(level, message, context, timestamp)

    def _format_iteration_header(
        self, iteration: int, island_id: Optional[int], max_iterations: Optional[int]
    ) -> str:
        """Format iteration header consistently."""
        if max_iterations is None:
            if island_id is not None:
                return f"[Island {island_id} | Iter {iteration}]"
            return f"[Iter {iteration}]"

        if island_id is not None:
            return f"[Island {island_id} | Iter {iteration}/{max_iterations}]"
        return f"[Iter {iteration}/{max_iterations}]"

    def _format_program_summary(self, program: Program) -> str:
        """
        Format program as concise summary (no code).

        Includes: program_id, generation, iteration, island, score, metrics, error_info
        """
        parts = [f"ID={program.id}"]

        if program.generation is not None:
            parts.append(f"gen={program.generation}")

        if program.iteration is not None:
            parts.append(f"iter={program.iteration}")

        # Extract island from metadata
        island_id = program.meta.get("assigned_island")
        if island_id is not None:
            parts.append(f"island={island_id}")

        if program.combined_score is not None:
            parts.append(f"success={program.combined_score:.4f}")

        if program.search_score is not None:
            parts.append(f"search={program.search_score:.4f}")

        if program.validity is not None:
            parts.append(f"validity={program.validity:.2f}")

        if program.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in program.metrics.items()
            )
            parts.append(f"metrics=[{metrics_str}]")

        if program.error_info:
            # Truncate long errors
            error_preview = (
                program.error_info[:100] + "..."
                if len(program.error_info) > 100
                else program.error_info
            )
            parts.append(f"error={error_preview}")

        return ", ".join(parts)

    def _format_console_message(
        self, message: str, context: Dict[str, Any], color: str, timestamp: float
    ) -> str:
        """
        Format message for console output with smart context-aware formatting.

        Detects iteration/island context and formats concisely.
        Returns Rich-formatted string with full datetime, level, and context.
        """
        # Format timestamp (full datetime)
        dt = datetime.datetime.fromtimestamp(timestamp)
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Get log level name
        level_name = {
            "blue": "INFO",
            "yellow": "WARNING",
            "red": "ERROR",
            "dim": "DEBUG",
        }.get(color, "INFO")

        # Build base prefix: timestamp + level
        prefix = f"[dim]{ts_str}[/dim] [{color}]{level_name:7s}[/{color}]"

        # Make a copy of context to avoid modifying original
        ctx = dict(context)

        # Detect and format iteration context
        iteration_header = ""
        if "iteration" in ctx:
            iteration = ctx.pop("iteration")
            island_id = ctx.pop("island_id", None)
            max_iterations = ctx.pop("max_iterations", None)
            iteration_header = self._format_iteration_header(iteration, island_id, max_iterations)

        # Format remaining context with type-aware coloring
        ctx_parts = []
        if ctx:
            for k, v in sorted(ctx.items()):
                if isinstance(v, float):
                    formatted = f"{k}=[cyan]{v:.4f}[/cyan]"
                elif isinstance(v, int):
                    formatted = f"{k}=[yellow]{v}[/yellow]"
                elif isinstance(v, str) and len(v) > 40:
                    formatted = f"{k}=[green]{v[:37]}...[/green]"
                else:
                    formatted = f"{k}=[green]{v}[/green]"
                ctx_parts.append(formatted)

        # Assemble final message
        parts = [prefix]
        if iteration_header:
            parts.append(iteration_header)
        if ctx_parts:
            parts.append(f"[{' '.join(ctx_parts)}]")
        parts.append(message)

        return " ".join(parts)

    def _write_text_log(self, formatted_message: str) -> None:
        """
        Write formatted message to plain text log file.

        Strips Rich markup before writing.
        """
        if not self.text_log_file:
            return

        try:
            # Strip Rich markup tags only (not regular brackets)
            # Pattern matches: [tag] or [/tag] where tag is alphabetic
            import re
            plain_text = re.sub(r"\[/?[a-zA-Z_]+\]", "", formatted_message)

            with open(self.text_log_file, "a") as f:
                f.write(plain_text + "\n")
        except Exception as e:
            # Fallback: print error to console
            if self.console:
                self.console.print(f"[red]Failed to write text log: {e}[/red]")

    def _write_jsonl_log(
        self, level: str, message: str, context: Dict[str, Any], timestamp: float
    ) -> None:
        """
        Write structured log entry to JSONL file.

        Format: One JSON object per line, easy to parse with jq, pandas, etc.
        """
        if not self.jsonl_log_file:
            return

        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
        }

        # Add experiment_id if set
        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        # Flatten context into log entry
        log_entry.update(context)

        # Write to file (append mode)
        try:
            with open(self.jsonl_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            # Fallback: print error to console
            if self.console:
                self.console.print(f"[red]Failed to write JSONL log: {e}[/red]")

    def separator(self, char: str = "-", length: int = 80) -> None:
        """
        Log a visual separator line.

        Args:
            char: Character to use for separator (default: -)
            length: Length of separator line (default: 80)
        """
        line = char * length
        # Write directly to console and files without timestamp/level
        if self.console_enabled and self.console:
            self.console.print(f"[dim]{line}[/dim]")

        if self.file_enabled and self.text_log_file:
            with open(self.text_log_file, "a") as f:
                f.write(line + "\n")

    def program(self, program: Program, title: Optional[str] = None) -> None:
        """
        Log program metadata (no code snippets).

        Logs concise program summary with ID, generation, iteration, island,
        score, metrics, and error_info (if present).

        Uses multi-line format for readability.
        """
        timestamp = time.time()

        # Format program summary
        summary = self._format_program_summary(program)

        # Determine color based on score and errors
        if program.error_info:
            color = "red"
            level = "ERROR"
        elif program.combined_score is not None:
            if program.combined_score >= 1.0:
                color = "green"
                level = "INFO"
            elif (program.search_score or 0.0) >= 0.3:
                color = "yellow"
                level = "WARNING"
            else:
                color = "red"
                level = "WARNING"
        else:
            color = "blue"
            level = "INFO"

        # Build message with title
        title_str = title or "Program"
        message = f"{title_str}: {summary}"

        # Console output
        if self.console_enabled and self.console:
            dt = datetime.datetime.fromtimestamp(timestamp)
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            formatted = f"[dim]{ts_str}[/dim] [{color}]{level:7s}[/{color}] {message}"
            self.console.print(formatted)

        # Plain text file output
        if self.file_enabled and self.text_log_file:
            import re
            plain_text = re.sub(r"\[/?[^\]]+\]", "",
                               f"{datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {level:7s} {message}")
            self._write_text_log(plain_text)

        # JSONL file output
        if self.jsonl_enabled and self.jsonl_log_file:
            self._log_program_jsonl(program, title)

    def _log_program_jsonl(self, program: Program, title: Optional[str] = None) -> None:
        """Log program details to JSONL file."""
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "event": "program_display",
            "logger": self.name,
            "title": title,
            "program_id": program.id,
            "generation": program.generation,
            "iteration": program.iteration,
            "combined_score": program.combined_score,
            "search_score": program.search_score,
            "validity": program.validity,
            "has_error": program.error_info is not None,
            "error_info": program.error_info if program.error_info else None,
            "metrics": program.metrics,
        }

        # Add island info from metadata
        if "assigned_island" in program.meta:
            log_entry["island_id"] = program.meta["assigned_island"]

        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        try:
            with open(self.jsonl_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Failed to log program to JSONL: {e}[/red]")

    def stats(self, stats: Dict[str, Any], title: str = "Statistics") -> None:
        """
        Display statistics.

        Console: Rich table
        Text file: Plain text table
        JSONL file: Structured JSON
        """
        if not stats:
            return

        # Console output (Rich table)
        if self.console_enabled and self.console:
            self._display_stats_console(stats, title)

        # Plain text file output
        if self.file_enabled and self.text_log_file:
            self._write_stats_text(stats, title)

        # JSONL file output
        if self.jsonl_enabled and self.jsonl_log_file:
            self._log_stats_jsonl(stats, title)

    def _display_stats_console(self, stats: Dict[str, Any], title: str) -> None:
        """Display stats as Rich table."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        for key, value in stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            table.add_row(key, formatted_value)

        self.console.print(table)

    def _write_stats_text(self, stats: Dict[str, Any], title: str) -> None:
        """Write stats to plain text file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        lines = [
            f"{timestamp} INFO    === {title} ===",
        ]

        for key, value in stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            lines.append(f"{timestamp} INFO      {key}: {formatted_value}")

        try:
            with open(self.text_log_file, "a") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def _log_stats_jsonl(self, stats: Dict[str, Any], title: str) -> None:
        """Log stats to JSONL file."""
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "event": "statistics",
            "logger": self.name,
            "title": title,
            "stats": stats,
        }

        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        try:
            with open(self.jsonl_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def iteration_result(
        self,
        iteration: int,
        child_id: str,
        parent_id: str,
        metrics: Dict[str, Any],
        elapsed_time: float,
        island_id: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Log iteration result in concise format.

        Args:
            iteration: Current iteration number
            child_id: ID of newly generated program
            parent_id: ID of parent program
            metrics: Program metrics (success/search score, validity, etc.)
            elapsed_time: Total elapsed time for this iteration
            island_id: Island ID (for multi-island)
            max_iterations: Total iterations (for progress tracking)
        """
        # Build concise message
        combined_score = metrics.get("combined_score", 0.0)
        search_score = metrics.get("search_score", combined_score)
        validity = metrics.get("validity", 1.0)

        # Format: Iteration X: child_id (success=0.00, search=0.85, validity=1.0) from parent_id in 5.2s
        message = (
            f"Iteration {iteration}: {child_id[:8]}... "
            f"(success={combined_score:.4f}, search={search_score:.4f}, validity={validity:.2f}) "
            f"from {parent_id[:8]}... in {elapsed_time:.2f}s"
        )

        # Add context
        context = {}
        if island_id is not None:
            context["island_id"] = island_id
        if max_iterations is not None:
            context["iteration"] = iteration
            context["max_iterations"] = max_iterations

        self.info(message, **context)

    def new_best_program(
        self,
        program_id: str,
        old_program_id: str,
        old_score: float,
        new_score: float,
        island_id: Optional[int] = None,
    ) -> None:
        """
        Log new best program announcement.

        Args:
            program_id: ID of new best program
            old_program_id: ID of previous best program
            old_score: Previous best score
            new_score: New best score
            island_id: Island ID (for multi-island)
        """
        improvement = new_score - old_score
        island_prefix = f"Island {island_id}: " if island_id is not None else ""

        self.info(
            f"{island_prefix}New best program {program_id} replaces {old_program_id} "
            f"(ranking_score: {old_score:.4f} → {new_score:.4f}, +{improvement:.4f})"
        )

    def get_log_path(self) -> Dict[str, Optional[Path]]:
        """Get the log file paths."""
        return {
            "text": self.text_log_file,
            "jsonl": self.jsonl_log_file,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LocalLogger(name={self.name}, "
            f"experiment_id={self.experiment_id}, "
            f"text_log={self.text_log_file}, "
            f"jsonl_log={self.jsonl_log_file})"
        )
