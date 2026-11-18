"""
Core type definitions for Famou 2.0.

Defines enums and type aliases used throughout the framework.
"""

from enum import Enum


class RolloutStatus(str, Enum):
    """
    Status of a rollout pipeline execution.

    Set by RolloutEngine based on module execution results.

    SUCCESS: All modules completed successfully
    FAILED: One or more modules failed after retries
    """

    SUCCESS = "success"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


class ModuleValidationError(ValueError):
    """
    Exception raised when a module's validation fails.

    This is raised by Module.validate_input() and Module.validate_output()
    for logical validation errors that should NOT be retried:
    - Empty population when selection is required
    - Missing parent program for generation
    - Invalid program structure after generation

    Note: This inherits from ValueError but is explicitly excluded from
    retry logic in RolloutEngine. Other ValueErrors (like JSONDecodeError)
    will still be retried.
    """

    pass


class Language(str, Enum):
    """
    Supported programming languages for code evolution.

    Currently focused on Python, but designed for future extension.
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    LEAN = "lean"
    SCRATCH = "scratch"

    def __str__(self) -> str:
        return self.value

    def to_scratch_language(self) -> "Language":
        mapping = {
            Language.LEAN: Language.SCRATCH,
        }
        return mapping.get(self, self)
