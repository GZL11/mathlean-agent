"""
Judging base modules for Famou 2.0.

**SIMPLIFIED: One Program Per Rollout**

Provides LLM-based code review and feedback.
"""

from abc import abstractmethod
from typing import Any, Dict

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module, RequiresLLM
from famou.infrastructure.llm.base import LLMClient


class JudgeModule(Module, RequiresLLM):
    """
    Base class for LLM-based code review and feedback.

    **SIMPLIFIED: One Program Per Rollout**

    BEHAVIOR:
    - Provides textual feedback only (NO scoring)
    - Sets program.llm_feedback field
    - Optional module (can be skipped to save costs)

    Note: combined_score is now set by EvaluateModule, not JudgeModule.

    Data Flow:
    - Input: result.generated_program (enriched with metrics/scores from EvaluateModule)
    - Output: Same program enriched with llm_feedback

    Subclasses must implement execute() to generate feedback.
    """

    llm_client: LLMClient  # Injected by RolloutEngine

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program is available."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No program for feedback generation. "
                "Make sure EvaluateModule has run first."
            )

    @abstractmethod
    def judge(self, context: Context, result: RolloutResult) -> Dict[str, Any]:
        """Main entry point for the judge module."""
        pass

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Execute the module on the given program."""
        try:
            feedback = self.judge(context, result)
            result.generated_program.llm_feedback = feedback
            return result
        except Exception as e:
            self.log_error(e)
            raise
