"""
Evaluation modules for Famou 2.0.

**SIMPLIFIED: One Program Per Rollout**

Evaluates a single program using custom evaluator functions via ExecutionEnvironment.

The evaluator function receives a program file path and returns evaluation results
in flat format (all metrics as top-level keys):

    def my_evaluator(program_path: str) -> Dict[str, Any]:
        return {
            "combined_score": float,      # Required - overall fitness
            "validity": float,             # Required - 0.0-1.0 validity score
            "error_info": Optional[str],   # Optional - error details
            "metric1": value1,             # Optional - any custom metrics
            "metric2": value2,             # Optional - any custom metrics
        }

The evaluation module reads from result.generated_program and enriches it IN PLACE
with metrics, combined_score, validity, and error_info.
"""

from typing import Callable, Dict, Any

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module, RequiresEnv
from famou.infrastructure.env import ExecutionEnvironment


class EvaluateModule(Module, RequiresEnv):
    """
    Evaluation module using user-provided evaluator functions.

    **SIMPLIFIED: One Program Per Rollout**

    Evaluates a single program by calling a custom evaluator function that returns
    evaluation metrics, combined_score, and validity.

    Data Flow:
    - Input: result.generated_program (single Program)
    - Call evaluate_fn(program_path) → enriches program in place
    - Output: Program enriched with metrics, combined_score, validity, error_info

    Dependencies (Injected):
        env: ExecutionEnvironment - Injected by RolloutEngine (default: LocalEnv)

    Config:
        evaluate_fn: Callable[[str], Dict[str, Any]] - REQUIRED user evaluator function
        timeout: Optional[float] - Execution timeout in seconds (passed to env)

    Evaluator Function Interface (Flat Format):

        The evaluator returns a flat dictionary where all keys except reserved
        ones (combined_score, validity, error_info) are treated as metrics.

            def my_evaluator(program_path: str) -> Dict[str, Any]:
                return {
                    "combined_score": float,      # Required - overall fitness score
                    "validity": float,             # Required - 0.0-1.0 validity score
                    "error_info": Optional[str],   # Optional - error details if failed
                    "metric1": value1,             # Optional - any custom metrics
                    "metric2": value2,             # Optional - any custom metrics
                }

        If the evaluator raises an exception, EvaluateModule catches it and sets:
        - validity = 0.0
        - combined_score = 0.0
        - error_info = error message
        - metrics = {"error": True, "error_type": exception class name}

    Example:
        >>> from examples.circle_packing.evaluator import evaluate
        >>> evaluator = EvaluateModule(name="eval", evaluate_fn=evaluate)
        >>> result = evaluator(context, result)
        >>> # Program now has metrics, combined_score, validity set
    """

    env: ExecutionEnvironment  # Injected by RolloutEngine

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program is available and evaluator is configured."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No program to evaluate. "
                "Make sure GenerateModule has run first."
            )

        evaluate_fn = self.config.get("evaluate_fn")
        if evaluate_fn is None:
            raise ValueError(
                f"{self.name}: evaluate_fn is required. "
                "Provide evaluate_fn=your_evaluator in config."
            )

        if not callable(evaluate_fn):
            raise ValueError(
                f"{self.name}: evaluate_fn must be callable, got {type(evaluate_fn)}"
            )

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Execute program using custom evaluator function via execution environment."""

        evaluate_fn: Callable[[str], Dict[str, Any]] = self.config["evaluate_fn"]
        timeout = self.config.get("timeout")

        program = result.generated_program

        try:
            # Execute program via environment abstraction
            # The env handles temp file creation/cleanup
            self.log_info(f"Evaluating {program.id}")
            print(f"evaluate function: {evaluate_fn}")
            eval_result = self.env.execute_with_evaluator(
                program_code=program.code,
                evaluate_fn=evaluate_fn,
                language=context.language,
                timeout=timeout,
            )

            # Validate that required fields are present
            if "combined_score" not in eval_result:
                raise ValueError(
                    f"Evaluator must return 'combined_score'. "
                    f"Got keys: {list(eval_result.keys())}"
                )
            if "validity" not in eval_result:
                raise ValueError(
                    f"Evaluator must return 'validity'. "
                    f"Got keys: {list(eval_result.keys())}"
                )

            # Extract metrics from flat format
            # All keys except reserved ones become metrics
            reserved_keys = {"combined_score", "validity", "error_info"}
            program.metrics = {
                k: v for k, v in eval_result.items()
                if k not in reserved_keys
            }

            # Set evaluation results
            program.combined_score = eval_result["combined_score"]
            program.validity = eval_result["validity"]
            program.error_info = eval_result.get("error_info")

            self.log_info(
                f"Evaluated {program.id}",
                program_id=program.id,
                combined_score=program.combined_score,
                validity=program.validity,
            )

        except Exception as e:
            # Handle evaluator errors
            # Set program as invalid with error info
            program.validity = 0.0
            program.error_info = f"Evaluator error: {str(e)}"
            program.metrics = {"error": True, "error_type": type(e).__name__}
            program.combined_score = 0.0

            self.log_error(
                f"Evaluator failed for {program.id}: {e}",
                program_id=program.id,
            )

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that program was evaluated."""
        if result.generated_program and result.generated_program.combined_score is None:
            self.log_warning(
                f"{self.name}: Program does not have combined_score set. "
                "Check evaluator function."
            )
