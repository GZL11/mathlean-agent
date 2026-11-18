"""
Local execution environment implementation.

Executes programs via user-provided evaluator functions.
"""

import os
import tempfile
from typing import List, Optional

from famou.utils import install_packages


class LocalEnv:
    """
    Local execution environment using user-provided evaluator functions.

    Writes program code to temporary file, calls user's evaluator function,
    and returns evaluation results. The evaluator handles all execution logic,
    validation, and metrics computation.

    Supports automatic installation of required Python packages before execution.

    Example:
        >>> from examples.circle_packing.evaluator import evaluate
        >>> env = LocalEnv()
        >>> result = env.execute_with_evaluator(
        ...     program_code="def fibonacci(n): ...",
        ...     evaluate_fn=evaluate,
        ...     language="python",
        ...     required_packages=["numpy", "matplotlib"],
        ...     experiment_id="exp_123",
        ... )
        >>> print(result["combined_score"])
        0.95
    """

    def __init__(self):
        """Initialize local execution environment."""
        pass

    def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn,  # Callable[[str], Dict[str, Any]]
        language: str = "python",
        timeout: Optional[float] = None,
        required_packages: Optional[List[str]] = None,
    ):
        """
        Execute program using user-provided evaluator function.

        This method:
        1. Installs required packages (Python only, if provided)
        2. Writes program code to a temporary file
        3. Calls user's evaluate_fn(program_path)
        4. Returns the evaluation results
        5. Cleans up the temporary file

        Packages are installed to an experiment-specific directory to avoid
        polluting the user's Python environment. The target directory is
        added to PYTHONPATH during execution.

        The evaluator function handles all execution logic, validation,
        and metrics computation. This makes it easy to support different
        evaluation patterns (subprocess, validation, benchmarking, etc.)
        without changing the environment code.

        Args:
            program_code: Source code to execute
            evaluate_fn: User's evaluator function(program_path) -> results_dict
            language: Programming language (default: "python")
            timeout: Optional timeout (not enforced here, evaluator should handle it)
            required_packages: Optional list of packages to install (Python only)
            experiment_id: Optional experiment ID for package installation directory

        Returns:
            Dictionary with evaluation results from evaluate_fn (flat format):
            {
                "combined_score": float,      # Required - overall fitness
                "validity": float,             # Required - 0.0-1.0 validity score
                "error_info": str,             # Optional - error details
                "metric1": value1,             # Optional - custom metrics
                "metric2": value2,             # Optional - custom metrics
            }

        Example:
            >>> def my_evaluator(program_path: str):
            ...     # User handles execution, validation, metrics
            ...     result = subprocess.run(['python', program_path], ...)
            ...     score = validate(result)
            ...     return {
            ...         "combined_score": score,
            ...         "validity": 1.0,
            ...         "accuracy": 0.98,
            ...         "speed": 1.2,
            ...     }
            >>>
            >>> env = LocalEnv()
            >>> result = env.execute_with_evaluator(
            ...     "def f(): pass",
            ...     my_evaluator,
            ...     required_packages=["numpy"],
            ...     experiment_id="exp_123"
            ... )
            >>> print(result["combined_score"])
            0.95
        """
        # Install packages before execution (Python only)
        if language == "python" and required_packages:
            install_packages(packages=required_packages)

        # Determine file extension based on language
        extension_map = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c",
            "rust": ".rs",
            "go": ".go",
            "scratch": ".scratch",
            "lean": ".lean",
        }
        extension = extension_map.get(language.lower(), ".txt")

        # Write program to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=extension,
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(program_code)
            temp_path = f.name
        
        # print(f"Language: {language} & Temp Path: {temp_path} & Program Code: {program_code}")

        try:
            # Call user's evaluator function
            # The evaluator handles all execution logic, validation, metrics
            result = evaluate_fn(temp_path)
            if isinstance(result, dict) and "generated_folder" in result:
                print("temp_folder:", result["generated_folder"])
            return result

        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            return None
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                # Ignore cleanup errors
                pass

    def __repr__(self) -> str:
        """Concise representation."""
        return f"LocalEnv()"
