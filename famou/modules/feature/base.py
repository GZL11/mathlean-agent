"""
Base feature extraction modules for Famou 2.0.

The feature module reads from result.generated_program and enriches it with
feature_vector. It modifies the program IN PLACE.
"""

from abc import abstractmethod
from typing import List

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module


class FeatureModule(Module):
    """
    Base class for feature extraction modules.

    BEHAVIOR:
    - Extracts feature vector from generated program
    - Sets program.feature_vector field
    - Optional module (if not included, RolloutResult.finalize() computes default)

    Data Flow:
    - Input: result.generated_program (enriched with metrics/scores from EvaluateModule)
    - Output: Same program enriched with feature_vector

    Subclasses must implement extract_features() to compute the feature vector.
    """

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program is available."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No program for feature extraction. "
                "Make sure EvaluateModule has run first."
            )

    @abstractmethod
    def extract_features(self, context: Context, result: RolloutResult) -> List[float]:
        """
        Extract feature vector from the generated program.

        This is the core method that subclasses must implement.
        Should return a list of floats representing the feature vector.

        Args:
            context: Experiment context (read-only)
            result: Rollout result with generated_program

        Returns:
            Feature vector as list of floats
        """
        pass

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Execute the module on the generated program."""
        # print("executionationationation")
        try:
            # print("start extraction")
            feature_vector = self.extract_features(context, result)
            # print("extraction finish")
            result.generated_program.feature_vector = feature_vector

            self.log_info(
                f"Extracted features for {result.generated_program.id}",
                program_id=result.generated_program.id,
                feature_dim=len(feature_vector),
            )

            return result
        except Exception as e:
            self.log_error(f"Feature extraction failed: {e}")
            raise
