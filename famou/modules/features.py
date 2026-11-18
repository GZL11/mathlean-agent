"""
Feature extraction modules for Famou 2.0.

Implements feature vector extraction for similarity/diversity computations:
- FeatureModule: Base class with abstract extract_features method
- EmbeddingFeature: Extract features using embedding models

The feature module reads from result.generated_program and enriches it with feature_vector.
It modifies the program IN PLACE.

Note: If FeatureModule is not included in the pipeline, RolloutResult.finalize()
will compute a default feature_vector from metrics + combined_score.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Module

if TYPE_CHECKING:
    from famou.infrastructure.embedding.base import EmbeddingClient


# ============================================================================
# Dependency Injection Protocol for Embedding
# ============================================================================

@runtime_checkable
class RequiresEmbedding(Protocol):
    """
    Protocol mixin for modules that require embedding client.

    Modules that need to generate embeddings should inherit from this protocol.
    The RolloutEngine will automatically inject the embedding client.

    Example:
        >>> class EmbeddingFeature(FeatureModule, RequiresEmbedding):
        ...     embedding_client: EmbeddingClient
        ...
        ...     def extract_features(self, context, result):
        ...         response = self.embedding_client.embed(program.code)
        ...         return response.embedding
    """

    embedding_client: "EmbeddingClient"


def has_embedding_dependency(module: Module) -> bool:
    """Check if a module requires embedding client."""
    return isinstance(module, RequiresEmbedding)


def inject_embedding(module: Module, embedding_client: "EmbeddingClient") -> None:
    """
    Inject embedding client into module if it requires one.

    Args:
        module: Module instance
        embedding_client: Embedding client to inject
    """
    if has_embedding_dependency(module):
        module.embedding_client = embedding_client  # type: ignore


# ============================================================================
# Base FeatureModule
# ============================================================================

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

    Example:
        >>> class MyFeatureExtractor(FeatureModule):
        ...     def extract_features(self, context, result) -> List[float]:
        ...         program = result.generated_program
        ...         # Custom feature extraction logic
        ...         return [program.combined_score, len(program.code), ...]
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
        try:
            feature_vector = self.extract_features(context, result)
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


# ============================================================================
# Concrete Implementations
# ============================================================================

class EmbeddingFeature(FeatureModule, RequiresEmbedding):
    """
    Extract features using embedding models.

    Uses an embedding client (OpenAI, local model, etc.) to generate
    semantic embeddings of the program code.

    Config:
        embed_field: Which field to embed ("code", "prompt", "combined")
            - "code": Embed program.code (default)
            - "prompt": Embed program.prompt
            - "combined": Embed concatenation of prompt + code
    """

    embedding_client: "EmbeddingClient"  # Injected by RolloutEngine

    def __init__(self, name: Optional[str] = None, embed_field: str = "code", **config):
        """
        Initialize embedding feature extractor.

        Args:
            name: Module name (default: "EmbeddingFeature")
            embed_field: Field to embed ("code", "prompt", "combined")
            **config: Additional configuration
        """
        super().__init__(name=name, **config)
        self.embed_field = embed_field

    def extract_features(self, context: Context, result: RolloutResult) -> List[float]:
        """Extract embedding from program code/prompt."""
        print("Is in feature.py!!!")
        # self.embed_field = None
        program = result.generated_program

        # Get text to embed based on configured field
        if self.embed_field == "code":
            text = program.code
        elif self.embed_field == "prompt":
            text = program.prompt or ""
        elif self.embed_field == "combined":
            prompt_text = program.prompt or ""
            text = f"{prompt_text}\n\n{program.code}"
        else:
            raise ValueError(f"Unknown embed_field: {self.embed_field}")
        # print(f"Text to embed: {text}")
        if not text:
            self.log_warning(f"Empty text for embedding, using empty code")
            text = "# empty"

        # Generate embedding
        response = self.embedding_client.embed(text)
        return response.embedding
        # return None