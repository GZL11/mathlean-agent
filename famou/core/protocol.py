"""
Module protocol and dependency injection for Famou 2.0.

Defines:
- Module: Base class for all algorithm components
- Protocol mixins for dependency injection (RequiresLLM, RequiresEnv)

Key Design Pattern: Program Enrichment Pipeline (Simplified)
-------------------------------------------------------------
**One Program Per Rollout**

Programs flow through modules and are gradually enriched with data:

1. SelectModule: population → selected_parent_id (single ID, str)
2. GenerateModule: selected_parent_id → generated_program (ONE Program)
3. EvaluateModule: generated_program (ENRICHES IN PLACE with metrics, validity, combined_score, error_info)
4. JudgeModule: generated_program (ENRICHES IN PLACE with llm_feedback)

Each module either:
- Routes Programs (SelectModule): Selects one parent ID, stores in selected_parent_id
- Creates new Program (GenerateModule): Looks up parent, creates one child
- Enriches existing Program IN PLACE (EvaluateModule, JudgeModule): No separate output

Inspired by gigaevo's Stage abstraction - modules are composable, reorderable
components that transform RolloutResult through the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from famou.core.data import Context, Program, RolloutResult
from famou.core.types import ModuleValidationError

# Avoid circular imports
if TYPE_CHECKING:
    from famou.infrastructure.llm.base import LLMClient
    from famou.infrastructure.env.base import ExecutionEnvironment
    from famou.infrastructure.embedding.base import EmbeddingClient
    from famou.infrastructure.logger.base import Logger


class Module(ABC):
    """
    Base class for all algorithm modules.

    Modules are the building blocks of the evolutionary pipeline.
    Each module transforms RolloutContext by either:
    1. Creating new Programs
    2. Enriching existing Programs
    3. Routing Programs between context fields

    Pipeline Data Flow Pattern:
    ---------------------------
    context.population (Dict[str, List[Program]] - read-only)
        ↓
    [SelectModule] → result.selected_parent_id (str) + result.inspiration_ids (List[str])
        ↓
    [GenerateModule] → result.generated_program (Program - NEW)
        ↓
    [EvaluateModule] → result.generated_program (ENRICHED IN PLACE with metrics)
        ↓
    [JudgeModule] → result.generated_program (ENRICHED IN PLACE with feedback)
        ↓
    Final: result.generated_program fully enriched

    Module Types:
    -------------
    - **Routing Modules** (SelectModule): Select program IDs from population
      - Input: context.population (Dict[str, List[Program]])
      - Output: result.selected_parent_id (str) + result.inspiration_ids (List[str])
      - Returns IDs only, not full Program objects

    - **Creation Modules** (GenerateModule): Create new Programs from parents
      - Input: result.selected_parent_id (str)
      - Looks up: context.get_program_by_id(parent_id) to get parent Program
      - Output: result.generated_program (Program - NEW)
      - Sets: id, code, prompt, response, thinking, generation, iteration, parent_ids

    - **Enrichment Modules** (EvaluateModule, JudgeModule): Enrich Programs IN PLACE
      - Input: result.generated_program (Program)
      - Output: Same program enriched in place
      - EvaluateModule enriches: metrics, validity, combined_score, error_info
      - JudgeModule enriches: llm_feedback
      - No separate output list - all mutations happen in place

    Modules should be:
    - Composable: Can be chained in any order
    - Stateless: Don't maintain state between rollouts (config is OK)

    Attributes:
        name: Module name (for logging and debugging)
        config: Configuration dictionary
        logger: Logger instance (injected by RolloutEngine)
    """

    def __init__(self, name: Optional[str] = None, **config):
        """
        Initialize module.

        Args:
            name: Optional module name (defaults to class name)
            **config: Configuration parameters specific to this module
        """
        self.name = name or self.__class__.__name__
        self.config = config
        self.logger: Optional["Logger"] = None  # Injected by RolloutEngine

    def __call__(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Entry point for module execution.

        Flow:
        1. Validate inputs (check prerequisites)
        2. Execute module logic (read from context, write to result)
        3. Post-validate output (check results)
        4. Return updated result

        **Key Design:**
        - **context**: Experiment-level data (READ-ONLY) - prompts, population, config
        - **result**: Rollout-level data (MUTABLE) - selected/generated/evaluated programs

        **Thread-Safety:**
        Each rollout has its own RolloutResult, so parallel rollouts are naturally
        isolated. Context is shared (read-only), which is safe.

        Args:
            context: Experiment context (read-only for modules)
            result: Rollout result (modules write to this)
            **kwargs: Additional runtime arguments

        Returns:
            Updated rollout result

        Raises:
            ModuleValidationError: If validation fails (not retried by RolloutEngine)

        Example:
            >>> # SelectModule reads context.population, writes result.selected_parent_id
            >>> result = select_module(context, result)
            >>> # GenerateModule reads result.selected_parent_id, writes result.generated_program
            >>> result = generate_module(context, result)
        """
        # Pre-execution validation
        # Wrap ValueError in ModuleValidationError so it's not retried
        try:
            self.validate_input(context, result)
        except ValueError as e:
            raise ModuleValidationError(str(e)) from e

        # print("validation 2")
        # Execute module logic
        # Errors here (including JSONDecodeError) WILL be retried
        result = self.execute(context, result, **kwargs)

        # print("validation 3")
        # Post-execution validation
        # Wrap ValueError in ModuleValidationError so it's not retried
        try:
            self.validate_output(context, result)
        except ValueError as e:
            raise ModuleValidationError(str(e)) from e

        return result

    @abstractmethod
    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Execute module logic.

        This is the core method that subclasses must implement.
        Modules READ from context (experiment data) and WRITE to result (rollout data).

        Args:
            context: Experiment context (read-only)
            result: Rollout result (mutable)
            **kwargs: Additional runtime arguments

        Returns:
            Updated rollout result
        """
        pass

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """
        Validate that inputs have required data.

        Override this method to check prerequisites before execution.
        Raise ValueError with a clear message if validation fails.

        Args:
            context: Experiment context (for checking population, config, etc.)
            result: Rollout result (for checking previous module outputs)

        Raises:
            ValueError: If inputs are missing required data

        Example:
            >>> class SelectModule(Module):
            ...     def validate_input(self, context, result):
            ...         if not context.population:
            ...             raise ValueError(
            ...                 f"{self.name}: Cannot select from empty population"
            ...             )
            >>> class GenerateModule(Module):
            ...     def validate_input(self, context, result):
            ...         if not result.selected_parent_id:
            ...             raise ValueError(
            ...                 f"{self.name}: No parent selected for generation"
            ...             )
        """
        pass

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """
        Validate that module produced valid output.

        Override this method to check postconditions after execution.
        Useful for catching bugs and ensuring data integrity.

        Args:
            context: Experiment context
            result: Rollout result (after execution)

        Raises:
            ValueError: If output is invalid

        Example:
            >>> class GenerateModule(Module):
            ...     def validate_output(self, context, result):
            ...         if not result.generated_program:
            ...             raise ValueError(
            ...                 f"{self.name}: Failed to generate program"
            ...             )
            ...         if not result.generated_program.code:
            ...             raise ValueError(
            ...                 f"{self.name}: Generated program has no code"
            ...             )
        """
        pass

    def log_info(self, message: str, **context_data):
        """Helper to log debug message with context (module-level logs are DEBUG)."""
        if self.logger:
            self.logger.debug(f"[{self.name}] {message}", **context_data)

    def log_error(self, message: str, **context_data):
        """Helper to log error message with context."""
        if self.logger:
            self.logger.error(f"[{self.name}] {message}", **context_data)

    def log_warning(self, message: str, **context_data):
        """Helper to log warning message with context."""
        if self.logger:
            self.logger.warning(f"[{self.name}] {message}", **context_data)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return f"{self.__class__.__name__}(name={self.name})"


# ============================================================================
# Rollout: Pipeline Encapsulation
# ============================================================================

class Rollout:
    """
    Encapsulates a reusable evolutionary pipeline.

    A Rollout is a sequence of modules that defines a single iteration of evolution.
    It provides a clean abstraction for users to define and share pipeline strategies.

    **Design Philosophy:**
    - Rollout is a lightweight data container (no execution logic)
    - RolloutEngine handles execution and dependency injection
    - Users define pipelines declaratively, framework handles plumbing

    **Typical Pipeline Structure:**
    1. SelectModule - Choose parents from population (optional)
    2. GenerateModule - Create new programs (required)
    3. EvaluateModule - Execute and score programs (required)
    4. JudgeModule - Provide LLM feedback (optional)

    Attributes:
        modules: List of modules to execute in sequence
        name: Optional name for this rollout strategy

    Example:
        >>> # Define reusable pipeline strategies
        >>> standard_flow = Rollout([
        ...     TopKSelect(k=2),
        ...     MutationGenerate(num_children_per_parent=2),
        ...     EvaluateModule(evaluate_fn=my_evaluator),
        ... ], name="standard")
        >>>
        >>> experimental_flow = Rollout([
        ...     RandomSelect(k=3),
        ...     MutationGenerate(temperature=0.9),
        ...     EvaluateModule(evaluate_fn=my_evaluator),
        ...     LLMJudge(criteria=["elegance", "efficiency"]),
        ... ], name="experimental")
        >>>
        >>> # Run experiment with explicit pipeline
        >>> evolver.run(standard_flow, config, data_service)
    """

    def __init__(
        self,
        modules: List[Module],
        name: Optional[str] = None,
        execution_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize rollout pipeline.

        Args:
            modules: List of modules to execute in sequence
            name: Optional name for this rollout strategy
            execution_config: Optional rollout-scoped execution overrides
        """
        self.modules = modules
        self.name = name or "rollout"
        self.execution_config = dict(execution_config or {})

    def validate(self) -> None:
        """
        Validate that pipeline has required module types.

        Checks:
        - Pipeline is not empty
        - Pipeline has at least one GenerateModule
        - Pipeline has at least one EvaluateModule (or judge module)

        Raises:
            ValueError: If pipeline is invalid
        """
        if not self.modules:
            raise ValueError("Rollout pipeline cannot be empty")

        # Import here to avoid circular dependency
        from famou.modules.generate import GenerateModule
        from famou.modules.evaluate import EvaluateModule

        has_generate = any(isinstance(m, GenerateModule) for m in self.modules)
        has_evaluate = any(isinstance(m, EvaluateModule) for m in self.modules)

        if not has_generate:
            raise ValueError(
                "Rollout pipeline must include at least one GenerateModule. "
                f"Current modules: {[m.name for m in self.modules]}"
            )

        if not has_evaluate:
            raise ValueError(
                "Rollout pipeline must include at least one EvaluateModule. "
                f"Current modules: {[m.name for m in self.modules]}"
            )

    def __repr__(self) -> str:
        """Concise representation."""
        module_names = [m.name for m in self.modules]
        return f"Rollout(name={self.name}, modules={module_names})"


# ============================================================================
# Convenience base classes for common module patterns
# ============================================================================

# ============================================================================
# Dependency Injection Protocol Mixins
# ============================================================================

@runtime_checkable
class RequiresLLM(Protocol):
    """
    Protocol mixin for modules that require LLM client.

    Modules that need to call LLMs should inherit from this protocol.
    The RolloutEngine will automatically inject the LLM client.

    Example:
        >>> class MutationGenerate(Module, RequiresLLM):
        ...     llm_client: LLMClient
        ...
        ...     def execute(self, context, **kwargs):
        ...         response = self.llm_client.generate(...)
        ...         # ...
    """

    llm_client: "LLMClient"

@runtime_checkable
class RequiresEnv(Protocol):
    """
    Protocol mixin for modules that require execution environment.

    Modules that need to execute code should inherit from this protocol.
    The RolloutEngine will automatically inject the execution environment.

    Example:
        >>> class LocalEvaluate(Module, RequiresEnv):
        ...     env: ExecutionEnvironment
        ...
        ...     def execute(self, context, **kwargs):
        ...         result = self.env.execute(program)
        ...         # ...
    """

    env: "ExecutionEnvironment"

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


# ============================================================================
# Helper functions for dependency injection
# ============================================================================

def has_dependency(module: Module, protocol: type) -> bool:
    """
    Check if a module requires a specific dependency protocol.

    Args:
        module: Module instance
        protocol: Protocol class (RequiresLLM, RequiresEnv)

    Returns:
        True if module implements the protocol

    Example:
        >>> class MyModule(Module, RequiresLLM):
        ...     llm_client: LLMClient
        >>> module = MyModule()
        >>> has_dependency(module, RequiresLLM)
        True
    """
    return isinstance(module, protocol)



def inject_llm(module: Module, llm_client: "LLMClient") -> None:
    """
    Inject LLM client into module if it requires one.

    Args:
        module: Module instance
        llm_client: LLM client to inject
    """
    if has_dependency(module, RequiresLLM):
        module.llm_client = llm_client  # type: ignore


def inject_env(module: Module, env: "ExecutionEnvironment") -> None:
    """
    Inject execution environment into module if it requires one.

    Args:
        module: Module instance
        env: Execution environment to inject
    """
    if has_dependency(module, RequiresEnv):
        module.env = env  # type: ignore

def inject_embedding(module: Module, embedding_client: "EmbeddingClient") -> None:
    """
    Inject embedding client into module if it requires one.

    Args:
        module: Module instance
        embedding_client: Embedding client to inject
    """
    # print("Embedding_client injectinginging")
    if has_dependency(module, RequiresEmbedding):
        print(f"module {module}: injectinginging embedding_client")
        module.embedding_client = embedding_client  # type: ignore
