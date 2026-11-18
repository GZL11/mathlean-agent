"""
RolloutEngine for Famou 2.0.

Orchestrates module execution in a pipeline.
Handles dependency injection and error recovery.
"""

import time
from typing import Optional, Tuple

from famou.core.data import Context, ModuleStat, RolloutResult
from famou.core.types import ModuleValidationError
from famou.core.protocol import Module, Rollout, inject_embedding, inject_env, inject_llm
from famou.infrastructure.embedding import EmbeddingClient
from famou.infrastructure.env import ExecutionEnvironment
from famou.infrastructure.llm import LLMClient
from famou.infrastructure.logger import Logger
from famou.utils import generate_rollout_id
from famou.utils.color import Colors


class RolloutEngine:
    """
    Reusable execution engine for running Rollout pipelines.

    The RolloutEngine:
    1. Provides reusable infrastructure (LLM, logger, env)
    2. Accepts different Rollout objects at runtime
    3. Injects dependencies and caches injected rollouts for efficiency
    4. Executes modules sequentially in the pipeline
    5. Handles errors according to strategy

    **Key Improvement:**
    The engine is now decoupled from specific rollouts - you can reuse the same
    engine to execute different rollout strategies without recreating infrastructure.

    **Pipeline Pattern:**
    ```
    Context (experiment-level, read-only)
        ↓
    [SelectModule] → result.selected_programs (List[str] - IDs)
        ↓
    [GenerateModule] → result.generated_programs (List[Program] - NEW)
        ↓
    [EvaluateModule] → result.generated_programs (ENRICHED with metrics)
        ↓
    [JudgeModule] → result.generated_programs (ENRICHED with scores)
        ↓
    RolloutResult (finalized)
    ```

    Example:
        >>> # Create engine once with infrastructure
        >>> engine = RolloutEngine(
        ...     llm_client=openai_client,
        ...     env=local_env,
        ...     logger=logger,
        ... )
        >>>
        >>> # Run different rollouts with same engine (efficient!)
        >>> exploit_rollout = Rollout([TopKSelect(k=1), MutationGenerate(), ...])
        >>> explore_rollout = Rollout([RandomSelect(k=3), MutationGenerate(), ...])
        >>>
        >>> result1 = engine.execute_rollout(exploit_rollout, context, iteration=1)
        >>> result2 = engine.execute_rollout(explore_rollout, context, iteration=2)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        env: Optional[ExecutionEnvironment] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        logger: Optional[Logger] = None,
        retry_enabled: bool = True,
        max_retries: int = 3,
        retry_wait_min: float = 2.0,
        retry_wait_max: float = 10.0,
        retry_multiplier: float = 1.5,
    ):
        """
        Initialize RolloutEngine with infrastructure.

        Args:
            llm_client: LLM client for generation/judging modules
            env: Execution environment for evaluation modules
            embedding_client: Embedding client for feature extraction modules
            logger: Logger for output
            retry_enabled: Enable retry logic for module execution (default: True)
            max_retries: Maximum retry attempts for any module (default: 3)
            retry_wait_min: Minimum wait between retries in seconds (default: 2.0)
            retry_wait_max: Maximum wait between retries in seconds (default: 10.0)
            retry_multiplier: Exponential backoff multiplier (default: 1.5)

        Failure Strategy:
            When a module fails after all retries, the rollout is marked as failed
            and the iteration continues with the next rollout. Failed rollouts do
            not update the population.
        """
        self.llm_client = llm_client
        self.env = env
        self.embedding_client = embedding_client
        self.logger = logger
        self.retry_enabled = retry_enabled
        self.max_retries = max_retries
        self.retry_wait_min = retry_wait_min
        self.retry_wait_max = retry_wait_max
        self.retry_multiplier = retry_multiplier

        # Cache for injected rollouts (only inject dependencies once per rollout)
        self._injected_rollouts: dict[int, bool] = {}

    def execute_rollout(self, rollout: Rollout, context: Context, iteration: int) -> RolloutResult:
        """
        Execute complete evolutionary rollout.

        Pipeline: Select → Generate → Evaluate → Judge
        Used during evolution iterations.

        The rollout is validated and dependencies are injected (with caching).
        Then all modules are executed sequentially with retry logic.

        Failure Strategy (Fail Rollout, Continue Iteration):
            - Modules are retried on transient failures (not validation errors)
            - If a module fails after all retries, the rollout is marked as failed
            - Failed rollouts are returned with status="failed"
            - The iteration continues (failed rollouts don't update population)

        Args:
            rollout: Rollout pipeline to execute
            context: Experiment context (shared, read-only)
            iteration: Current iteration number

        Returns:
            Finalized RolloutResult (status="success" or "failed")
        """
        # Validate and inject dependencies (cached, only happens once per unique rollout)
        rollout_id_hash = id(rollout)
        if rollout_id_hash not in self._injected_rollouts:
            rollout.validate()
            self._inject_dependencies(rollout)
            self._injected_rollouts[rollout_id_hash] = True

        # Create fresh RolloutResult for this rollout
        # Include island_id in rollout_id for multi-island experiments (for uniqueness)
        island_id_for_rollout = None
        if context.experiment_config and context.experiment_config.num_islands > 1:
            island_id_for_rollout = context.island_id

        result = RolloutResult(
            rollout_id=generate_rollout_id(
                experiment_id=context.experiment_id,
                iteration=iteration,
                island_id=island_id_for_rollout
            ),
            iteration=iteration,
            created_at=time.time(),
        )

        if self.logger:
            self.logger.debug(
                f"Starting rollout {result.rollout_id} (strategy: {rollout.name})",
                iteration=iteration,
                experiment_id=context.experiment_id,
                rollout_name=rollout.name,
            )

        # Execute pipeline
        return self._execute_pipeline(rollout.modules, context, result, iteration)

    def _inject_dependencies(self, rollout: Rollout) -> None:
        """
        Inject infrastructure dependencies into rollout's modules.

        Uses protocol checking to inject only what each module needs:
        - RequiresLLM → llm_client
        - RequiresEnv → env
        - All modules → logger

        Args:
            rollout: Rollout whose modules need dependency injection
        """
        for module in rollout.modules:
            # Always inject logger
            if self.logger:
                module.logger = self.logger

            # Inject LLM client if module requires it
            if self.llm_client:
                inject_llm(module, self.llm_client)

            # Inject execution environment if module requires it
            if self.env:
                inject_env(module, self.env)
            
            # Inject embedding client if module requires it
            if self.embedding_client:
                inject_embedding(module, self.embedding_client)

    def _execute_module_with_retry(
        self,
        module: Module,
        context: Context,
        result: RolloutResult,
    ) -> Tuple[RolloutResult, int]:
        """
        Execute module with retry logic.

        Retries on all exceptions EXCEPT ValueError (validation errors).
        Uses exponential backoff between retry attempts.

        Args:
            module: Module to execute
            context: Experiment context
            result: Rollout result

        Returns:
            Tuple of (updated rollout result, retry count)

        Raises:
            Exception: If module fails after all retries
        """
        if not self.retry_enabled:
            # No retry, execute directly
            return module(context, result), 0

        # Import tenacity for retry logic
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_not_exception_type,
            RetryError,
        )

        # Track retry count
        retry_count = 0

        def _track_retry(retry_state):
            nonlocal retry_count
            # attempt_number is the attempt that just failed (1-based)
            # This equals the number of retries made so far
            retry_count = retry_state.attempt_number
            self._log_retry_attempt(module, retry_state)

        # Create retry decorator (retry all except ValueError)
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_multiplier,
                min=self.retry_wait_min,
                max=self.retry_wait_max,
            ),
            retry=retry_if_not_exception_type(ModuleValidationError),  # Don't retry validation errors
            before_sleep=_track_retry,
            reraise=True,
        )

        # Wrap module execution
        @retry_decorator
        def _execute():
            # if module.name == "EmbeddingFeature":
                # print("context:", context)
                # print("result:", result)
            # print(f"Running module: {module.name}")
            return module(context, result)

        try:
            result = _execute()
            return result, retry_count
        except RetryError as e:
            # All retries exhausted - retry_count already updated by _track_retry
            # But if max_retries was reached, set it explicitly
            retry_count = e.last_attempt.attempt_number
            original_exception = e.last_attempt.exception()
            raise RuntimeError(
                f"Module {module.name} failed after {self.max_retries} retries"
            ) from original_exception

    def execute_enrichment(
        self,
        program,
        modules: list,
        context: Context,
        rollout_id: str,
    ) -> RolloutResult:
        """
        Enrich existing program with enrichment modules.

        Pipeline: Evaluate → Judge (no Select/Generate)
        Used for initial programs and re-evaluation.

        This method is designed for enriching initial programs or re-evaluating
        existing programs. It wraps the program in a RolloutResult and executes
        enrichment modules (EvaluateModule, LLMJudge, etc.) with retry logic.

        Args:
            program: Program to enrich
            modules: List of enrichment modules to apply
            context: Experiment context
            rollout_id: Unique ID for this enrichment operation

        Returns:
            RolloutResult with enriched program

        Example:
            >>> enrichment_modules = [EvaluateModule(), LLMJudge()]
            >>> result = engine.execute_enrichment(
            ...     program=initial_program,
            ...     modules=enrichment_modules,
            ...     context=context,
            ...     rollout_id="init_0",
            ... )
            >>> enriched_program = result.generated_program
        """
        # Inject dependencies into modules
        for module in modules:
            # Always inject logger
            if self.logger and not hasattr(module, "logger"):
                module.logger = self.logger

            # Inject LLM client if module requires it
            if self.llm_client:
                inject_llm(module, self.llm_client)

            # Inject execution environment if module requires it
            if self.env:
                inject_env(module, self.env)
            
            if self.embedding_client:
                inject_embedding(module, self.embedding_client)

        # Create RolloutResult with the program
        result = RolloutResult(
            rollout_id=rollout_id,
            iteration=0,
            status="success",
            generated_program=program,
            created_at=time.time(),
        )

        if self.logger:
            self.logger.debug(
                f"Starting enrichment {rollout_id}",
                program_id=program.id,
                num_modules=len(modules),
            )

        # Execute pipeline
        print(f"{Colors.BOLD}{Colors.BLUE}Starting enrichment pipeline {Colors.RESET}")
        return self._execute_pipeline(modules, context, result, iteration=0)

    def _execute_pipeline(
        self,
        modules: list,
        context: Context,
        result: RolloutResult,
        iteration: int,
    ) -> RolloutResult:
        """
        Core execution: run modules with retry logic.

        Shared by both execute_rollout() and execute_enrichment().
        Executes modules sequentially, handles retries and errors,
        finalizes result.

        Args:
            modules: List of modules to execute
            context: Experiment context
            result: RolloutResult to update
            iteration: Current iteration number (for logging)

        Returns:
            Finalized RolloutResult
        """
        # Execute module pipeline with retry and failure tracking
        rollout_failed = False
        failed_module = None
        failure_error = None
        # print(f"pipeline 1, modules: {modules}")
        for module in modules:
            start_time = time.time()
            retry_count = 0
            success = True
            error_msg = None

            try:
                if self.logger:
                    self.logger.debug(
                        f"Executing module: {module.name}",
                        module=module.name,
                        iteration=iteration,
                    )

                start_time = time.time()
                print(f"pipline 1.5, modulename: {module.name} module: {module}")

                # Execute module with retry logic
                result, retry_count = self._execute_module_with_retry(module, context, result)

                elapsed_ms = (time.time() - start_time) * 1000

                if self.logger:
                    self.logger.debug(
                        f"Module {module.name} completed",
                        module=module.name,
                        elapsed=f"{elapsed_ms:.1f}ms",
                        retries=retry_count,
                    )

            except Exception as e:
                # Module failed after all retries
                elapsed_ms = (time.time() - start_time) * 1000
                success = False
                error_msg = str(e)
                rollout_failed = True
                failed_module = module.name
                failure_error = str(e)
                print(f"pipline 1.6, modulename: {module.name} module: {module} failed")

                if self.logger:
                    self.logger.error(
                        f"Module {module.name} failed after retries: {e}",
                        module=module.name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

            # Record module stats regardless of success/failure
            result.module_stats.append(ModuleStat(
                module_name=module.name,
                execution_time_ms=elapsed_ms,
                success=success,
                error_message=error_msg,
                retry_count=retry_count,
            ))

            # Stop processing if module failed
            if not success:
                break

        print("pipeline 2")
        # Set rollout status and error information
        if rollout_failed:
            result.status = "failed"
            result.failed_module = failed_module
            result.error_message = failure_error

        # Finalize result (compute stats)
        result.finalize()

        # Mark completion time
        result.completed_at = time.time()

        # Log completion
        if self.logger:
            if rollout_failed:
                self.logger.error(
                    f"Execution {result.rollout_id} FAILED",
                    iteration=iteration,
                    failed_module=failed_module,
                    error=failure_error,
                    status="failed",
                )
            else:
                self.logger.debug(
                    f"Execution {result.rollout_id} completed successfully",
                    iteration=iteration,
                    score=(
                        f"{result.generated_program.combined_score:.4f}"
                        if result.generated_program and result.generated_program.combined_score is not None
                        else "N/A"
                    ),
                    has_program=result.generated_program is not None,
                    validity=(
                        f"{result.generated_program.validity:.2f}"
                        if result.generated_program and result.generated_program.validity is not None
                        else "N/A"
                    ),
                    status="success",
                )
        # print(f"{Colors.BOLD}{Colors.BLUE}The result after one pipline: {Colors.CYAN}{result}{Colors.RESET}")
        return result

    def _log_retry_attempt(self, module: Module, retry_state):
        """Log retry attempt with details."""
        if not self.logger:
            return

        attempt = retry_state.attempt_number
        wait = retry_state.next_action.sleep if retry_state.next_action else 0
        exception = retry_state.outcome.exception()

        self.logger.warning(
            f"Module {module.name} failed (attempt {attempt}/{self.max_retries}), "
            f"retrying in {wait:.1f}s: {exception}",
            module=module.name,
            attempt=attempt,
            max_retries=self.max_retries,
            wait_time=f"{wait:.1f}s",
            error=str(exception),
            error_type=type(exception).__name__,
        )

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"RolloutEngine("
            f"llm={self.llm_client is not None}, "
            f"env={self.env is not None}, "
            f"retry={self.retry_enabled}, "
            f"max_retries={self.max_retries})"
        )
