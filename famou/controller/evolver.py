"""
Evolver for Famou 2.0.

Main controller for experiment lifecycle.
Uses streaming evolution: maintains worker pool, updates population as rollouts complete.
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from famou.config import ExperimentConfig
from famou.controller.engine import RolloutEngine
from famou.core.data import Context, Experiment, Program, RolloutResult
from famou.core.protocol import Rollout
from famou.infrastructure.env import ExecutionEnvironment, LocalEnv
from famou.infrastructure.llm import LLMClient
from famou.infrastructure.embedding import EmbeddingClient
from famou.infrastructure.logger import Logger
from famou.infrastructure.storage import DataService
from famou.utils.global_memory_store import (
    persist_program_memory,
    resolve_global_memory_dir,
)
from famou.utils.proof_experience import extract_and_attach_experiences
from famou.utils import generate_experiment_id
from famou.utils.color import Colors

from famou.modules.population import PopulationModule

Context.model_rebuild()


def _clip01(value: object) -> float:
    """Coerce arbitrary values into a stable [0, 1] range."""
    try:
        numeric = float(value or 0.0)
    except Exception:
        numeric = 0.0
    return max(0.0, min(1.0, numeric))

@dataclass
class _IslandTracker:
    """
    Tracks iteration assignment across islands using global iteration counter.

    Uses a global iteration counter that increments across all islands in round-robin fashion.
    This ensures:
    - Each program gets a unique iteration number
    - Iteration=0 remains reserved for seed programs
    - Rollout IDs can be unique per island-iteration pair
    - Archive ordering reflects chronological program creation

    The tracker automatically computes remaining iterations based on start_iteration
    and max_iterations. This correctly handles resume scenarios where start_iteration > 1.

    Usage (new experiment):
        tracker = _IslandTracker(num_islands=1, start_iteration=1, max_iterations=10)
        # remaining_iterations = 10 - 1 + 1 = 10
        # -> dispatches iterations 1, 2, ..., 10

    Usage (resume from iteration 5):
        tracker = _IslandTracker(num_islands=1, start_iteration=6, max_iterations=10)
        # remaining_iterations = 10 - 6 + 1 = 5
        # -> dispatches iterations 6, 7, 8, 9, 10

    Usage (multi-island):
        tracker = _IslandTracker(num_islands=3, start_iteration=1, max_iterations=10)
        # Round-robin with global counter
        # -> (0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6), (0, 7), (1, 8), (2, 9), (0, 10)
    """

    num_islands: int
    start_iteration: int
    max_iterations: int  # Target total iterations (from config)
    current_island_index: int = 0
    current_global_iteration: int = field(init=False)
    total_iterations_dispatched: int = 0
    remaining_iterations: int = field(init=False)  # Computed from start and max

    def __post_init__(self):
        # Global iteration counter starts from start_iteration
        self.current_global_iteration = self.start_iteration
        # Compute remaining iterations: max_iterations is the target total,
        # start_iteration - 1 is how many have been completed
        self.remaining_iterations = self.max_iterations - self.start_iteration + 1
        # Ensure non-negative (in case of misconfiguration)
        if self.remaining_iterations < 0:
            self.remaining_iterations = 0

    def get_next(self) -> Optional[Tuple[int, int]]:
        """
        Get the next (island_id, iteration) to execute using global iteration counter.

        Returns:
            (island_id, iteration) tuple, or None if all remaining iterations dispatched
        """
        # Check if we've dispatched all remaining iterations
        if self.total_iterations_dispatched >= self.remaining_iterations:
            return None

        # Get current island and global iteration
        island_id = self.current_island_index
        iteration = self.current_global_iteration

        # Advance counters
        self.current_global_iteration += 1
        self.total_iterations_dispatched += 1
        self.current_island_index = (self.current_island_index + 1) % self.num_islands

        return island_id, iteration

    def has_pending(self, island_id: int) -> bool:
        """Check if there are pending iterations."""
        return self.total_iterations_dispatched < self.remaining_iterations


class Evolver:
    """
    Main controller for evolutionary experiment.

    The Evolver:
    1. Manages the experiment lifecycle (iterations, population updates)
    2. Creates Context for each iteration (experiment-level, shared)
    3. Delegates rollout execution to RolloutEngine
    4. Updates Experiment state after each rollout
    5. Persists checkpoints (if storage provided)
    """

    _RESERVED_LOG_METRIC_KEYS = {
        "combined_score",
        "success_score",
        "search_score",
        "validity",
        "error_info",
    }

    @staticmethod
    def _assign_initial_programs_to_islands(
        initial_programs: List[Program],
        num_islands: int,
    ) -> Dict[int, List[Program]]:
        """
        Assign initial programs to islands using round-robin distribution.

        Args:
            initial_programs: List of initial programs to distribute
            num_islands: Number of islands to distribute across

        Returns:
            Dictionary mapping island_id to list of assigned initial programs
        """
        # Initialize empty lists for each island
        island_assignments = {i: [] for i in range(num_islands)}

        # Distribute programs using round-robin
        for i, program in enumerate(initial_programs):
            island_id = i % num_islands
            island_assignments[island_id].append(program)
        # The island would not be empty even if island number > initial program number
        for i in range(num_islands):
            if len(island_assignments[i]) == 0:
                island_assignments[i].append(initial_programs[i % len(initial_programs)])
        return island_assignments

    def _initialize_islands_from_initial_programs(
        self,
        initial_programs: List[Program],
        population_module: PopulationModule,
        num_islands: int,
        config: ExperimentConfig,
    ) -> Dict[int, Dict[str, List[Program]]]:
        """
        Initialize all islands from initial programs through PopulationModule.

        Works for both single island (num_islands=1) and multi-island cases.
        Initial programs are assigned to islands using round-robin distribution.
        Each program is added to its island through PopulationModule.update_population()
        to respect the PopulationModule's internal structure.

        Args:
            initial_programs: Initial programs to populate the islands
            population_module: Module that manages population structure
            num_islands: Number of islands to create
            config: Experiment configuration

        Returns:
            Dictionary mapping island_id to population dict
        """
        # Assign initial programs to islands (round-robin)
        island_program_assignments = self._assign_initial_programs_to_islands(
            initial_programs=initial_programs,
            num_islands=num_islands,
        )

        # Initialize each island by adding programs through PopulationModule
        island_populations = {}

        for island_id in range(num_islands):
            assigned_programs = island_program_assignments[island_id]
            population = {}  # Start with empty population

            # Add each program through PopulationModule
            for program in assigned_programs:
                # Deep copy to avoid shared references between islands
                program_copy = program.model_copy(deep=True)
                program_copy.meta["assigned_island"] = island_id

                # Let PopulationModule decide where/how to store it
                population = population_module.update_population(
                    current_population=population,
                    new_program=program_copy,
                    experiment_config=config,
                    island_id=island_id,
                )

            island_populations[island_id] = population

        return island_populations

    def __init__(
        self,
        config: ExperimentConfig,
        population_module: PopulationModule,
        initial_programs: Optional[List[Program]] = None,
        experiment: Optional[Experiment] = None,
        logger: Optional[Logger] = None,
        data_service: Optional[DataService] = None,
        llm_client: Optional[LLMClient] = None,
        env: Optional[ExecutionEnvironment] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        experiment_id: Optional[str] = None,
    ):
        """
        Initialize Evolver for new experiment or resume from checkpoint.

        Two usage modes:
        1. New experiment: Provide config + initial_programs
        2. Resume: Provide config + experiment

        Args:
            config: Experiment configuration (contains name, task_description, etc.)
            population_module: Population management strategy (REQUIRED)
            initial_programs: Initial programs for new experiments (unenriched)
            experiment: Loaded experiment for resume
            logger: Logger for output
            data_service: Storage service for checkpoints
            llm_client: LLM client for code generation
            env: Execution environment
            experiment_id: Optional experiment ID (only for new experiments)

        Raises:
            ValueError: If both or neither of initial_programs/experiment provided
            ValueError: If initial_programs is empty

        Examples:
            New experiment:
                >>> evolver = Evolver(
                ...     config=config,  # config.name used for experiment
                ...     population_module=TopKPopulation(),
                ...     initial_programs=[program1, program2],
                ...     ...
                ... )

            Resume from checkpoint:
                >>> experiment = storage.load_experiment("data/exp_123")
                >>> config = ExperimentConfig.from_yaml("data/exp_123/config.yaml")
                >>> evolver = Evolver(
                ...     config=config,
                ...     population_module=TopKPopulation(),
                ...     experiment=experiment,
                ...     ...
                ... )
        """
        # Validate usage mode
        if experiment is not None and initial_programs is not None:
            raise ValueError(
                "Cannot provide both 'experiment' and 'initial_programs'. "
                "Use 'initial_programs' for new experiments or 'experiment' for resume."
            )

        if experiment is None and initial_programs is None:
            raise ValueError(
                "Must provide either 'initial_programs' (new experiment) "
                "or 'experiment' (resume from checkpoint)"
            )

        # Store common attributes
        self.config = config
        self.logger = logger
        self.data_service = data_service
        self.population_module = population_module
        self.llm_client = llm_client
        self.env = env
        self.embedding_client = embedding_client

        # Mode 1: New experiment from initial programs
        if initial_programs is not None:
            if not initial_programs:
                raise ValueError("initial_programs cannot be empty")

            # Generate experiment ID
            if experiment_id is None:
                experiment_id = generate_experiment_id(config.name)

            # Initialize island populations from initial programs
            island_populations = self._initialize_islands_from_initial_programs(
                initial_programs=initial_programs,
                population_module=population_module,
                num_islands=config.num_islands,
                config=config,
            )

            # Create new experiment
            self.experiment = Experiment(
                id=experiment_id,
                name=config.name,
                island_populations=island_populations,
                num_islands=config.num_islands,
                archive=list(initial_programs),
                current_iteration=0,
            )

            # Initialize best program tracking from initial programs
            for program in initial_programs:
                if program.combined_score is not None or program.search_score is not None:
                    self.experiment.update_best_program(program)

            if logger:
                logger.info(
                    "Evolver initialized from initial programs",
                    experiment_id=self.experiment.id,
                    num_programs=len(initial_programs),
                    num_islands=config.num_islands,
                )

        # Mode 2: Resume from loaded experiment
        else:
            self.experiment = experiment

            if logger:
                logger.info(
                    "Evolver initialized from checkpoint",
                    experiment_id=self.experiment.id,
                    current_iteration=self.experiment.current_iteration,
                    archive_size=len(self.experiment.archive),
                )

        # Create RolloutEngine
        self.engine = RolloutEngine(
            llm_client=llm_client,
            env=env or LocalEnv(),
            logger=logger,
            embedding_client=embedding_client
        )
        self.global_memory_dir = self._resolve_global_memory_dir()

    def _save_initial_programs(self) -> None:
        """
        Save initial programs to storage.

        This should be called before evolution starts to ensure all programs
        in the initial archive are saved to individual files.

        Adds experiment_id to program metadata if not already present.
        """
        for program in self.experiment.archive:
            # Add experiment_id if not present
            if "experiment_id" not in program.meta:
                program.meta["experiment_id"] = self.experiment.id

            # Save program to file
            # print(f"Saving initial program to storage: {program}")
            self.data_service.save_program(program)

    def _resolve_global_memory_dir(self):
        """Resolve global persistent memory directory from runtime/storage settings."""
        storage_base_path = None
        if self.data_service is not None and hasattr(self.data_service, "base_path"):
            storage_base_path = getattr(self.data_service, "base_path")
        runtime_config = {}
        explicit_dir = getattr(self.config, "global_memory_dir", None)
        if explicit_dir:
            runtime_config["global_memory_dir"] = explicit_dir
        return resolve_global_memory_dir(runtime_config, default_base_path=storage_base_path)

    def _has_solved_program(self) -> bool:
        """Check whether the archive already contains a solved theorem candidate."""
        for program in self.experiment.archive:
            if self._program_is_solved(program):
                return True
        return False

    @staticmethod
    def _program_is_solved(program: Optional[Program]) -> bool:
        """Treat a candidate as solved when either top-level success or theorem flag is final."""
        if program is None:
            return False
        if float(program.success_score or 0.0) >= 0.999999:
            return True
        if not isinstance(program.metrics, dict):
            return False
        return float(program.metrics.get("theorem_solved") or 0.0) >= 0.999999

    def _result_is_solved(self, result: Optional[RolloutResult]) -> bool:
        """Check whether a rollout produced a theorem that is already fully solved."""
        if result is None or result.generated_program is None:
            return False
        return self._program_is_solved(result.generated_program)

    def _extract_enrichment_modules(self, rollout: Rollout) -> List:
        """
        Extract enrichment modules from rollout.

        Enrichment modules are those that add data to programs:
        - EvaluateModule: Adds scores/metrics
        - LLMJudge: Adds LLM analysis
        - (Future: VerifyModule, ProfileModule, etc.)

        Selection/Generation modules are skipped (they create new programs).

        Args:
            rollout: Rollout pipeline

        Returns:
            List of enrichment modules
        """
        from famou.modules.evaluate import EvaluateModule
        from famou.modules.judge import JudgeModule
        from famou.modules.feature import FeatureModule

        enrichment_modules = []

        for module in rollout.modules:
            if isinstance(module, (EvaluateModule, JudgeModule, FeatureModule)):
                enrichment_modules.append(module)

        return enrichment_modules

    def _enrich_initial_programs(self, rollout: Rollout) -> None:
        """
        Enrich initial programs using enrichment modules from rollout.

        Extracts enrichment modules (EvaluateModule, LLMJudge, etc.) from
        the rollout and applies them to initial programs in archive using
        RolloutEngine for consistent execution, retry logic, and error handling.

        This ensures initial programs go through the same enrichment pipeline
        as generated programs, providing consistent treatment.

        Args:
            rollout: Rollout pipeline containing enrichment modules
        """
        enrichment_modules = self._extract_enrichment_modules(rollout)

        if self.logger:
            self.logger.info(
                f"Enriching {len(self.experiment.archive)} initial programs",
                num_modules=len(enrichment_modules),
            )

        for i, program in enumerate(self.experiment.archive):
            # Create a minimal context for enrichment
            context = Context(
                experiment_id=self.experiment.id,
                task_description=self.config.task_description,
                task_constraint=self.config.task_constraint,
                population={},  # Empty for initial enrichment
                island_id=0,
                config=self._build_runtime_config(island_id=0),
                experiment_config=self.config,
            )

            # Use RolloutEngine to enrich program (includes retry logic, error handling)
            result = self.engine.execute_enrichment(
                program=program,
                modules=enrichment_modules,
                context=context,
                rollout_id=f"init_{i}",
            )
            print(f"{Colors.BOLD}{Colors.BLUE}  Enrichment result: {Colors.CYAN}{result}{Colors.RESET}")

            if result.generated_program is not None:
                extract_and_attach_experiences(result.generated_program)
                self._persist_global_memory(result.generated_program, parent_program=None)

            # Update archive with enriched program
            self.experiment.archive[i] = result.generated_program

            if self.logger:
                score = result.generated_program.combined_score or 0.0
                search_score = result.generated_program.effective_search_score
                validity = result.generated_program.validity or 0.0
                status = result.status
                self.logger.info(
                    f"Initial program enriched",
                    program_id=program.id,
                    score=f"{score:.6f}",
                    search_score=f"{search_score:.6f}",
                    validity=f"{validity:.2f}",
                    status=status,
                )

        # print("# of Enriched Programs:", len(self.experiment.archive))
        # Update island populations with enriched programs
        island_populations = self._initialize_islands_from_initial_programs(
            initial_programs=self.experiment.archive,
            population_module=self.population_module,
            num_islands=self.config.num_islands,
            config=self.config,
        )
        self.experiment.island_populations = island_populations
        # print(f"Initialized island_populations: {island_populations}")

        if self.logger:
            self.logger.info("Initial programs enrichment complete")

    def run(self, rollout: Rollout) -> Experiment:
        """
        Run the evolutionary experiment with streaming execution.

        Uses the configured experiment, config, logger, and storage.
        Takes a Rollout pipeline that defines the evolution steps.

        Works for both new and resumed experiments:
        - New (iteration=0): Enriches initial programs first
        - Resume (iteration>0): Continues evolution directly

        Args:
            rollout: Rollout pipeline with modules (Select -> Generate -> Evaluate)

        Returns:
            Updated Experiment with final population
        """
        # Enrich initial programs ONLY for new experiments
        if self.experiment.current_iteration == 0:
            if self.logger:
                self.logger.info("New experiment detected, enriching initial programs")
            self._enrich_initial_programs(rollout)

            # Save enriched programs
            if self.data_service:
                self._save_initial_programs()

            if self._has_solved_program():
                if self.logger:
                    self.logger.info(
                        "Stopping after initial enrichment because a solved program already exists",
                        experiment_id=self.experiment.id,
                    )
                return self.experiment

        # Store rollout for use in _execute_rollout
        self.current_rollout = rollout

        # Unified execution - single method handles both modes
        return self._run_evolution()

    def _run_evolution(self) -> Experiment:
        """
        Unified streaming evolution for both single and multi-island modes.

        This method replaces _run_single_island() and _run_multi_island(),
        handling both cases through the _IslandTracker abstraction.

        Returns:
            Updated Experiment with final population
        """
        # After the initial-program-enrichment finished
        max_iterations = self.config.max_iterations
        # Start from current_iteration + 1 (iteration 0 is reserved for seeds)
        # For new experiments: current_iteration=0 → start_iteration=1
        # For resume at iter 5: current_iteration=5 → start_iteration=6
        start_iteration = self.experiment.current_iteration + 1
        max_workers = self.config.max_workers
        num_islands = self.config.num_islands
        execution_config = getattr(self.current_rollout, "execution_config", {}) or {}
        global_stop_on_solved = bool(execution_config.get("global_stop_on_solved", False))

        # Initialize tracker with global iteration counter
        tracker = _IslandTracker(
            num_islands=num_islands,
            start_iteration=start_iteration,
            max_iterations=max_iterations
        )

        #region ===  Log experiment details (global information) ===
        if self.logger:
            self.logger.separator("=", 80)
            log_msg = f"Starting evolution: {self.experiment.name}"
            if num_islands > 1:
                log_msg = f"Starting multi-island evolution: {self.experiment.name}"

            self.logger.info(
                log_msg,
                experiment_id=self.experiment.id,
                num_islands=num_islands,
                island_size=self.config.island_size if num_islands > 1 else None,
                migration_interval=self.config.migration_interval if num_islands > 1 else None,
                start_iteration=start_iteration,
                max_iterations=max_iterations,
                max_workers=max_workers,
            )
            self.logger.separator("=", 80)
        #endregion

        # Use ThreadPoolExecutor for parallel rollouts
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Track futures: future → (island_id, iteration)
            futures = {}

            # Track completion for checkpointing and migration
            total_completed = 0
            completed_per_island = {i: 0 for i in range(num_islands)}

            # Submit initial batch of rollouts
            # Limit by max_workers (parallel capacity) and remaining iterations
            initial_batch_size = min(max_workers, tracker.remaining_iterations)
            for _ in range(initial_batch_size):
                next_task = tracker.get_next()
                if next_task is None:
                    break
                # print(f"Next_task: {next_task}")
                island_id, iteration = next_task
                future = executor.submit(self._execute_rollout, island_id, iteration)
                futures[future] = (island_id, iteration)

            # print(f"{Colors.BLUE}{Colors.BOLD}Futures: {Colors.CYAN}{futures}{Colors.RESET}")
            # Process rollouts as they complete
            # Use while loop because as_completed only iterates over initial futures
            stop_requested = False
            while futures:
                # Wait for the next completed future, as long as there a finished task, the program won't wait
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    island_id, iteration = futures[future]
                    del futures[future]  # Remove from tracking

                    try:
                        result = future.result()

                        # Update experiment with rollout result
                        self._update_experiment(result, island_id)
                        completed_per_island[island_id] += 1
                        total_completed += 1

                        # Perform migration if needed (multi-island only)
                        if num_islands > 1:
                            migration_interval = self.config.migration_interval
                            migration_size = self.config.migration_size
                            reset_interval = self.config.reset_interval
                            reset_size = self.config.reset_size
                            if (
                                reset_interval
                                and total_completed % reset_interval == 0
                            ):
                                self._perform_reset(reset_size)
                            elif (
                                migration_interval 
                                and total_completed % migration_interval == 0
                            ):
                                self._perform_migration(migration_size)

                        # Save checkpoint if needed
                        if self.data_service and self.config.checkpoint_interval > 0:
                            if total_completed % self.config.checkpoint_interval == 0:
                                self._save_checkpoint()

                        # Stats table removed for cleaner logs (metrics already shown in iteration_result)
                        if global_stop_on_solved and self._result_is_solved(result):
                            stop_requested = True
                            if self.logger:
                                self.logger.info(
                                    "Stopping evolution early because a solved theorem was found",
                                    experiment_id=self.experiment.id,
                                    iteration=iteration,
                                    island_id=island_id if num_islands > 1 else None,
                                    program_id=result.generated_program.id,
                                )

                            for pending_future in list(futures.keys()):
                                pending_future.cancel()
                                del futures[pending_future]
                            break

                        # Launch new rollout if work remains
                        next_task = tracker.get_next()
                        if next_task is not None:
                            new_island_id, new_iteration = next_task
                            new_future = executor.submit(
                                self._execute_rollout, new_island_id, new_iteration
                            )
                            futures[new_future] = (new_island_id, new_iteration)

                    except Exception as e:
                        if self.logger:
                            log_context = {
                                "iteration": iteration,
                                "island_id": island_id,
                                "error": str(e),
                            }
                            self.logger.error("Rollout failed", **log_context)
                        # Continue with other rollouts

                if stop_requested:
                    break

        #region === Log completion === (The Completion of the total program)
        if self.logger:
            # Calculate total population across all islands
            total_population = sum(
                len(bucket_progs)
                for island_pop in self.experiment.island_populations.values()
                for bucket_progs in island_pop.values()
            )
            self.logger.separator("=", 80)

            log_msg = "Evolution complete" if num_islands == 1 else "Multi-island evolution complete"
            self.logger.info(
                log_msg,
                experiment_name=self.experiment.name,
                experiment_id=self.experiment.id,
                final_iteration=self.experiment.current_iteration,
                num_islands=num_islands,
                total_population=total_population,
                archive_size=len(self.experiment.archive),
            )

            # Log best program per island (for all cases, including single-island)
            for island_id in range(num_islands):
                island_pop = self.experiment.get_island_population(island_id)
                all_programs = [
                    p for bucket_progs in island_pop.values() for p in bucket_progs
                ]
                if all_programs:
                    best = max(all_programs, key=lambda p: p.ranking_score)
                    msg = (
                        f"Island {island_id}: best program {best.id[:8]}... "
                        f"(success={best.success_score:.4f}, "
                        f"search={best.effective_search_score:.4f}, "
                        f"population={len(all_programs)})"
                    )
                    self.logger.info(msg)
        #endregion

        # Final checkpoint
        if self.data_service:
            self._save_checkpoint()

        return self.experiment

    def _execute_rollout(self, island_id: int, iteration: int) -> RolloutResult:
        """
        Execute a single rollout for a specific island.

        Unified method that handles both single-island (island_id=0) and
        multi-island execution.

        Args:
            island_id: Island ID (0 for single-island mode)
            iteration: Iteration number

        Returns:
            RolloutResult with evaluated programs

        Thread Safety:
            - Reads island population (thread-safe, no mutations)
            - Does not write to shared state
            - All rollout state is isolated in Context and RolloutResult
        """
        # See LocalLogger._format_iteration_header for more information, at local_logger.py
        if self.logger:
            # print("have logger")
            # Always include island_id for consistency (single-island uses island 0)
            log_context = {
                "iteration": iteration,
                "max_iterations": self.config.max_iterations,
            }
            # Only show island_id in context if multi-island
            if self.config.num_islands > 1:
                log_context["island_id"] = island_id

            # print(f"logger information: {log_context}")
            self.logger.info("Starting iteration", **log_context)

        # Create context with appropriate population snapshot
        context = self._create_context(iteration, island_id)

        execution_config = getattr(self.current_rollout, "execution_config", {}) or {}
        local_island_budget, budget_summary = self._allocate_local_island_budget(
            context,
            execution_config,
        )
        context.config = {
            **context.config,
            "allocated_local_island_budget": local_island_budget,
            "local_island_budget_summary": budget_summary,
        }

        if local_island_budget > 1:
            result = self._execute_local_island_rollout(
                context=context,
                island_id=island_id,
                iteration=iteration,
                budget=local_island_budget,
                execution_config=execution_config,
                budget_summary=budget_summary,
            )
        else:
            result = self.engine.execute_rollout(self.current_rollout, context, iteration)
            result.stats["local_island_budget"] = local_island_budget
            result.stats["local_island_budget_summary"] = budget_summary
            if result.generated_program is not None:
                result.generated_program.meta["local_island_budget"] = local_island_budget
                result.generated_program.meta["local_island_budget_summary"] = budget_summary

        return result

    def _allocate_local_island_budget(
        self,
        context: Context,
        execution_config: Dict[str, object],
    ) -> Tuple[int, Dict[str, object]]:
        """
        Resolve the local island budget for the current outer rollout.

        Default behavior stays fixed. Adaptive mode allocates more budget to plans
        that already show strong realization signal under bounded cost.
        """

        def _clip01(value: object) -> float:
            try:
                numeric = float(value or 0.0)
            except Exception:
                numeric = 0.0
            return min(1.0, max(0.0, numeric))

        base_budget = max(1, int(execution_config.get("local_island_budget", 1) or 1))
        mode = str(execution_config.get("local_island_budget_mode", "fixed") or "fixed").strip().lower()
        min_budget = max(1, int(execution_config.get("local_island_budget_min", 1) or 1))
        max_budget = max(min_budget, int(execution_config.get("local_island_budget_max", base_budget) or base_budget))
        if base_budget < min_budget:
            base_budget = min_budget
        if base_budget > max_budget:
            base_budget = max_budget

        if base_budget <= 1 or mode != "adaptive":
            return base_budget, {
                "mode": "fixed" if mode != "adaptive" else "adaptive",
                "base_budget": base_budget,
                "min_budget": min_budget,
                "max_budget": max_budget,
                "allocated_budget": base_budget,
                "anchor_program_id": None,
                "reason": "fixed_budget" if mode != "adaptive" else "single_step_budget",
            }

        candidates = context.get_all_programs()
        if not candidates:
            return base_budget, {
                "mode": "adaptive",
                "base_budget": base_budget,
                "min_budget": min_budget,
                "max_budget": max_budget,
                "allocated_budget": base_budget,
                "anchor_program_id": None,
                "reason": "no_population_snapshot",
            }

        anchor = max(candidates, key=lambda program: program.ranking_score)
        metrics = anchor.metrics or {}
        success_score = _clip01(anchor.success_score)
        theorem_solved = _clip01(metrics.get("theorem_solved", success_score))
        search_score = _clip01(anchor.effective_search_score)
        progress_score = _clip01(metrics.get("progress_score", search_score))
        cost_penalty = _clip01(metrics.get("cost_penalty", 0.0))
        frontier_size = max(0.0, float(metrics.get("frontier_size", 0.0) or 0.0))
        blocked_size = max(0.0, float(metrics.get("blocked_size", 0.0) or 0.0))

        if theorem_solved >= 1.0 or success_score >= 1.0:
            return min_budget, {
                "mode": "adaptive",
                "base_budget": base_budget,
                "min_budget": min_budget,
                "max_budget": max_budget,
                "allocated_budget": min_budget,
                "anchor_program_id": anchor.id,
                "reason": "anchor_already_solved",
                "success_score": success_score,
                "search_score": search_score,
                "progress_score": progress_score,
                "cost_penalty": cost_penalty,
                "frontier_size": frontier_size,
                "blocked_size": blocked_size,
            }

        total_open = frontier_size + blocked_size
        topology_opportunity = 0.5 if total_open <= 0.0 else frontier_size / total_open
        promise_score = max(search_score, progress_score)
        signal = (
            0.55 * promise_score
            + 0.25 * topology_opportunity
            + 0.20 * (1.0 - cost_penalty)
        )
        if blocked_size > frontier_size and total_open > 0.0:
            signal -= 0.10
        signal = min(1.0, max(0.0, signal))

        allocated_budget = min_budget + round((max_budget - min_budget) * signal)
        allocated_budget = max(min_budget, min(max_budget, allocated_budget))

        return allocated_budget, {
            "mode": "adaptive",
            "base_budget": base_budget,
            "min_budget": min_budget,
            "max_budget": max_budget,
            "allocated_budget": allocated_budget,
            "anchor_program_id": anchor.id,
            "reason": "adaptive_realization_allocation",
            "success_score": success_score,
            "search_score": search_score,
            "progress_score": progress_score,
            "cost_penalty": cost_penalty,
            "frontier_size": frontier_size,
            "blocked_size": blocked_size,
            "topology_opportunity": topology_opportunity,
            "allocation_signal": signal,
        }

    def _execute_local_island_rollout(
        self,
        context: Context,
        island_id: int,
        iteration: int,
        budget: int,
        execution_config: Dict[str, object],
        budget_summary: Optional[Dict[str, object]] = None,
    ) -> RolloutResult:
        """
        Execute fixed-budget local search for one plan island.

        This keeps the controller change minimal: reuse the existing rollout as the
        inner operator, but maintain a temporary local population and only commit the
        selected best result back to the global experiment.
        """
        stop_on_solved = bool(execution_config.get("local_island_stop_on_solved", True))
        return_mode = str(execution_config.get("local_island_return_mode", "best"))
        inner_rollout = execution_config.get("local_island_inner_rollout")

        local_population = {
            bucket_id: [program.model_copy(deep=True) for program in programs]
            for bucket_id, programs in context.population.items()
        }
        local_history: List[Dict[str, object]] = []
        best_result: Optional[RolloutResult] = None
        best_score = float("-inf")
        last_result: Optional[RolloutResult] = None

        def _record_result(local_step: int, phase: str, rollout_used, result: RolloutResult) -> bool:
            nonlocal local_population, best_result, best_score

            score = None
            theorem_solved = False
            if result.generated_program is not None:
                score = result.generated_program.effective_search_score
                theorem_solved = bool(result.generated_program.metrics.get("theorem_solved"))
                result.generated_program.meta["outer_iteration"] = iteration
                result.generated_program.meta["local_island_step"] = local_step
                result.generated_program.meta["local_island_budget"] = budget
                result.generated_program.meta["local_island_phase"] = phase
                result.generated_program.meta["local_island_rollout_name"] = rollout_used.name

            local_history.append({
                "local_step": local_step,
                "phase": phase,
                "rollout_name": rollout_used.name,
                "rollout_id": result.rollout_id,
                "status": result.status,
                "score": score,
                "theorem_solved": theorem_solved,
                "program_id": result.generated_program.id if result.generated_program else None,
            })

            if result.status == "success" and result.generated_program is not None:
                local_population = self.population_module.update_population(
                    current_population=local_population,
                    new_program=result.generated_program,
                    experiment_config=self.config,
                    island_id=island_id,
                )

                current_score = result.generated_program.ranking_score
                if current_score is not None and current_score >= best_score:
                    best_result = result
                    best_score = current_score

                if stop_on_solved and theorem_solved:
                    return True
            return False

        next_local_step = 0

        if inner_rollout is not None and budget >= 1:
            local_context = context.model_copy(deep=True)
            local_context.population = local_population
            local_context.config = {
                **local_context.config,
                "local_island_step": 0,
                "local_island_budget": budget,
                "local_island_phase": "outer_mutation",
            }
            local_context.iteration = iteration

            inner_iteration = self._make_local_iteration_id(iteration, 0)
            result = self.engine.execute_rollout(
                self.current_rollout,
                local_context,
                inner_iteration,
            )
            last_result = result
            next_local_step = 1
            if _record_result(0, "outer_mutation", self.current_rollout, result):
                chosen = best_result if return_mode == "best" else last_result
                if chosen is None:
                    assert last_result is not None
                    chosen = last_result

                chosen.stats["local_island_history"] = local_history
                chosen.stats["local_island_budget"] = budget
                chosen.stats["local_island_budget_summary"] = budget_summary or {}
                chosen.stats["local_island_steps_executed"] = len(local_history)
                chosen.stats["local_island_return_mode"] = return_mode
                chosen.stats["local_island_best_score"] = None if best_score == float("-inf") else best_score

                if chosen.generated_program is not None:
                    chosen.generated_program.meta["local_island_summary"] = {
                        "budget": budget,
                        "steps_executed": len(local_history),
                        "return_mode": return_mode,
                        "best_score": None if best_score == float("-inf") else best_score,
                        "budget_summary": budget_summary or {},
                        "history": local_history,
                    }

                return chosen

        for local_step in range(next_local_step, budget):
            rollout_used = inner_rollout or self.current_rollout
            phase = "inner_repair" if inner_rollout is not None else "local_search"
            local_context = context.model_copy(deep=True)
            local_context.population = local_population
            local_context.config = {
                **local_context.config,
                "local_island_step": local_step,
                "local_island_budget": budget,
                "local_island_phase": phase,
            }
            local_context.iteration = iteration

            inner_iteration = self._make_local_iteration_id(iteration, local_step)
            result = self.engine.execute_rollout(
                rollout_used,
                local_context,
                inner_iteration,
            )
            last_result = result
            if _record_result(local_step, phase, rollout_used, result):
                break

        chosen = best_result if return_mode == "best" else last_result
        if chosen is None:
            assert last_result is not None
            chosen = last_result

        chosen.stats["local_island_history"] = local_history
        chosen.stats["local_island_budget"] = budget
        chosen.stats["local_island_budget_summary"] = budget_summary or {}
        chosen.stats["local_island_steps_executed"] = len(local_history)
        chosen.stats["local_island_return_mode"] = return_mode
        chosen.stats["local_island_best_score"] = None if best_score == float("-inf") else best_score

        if chosen.generated_program is not None:
            chosen.generated_program.meta["local_island_summary"] = {
                "budget": budget,
                "steps_executed": len(local_history),
                "return_mode": return_mode,
                "best_score": None if best_score == float("-inf") else best_score,
                "budget_summary": budget_summary or {},
                "history": local_history,
            }

        return chosen

    def _make_local_iteration_id(self, iteration: int, local_step: int) -> int:
        """Generate a rollout-local iteration id without colliding inside one outer step."""
        return iteration * 1000 + local_step


    def _perform_migration(self, migration_size: int) -> None:
        """
        Perform migration between islands using configured topology.

        Programs are sampled based on combined_score (weighted random sampling)
        rather than deterministically selecting top programs. This introduces
        diversity while still favoring higher-scored programs.

        Args:
            migration_size: Number of programs to migrate per island

        Currently supports ring topology: island i → island (i+1) % num_islands
        """
        num_islands = self.config.num_islands
        topology = self.config.migration_topology

        if topology == "ring":
            # Ring topology: each island sends to next island


            total_added = 0
            to_migrant = []
            for island_id in range(num_islands):
                source_pop = self.experiment.get_island_population(island_id)
                source_programs = self._unique_programs_by_id(
                    self._flatten_population(source_pop)
                )
                # Sample migrants based on combined_score (weighted random)
                migrants = self._sample_by_score(source_programs, migration_size)
                to_migrant.append(migrants)
            for island_id in range(num_islands):
                target_id = (island_id + 1) % num_islands
                migrants = to_migrant[island_id]
                target_pop = self.experiment.get_island_population(target_id)
                updated_pop, added = self._apply_programs_to_population(
                    current_population=target_pop,
                    programs=migrants,
                    island_id=target_id,
                )
                total_added += added
                self.experiment.island_populations[target_id] = updated_pop

            if self.logger:
                self.logger.info(
                    "Migration performed",
                    topology="ring",
                    num_islands=num_islands,
                    programs_per_island=migration_size,
                    programs_added=total_added,
                )

        # TODO: Support other topologies (full, star)

    def _perform_reset(self, reset_size: int) -> None:
        """
        Reset the worst-performing half of islands and reset from top donors.

        Worst-performing is determined by each island's best program score.
        """
        num_islands = self.config.num_islands
        if num_islands < 2:
            return

        worst_count = num_islands // 2
        if worst_count == 0:
            return

        island_scores = []
        for island_id in range(num_islands):
            island_pop = self.experiment.get_island_population(island_id)
            best_program = self._best_program_in_population(island_pop)
            best_score = best_program.ranking_score if best_program is not None else float("-inf")
            island_scores.append((island_id, best_score))

        island_scores.sort(key=lambda item: item[1])
        worst_islands = [island_id for island_id, _ in island_scores[:worst_count]]
        donor_islands = [island_id for island_id, _ in island_scores[worst_count:]]

        donor_programs: List[Program] = []
        for donor_id in donor_islands:
            donor_pop = self.experiment.get_island_population(donor_id)
            donor_programs.extend(self._flatten_population(donor_pop))

        donor_programs = self._unique_programs_by_id(donor_programs)
        donors = self._top_scored_programs(donor_programs, reset_size)
        if not donors:
            return

        for island_id in worst_islands:
            self.experiment.island_populations[island_id] = {}
            updated_pop, added = self._apply_programs_to_population(
                current_population={},
                programs=donors,
                island_id=island_id,
            )
            self.experiment.island_populations[island_id] = updated_pop

        self.experiment.updated_at = time.time()

        if self.logger:
            self.logger.info(
                "reset performed",
                num_islands=num_islands,
                reseted_islands=worst_islands,
                donor_islands=donor_islands,
                programs_per_island=reset_size,
            )

    def _build_runtime_config(self, island_id: int = 0) -> Dict[str, object]:
        """Build lightweight runtime config shared with selection/generation modules."""
        config: Dict[str, object] = {
            "language": self.config.language,
        }
        if self.global_memory_dir is not None:
            config["global_memory_dir"] = str(self.global_memory_dir)
        if self.config.num_islands == 1:
            config["max_iterations"] = self.config.max_iterations
        else:
            config["island_id"] = island_id
            specialization = self._build_island_specialization(island_id)
            if specialization is not None:
                config["island_specialization"] = specialization
        return config

    def _build_island_specialization(self, island_id: int) -> Optional[Dict[str, object]]:
        """
        Assign a soft outer-search role to each island.

        The specialization only exists in multi-island mode. It is deliberately a
        prompt/runtime prior instead of a hard controller rule so that strong
        verifier evidence can still override the island's default behavior.
        """
        if self.config.num_islands <= 1:
            return None

        specializations = [
            {
                "mode": "conservative_refinement",
                "reason": "exploit_current_plan",
                "instruction": (
                    "Prefer refining the current plan, reuse verified lemmas, and keep "
                    "the subgoal DAG stable unless the verifier indicates the plan is blocked."
                ),
            },
            {
                "mode": "memory_guided_replanning",
                "reason": "restructure_with_memory",
                "instruction": (
                    "When progress stalls, use retrieved theorem and lemma memories to "
                    "rebuild the plan more aggressively instead of only polishing local nodes."
                ),
            },
            {
                "mode": "crossover_recombination",
                "reason": "merge_complementary_plans",
                "instruction": (
                    "Actively look for compatible inspirations with complementary subgoals, "
                    "verified lemmas, or strategy types, and prefer coherent crossover over "
                    "isolated mutation when that structure looks promising."
                ),
            },
        ]
        selected = dict(specializations[island_id % len(specializations)])
        selected["island_id"] = island_id
        selected["num_islands"] = self.config.num_islands
        return selected

    def _create_context(self, iteration: int, island_id: int = 0) -> Context:
        """
        Create Context for a rollout.

        Context is shared across all modules in the rollout and contains:
        - Experiment ID
        - Task description (what problem to solve)
        - Current population (for selection)
        - Config (test cases, etc.)

        Args:
            iteration: Current iteration number
            island_id: Island ID (default: 0). For multi-island mode, only this
                      island's population is included in the context.

        Returns:
            Context instance
        """
        # Always use get_island_population for consistency
        island_population = self.experiment.get_island_population(island_id)

        # Compute population diversity from feature vectors
        pop_programs = []
        for bucket_progs in island_population.values():
            pop_programs.extend(bucket_progs)
        diversity = self._compute_population_diversity(pop_programs)

        config = self._build_runtime_config(island_id=island_id)

        return Context(
            experiment_id=self.experiment.id,
            task_description=self.config.task_description,
            task_constraint=self.config.task_constraint,
            population=island_population,
            island_id=island_id,
            config=config,
            experiment_config=self.config,
            diversity=diversity,
            iteration=iteration,
            created_at=time.time(),
        )

    def _flatten_population(
        self, population: Dict[str, List[Program]]
    ) -> List[Program]:
        """Flatten population buckets into a list of programs."""
        programs: List[Program] = []
        for bucket_progs in population.values():
            programs.extend(bucket_progs)
        return programs

    def _unique_programs_by_id(self, programs: List[Program]) -> List[Program]:
        """Deduplicate programs by ID while preserving order."""
        seen_ids = set()
        unique: List[Program] = []
        for program in programs:
            if program.id in seen_ids:
                continue
            seen_ids.add(program.id)
            unique.append(program)
        return unique

    def _top_scored_programs(
        self, programs: List[Program], limit: int
    ) -> List[Program]:
        """Select top scored programs, skipping those without scores."""
        scored = [p for p in programs if p.combined_score is not None or p.search_score is not None]
        scored.sort(key=lambda p: p.ranking_score, reverse=True)
        return scored[:limit]

    def _sample_by_score(
        self, programs: List[Program], count: int
    ) -> List[Program]:
        """
        Sample programs with probability proportional to search score.

        Uses softmax-style weighting to convert scores to probabilities.
        Programs without scores are excluded.

        Args:
            programs: List of programs to sample from
            count: Number of programs to sample

        Returns:
            List of sampled programs (without replacement)
        """
        import random
        import math

        # Filter to programs with scores
        scored = [p for p in programs if p.combined_score is not None or p.search_score is not None]
        if not scored:
            return []

        # Limit count to available programs
        count = min(count, len(scored))
        if count == 0:
            return []

        # Extract scores and compute softmax weights
        scores = [p.effective_search_score for p in scored]
        max_score = max(scores)

        # Use softmax with temperature=1 to convert scores to probabilities
        # Shift by max_score for numerical stability
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        weights = [e / total for e in exp_scores]

        # Sample without replacement using weighted random selection
        sampled: List[Program] = []
        remaining = list(zip(scored, weights))

        for _ in range(count):
            if not remaining:
                break

            # Normalize weights for remaining programs
            total_weight = sum(w for _, w in remaining)
            if total_weight == 0:
                break

            # Select one program based on weights
            r = random.random() * total_weight
            cumulative = 0.0
            selected_idx = 0

            for idx, (_, w) in enumerate(remaining):
                cumulative += w
                if r <= cumulative:
                    selected_idx = idx
                    break

            # Add selected program and remove from remaining
            selected_program, _ = remaining.pop(selected_idx)
            sampled.append(selected_program)

        return sampled

    def _best_program_in_population(
        self, population: Dict[str, List[Program]]
    ) -> Optional[Program]:
        """Get the best program in a population (by ranking score)."""
        programs = [
            p
            for p in self._flatten_population(population)
            if p.combined_score is not None or p.search_score is not None
        ]
        if not programs:
            return None
        return max(programs, key=lambda p: p.ranking_score)

    @classmethod
    def _build_iteration_metrics(cls, program: Program) -> Dict[str, object]:
        """
        Build iteration metrics for logging while preserving evaluator-owned
        top-level results.
        """
        metrics = {
            "combined_score": program.combined_score,
            "success_score": program.success_score,
            "search_score": program.effective_search_score,
            "validity": program.validity,
        }
        if program.error_info is not None:
            metrics["error_info"] = program.error_info
        if program.metrics:
            metrics.update(
                {
                    key: value
                    for key, value in program.metrics.items()
                    if key not in cls._RESERVED_LOG_METRIC_KEYS
                }
            )
        return metrics

    def _persist_global_memory(
        self,
        program: Optional[Program],
        parent_program: Optional[Program] = None,
    ) -> None:
        """Persist theorem/subgoal/lemma memories when global memory is enabled."""
        if program is None or self.global_memory_dir is None:
            return
        counts = persist_program_memory(
            program,
            self.global_memory_dir,
            parent_program=parent_program,
        )
        program.meta["global_memory_counts"] = counts

    def _apply_programs_to_population(
        self,
        current_population: Dict[str, List[Program]],
        programs: List[Program],
        island_id: int,
    ) -> Tuple[Dict[str, List[Program]], int]:
        """
        Add programs to a population via PopulationModule with dedupe.

        Returns updated population and number of programs retained.
        """
        existing_ids = {p.id for p in self._flatten_population(current_population)}
        updated_population = current_population
        added = 0

        for program in programs:
            if program.id in existing_ids:
                continue
            program_copy = program.model_copy(deep=True)
            program_copy.meta["assigned_island"] = island_id
            updated_population = self.population_module.update_population(
                current_population=updated_population,
                new_program=program_copy,
                experiment_config=self.config,
                island_id=island_id,
            )
            updated_ids = {p.id for p in self._flatten_population(updated_population)}
            if program.id in updated_ids:
                added += 1
            existing_ids = updated_ids

        return updated_population, added

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine distance between two vectors.

        Cosine distance = 1 - cosine_similarity
        Cosine similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine distance in range [0, 2], where 0 means identical direction
        """
        import math

        if not vec1 or not vec2:
            return 1.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 1.0

        return 1.0 - (dot / (norm1 * norm2))

    def _compute_population_diversity(self, programs: List[Program]) -> float:
        """
        Compute population diversity as average pairwise cosine distance.

        Uses feature vectors of programs to measure diversity. Higher values
        indicate more diverse populations.

        Args:
            programs: List of programs to analyze

        Returns:
            Average pairwise cosine distance (0 = all identical, 2 = maximally diverse)
        """
        if len(programs) < 2:
            return 0.0

        # Extract feature vectors from programs
        vectors = [p.feature_vector for p in programs if p.feature_vector]
        if len(vectors) < 2:
            return 0.0

        # Compute average pairwise cosine distance
        total_dist = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_dist += self._cosine_distance(vectors[i], vectors[j])
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _update_experiment(self, result: RolloutResult, island_id: int = 0) -> None:
        """
        Update experiment state with rollout result.

        Only successful rollouts update the population. Failed rollouts are
        recorded in history but do not add programs to the population.

        Updates:
        - Add new programs to population (only if rollout succeeded)
        - Append result to rollout history
        - Increment iteration counter
        - Update timestamp

        Args:
            result: RolloutResult from completed rollout
            island_id: Island ID to update (default: 0 for single-island mode)
        """
        # Track old best for comparison
        old_best_score = None
        old_best_id = None
        if self.logger:
            island_pop = self.experiment.get_island_population(island_id)
            all_programs_before = [
                p for bucket_progs in island_pop.values() for p in bucket_progs
            ]
            if all_programs_before:
                best_before = max(all_programs_before, key=lambda p: p.ranking_score)
                if best_before.ranking_score:
                    old_best_score = best_before.ranking_score
                    old_best_id = best_before.id

        # Only update population if rollout succeeded
        if result.status == "success" and result.generated_program:
            parent_program = None
            if result.selected_parent_id:
                parent_program = self.experiment.get_program_by_id(result.selected_parent_id)
            extract_and_attach_experiences(
                result.generated_program,
                parent_program=parent_program,
            )
            self._persist_global_memory(
                result.generated_program,
                parent_program=parent_program,
            )

            # Save program to individual file
            if self.data_service:
                self.data_service.save_program(result.generated_program)

            # Add new program to archive (all programs ever created)
            self.experiment.archive.append(result.generated_program)

            # Update best program tracking
            self.experiment.update_best_program(result.generated_program)

            # Get current island population
            current_pop = self.experiment.get_island_population(island_id)

            # Update population using PopulationModule (required)
            updated_population = self.population_module.update_population(
                current_population=current_pop,
                new_program=result.generated_program,
                experiment_config=self.config,
                island_id=island_id,
            )
            # Update experiment with new island population
            self.experiment.island_populations[island_id] = updated_population
        elif result.status == "failed":
            # Log failed rollout
            if self.logger:
                self.logger.warning(
                    f"Skipping population update for failed rollout",
                    island_id=island_id,
                    failed_module=result.failed_module,
                    error=result.error_message,
                )

        # Store rollout result in history
        self.experiment.rollout_history.append(result)

        # Save rollout result to individual file
        if self.data_service:
            self.data_service.save_result(result)

        # Increment iteration counter
        self.experiment.current_iteration += 1

        # Update timestamp
        self.experiment.updated_at = time.time()

        if self.logger and result.status == "success" and result.generated_program:
            # Log iteration result with detailed metrics
            program = result.generated_program
            metrics = self._build_iteration_metrics(program)

            # Calculate elapsed time
            elapsed_time = 0.0
            if result.stats and "total_time" in result.stats:
                elapsed_time = result.stats["total_time"]

            parent_id = result.selected_parent_id or "unknown"

            # Log iteration result
            # Only include island_id in context if multi-island
            iter_context = {
                "iteration": self.experiment.current_iteration,
                "child_id": program.id,
                "parent_id": parent_id,
                "metrics": metrics,
                "elapsed_time": elapsed_time,
                "max_iterations": self.config.max_iterations,
            }
            if self.config.num_islands > 1:
                iter_context["island_id"] = island_id

            self.logger.iteration_result(**iter_context)

            # Log population statistics
            self._log_population_stats(island_id, old_best_score, old_best_id)

    def _log_population_stats(
        self,
        island_id: int,
        old_best_score: Optional[float],
        old_best_id: Optional[str],
    ) -> None:
        """
        Log population statistics for an island.

        Calculates and logs:
        - Population size, best/average scores
        - Diversity (average pairwise cosine distance)
        - Max generation, error count
        - New best program notification if applicable

        Args:
            island_id: Island to log stats for
            old_best_score: Previous best score (for detecting improvement)
            old_best_id: Previous best program ID
        """
        if not self.logger:
            return

        island_pop = self.experiment.get_island_population(island_id)
        all_programs = [
            p for bucket_progs in island_pop.values() for p in bucket_progs
        ]
        scores = [p.effective_search_score for p in all_programs]

        if not scores:
            return

        avg_score = sum(scores) / len(scores)
        best_program = max(all_programs, key=lambda p: p.ranking_score)
        best_search_score = best_program.effective_search_score
        best_ranking_score = best_program.ranking_score

        # Calculate additional population metrics
        diversity = self._compute_population_diversity(all_programs)
        max_gen = max((p.generation for p in all_programs), default=0)
        error_count = sum(1 for p in all_programs if p.error_info is not None)

        # Check if this is a new best program
        if (
            old_best_score is not None
            and best_ranking_score > old_best_score
            and old_best_id != best_program.id
        ):
            best_context = {
                "program_id": best_program.id,
                "old_program_id": old_best_id,
                "old_score": old_best_score,
                "new_score": best_ranking_score,
            }
            if self.config.num_islands > 1:
                best_context["island_id"] = island_id

            self.logger.new_best_program(**best_context)

        # Log population status
        island_prefix = f"Island {island_id}: "
        self.logger.info(
            f"{island_prefix}{len(all_programs)} programs, "
            f"best_search={best_search_score:.4f}, avg_search={avg_score:.4f}, "
            f"best_success={best_program.success_score:.4f}, "
            f"diversity={diversity:.2f}, gen={max_gen}, "
            f"error_count={error_count}",
        )

    def _save_checkpoint(self) -> None:
        """
        Save experiment checkpoint to storage.

        Persists:
        - Experiment state (population, history, etc.)
        - All programs in the archive

        Saves to experiment_checkpoint_{iteration}.json where iteration is
        taken from experiment.current_iteration.
        """
        if not self.data_service:
            return

        try:
            # Save experiment checkpoint (uses iteration from experiment)
            self.data_service.save_experiment(self.experiment)

            if self.logger:
                self.logger.info(
                    "Checkpoint saved",
                    iteration=self.experiment.current_iteration,
                    archive_size=len(self.experiment.archive),
                )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to save checkpoint",
                    iteration=self.experiment.current_iteration,
                    error=str(e),
                )

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"Evolver("
            f"experiment={self.experiment.name}, "
            f"current_iteration={self.experiment.current_iteration})"
        )
