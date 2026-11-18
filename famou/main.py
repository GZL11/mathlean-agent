#!/usr/bin/env python3
"""
Main entry point for running Famou experiments.

Usage (new experiment):
    python -m famou -c CONFIG -p PROGRAM [PROGRAM ...] -e EVALUATOR

Usage (resume from checkpoint):
    python -m famou --resume EXPERIMENT_PATH -e EVALUATOR
    python -m famou --resume EXPERIMENT_PATH -e EVALUATOR -c CONFIG  # override config

Example (single seed):
    python -m famou \\
        --config examples/circle_packing/config.yaml \\
        --programs examples/circle_packing/init.py \\
        --evaluator examples/circle_packing/evaluator.py

Example (multiple seeds):
    python -m famou \\
        --config config.yaml \\
        --programs init1.py init2.py init3.py \\
        --evaluator evaluator.py

Example (resume):
    python -m famou \\
        --resume famou_data/circle_packing_evolution_abc123 \\
        --evaluator examples/circle_packing/evaluator.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, List, Optional

from famou.config import FamouConfig
from famou.core.data import Program
from famou.controller import Evolver
from famou.infrastructure.factory import InfraFactory
from famou.infrastructure.logger import LocalLogger
from famou.strategies import StrategyRegistry
from famou.utils import generate_experiment_id
from famou.utils.color import Colors

from famou.core.types import Language


def load_config(yaml_path: str) -> FamouConfig:
    """
    Load experiment configuration from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        FamouConfig instance
    """
    return FamouConfig.from_yaml(yaml_path)

def load_task_description(task: Path | str, logger: LocalLogger, config_problem: Optional[dict]) -> str:
    path = Path(task)
    if path.exists():
        if path.is_file():
            logger.info("task_description is a file, loading from file")
            with open(path, "r") as f:
                task_description = f.read()
        elif path.is_dir():
            raise NotImplementedError("Directory as task_description is not supported yet.")
    else:
        logger.info("task_description is not a file, directly using the text.")
        task_description = task.format(**config_problem)
    return task_description

def load_initial_programs(
    program_paths: List[str],
    experiment_id: str,
    language: Language
) -> List[Program]:
    """
    Load initial program(s) from one or more files.

    This function ONLY loads the code and creates Program objects.
    Enrichment (evaluation, judging, etc.) happens later in Evolver.

    Args:
        program_paths: List of paths to program files
        experiment_id: Experiment ID to add to program metadata
        language: Programming language of the programs

    Returns:
        List of Program objects with unique IDs (seed_0, seed_1, ...)

    Example:
        >>> programs = load_initial_programs(
        ...     ["init1.py", "init2.py"],
        ...     "exp_123",
        ...     Language.PYTHON
        ... )
        >>> [p.id for p in programs]
        ['seed_0', 'seed_1']
    """
    programs = []

    for i, program_path in enumerate(program_paths):
        if language == Language.LEAN:
            assert program_path.endswith('.scratch'), f"Program file must be a Lean file: {program_path}"
        elif language == Language.PYTHON:
            assert program_path.endswith('.py'), f"Program file must be a Python file: {program_path}"
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program file not found: {program_path}")
        code = path.read_text()

        # Optional `explanation` setup
        # txt_path = Path(program_path[:-3] + ".txt")
        # md_path = Path(program_path[:-3] + ".md")
        txt_path = path.with_suffix(".txt")
        md_path = path.with_suffix(".md")
        explanation = None
        if txt_path.exists():
            explanation = txt_path.read_text()
        elif md_path.exists():
            explanation = md_path.read_text()
        else:
            explanation = None
            # raise FileNotFoundError(f"Neither text file nor markdown file found for initial program explanation {path}")

        # Create initial program WITHOUT scores/evaluation
        initial_program = Program(
            id=f"seed_{i}",
            code=code,
            explanation=explanation,
            generation=0,
            iteration=0,
            language=language,
        )
        # print(f"Initial program id: {initial_program.id}, explanation: {explanation}")

        # Add experiment_id to metadata
        initial_program.meta["experiment_id"] = experiment_id
        initial_program.meta["source_file"] = str(path.name)

        programs.append(initial_program)

    return programs

def load_evaluator(evaluator_path: str) -> Callable:
    """
    Dynamically load evaluator function from a Python file.

    The file must contain an 'evaluate' function with signature:
        def evaluate(program_path: str) -> Dict[str, Any]

    Args:
        evaluator_path: Path to Python file containing evaluate function

    Returns:
        The evaluate function

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file doesn't contain 'evaluate' function
    """
    path = Path(evaluator_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluator file not found: {evaluator_path}")

    spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate"):
        raise ValueError(
            f"Evaluator file must contain 'evaluate' function: {evaluator_path}"
        )

    return module.evaluate


def get_evaluator_name(evaluate_fn: Callable) -> str:
    """Return a readable evaluator name for functions and callable objects."""
    return getattr(evaluate_fn, "__name__", evaluate_fn.__class__.__name__)


def main():
    """Main entry point."""
    #region === Argument Parsing ===
    parser = argparse.ArgumentParser(
        description="Run Famou evolutionary optimization experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # New experiment (single seed)
    python -m famou \\
        --config examples/circle_packing/config.yaml \\
        --programs examples/circle_packing/init.py \\
        --evaluator examples/circle_packing/evaluator.py

    # New experiment (multiple seeds for diversity)
    python -m famou \\
        --config config.yaml \\
        --programs init1.py init2.py init3.py \\
        --evaluator evaluator.py

    # Resume from checkpoint
    python -m famou \\
        --resume famou_data/circle_packing_evolution_abc123 \\
        --evaluator examples/circle_packing/evaluator.py

    # Resume with config override (e.g., increase max_iterations)
    python -m famou \\
        --resume famou_data/circle_packing_evolution_abc123 \\
        --config new_config.yaml \\
        --evaluator examples/circle_packing/evaluator.py
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML configuration file (required for new, optional for resume)",
    )
    parser.add_argument(
        "--programs",
        "-p",
        nargs="+",
        help="Path(s) to initial program file(s). Multiple files supported for multi-seed experiments.",
    )
    parser.add_argument(
        "--evaluator",
        "-e",
        required=True,
        help="Path to evaluator module (Python file with 'evaluate' function)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        help="Path to experiment directory to resume from checkpoint",
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=-1,
        help="Checkpoint iteration to resume from (default: latest)",
    )

    args = parser.parse_args()
    #endregion

    #region === Validate arguments based on mode ===
    if args.resume:
        # Resume mode - config and programs are optional
        pass
    else:
        # New experiment mode - config and programs are required
        if not args.config:
            parser.error("--config is required for new experiments")
        if not args.programs:
            parser.error("--programs is required for new experiments")
    #endregion

    print("\n" + "=" * 70)
    print("  FAMOU 2.0 - Evolutionary Code Optimization")
    print("=" * 70)

    #region === Create basic logger for early logging ===
    basic_logger = LocalLogger(name="famou", level="INFO", file_enabled=False)
    #endregion

    #region === Load evaluator (required for both new and resume) ===
    print(f"\n[1/6] Loading evaluator from: {args.evaluator}")
    evaluate_fn = load_evaluator(args.evaluator)
    basic_logger.info("Evaluator loaded", evaluator=get_evaluator_name(evaluate_fn))
    #endregion

    # Branch based on mode: resume or new experiment
    if args.resume:
        # =================================================================
        # RESUME MODE
        # =================================================================
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume path not found: {args.resume}")
            sys.exit(1)

        # === Extract original experiment_id from path ===
        original_experiment_id = resume_path.name

        print(f"\n[2/6] Loading experiment: {original_experiment_id}")

        # Always load saved config first (for comparison)
        saved_config_path = resume_path / "config.yaml"
        if not saved_config_path.exists():
            if not args.config:
                print(f"Error: No saved config found at {saved_config_path}")
                print("       Please provide --config to specify configuration")
                sys.exit(1)
            saved_config = None
        else:
            saved_config = load_config(str(saved_config_path))

        # Determine effective config and whether it changed
        if args.config:
            print(f"      Loading config override from: {args.config}")
            config = load_config(args.config)
            config_changed = saved_config is None or config.to_dict() != saved_config.to_dict()
        else:
            print(f"      Loading saved config from: {saved_config_path}")
            config = saved_config
            config_changed = False

        basic_logger.info(
            "Configuration loaded for resume",
            experiment=config.experiment.name,
            max_iterations=config.experiment.max_iterations,
            config_changed=config_changed,
        )

        # Setup initial infrastructure (may be replaced if forking)
        print("\n[3/6] Setting up infrastructure...")
        infra = InfraFactory.create_all(
            config.infrastructure,
            experiment_name=config.experiment.name,
            experiment_id=original_experiment_id,
        )
        config.experiment.name = config.experiment.name.format(**config.problem)
        config.experiment.task_description = config.experiment.task_description.format(**config.problem)
        config.experiment.task_constraint = config.experiment.task_constraint.format(**config.problem)

        # Load experiment from checkpoint
        print(f"\n[4/6] Loading checkpoint (iteration={args.iteration})...")
        experiment = infra.storage.load_experiment(original_experiment_id, iteration=args.iteration)
        loaded_iteration = experiment.current_iteration
        print(f"      Loaded iteration: {loaded_iteration}")
        print(f"      Archive size: {len(experiment.archive)}")

        # Detect if this is the latest checkpoint
        is_latest = True
        if args.iteration != -1:
            # User specified explicit iteration - check if it's actually the latest
            latest_experiment = infra.storage.load_experiment(original_experiment_id, iteration=-1)
            is_latest = (latest_experiment.current_iteration == loaded_iteration)

        # Determine if we should fork
        should_fork = (not is_latest) or config_changed
        fork_reason = None
        if not is_latest:
            fork_reason = "previous_checkpoint"
        elif config_changed:
            fork_reason = "config_changed"

        if should_fork:
            # =============================================================
            # FORK: Create new experiment from checkpoint
            # =============================================================
            new_experiment_id = generate_experiment_id(config.experiment.name)

            print(f"\n      Forking experiment (reason: {fork_reason})")
            print(f"      Original: {original_experiment_id} (iteration {loaded_iteration})")
            print(f"      New:      {new_experiment_id}")

            # Set resume_from tracking
            experiment.resume_from = {
                "experiment_id": original_experiment_id,
                "iteration": loaded_iteration,
            }
            experiment.id = new_experiment_id

            # Re-create infrastructure with new experiment_id
            infra = InfraFactory.create_all(
                config.infrastructure,
                experiment_name=config.experiment.name,
                experiment_id=new_experiment_id,
            )

            # Save config to new folder
            infra.storage.save_config(new_experiment_id, config)

            # Copy programs to new folder (update experiment_id in meta)
            print(f"      Copying {len(experiment.archive)} programs to new folder...")
            for program in experiment.archive:
                program.meta["experiment_id"] = new_experiment_id
                infra.storage.save_program(program)

            infra.logger.info(
                "Forked experiment",
                original_id=original_experiment_id,
                original_iteration=loaded_iteration,
                new_id=new_experiment_id,
                reason=fork_reason,
            )

            experiment_id = new_experiment_id
        else:
            # =============================================================
            # PURE RESUME: Continue same experiment
            # =============================================================
            experiment_id = original_experiment_id
            infra.logger.info(
                "Resuming experiment",
                experiment_id=experiment_id,
                iteration=loaded_iteration,
            )

        print(infra)

        # Load strategy
        print(f"\n[5/6] Loading strategy '{config.experiment.strategy}'...")
        strategy = StrategyRegistry.get(
            config.experiment.strategy,
            evaluate_fn=evaluate_fn,
            params=config.modules,
        )
        print(f"      Strategy: {strategy}")

        # Create evolver in resume mode (with experiment, not initial_programs)
        evolver = Evolver(
            config=config.experiment,
            population_module=strategy.population_module,
            experiment=experiment,  # Resume mode
            logger=infra.logger,
            llm_client=infra.llm,
            env=infra.env,
            data_service=infra.storage,
            embedding_client=infra.embedding,
        )

        if should_fork:
            print(f"\n[6/6] Starting forked evolution from iteration {experiment.current_iteration}...")
        else:
            print(f"\n[6/6] Resuming evolution from iteration {experiment.current_iteration}...")
        print("=" * 70)

    else:
        # =================================================================
        # NEW EXPERIMENT MODE
        # =================================================================
        #region === Load configuration ===
        print(f"\n[2/6] Loading configuration from: {args.config}")
        config = load_config(args.config)
        basic_logger.info(
            "Configuration loaded",
            experiment=config.experiment.name,
            max_iterations=config.experiment.max_iterations,
            island_size=config.experiment.island_size,
            language=config.experiment.language,
        )
        # problem configuration & change task description based on problem configuration
        print(f"{Colors.BLUE}{Colors.BOLD}Configuration: {config.problem}{Colors.RESET}")
        config.experiment.name = config.experiment.name.format(**config.problem)
        # config.experiment.task_description = config.experiment.task_description.format(**config.problem)
        # Load task_description. Supporting 'Path' format
        config.experiment.task_description = load_task_description(config.experiment.task_description, basic_logger, config.problem)
        config.experiment.task_constraint = config.experiment.task_constraint.format(**config.problem)
    
        # print(f"{config.experiment.task_description}")
        # sys.exit()
        # Generate experiment ID
        experiment_id = generate_experiment_id(config.experiment.name)
        basic_logger.info("Experiment ID generated", experiment_id=experiment_id)
        #endregion

        #region === Setup infrastructure ===
        print("\n[3/6] Setting up infrastructure...")
        infra = InfraFactory.create_all(
            config.infrastructure,
            experiment_name=config.experiment.name,
            experiment_id=experiment_id,
        )
        print(infra)
        infra.logger.info(
            "Infrastructure initialized",
            llm_provider=config.infrastructure.llm.provider,
            llm_model=config.infrastructure.llm.model,
            storage_type=config.infrastructure.storage.type,
            storage_path=config.infrastructure.storage.base_path,
        )

        # Save config for future resume
        infra.storage.save_config(experiment_id, config)
        infra.logger.info("Config saved for resume support")
        #endregion

        #region === Load initial programs ===
        num_files = len(args.programs)
        if num_files == 1:
            print(f"\n[4/6] Loading initial program from: {args.programs[0]}")
        else:
            print(f"\n[4/6] Loading {num_files} initial programs:")
            for p in args.programs:
                print(f"      - {p}")
        initial_programs = load_initial_programs(
            args.programs,
            experiment_id,
            config.experiment.language,
        )
        infra.logger.info(
            "Initial programs loaded",
            num_programs=len(initial_programs),
            files=[str(Path(p).name) for p in args.programs],
        )
        #endregion

        #region === Load strategy ===
        print(f"\n[5/6] Loading strategy '{config.experiment.strategy}'...")
        strategy = StrategyRegistry.get(
            config.experiment.strategy,
            evaluate_fn=evaluate_fn,
            params=config.modules,
        )
        print(f"      Strategy: {strategy}")
        #endregion

        #region === Create evolver with initial programs (new experiment mode) ===
        evolver = Evolver(
            config=config.experiment,
            population_module=strategy.population_module,
            initial_programs=initial_programs,
            logger=infra.logger,
            llm_client=infra.llm,
            env=infra.env,
            data_service=infra.storage,
            experiment_id=experiment_id,
            embedding_client=infra.embedding,
        )
        #endregion

        print(f"\n[6/6] Starting evolution...")
        print("=" * 70)

    try:
        print(f"{Colors.GREEN}{Colors.BOLD}The rollout configuration{Colors.RESET} {strategy.rollout}")
        final_experiment = evolver.run(strategy.rollout)

        # Print results
        print("\n" + "=" * 70)
        print("  EVOLUTION COMPLETE")
        print("=" * 70)

        # Get best program
        population = final_experiment.get_all_programs()
        if population:
            best_program = max(population, key=lambda p: p.ranking_score)

            print(f"\n  Iterations completed: {final_experiment.current_iteration}")
            print(f"  Archive size: {len(final_experiment.archive)}")
            print(f"  Final population size: {len(population)}")
            print(f"\n  Best program:")
            print(f"    ID: {best_program.id}")
            print(f"    Success score: {best_program.success_score:.6f}")
            print(f"    Search score: {best_program.effective_search_score:.6f}")
            print(f"    Generation: {best_program.generation}")

            if best_program.metrics:
                print(f"    Metrics: {best_program.metrics}")

            # Print top 5 programs
            print(f"\n  Top 5 programs:")
            sorted_pop = sorted(
                population,
                key=lambda p: p.ranking_score,
                reverse=True,
            )
            for i, prog in enumerate(sorted_pop[:5]):
                print(
                    f"    {i+1}. {prog.id}: "
                    f"success={prog.success_score:.6f}, "
                    f"search={prog.effective_search_score:.6f}"
                )

        # Print storage location
        storage_path = config.infrastructure.storage.base_path
        print("\n" + "=" * 70)
        print("  DATA STORAGE")
        print("=" * 70)
        print(f"\n  All results saved to: {storage_path}/{final_experiment.id}/")
        print(f"    - Experiment state: experiment.json")
        print(f"    - Programs: programs/{len(final_experiment.archive)} files")
        print(f"    - Rollout results: results/{len(final_experiment.rollout_history)} files")
        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("  EVOLUTION INTERRUPTED BY USER")
        print("=" * 70)

        infra.logger.warning(
            "Evolution interrupted by user",
            current_iteration=evolver.experiment.current_iteration,
            archive_size=len(evolver.experiment.archive),
        )

        # Save checkpoint before exiting
        if infra.storage:
            storage_path = config.infrastructure.storage.base_path
            infra.logger.info(f"Saving checkpoint to: {storage_path}/{evolver.experiment.id}/")
            try:
                infra.storage.save_experiment(evolver.experiment)
                infra.logger.info(
                    "Checkpoint saved successfully",
                    experiment_id=evolver.experiment.id,
                    num_programs=len(evolver.experiment.archive),
                    num_rollouts=len(evolver.experiment.rollout_history),
                )
                print(f"\n  Checkpoint saved to: {storage_path}/{evolver.experiment.id}/")
                print(f"    - Experiment state: experiment.json")
                print(f"    - Programs: programs/{len(evolver.experiment.archive)} files")
                print(f"    - Rollout results: results/{len(evolver.experiment.rollout_history)} files")
            except Exception as e:
                infra.logger.error(f"Failed to save checkpoint: {e}")
                print(f"  Failed to save checkpoint: {e}")

        print("\n" + "=" * 70)

    except Exception as e:
        print("\n\n" + "=" * 70)
        print("  EVOLUTION FAILED")
        print("=" * 70)

        infra.logger.error(
            "Evolution failed with error",
            error=str(e),
            current_iteration=evolver.experiment.current_iteration,
        )

        import traceback
        traceback.print_exc()

        # Try to save checkpoint even on failure
        if infra.storage:
            storage_path = config.infrastructure.storage.base_path
            infra.logger.info("Attempting to save checkpoint after failure")
            try:
                infra.storage.save_experiment(evolver.experiment)
                infra.logger.info(
                    "Checkpoint saved successfully",
                    experiment_id=evolver.experiment.id,
                )
                print(f"\n  Checkpoint saved to: {storage_path}/{evolver.experiment.id}/")
            except Exception as save_error:
                infra.logger.error(f"Failed to save checkpoint: {save_error}")
                print(f"  Failed to save checkpoint: {save_error}")

        print("\n" + "=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
