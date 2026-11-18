"""
Local file-based storage implementation.

Uses JSON for individual programs and experiments.

Directory structure:
    base_path/
    ├── {experiment_id}/
    │   ├── programs/
    │   │   ├── program_0.json      # Program from iteration 0
    │   │   ├── program_1.json      # Program from iteration 1
    │   │   └── ...
    │   ├── results/
    │   │   ├── rollout_0.json     # Rollout result for iteration 0
    │   │   ├── rollout_1.json     # Rollout result for iteration 1
    │   │   └── ...
    │   └── experiment.json         # Latest experiment checkpoint
    └── ...
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from famou.core.data import Experiment, Program, RolloutResult
from famou.core.types import Language

# Import FamouConfig for type hints (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from famou.config.settings import FamouConfig


class LocalStorage:
    """
    File-based storage using JSON with compact checkpoints.

    Implements: DataService protocol (structural subtyping)

    Features:
    - JSON for individual programs (one file per program)
    - Code files with appropriate extensions (.py, .js, .java, .cpp, .go, .rs)
    - JSON for experiments with compact checkpoints (10-100x smaller)
    - JSON for rollout results
    - Organized by experiment ID and iteration
    - Automatic directory creation

    Compact Checkpoint Design:
    - Archive stores full Program objects (single source of truth)
    - Populations store only Program IDs
    - RolloutResults store only Program IDs
    - Reconstruction on load: lookup IDs in archive
    - Reduces file size from ~145MB to ~1.5MB for 50 programs × 3 islands

    Directory Structure:
        base_path/
        ├── {experiment_id}/
        │   ├── programs/
        │   │   ├── program_0.json      # Full Program object
        │   │   ├── program_0.py        # Code file (if language is Python)
        │   │   ├── program_1.json
        │   │   ├── program_1.cpp       # Code file (if language is C++)
        │   │   └── ...
        │   ├── results/
        │   │   ├── rollout_0.json      # RolloutResult (compact: IDs only)
        │   │   ├── rollout_1.json
        │   │   └── ...
        │   └── experiment.json          # Compact checkpoint (IDs only)

    Example:
        >>> storage = LocalStorage("./famou_data")
        >>> storage.save_program(program)
        >>> storage.save_experiment(experiment)  # Compact format
        >>> loaded = storage.load_experiment("exp_123")  # Reconstructed
        >>> len(loaded.archive)  # All programs available
        50
    """

    # Language to file extension mapping
    LANGUAGE_EXTENSIONS: Dict[Language, str] = {
        Language.PYTHON: ".py",
        Language.JAVASCRIPT: ".js",
        Language.JAVA: ".java",
        Language.CPP: ".cpp",
        Language.GO: ".go",
        Language.RUST: ".rs",
        Language.LEAN: ".lean",
        Language.SCRATCH: ".scratch",
    }

    def __init__(self, base_path: str = "./famou_data"):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for all data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_experiment_dir(self, experiment_id: str) -> Path:
        """Get directory for experiment."""
        exp_dir = self.base_path / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _get_programs_dir(self, experiment_id: str) -> Path:
        """Get programs directory for experiment."""
        programs_dir = self._get_experiment_dir(experiment_id) / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        return programs_dir

    def _get_results_dir(self, experiment_id: str) -> Path:
        """Get results directory for experiment."""
        results_dir = self._get_experiment_dir(experiment_id) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _get_code_file_extension(self, language: Language) -> str:
        """Get file extension for a given language."""
        return self.LANGUAGE_EXTENSIONS.get(language, ".py")

    def _get_code_file_path(self, experiment_id: str, program_id: str, language: Language) -> Path:
        """Get the path for the code file with appropriate extension."""
        programs_dir = self._get_programs_dir(experiment_id)
        extension = self._get_code_file_extension(language)
        return programs_dir / f"{program_id}{extension}"

    # ========================================================================
    # Program Storage
    # ========================================================================

    def save_program(self, program: Program) -> None:
        """
        Save program to JSON file and code file with appropriate extension.

        One JSON file and one code file per program, named by program ID.

        Args:
            program: Program to save
        """
        # Get experiment ID from program metadata
        experiment_id = program.meta.get("experiment_id")
        if not experiment_id:
            raise ValueError("Program must have experiment_id in meta")

        # Get programs directory for this experiment
        programs_dir = self._get_programs_dir(experiment_id)

        # === Save to program-specific JSON file (use program ID for unique filename) ===
        program_file = programs_dir / f"{program.id}.json"
        program_file.write_text(json.dumps(program.model_dump(), indent=2, ensure_ascii=False))

        # === Save code file with appropriate extension ===
        code_file = self._get_code_file_path(experiment_id, program.id, program.language)
        code_file.write_text(program.code)

        # === Transfer generated_folder (temp folder storing lean file) into experiment folder ===
        if program.generated_folder is not None:
            from shutil import copytree, rmtree
            generated_folder = program.generated_folder
            save_folder = programs_dir / f"{program.id}_lean"
            if save_folder.exists():
                rmtree(save_folder)
            copytree(generated_folder, save_folder)
            rmtree(generated_folder)

    def load_program(self, program_id: str) -> Program:
        """
        Load program by ID.

        Direct lookup by program ID filename.

        Args:
            program_id: Program ID to load

        Returns:
            Program instance

        Raises:
            FileNotFoundError if program not found
        """
        # Search all experiment directories
        for exp_dir in self.base_path.iterdir():
            if not exp_dir.is_dir():
                continue

            programs_dir = exp_dir / "programs"
            if not programs_dir.exists():
                continue

            # Direct lookup by program_id filename
            program_file = programs_dir / f"{program_id}.json"
            if program_file.exists():
                program_data = json.loads(program_file.read_text())
                return Program.model_validate(program_data)

        raise FileNotFoundError(f"Program {program_id} not found")

    def list_programs(
        self,
        experiment_id: str,
        iteration: Optional[int] = None,
        generation: Optional[int] = None,
    ) -> List[Program]:
        """
        List programs matching filters.

        Args:
            experiment_id: Experiment ID
            iteration: Optional iteration filter
            generation: Optional generation filter

        Returns:
            List of matching programs
        """
        programs_dir = self._get_programs_dir(experiment_id)

        programs: List[Program] = []

        # Read all program files (filenames are program IDs)
        json_files = sorted(programs_dir.glob("*.json"))

        # Read programs from JSON files
        for program_file in json_files:
            if not program_file.exists():
                continue

            program_data = json.loads(program_file.read_text())

            # Apply iteration filter if specified
            if iteration is not None and program_data.get("iteration") != iteration:
                continue

            # Apply generation filter if specified
            if generation is not None and program_data.get("generation") != generation:
                continue

            programs.append(Program.model_validate(program_data))

        return programs

    # ========================================================================
    # RolloutResult Storage (Compact Format)
    # ========================================================================

    def save_result(self, result: RolloutResult) -> None:
        """
        Save rollout result using compact format.

        Compact format:
        - Stores only Program IDs instead of full Program objects
        - Reduces file size significantly
        - Programs are reconstructed from experiment archive on load

        Args:
            result: RolloutResult to save
        """
        # Extract experiment ID from generated program
        experiment_id = None
        if result.generated_program:
            experiment_id = result.generated_program.meta.get("experiment_id")

        if not experiment_id:
            raise ValueError("Cannot determine experiment_id from RolloutResult")

        results_dir = self._get_results_dir(experiment_id)
        # Use rollout_id as filename (unique identifier)
        result_file = results_dir / f"{result.rollout_id}.json"

        # Convert to compact representation
        compact_data = result.to_compact_dict()
        # Add experiment_id for reconstruction
        compact_data["experiment_id"] = experiment_id

        # Save as JSON
        result_file.write_text(json.dumps(compact_data, indent=2, ensure_ascii=False))

    def load_result(self, rollout_id: str) -> RolloutResult:
        """
        Load rollout result from compact format.

        Reconstructs full Program objects by loading experiment archive.

        Args:
            rollout_id: Rollout ID

        Returns:
            Fully reconstructed RolloutResult instance

        Raises:
            FileNotFoundError: If result not found
        """
        # Search all experiment directories
        for exp_dir in self.base_path.iterdir():
            if not exp_dir.is_dir():
                continue

            results_dir = exp_dir / "results"
            if not results_dir.exists():
                continue

            # Direct lookup by rollout_id filename
            result_file = results_dir / f"{rollout_id}.json"
            if result_file.exists():
                compact_data = json.loads(result_file.read_text())
                # Get experiment_id from compact data
                experiment_id = compact_data.get("experiment_id")
                if not experiment_id:
                    # Fallback: try to infer from directory structure
                    experiment_id = exp_dir.name

                # Load experiment to get archive
                experiment = self.load_experiment(experiment_id)
                archive_dict = {p.id: p for p in experiment.archive}

                # Reconstruct from compact format
                return RolloutResult.from_compact_dict(compact_data, archive_dict)

        raise FileNotFoundError(f"RolloutResult {rollout_id} not found")

    # ========================================================================
    # Experiment Storage (Compact Checkpoints)
    # ========================================================================

    def save_experiment(self, experiment: Experiment) -> None:
        """
        Save experiment state (checkpoint) using compact format.

        Saves to experiment_checkpoint_{iteration}.json where iteration is
        taken from experiment.current_iteration.

        Compact format:
        - Stores full Program objects only in archive
        - Stores only Program IDs in populations and rollout history
        - Reduces checkpoint file size by 10-100x

        Args:
            experiment: Experiment to save
        """
        exp_dir = self._get_experiment_dir(experiment.id)
        iteration = experiment.current_iteration
        checkpoint_file = exp_dir / f"experiment_checkpoint_{iteration}.json"

        # Convert to compact representation
        compact_data = experiment.to_compact_dict()

        # Save as JSON
        checkpoint_file.write_text(json.dumps(compact_data, indent=2, ensure_ascii=False))

    def load_experiment(self, experiment_id: str, iteration: int = -1) -> Experiment:
        """
        Load experiment state from compact checkpoint.

        Loads programs from individual files in programs/ directory
        and rollout results from individual files in results/ directory.

        Args:
            experiment_id: Experiment ID
            iteration: Iteration number to load (default: -1 for latest checkpoint)

        Returns:
            Fully reconstructed Experiment instance

        Raises:
            FileNotFoundError: If experiment not found
        """
        exp_dir = self._get_experiment_dir(experiment_id)

        # Determine checkpoint file to load
        if iteration == -1:
            # Find the latest checkpoint
            checkpoint_files = list(exp_dir.glob("experiment_checkpoint_*.json"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoints found for experiment {experiment_id}")

            # Extract iteration numbers and find the maximum
            iterations = []
            for f in checkpoint_files:
                # Extract iteration number from filename
                stem = f.stem  # experiment_checkpoint_5
                try:
                    iter_num = int(stem.split("_")[-1])
                    iterations.append((iter_num, f))
                except (ValueError, IndexError):
                    continue

            if not iterations:
                raise FileNotFoundError(f"No valid checkpoints found for experiment {experiment_id}")

            # Get the file with the highest iteration number
            _, checkpoint_file = max(iterations, key=lambda x: x[0])
        else:
            # Load specific iteration checkpoint
            checkpoint_file = exp_dir / f"experiment_checkpoint_{iteration}.json"

        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint for experiment {experiment_id} at iteration {iteration} not found"
            )

        # Load compact data
        compact_data = json.loads(checkpoint_file.read_text())

        # Get the list of program IDs and rollout IDs from checkpoint
        # This ensures we only load programs/rollouts that existed at checkpoint time
        archive_ids = set(compact_data.get("archive_ids", []))
        rollout_ids = set(compact_data.get("rollout_ids", []))

        # Load ONLY programs listed in archive_ids (not all programs in folder)
        programs_dir = self._get_programs_dir(experiment_id)
        archive_dict = {}

        if programs_dir.exists() and archive_ids:
            for program_id in archive_ids:
                program_file = programs_dir / f"{program_id}.json"
                if program_file.exists():
                    program_data = json.loads(program_file.read_text())
                    program = Program.model_validate(program_data)
                    archive_dict[program.id] = program

        # Load ONLY rollout results listed in rollout_ids
        results_dir = self._get_results_dir(experiment_id)
        rollout_history = []

        if results_dir.exists() and rollout_ids:
            for rollout_id in rollout_ids:
                result_file = results_dir / f"{rollout_id}.json"
                if result_file.exists():
                    result_data = json.loads(result_file.read_text())
                    result = RolloutResult.from_compact_dict(result_data, archive_dict)
                    rollout_history.append(result)

        # Reconstruct experiment from compact format with loaded programs and rollouts
        return Experiment.from_compact_dict(
            compact_data,
            archive_dict=archive_dict,
            rollout_history=rollout_history
        )

    # ========================================================================
    # Config Storage
    # ========================================================================

    def save_config(self, experiment_id: str, config: "FamouConfig") -> None:
        """
        Save experiment configuration to YAML file.

        Saves to {experiment_id}/config.yaml for resume support.

        Args:
            experiment_id: Experiment ID
            config: FamouConfig to save
        """
        exp_dir = self._get_experiment_dir(experiment_id)
        config_file = exp_dir / "config.yaml"

        # Use FamouConfig's built-in to_yaml method
        config.to_yaml(str(config_file))

    def load_config(self, experiment_id: str) -> "FamouConfig":
        """
        Load experiment configuration from YAML file.

        Args:
            experiment_id: Experiment ID

        Returns:
            FamouConfig instance

        Raises:
            FileNotFoundError: If config not found
        """
        from famou.config.settings import FamouConfig

        exp_dir = self._get_experiment_dir(experiment_id)
        config_file = exp_dir / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Config not found for experiment {experiment_id}. "
                f"Expected at: {config_file}"
            )

        return FamouConfig.from_yaml(str(config_file))

    def __repr__(self) -> str:
        """Concise representation."""
        return f"LocalStorage(base_path={self.base_path})"
