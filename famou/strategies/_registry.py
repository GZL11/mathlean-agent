"""
Strategy Registry for Famou 2.0.

Pure file-based registry with auto-discovery from directories.
Strategies are shared via Git commit/push/pull.

A strategy combines:
- Rollout pipeline: How to select, generate, and evaluate programs
- PopulationModule: How to manage the population

Usage:
    # Get a strategy
    >>> from famou.strategies import StrategyRegistry
    >>> strategy = StrategyRegistry.get("standard")
    >>> strategy.rollout  # Rollout pipeline
    >>> strategy.population_module  # PopulationModule

    # List all available strategies
    >>> StrategyRegistry.list_strategies()
    ['standard', 'my_custom_strategy', ...]

    # Use in experiment
    >>> evolver = Evolver(
    ...     experiment=experiment,
    ...     config=config,
    ...     population_module=strategy.population_module,
    ...     ...
    ... )
    >>> evolver.run(strategy.rollout)
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules import PopulationModule


@dataclass
class Strategy:
    """
    A complete evolutionary strategy.

    Combines:
    - Rollout pipeline: How to select, generate, and evaluate programs
    - PopulationModule: How to manage the population

    Attributes:
        name: Strategy name (e.g., "standard", "explore")
        rollout: Rollout pipeline with modules
        population_module: Population management strategy
        description: Human-readable description
        tags: Optional tags for categorization
        author: Optional author name
    """

    name: str
    rollout: Rollout
    population_module: PopulationModule
    description: str
    tags: Optional[List[str]] = None
    author: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        modules_str = " → ".join(m.name for m in self.rollout.modules)
        return f"{self.name}: {modules_str} + {self.population_module.name}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Strategy(name='{self.name}', "
            f"rollout={len(self.rollout.modules)} modules, "
            f"population={self.population_module.name})"
        )


class StrategyRegistry:
    """
    Pure file-based strategy registry.

    Scans strategy directory for .py files with create_strategy() function.
    No in-memory caching - always reads from disk (simple & reliable).

    Strategy directory: ./famou/strategies/

    To add a new strategy:
    1. Create file: famou/strategies/my_strategy.py
    2. Define create_strategy() function that returns strategy dict
    3. Commit to Git: git add, git commit, git push
    4. Others pull: git pull
    5. Use: StrategyRegistry.get("my_strategy")
    """

    # Strategy directory to scan
    _builtin_strategy_dir = str(Path(__file__).resolve().parent)
    _strategy_dirs = [
        _builtin_strategy_dir,
    ]

    @classmethod
    def get(
        cls,
        name: str,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
    ) -> Strategy:
        """
        Load strategy from file with optional evaluator and params.

        Searches strategy directories for {name}.py file and loads it.
        The strategy's create_strategy() function receives evaluate_fn and params
        to configure modules.

        Args:
            name: Strategy name (filename without .py extension)
            evaluate_fn: Evaluator function to inject into EvaluateModule
            params: Module parameters from config (overrides strategy defaults)

        Returns:
            Strategy object

        Raises:
            KeyError: If strategy not found

        Example:
            >>> strategy = StrategyRegistry.get(
            ...     "standard",
            ...     evaluate_fn=my_evaluator,
            ...     params=config.modules,
            ... )
        """
        # print("cls._strategy_dirs:", cls._strategy_dirs)
        for directory in cls._strategy_dirs:
            strategy_file = Path(directory) / f"{name}.py"
            # print("Strategy file path: ", strategy_file, "directory:", directory)
            if strategy_file.exists():
                return cls._load_from_file(strategy_file, name, evaluate_fn, params)

        # Not found - provide helpful error message
        available = cls.list_strategies()
        available_str = ", ".join(available) if available else "(none found)"
        raise KeyError(
            f"Strategy '{name}' not found.\n"
            f"Available strategies: {available_str}\n"
            f"Searched in: {cls._strategy_dirs}\n\n"
            f"To create a new strategy, see: famou/strategies/README.md"
        )

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all available strategies by scanning directories.

        Returns:
            List of strategy names (sorted)

        Example:
            >>> StrategyRegistry.list_strategies()
            ['standard', 'explore', 'my_custom']
        """
        strategies = []
        for directory in cls._strategy_dirs:
            path = Path(directory)
            if not path.exists():
                continue

            for strategy_file in path.glob("*.py"):
                # Skip __init__.py and private files
                if strategy_file.name.startswith("_"):
                    continue

                # Strategy name is filename without .py
                strategy_name = strategy_file.stem
                strategies.append(strategy_name)

        # Remove duplicates (custom overrides builtin), sort
        return sorted(set(strategies))

    @classmethod
    def list_all(cls) -> Dict[str, Strategy]:
        """
        Load all strategies with full details.

        Returns:
            Dictionary mapping names to Strategy objects

        Example:
            >>> strategies = StrategyRegistry.list_all()
            >>> for name, strategy in strategies.items():
            ...     print(f"{name}: {strategy.description}")
        """
        all_strategies = {}
        for name in cls.list_strategies():
            try:
                all_strategies[name] = cls.get(name)
            except Exception as e:
                # Skip strategies that fail to load
                print(f"Warning: Failed to load strategy '{name}': {e}")
        return all_strategies

    @classmethod
    def add_strategy_directory(cls, directory: str) -> None:
        """
        Add a custom directory to scan for strategies.

        Useful for loading strategies from external locations.

        Args:
            directory: Path to directory containing strategy files

        Example:
            >>> StrategyRegistry.add_strategy_directory("/shared/team-strategies")
            >>> strategy = StrategyRegistry.get("team_best")
        """
        if directory not in cls._strategy_dirs:
            cls._strategy_dirs.append(directory)

    @classmethod
    def _load_from_file(
        cls,
        file_path: Path,
        name: str,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
    ) -> Strategy:
        """
        Load strategy from Python file.

        The file must define a create_strategy() function that accepts
        evaluate_fn and params, and returns a dictionary with keys:
        rollout, population_module, description, tags.

        Args:
            file_path: Path to strategy file
            name: Strategy name
            evaluate_fn: Evaluator function to inject into EvaluateModule
            params: Module parameters from config

        Returns:
            Strategy object

        Raises:
            ValueError: If file doesn't define create_strategy()
            Exception: If strategy creation fails
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location(f"strategy_{name}", file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load strategy file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check for create_strategy() function
        if not hasattr(module, "create_strategy"):
            raise ValueError(
                f"Strategy file {file_path} must define create_strategy() function.\n"
                f"See famou/strategies/README.md for template."
            )

        # Call factory function with evaluate_fn and params
        strategy_dict = module.create_strategy(
            evaluate_fn=evaluate_fn,
            params=params or ModulesConfig(),
        )

        # Validate required keys
        required_keys = ["rollout", "population_module", "description"]
        missing_keys = [k for k in required_keys if k not in strategy_dict]
        if missing_keys:
            raise ValueError(
                f"Strategy '{name}' missing required keys: {missing_keys}\n"
                f"create_strategy() must return dict with keys: {required_keys}"
            )

        # Extract metadata
        description = strategy_dict["description"]
        tags = strategy_dict.get("tags", [])
        author = strategy_dict.get("author", "")

        # Extract author from module docstring if not provided
        if not author and module.__doc__:
            for line in module.__doc__.split("\n"):
                if line.strip().startswith("Author:"):
                    author = line.split("Author:")[1].strip()
                    break

        # Create Strategy object
        return Strategy(
            name=name,
            rollout=strategy_dict["rollout"],
            population_module=strategy_dict["population_module"],
            description=description,
            tags=tags,
            author=author,
        )


# Convenience function for creating strategies
def create_strategy(
    rollout_modules: List,
    population_module: PopulationModule,
    description: str,
    tags: Optional[List[str]] = None,
    author: Optional[str] = None,
) -> Dict:
    """
    Helper to create a strategy dictionary from components.

    Use this in your strategy file's create_strategy() function.

    Args:
        rollout_modules: List of modules for the rollout pipeline
        population_module: Population management module
        description: Human-readable description
        tags: Optional tags for categorization
        author: Optional author name

    Returns:
        Dictionary ready to return from create_strategy()

    Example:
        >>> def create_strategy():
        ...     from famou.strategies import create_strategy as make_strategy
        ...     return make_strategy(
        ...         rollout_modules=[TopKSelect(k=1), MutationGenerate()],
        ...         population_module=TopKPopulation(),
        ...         description="My custom strategy",
        ...         tags=["exploitation"]
        ...     )
    """
    from famou.core.protocol import Rollout

    rollout = Rollout(rollout_modules, name="custom_rollout")

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": description,
        "tags": tags or [],
        "author": author or "",
    }
