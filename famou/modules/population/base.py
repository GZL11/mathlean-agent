"""
Population management base modules for Famou 2.0.

**SIMPLIFIED: One Program Per Rollout**

These modules handle how a new program is merged into the population.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from famou.core.data import Program
from famou.core.protocol import Module

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig


class PopulationModule(Module):
    """
    Base class for population management strategies.
    Decide whether the new program from the current rollout should be added to
    the population and which existing programs should be kept/removed.

    **Data Flow:**
    - Reads: result.generated_program (new program with score)
    - Reads: context.population (current population)
    - Returns: Updated population via update_population() (called by Evolver)

    **Design Pattern:**
    PopulationModule update_population() is called by Evolver after a rollout completes,
    not as part of the Rollout pipeline. This is because population updates happen
    between rollouts, not during them.
    """

    def execute(self, context, result, **kwargs):
        """
        PopulationModule is not executed in the rollout pipeline.

        It is called by Evolver.update_population() after a rollout completes.
        This method should not be called directly.
        """
        raise NotImplementedError(
            f"{self.name}: PopulationModule.execute() should not be called. "
            "Use update_population() instead, which is called by Evolver."
        )

    @abstractmethod
    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Update population with a new program.

        Implements the strategy for merging the new program into the existing
        population (e.g., Top-K, diversity-based, MAP-Elites, etc.)

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: Newly generated program (with score)
            experiment_config: Experiment configuration for accessing settings
                like island_size. Optional for backwards compatibility.
            island_id: Island identifier for multi-island evolution (default: 0).

        Returns:
            Updated population {bucket_id: [programs]}
        """
        pass
