"""Random selection strategy for Famou 2.0."""

import random
from typing import List

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


class RandomSelect(SelectModule):
    """
    Random selection strategy.

    **SIMPLIFIED: One Program Per Rollout**

    Randomly selects:
    - Parent: Single random program from population
    - Inspirations: N random programs (excluding parent)

    Config:
        num_inspirations: Number of random inspirations to select (default: 0)

    Example:
        >>> selector = RandomSelect(num_inspirations=2)
        >>> result = selector(context, result)
        >>> result.selected_parent_id  # Single program ID
        >>> result.inspiration_ids  # List of 2 program IDs
    """

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Randomly select one program ID from population.

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs

        Returns:
            Selected program ID (single string)
        """
        # Get list of program IDs
        program_ids = [p.id for p in population]

        # Randomly select one
        selected_id = random.choice(program_ids)

        self.log_info(
            f"Randomly selected program {selected_id} from population of {len(program_ids)}"
        )

        return selected_id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Randomly select N inspiration programs from population (excluding parent).

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs
            parent_id: ID of the selected parent (excluded from inspirations)

        Returns:
            List of randomly selected inspiration program IDs
        """
        # Get config
        num_inspirations = self.config.get("num_inspirations", 0)

        # Filter out parent
        available = [p for p in population if p.id != parent_id]

        # No inspirations requested or not enough programs
        if num_inspirations <= 0 or not available:
            return []

        # Randomly select up to num_inspirations
        num_to_select = min(num_inspirations, len(available))
        selected = random.sample(available, num_to_select)
        inspiration_ids = [p.id for p in selected]

        if inspiration_ids:
            self.log_info(f"Randomly selected {len(inspiration_ids)} inspirations: {inspiration_ids}")

        return inspiration_ids
