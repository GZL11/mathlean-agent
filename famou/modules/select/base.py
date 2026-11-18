"""
Selection modules for Famou 2.0.

**SIMPLIFIED: One Program Per Rollout**

Implements strategies for selecting:
- Parent program: The program to modify/mutate
- Inspiration programs: Other programs that serve as examples/reference for the LLM
"""

from abc import abstractmethod
from typing import List

from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Module


class SelectModule(Module):
    """
    Base class for selection strategies.

    **SIMPLIFIED: One Program Per Rollout**

    Selection modules choose:
    - Parent: The program to modify/mutate (single ID)
    - Inspirations: Other programs that serve as examples for the LLM (list of IDs)

    Design:
    - Subclasses implement select_parent() to return str (single program ID)
    - Subclasses implement select_inspirations() to return List[str] (program IDs)
    - Base class execute() handles the module protocol and calls both methods
    - result.selected_parent_id stores the parent program ID
    - result.inspiration_ids stores the inspiration program IDs

    This abstraction makes it easy to implement new selection strategies:
    just override select_parent() and select_inspirations() with your logic.
    """

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that population is available."""
        all_programs = context.get_all_programs()
        if not all_programs:
            raise ValueError(
                f"{self.name}: Cannot select from empty population. "
                "Make sure Experiment has programs in the current population."
            )

    @abstractmethod
    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select a single parent program from population.

        Args:
            context: Rollout context with population structure, diversity, iteration, etc.
            population: List of available programs (flattened from all buckets)

        Returns:
            Selected program ID (single string)
        """
        pass

    @abstractmethod
    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Select inspiration programs from population.

        Inspirations are high-quality programs that serve as examples/reference
        for the LLM during generation. They are NOT the program being modified.

        Default implementation selects top-K programs by score, excluding parent.
        Subclasses can override for different strategies.

        Args:
            context: Rollout context with population structure, diversity, iteration, etc.
            population: List of available programs (flattened from all buckets)
            parent_id: ID of the selected parent (excluded from inspirations)

        Returns:
            List of inspiration program IDs (may be empty)
        """
        pass

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Execute selection logic.

        Reads population, calls select_parent() and select_inspirations(),
        and writes to result.selected_parent_id and result.inspiration_ids.
        """
        # Flatten population from all buckets
        all_programs = context.get_all_programs()

        # Select parent
        selected_id = self.select_parent(context, all_programs)
        result.selected_parent_id = selected_id

        self.log_info(f"Selected parent: {selected_id}")

        # Select inspirations
        inspiration_ids = self.select_inspirations(context, all_programs, selected_id)
        self.log_info(f"Selected inspirations: {inspiration_ids}")
        result.inspiration_ids = inspiration_ids

        return result
