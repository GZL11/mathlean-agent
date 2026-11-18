"""Top-K selection strategy for Famou 2.0."""

from typing import List

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


class TopKSelect(SelectModule):
    """
    Top-K selection strategy (best programs).

    Selects:
    - Parent: The single best program by ranking_score
    - Inspirations: Next K best programs (excluding parent)

    Config:
        k: Number of top programs to consider as parent (default: 1)
        num_inspirations: Number of additional top programs as inspirations (default: 0)
        reverse: If True, select worst instead of best (default: False)

    Example:
        >>> selector = TopKSelect(k=1, num_inspirations=2)
        >>> result = selector(context, result)
        >>> result.selected_parent_id  # ID of best program
        >>> result.inspiration_ids  # IDs of 2nd and 3rd best programs
    """

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select the best program ID by score.

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs

        Returns:
            Selected program ID (best by score)
        """
        reverse = self.config.get("reverse", False)

        # Sort programs by ranking score (solved status first, then search score)
        sorted_programs = sorted(
            population,
            key=lambda p: p.ranking_score,
            reverse=not reverse,  # reverse=False for best (high scores first)
        )

        # Select best program (top of sorted list)
        best_program = sorted_programs[0]
        selected_id = best_program.id

        # Log selection info
        self.log_info(
            f"Selected best program {selected_id} with success/search: "
            f"{best_program.success_score:.3f}/{best_program.effective_search_score:.3f}"
        )

        return selected_id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Select next K best programs as inspirations (excluding parent).

        Uses the same ordering as parent selection (best scores first).

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs
            parent_id: ID of the selected parent (excluded from inspirations)

        Returns:
            List of inspiration program IDs (next best programs after parent)
        """
        reverse = self.config.get("reverse", False)
        num_inspirations = self.config.get("num_inspirations", 0)
        exclude_parent = self.config.get("exclude_parent", True)

        # Filter out parent if needed
        if exclude_parent:
            available = [p for p in population if p.id != parent_id]
        else:
            available = population

        # No inspirations requested
        if num_inspirations <= 0 or not available:
            return []

        # Sort by score using same logic as parent selection
        sorted_programs = sorted(available, key=lambda p: p.ranking_score, reverse=not reverse)

        # Select top-K as inspirations
        selected = sorted_programs[:num_inspirations]
        inspiration_ids = [p.id for p in selected]

        if inspiration_ids:
            scores_str = ", ".join([
                f"{pid}({self._get_score(population, pid):.3f})"
                for pid in inspiration_ids
            ])
            self.log_info(f"Selected {len(inspiration_ids)} inspirations: {scores_str}")

        return inspiration_ids

    def _get_score(self, population: List[Program], program_id: str) -> float:
        """Helper to get score for a program ID."""
        for p in population:
            if p.id == program_id:
                return p.effective_search_score
        return 0.0
