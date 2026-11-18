"""Cluster adaptive selection strategy for Famou 2.0.

This module implements an adaptive selection strategy designed to work with
ClusterPopulation. It dynamically switches between exploration and exploitation
modes based on population diversity and iteration progress.

Key ideas from sample.py methodology:
- Exploration mode: Encourage diversity by selecting from different clusters
- Exploitation mode: Refine best solutions by focusing on high-performing clusters
- Temperature-controlled softmax for probabilistic cluster selection
"""

import math
from typing import Dict, List, Optional

import numpy as np

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


class ClusterAdaptiveSelect(SelectModule):
    """
    Adaptive cluster-based selection that balances exploration and exploitation.

    Designed to work with ClusterPopulation's bucket structure:
        {"cluster_0": [Program, ...], "cluster_1": [Program, ...], ...}

    Strategy switching:
    - Exploration (low diversity or early iterations): Sample diverse parents,
      inspirations from OTHER clusters to encourage diversity
    - Exploitation (high diversity or late iterations): Focus on best clusters,
      inspirations from SAME cluster to refine solutions

    Config:
        diversity_threshold: Switch to exploitation when diversity >= this (default: 0.25)
        progress_threshold: Switch to exploitation when iteration >= this fraction (default: 0.8)
        initial_temperature: Base temperature for softmax (default: 1.0)
        num_inspirations: Number of inspiration programs (default: 2)
    """

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select parent using cluster-aware adaptive strategy.

        1. Determine strategy (exploration vs exploitation) from context
        2. Extract cluster structure from context.population
        3. Score each cluster by its elite (best program)
        4. Use temperature-scaled softmax to select cluster
        5. Exploitation: return cluster elite
        6. Exploration: weighted sample within selected cluster

        Args:
            context: Rollout context with population, diversity, iteration
            population: List of available programs (flattened)

        Returns:
            Selected program ID
        """
        strategy = self._get_strategy(context)
        clusters = self._extract_clusters(context)

        if not clusters:
            self.log_warning("No cluster structure found, using best program")
            return self._select_best(population)

        # Find elite (best program) for each non-empty cluster
        cluster_elites = {}
        cluster_scores = []
        non_empty_cluster_ids = []

        for cluster_id in sorted(clusters.keys()):
            programs = clusters[cluster_id]
            if programs:  # Only consider non-empty clusters
                elite = max(programs, key=lambda p: p.combined_score or 0.0)
                cluster_elites[cluster_id] = elite
                cluster_scores.append(elite.combined_score or 0.0)
                non_empty_cluster_ids.append(cluster_id)

        if not cluster_elites:
            self.log_warning("All clusters empty, using best from population")
            return self._select_best(population)

        # replace zeros with min non-zero score
        if max(cluster_scores) > 0:
            non_zero_scores = [s for s in cluster_scores if s > 0]
            if non_zero_scores:
                min_score = min(non_zero_scores)
                cluster_scores = [s if s > 0 else min_score for s in cluster_scores]

        # Normalize scores to reasonable range (avoid overflow in exp)
        while max(cluster_scores) > 2:
            cluster_scores = [s / 2 for s in cluster_scores]

        # Compute temperature based on diversity and progress
        temperature = self._compute_temperature(context)

        # Softmax selection of cluster
        cluster_scores = np.nan_to_num(cluster_scores, nan=0.0)
        probabilities = self._stable_softmax(cluster_scores, temperature)

        selected_idx = np.random.choice(len(non_empty_cluster_ids), p=probabilities)
        selected_cluster = non_empty_cluster_ids[selected_idx]

        # Get programs from selected cluster
        selected_programs = clusters.get(selected_cluster, [])
        if not selected_programs:
            return self._select_best(population)

        # Exploitation: return cluster elite
        if strategy == "exploitation":
            elite = cluster_elites.get(selected_cluster)
            if elite:

                self.log_info(
                    f"Exploitation: selected elite from {selected_cluster} "
                    f"(score={elite.combined_score:.4f})"
                )
                return elite.id

        # Exploration: weighted sample within cluster
        weights = [p.combined_score or 0.0 for p in selected_programs]
        weights = [max(w, 0.0) for w in weights]

        if sum(weights) == 0:
            weights = [1.0] * len(selected_programs)

        weights = np.array(weights)
        weights = weights / weights.sum()

        selected = np.random.choice(selected_programs, p=weights)
        self.log_info(
            f"Exploration: sampled from {selected_cluster} "
            f"(score={(selected.combined_score or 0):.4f})"
        )
        return selected.id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Select inspiration programs based on current strategy.

        - Exploitation: Sample high-scoring programs from SAME cluster as parent
          (to refine solutions within the local region)
        - Exploration: Select diverse programs from OTHER clusters
          (to encourage cross-pollination of ideas)

        Args:
            context: Rollout context with population, diversity, iteration
            population: List of available programs
            parent_id: ID of the selected parent

        Returns:
            List of inspiration program IDs
        """
        num_inspirations = self.config.get("num_inspirations", 2)
        if num_inspirations <= 0:
            return []

        strategy = self._get_strategy(context)
        clusters = self._extract_clusters(context)

        if not clusters:
            return self._select_top_inspirations(population, parent_id, num_inspirations)

        # Find parent's cluster
        parent_cluster = None
        for cluster_id, programs in clusters.items():
            if any(p.id == parent_id for p in programs):
                parent_cluster = cluster_id
                break

        if strategy == "exploitation":
            return self._exploitation_inspirations(
                clusters, parent_id, parent_cluster, num_inspirations
            )
        else:
            return self._exploration_inspirations(
                clusters, parent_id, parent_cluster, num_inspirations, population
            )

    # ========== Strategy Helpers ==========

    def _get_strategy(self, context: Context) -> str:
        """
        Determine strategy based on diversity and iteration progress.

        Switches to exploitation when:
        - Population diversity >= diversity_threshold, OR
        - Current iteration >= progress_threshold * max_iterations
        """
        diversity_threshold = self.config.get("diversity_threshold", 0.25)
        progress_threshold = self.config.get("progress_threshold", 0.8)

        diversity = context.diversity
        current_iter = context.iteration
        max_iter = (
            context.experiment_config.max_iterations
            if context.experiment_config
            else 100
        )

        progress = current_iter / max_iter if max_iter > 0 else 0

        if diversity >= diversity_threshold or progress >= progress_threshold:
            strategy = "exploitation"
        else:
            strategy = "exploration"

        self.log_info(
            f"Strategy: {strategy} (diversity={diversity:.3f}, progress={progress:.1%})"
        )
        return strategy

    def _compute_temperature(self, context: Context) -> float:
        """
        Compute temperature for softmax based on diversity and progress.

        Lower temperature -> more deterministic (favor best clusters)
        Higher temperature -> more random (explore more clusters)
        """
        initial_temp = self.config.get("initial_temperature", 1.0)

        diversity = context.diversity
        current_iter = context.iteration
        max_iter = (
            context.experiment_config.max_iterations
            if context.experiment_config
            else 100
        )

        progress = current_iter / max_iter * 4 if max_iter > 0 else 0

        # Temperature decreases as we progress and as diversity increases
        temperature = initial_temp * (1 - progress) * (1 - diversity)
        return max(temperature, 0.1)

    # ========== Inspiration Helpers ==========

    def _exploitation_inspirations(
        self,
        clusters: Dict[str, List[Program]],
        parent_id: str,
        parent_cluster: Optional[str],
        n: int,
    ) -> List[str]:
        """Select high-scoring programs from parent's cluster."""
        if parent_cluster is None:
            return []

        candidates = [
            p for p in clusters.get(parent_cluster, [])
            if p.id != parent_id and (p.combined_score or 0.0) > 0
        ]

        if not candidates:
            return []

        scores = np.array([p.combined_score or 0.0 for p in candidates])
        scores = np.nan_to_num(scores, nan=0.0)

        if scores.sum() == 0:
            scores = np.ones(len(candidates))

        scores = scores / scores.sum()

        n = min(n, len(candidates))
        selected_indices = np.random.choice(
            len(candidates), size=n, replace=False, p=scores
        )

        inspirations = [candidates[i].id for i in selected_indices]
        self.log_info(f"Exploitation inspirations from {parent_cluster}: {inspirations}")
        return inspirations

    def _exploration_inspirations(
        self,
        clusters: Dict[str, List[Program]],
        parent_id: str,
        parent_cluster: Optional[str],
        n: int,
        population: List[Program],
    ) -> List[str]:
        """Select diverse programs from OTHER clusters."""
        candidates = []
        for cluster_id, programs in clusters.items():
            if cluster_id != parent_cluster:
                candidates.extend(programs)

        candidates = [p for p in candidates if p.id != parent_id]

        if not candidates:
            return self._select_top_inspirations(population, parent_id, n)

        diverse_ids = self._get_most_diverse(candidates, n)
        self.log_info(f"Exploration inspirations (diverse): {diverse_ids}")
        return diverse_ids

    # ========== Utility Methods ==========

    def _extract_clusters(self, context: Context) -> Dict[str, List[Program]]:
        """Extract cluster structure from context population."""
        clusters = {}
        for bucket_id, programs in context.population.items():
            if bucket_id.startswith("cluster_"):
                clusters[bucket_id] = list(programs)
        return clusters

    def _select_best(self, population: List[Program]) -> str:
        """Fallback: select best program by score."""
        if not population:
            raise ValueError("Cannot select from empty population")
        best = max(population, key=lambda p: p.combined_score or 0.0)
        return best.id

    def _select_top_inspirations(
        self, population: List[Program], parent_id: str, n: int
    ) -> List[str]:
        """Fallback: select top N programs by score, excluding parent."""
        candidates = [p for p in population if p.id != parent_id]
        sorted_candidates = sorted(
            candidates, key=lambda p: p.combined_score or 0.0, reverse=True
        )
        return [p.id for p in sorted_candidates[:n]]

    def _stable_softmax(self, x: List[float], temperature: float = 1.0) -> np.ndarray:
        """Numerically stable softmax with temperature scaling."""
        x = np.array(x)
        x = x / max(temperature, 1e-8)
        x_max = np.max(x)
        x_exp = np.exp(x - x_max)
        return x_exp / x_exp.sum()

    def _get_most_diverse(self, candidates: List[Program], n: int) -> List[str]:
        """Select n most diverse programs using greedy max-min diversity."""
        if not candidates:
            return []

        with_features = [p for p in candidates if p.feature_vector is not None]
        if not with_features:
            sorted_by_score = sorted(
                candidates, key=lambda p: p.combined_score or 0.0, reverse=True
            )
            return [p.id for p in sorted_by_score[:n]]

        # Greedy max-min selection
        selected: List[Program] = []
        remaining = list(with_features)

        remaining.sort(key=lambda p: p.combined_score or 0.0, reverse=True)
        selected.append(remaining.pop(0))

        while len(selected) < n and remaining:
            best_candidate = None
            best_min_dist = -1

            for candidate in remaining:
                min_dist = min(
                    self._cosine_distance(candidate.feature_vector, s.feature_vector)
                    for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return [p.id for p in selected]

    def _cosine_distance(
        self, vec1: Optional[List[float]], vec2: Optional[List[float]]
    ) -> float:
        """Compute cosine distance: 1 - cosine_similarity."""
        if not vec1 or not vec2:
            return 1.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 1.0

        return 1.0 - (dot / (norm1 * norm2))
