"""
Population management modules for Famou 2.0.

Base class:
- PopulationModule: Inherit to create custom population management strategies

Built-in implementations (import from files directly):
- topk.TopKPopulation: Keep top-K programs by combined_score
- cluster.ClusterPopulation: Cluster-based diversity maintenance
"""

from famou.modules.population.base import PopulationModule

__all__ = ["PopulationModule"]
