"""
Strategies module for Famou 2.0.

Strategies combine rollout pipelines with population modules.
"""

from famou.strategies._registry import Strategy, StrategyRegistry, create_strategy

__all__ = ["Strategy", "StrategyRegistry", "create_strategy"]
