"""
Selection modules for Famou 2.0.

Base class:
- SelectModule: Inherit to create custom selection strategies

Built-in implementations (import from files directly):
- topk.TopKSelect: Select from top-K programs by score
- random.RandomSelect: Random parent selection
"""

from famou.modules.select.base import SelectModule

__all__ = ["SelectModule"]
