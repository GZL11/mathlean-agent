"""
Feature extraction modules for Famou 2.0.

Base class:
- FeatureModule: Inherit to create custom feature extractors

Built-in implementations (import from files directly):
- embedding.EmbeddingFeature: LLM embedding-based feature extraction
"""

from famou.modules.feature.base import FeatureModule

__all__ = ["FeatureModule"]
