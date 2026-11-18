"""
Storage infrastructure for Famou 2.0.

Provides:
- DataService: Protocol for data persistence
- LocalStorage: Local file-based implementation
"""

from famou.infrastructure.storage.base import DataService
from famou.infrastructure.storage.local_storage import LocalStorage

__all__ = [
    "DataService",
    "LocalStorage",
]
