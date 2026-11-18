"""
ID generation and seeding utilities for Famou 2.0.

Provides functions for generating unique program IDs, managing random seeds
for reproducibility, and timestamp utilities.
"""

import random
import time
import uuid
from typing import Optional
from datetime import datetime, timedelta

now = datetime.now() + timedelta(hours=8)
current_time = now.strftime("%m-%d-%H-%M")
print("=" * 70)
print(f"    (Current time: {current_time})")

# Global seed for reproducibility
_GLOBAL_SEED: Optional[int] = None


def set_global_seed(seed: int = 42) -> None:
    """
    Set global seed for reproducibility.

    This sets the seed for:
    - Python's random module
    - UUID generation (via random)

    Args:
        seed: The seed value (default: 42, inspired by openevolve)
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    random.seed(seed)


def get_global_seed() -> Optional[int]:
    """Get the current global seed."""
    return _GLOBAL_SEED


def generate_short_uuid() -> str:
    """
    Generate a short UUID (8 characters).

    Returns:
        8-character hex string
    """
    return uuid.uuid4().hex[:8]


def generate_program_id(generation: int, iteration: int) -> str:
    """
    Generate a unique program ID in the format: {generation}_{iteration}_{uuid}.

    Args:
        generation: The genealogical generation (0=seed, 1=child, etc.)
        iteration: The iteration/round when this program was created

    Returns:
        Unique program ID string

    Example:
        >>> generate_program_id(0, 0)
        '0_0_a1b2c3d4'
    """
    short_id = generate_short_uuid()
    return f"{iteration}_{generation}_{short_id}"


def generate_experiment_id(name: Optional[str] = None) -> str:
    """
    Generate a unique experiment ID with datetime for readability.

    Format: {name}_{YYYYMMDD_HHMMSS}_{uuid4}

    The datetime prefix makes experiments:
    - Human-readable: You can tell when an experiment was created
    - Sortable: Experiments sort chronologically in file browsers
    - Unique: The UUID suffix guarantees uniqueness even if same second

    Args:
        name: Optional experiment name to include in ID

    Returns:
        Unique experiment ID

    Example:
        >>> generate_experiment_id("circle_packing")
        'circle_packing_20240115_143022_a1b2'
        >>> generate_experiment_id()
        'exp_20240115_143022_a1b2'
    """
    from datetime import datetime, timezone

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = generate_short_uuid()[:4]  # 4 chars for uniqueness

    if name:
        # Sanitize name (replace spaces with underscores, lowercase)
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        return f"{safe_name}_{current_time}_{short_id}"
    return f"exp_{short_id}"


def generate_rollout_id(experiment_id: str, iteration: int, island_id: Optional[int] = None) -> str:
    """
    Generate a rollout ID.

    Args:
        experiment_id: The experiment this rollout belongs to
        iteration: The iteration number (global iteration counter)
        island_id: Optional island ID for multi-island experiments

    Returns:
        Rollout ID string

    Example:
        >>> generate_rollout_id("exp_12345678", 3)
        'exp_12345678_rollout_3'
        >>> generate_rollout_id("exp_12345678", 3, island_id=1)
        'exp_12345678_island_1_rollout_3'
    """
    if island_id is not None:
        return f"{experiment_id}_island_{island_id}_rollout_{iteration}"
    return f"{experiment_id}_rollout_{iteration}"


def get_timestamp() -> float:
    """
    Get current UTC timestamp.

    Returns:
        Unix timestamp (seconds since epoch)
    """
    return time.time()


def timestamp_to_str(timestamp: float, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Convert timestamp to human-readable string.

    Args:
        timestamp: Unix timestamp
        format: strftime format string

    Returns:
        Formatted timestamp string
    """
    return time.strftime(format, time.gmtime(timestamp))


# Initialize with default seed on module import
set_global_seed(42)
