"""
Execution environment infrastructure for Famou 2.0.

Provides:
- ExecutionEnvironment protocol (base interface)
- LocalEnv implementation (local subprocess-based)

Execution environments run user programs via custom evaluator functions,
enabling easy extensibility to Docker, remote, or cloud-based execution.
"""

from famou.infrastructure.env.base import ExecutionEnvironment
from famou.infrastructure.env.local_env import LocalEnv

__all__ = [
    "ExecutionEnvironment",
    "LocalEnv",
]
