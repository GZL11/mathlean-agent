"""Lean evaluator for the namespace-scoped mathproof example."""

from __future__ import annotations

from pathlib import Path

from famou.utils.mathproof_example import create_mathproof_example_evaluator


CURRENT_DIR = Path(__file__).resolve().parent

evaluate = create_mathproof_example_evaluator(
    example_dir=CURRENT_DIR,
    problem_path="problem.lean",
)
