"""Helpers for packaging Lean mathproof tasks as self-contained examples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from famou.utils.plan_lean_evaluator import create_plan_lean_evaluator
from famou.utils.proof_runtime_bridge import make_proof_runtime_bridge_verifier


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment flag with a sane default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _find_repo_root(start: Path) -> Path:
    """Find the repository root by locating a supported shared Lean workspace."""
    candidates: list[Path] = [start, *start.parents]

    env_repo_root = os.environ.get("FAMOU_REPO_ROOT")
    if env_repo_root:
        candidates.extend([Path(env_repo_root).resolve()])

    module_repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([module_repo_root, *module_repo_root.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "_proof" / "project").exists():
            return candidate
        if (candidate / "proof_runtime" / "project").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate repo root containing '_proof/project' or 'proof_runtime/project' "
        f"from: {start}"
    )


def _default_workspace_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    """Resolve the shared Lean verification workspace for the current repository layout."""
    if (repo_root / "_proof" / "project").exists():
        project_root = repo_root / "_proof" / "project"
    elif (repo_root / "proof_runtime" / "project").exists():
        project_root = repo_root / "proof_runtime" / "project"
    else:
        raise FileNotFoundError(
            "Could not locate a shared Lean workspace under '_proof/project' or "
            f"'proof_runtime/project' in: {repo_root}"
        )

    artifact_root = project_root / "temp_plan_eval"
    staging_root = project_root / "temp" / "plan_bridge"
    return project_root.resolve(), artifact_root.resolve(), staging_root.resolve()


def _resolve_under(base_dir: Path, maybe_relative: str | Path) -> Path:
    """Resolve a path relative to a base directory when needed."""
    path = Path(maybe_relative)
    return path if path.is_absolute() else (base_dir / path).resolve()


def create_mathproof_example_evaluator(
    example_dir: str | Path,
    problem_path: str | Path = "problem.lean",
    *,
    project_root: Optional[str | Path] = None,
    artifact_root: Optional[str | Path] = None,
    staging_root: Optional[str | Path] = None,
    timeout: Optional[int] = None,
    verify_all_nodes: Optional[bool] = None,
    verify_lemmas: Optional[bool] = None,
    cleanup_artifacts: bool = False,
):
    """
    Build a plan-aware Lean evaluator for one example directory.

    The example only needs to provide a local `problem.lean`; the shared Lean
    workspace remains the single verification backend.
    """
    example_root = Path(example_dir).resolve()
    repo_root = _find_repo_root(example_root)

    resolved_problem_path = _resolve_under(example_root, problem_path)
    if not resolved_problem_path.exists():
        raise FileNotFoundError(f"Lean problem scaffold not found: {resolved_problem_path}")

    default_project_root, default_artifact_root, default_staging_root = _default_workspace_paths(
        repo_root
    )
    lean_project_root = _resolve_under(repo_root, project_root or default_project_root)
    lean_artifact_root = _resolve_under(repo_root, artifact_root or default_artifact_root)
    lean_staging_root = _resolve_under(repo_root, staging_root or default_staging_root)
    lean_timeout = int(timeout or os.environ.get("FAMOU_LEAN_VERIFY_TIMEOUT", "120"))

    return create_plan_lean_evaluator(
        verifier_fn=make_proof_runtime_bridge_verifier(
            project_root=lean_project_root,
            staging_root=lean_staging_root,
            timeout=lean_timeout,
        ),
        project_root=lean_project_root,
        artifact_root=lean_artifact_root,
        theorem_scaffold_path=resolved_problem_path,
        default_imports="import Mathlib",
        reward_mode="outer",
        verify_all_nodes=_env_flag("FAMOU_LEAN_VERIFY_ALL_NODES", False)
        if verify_all_nodes is None
        else verify_all_nodes,
        verify_lemmas=_env_flag("FAMOU_LEAN_VERIFY_LEMMAS", False)
        if verify_lemmas is None
        else verify_lemmas,
        cleanup_artifacts=cleanup_artifacts,
    )
