"""Batch materialization helpers for Lean mathproof benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from famou.config import FamouConfig
from famou.utils.mathproof_benchmarks import (
    BenchmarkProblem,
    MathproofTaskSpec,
    benchmark_digest,
    list_minif2f_problems,
    list_putnambench_problems,
    materialize_mathproof_tasks,
)


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    """Summary of one materialized benchmark suite."""

    benchmark_name: str
    split: Optional[str]
    suite_root: Path
    manifest_path: Path
    task_specs: tuple[MathproofTaskSpec, ...]


EVALUATOR_TEMPLATE = '''"""Generated evaluator for one benchmark-backed mathproof task."""

from __future__ import annotations

from pathlib import Path

from famou.utils.mathproof_example import create_mathproof_example_evaluator


CURRENT_DIR = Path(__file__).resolve().parent

evaluate = create_mathproof_example_evaluator(
    example_dir=CURRENT_DIR,
    problem_path="problem.lean",
)
'''


RUN_SH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python -m famou \\
  -c config.yaml \\
  -p init.scratch \\
  -e evaluator.py
"""


def load_benchmark_problems(
    benchmark_name: str,
    benchmark_root: str | Path,
    *,
    split: Optional[str] = None,
) -> list[BenchmarkProblem]:
    """Dispatch benchmark loading to the appropriate adapter."""
    normalized = benchmark_name.strip().lower()
    if normalized == "minif2f":
        return list_minif2f_problems(benchmark_root, split=split or "test")
    if normalized == "putnambench":
        return list_putnambench_problems(benchmark_root)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def _filter_problems(
    problems: Sequence[BenchmarkProblem],
    *,
    problem_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> list[BenchmarkProblem]:
    """Select a subset of benchmark problems while preserving source order."""
    selected = list(problems)
    if problem_ids:
        wanted = {problem_id.strip() for problem_id in problem_ids if problem_id.strip()}
        selected = [
            problem
            for problem in selected
            if problem.problem_id in wanted or problem.theorem_name in wanted
        ]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = selected[:limit]
    return selected


def materialize_benchmark_suite(
    benchmark_name: str,
    benchmark_root: str | Path,
    output_root: str | Path,
    *,
    split: Optional[str] = None,
    problem_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> BenchmarkSuiteSpec:
    """
    Materialize a benchmark slice into current-style task directories.

    The output tree contains:
    - one directory per theorem with `problem.lean`, `init.scratch`, `metadata.json`
    - one top-level `manifest.json` summarizing the generated task set
    """
    loaded = load_benchmark_problems(benchmark_name, benchmark_root, split=split)
    selected = _filter_problems(loaded, problem_ids=problem_ids, limit=limit)
    if not selected:
        raise ValueError("No benchmark problems selected for materialization")

    suite_name = f"{selected[0].benchmark_name.lower()}_{selected[0].split or 'all'}"
    suite_digest = benchmark_digest(selected)
    suite_root = Path(output_root).resolve() / f"{suite_name}_{suite_digest}"
    suite_root.mkdir(parents=True, exist_ok=True)

    task_specs = tuple(materialize_mathproof_tasks(selected, suite_root))
    manifest = {
        "benchmark_name": selected[0].benchmark_name,
        "split": selected[0].split,
        "num_tasks": len(task_specs),
        "suite_digest": suite_digest,
        "suite_root": str(suite_root),
        "tasks": [
            {
                "task_id": spec.task_id,
                "benchmark_name": spec.benchmark_name,
                "theorem_name": spec.theorem_name,
                "theorem_statement": spec.theorem_statement,
                "task_dir": str(spec.task_dir),
                "problem_path": str(spec.problem_path),
                "init_scratch_path": str(spec.init_scratch_path),
                "source_path": str(spec.source_path),
                "split": spec.split,
            }
            for spec in task_specs
        ],
    }
    manifest_path = suite_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return BenchmarkSuiteSpec(
        benchmark_name=selected[0].benchmark_name,
        split=selected[0].split,
        suite_root=suite_root,
        manifest_path=manifest_path,
        task_specs=task_specs,
    )


def _sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text).strip("_")


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def prepare_suite_runtime(
    suite: BenchmarkSuiteSpec,
    config_template_path: str | Path,
    *,
    storage_base_path: Optional[str] = None,
) -> BenchmarkSuiteSpec:
    """
    Create per-task `config.yaml`, `evaluator.py`, and `run.sh` files from a template config.

    This keeps benchmark task generation separate from user-specific LLM/runtime settings:
    the template config is treated as the source of truth for infrastructure and algorithm knobs.
    """
    template = FamouConfig.from_yaml(str(config_template_path))
    suite_slug = _sanitize_name(f"{suite.benchmark_name.lower()}_{suite.split or 'all'}")

    for task in suite.task_specs:
        task_config = FamouConfig.from_dict(deepcopy(template.to_dict()))
        task_name = _sanitize_name(f"{suite_slug}_{task.task_id}")
        task_config.experiment.name = task_name
        task_config.experiment.task_description = "problem.lean"
        if storage_base_path is not None:
            task_config.infrastructure.storage.base_path = storage_base_path

        task.task_dir.mkdir(parents=True, exist_ok=True)
        task_config.to_yaml(str(task.task_dir / "config.yaml"))
        _write_if_changed(task.task_dir / "evaluator.py", EVALUATOR_TEMPLATE)
        _write_if_changed(task.task_dir / "run.sh", RUN_SH_TEMPLATE)
        os.chmod(task.task_dir / "run.sh", 0o755)

    runtime_manifest = {
        "suite_root": str(suite.suite_root),
        "config_template_path": str(Path(config_template_path).resolve()),
        "tasks": [
            {
                "task_id": task.task_id,
                "task_dir": str(task.task_dir),
                "config_path": str(task.task_dir / "config.yaml"),
                "evaluator_path": str(task.task_dir / "evaluator.py"),
                "run_script_path": str(task.task_dir / "run.sh"),
            }
            for task in suite.task_specs
        ],
    }
    _write_if_changed(
        suite.suite_root / "runtime_manifest.json",
        json.dumps(runtime_manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return suite


def execute_suite_runtime(
    suite: BenchmarkSuiteSpec,
    *,
    task_ids: Optional[Sequence[str]] = None,
    per_task_timeout: Optional[int] = None,
) -> Path:
    """Run each generated task directory sequentially via `python -m famou`."""
    selected = list(suite.task_specs)
    if task_ids:
        wanted = {task_id.strip() for task_id in task_ids if task_id.strip()}
        selected = [task for task in selected if task.task_id in wanted]
    if not selected:
        raise ValueError("No tasks selected for execution")

    results = []
    for task in selected:
        command = [
            "python",
            "-m",
            "famou",
            "-c",
            "config.yaml",
            "-p",
            "init.scratch",
            "-e",
            "evaluator.py",
        ]
        start = time.time()
        timeout_hit = False
        try:
            completed = subprocess.run(
                command,
                cwd=task.task_dir,
                text=True,
                capture_output=True,
                timeout=per_task_timeout,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timeout_hit = True
            stdout = _coerce_text(exc.stdout)
            stderr = _coerce_text(exc.stderr)
            returncode = -9
        elapsed = time.time() - start
        (task.task_dir / "batch_stdout.log").write_text(stdout, encoding="utf-8")
        (task.task_dir / "batch_stderr.log").write_text(stderr, encoding="utf-8")
        results.append(
            {
                "task_id": task.task_id,
                "task_dir": str(task.task_dir),
                "returncode": returncode,
                "timeout": timeout_hit,
                "elapsed_seconds": elapsed,
                "stdout_path": str(task.task_dir / "batch_stdout.log"),
                "stderr_path": str(task.task_dir / "batch_stderr.log"),
            }
        )

    summary_path = suite.suite_root / "batch_run_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return _load_json(path)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_text(value: Any) -> str:
    """Normalize subprocess stdout/stderr payloads to text for logging."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _metric(program: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = program.get("metrics") or {}
    if key in metrics:
        return _coerce_float(metrics.get(key), default)
    plan_metrics = ((program.get("meta") or {}).get("plan_metrics") or {})
    if key in plan_metrics:
        return _coerce_float(plan_metrics.get(key), default)
    if key in program:
        return _coerce_float(program.get(key), default)
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(base_dir: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_task_runtime(task_dir: Path, config_path: Path) -> tuple[Path, str]:
    config = FamouConfig.from_yaml(str(config_path))
    storage_root = _resolve_path(task_dir, config.infrastructure.storage.base_path)
    experiment_name = config.experiment.name
    return storage_root, experiment_name


def _find_experiment_dirs(storage_root: Path, experiment_name: str) -> list[Path]:
    if not storage_root.exists():
        return []
    candidates = []
    for child in storage_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == experiment_name or child.name.startswith(f"{experiment_name}_"):
            candidates.append(child)
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return candidates


def _checkpoint_iteration(checkpoint_path: Path) -> int:
    stem = checkpoint_path.stem
    suffix = stem.removeprefix("experiment_checkpoint_")
    return _safe_int(suffix, default=-1)


def _latest_checkpoint(exp_dir: Path) -> Optional[Path]:
    checkpoint_paths = list(exp_dir.glob("experiment_checkpoint_*.json"))
    if not checkpoint_paths:
        return None
    checkpoint_paths.sort(key=_checkpoint_iteration)
    return checkpoint_paths[-1]


def _program_payloads(exp_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    programs_dir = exp_dir / "programs"
    if not programs_dir.exists():
        return payloads
    for program_path in sorted(programs_dir.glob("*.json")):
        payload = _maybe_load_json(program_path)
        if payload is None:
            continue
        payload["_program_json_path"] = str(program_path.resolve())
        payloads.append(payload)
    return payloads


def _program_rank_key(program: dict[str, Any]) -> tuple[float, float, float, float, int, int, str]:
    return (
        _metric(program, "success_score"),
        _metric(program, "search_score"),
        _metric(program, "combined_score"),
        _coerce_float(program.get("validity"), 0.0),
        _safe_int(program.get("iteration"), default=0),
        _safe_int(program.get("generation"), default=0),
        str(program.get("id") or ""),
    )


def _best_program_payload(exp_dir: Path, checkpoint: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    preferred_id = None if checkpoint is None else checkpoint.get("best_program_id")
    if preferred_id:
        preferred_path = exp_dir / "programs" / f"{preferred_id}.json"
        preferred_payload = _maybe_load_json(preferred_path)
        if preferred_payload is not None:
            preferred_payload["_program_json_path"] = str(preferred_path.resolve())
            return preferred_payload

    payloads = _program_payloads(exp_dir)
    if not payloads:
        return None
    return max(payloads, key=_program_rank_key)


def _iterations_completed(
    checkpoint: Optional[dict[str, Any]],
    programs: Sequence[dict[str, Any]],
) -> int:
    if checkpoint is not None:
        return _safe_int(checkpoint.get("current_iteration"), default=0)
    if not programs:
        return 0
    return max(_safe_int(program.get("iteration"), default=0) for program in programs)


def _default_summary_json_path(suite_root: Path) -> Path:
    return suite_root / "suite_result_summary.json"


def _default_summary_csv_path(suite_root: Path) -> Path:
    return suite_root / "suite_result_summary.csv"


def summarize_suite_results(
    *,
    suite_root: str | Path | None = None,
    runtime_manifest_path: str | Path | None = None,
    batch_summary_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Summarize benchmark task outputs into one suite-level JSON/CSV report.

    The summarizer is intentionally tolerant to partial runs:
    - tasks without any experiment directory are still reported
    - seed-only runs without checkpoints are supported
    - batch-level return code / timeout information is merged when available
    """
    if suite_root is None and runtime_manifest_path is None:
        raise ValueError("Either suite_root or runtime_manifest_path must be provided")

    runtime_manifest: dict[str, Any]
    if runtime_manifest_path is not None:
        runtime_manifest_file = Path(runtime_manifest_path).resolve()
        runtime_manifest = _load_json(runtime_manifest_file)
        suite_root_path = Path(runtime_manifest.get("suite_root") or runtime_manifest_file.parent).resolve()
    else:
        suite_root_path = Path(suite_root).resolve()
        runtime_manifest_file = suite_root_path / "runtime_manifest.json"
        runtime_manifest = _load_json(runtime_manifest_file)

    manifest_path = suite_root_path / "manifest.json"
    manifest = _maybe_load_json(manifest_path) or {}

    batch_summary_file: Optional[Path] = None
    if batch_summary_path is not None:
        batch_summary_file = Path(batch_summary_path).resolve()
    else:
        candidate = suite_root_path / "batch_run_summary.json"
        if candidate.exists():
            batch_summary_file = candidate

    batch_summary = _maybe_load_json(batch_summary_file) if batch_summary_file is not None else None
    batch_by_task = {
        str(entry.get("task_id")): entry
        for entry in (batch_summary or {}).get("results", [])
        if entry.get("task_id") is not None
    }

    task_results: list[dict[str, Any]] = []
    for task_entry in runtime_manifest.get("tasks", []):
        task_dir = Path(task_entry["task_dir"]).resolve()
        config_path = Path(task_entry["config_path"]).resolve()
        metadata = _maybe_load_json(task_dir / "metadata.json") or {}
        storage_root, experiment_name = _resolve_task_runtime(task_dir, config_path)
        experiment_dirs = _find_experiment_dirs(storage_root, experiment_name)
        experiment_dir = experiment_dirs[-1] if experiment_dirs else None
        checkpoint_path = _latest_checkpoint(experiment_dir) if experiment_dir is not None else None
        checkpoint = _maybe_load_json(checkpoint_path) if checkpoint_path is not None else None
        program_payloads = _program_payloads(experiment_dir) if experiment_dir is not None else []
        best_program = _best_program_payload(experiment_dir, checkpoint) if experiment_dir is not None else None
        batch_entry = batch_by_task.get(str(task_entry.get("task_id")))

        success_score = _metric(best_program or {}, "success_score")
        search_score = _metric(best_program or {}, "search_score")
        combined_score = _metric(best_program or {}, "combined_score")
        theorem_solved = _metric(best_program or {}, "theorem_solved")
        proof_validity = _metric(best_program or {}, "proof_validity")
        compile_success = _metric(best_program or {}, "compile_success")
        no_sorry = _metric(best_program or {}, "no_sorry")
        statement_match = _metric(best_program or {}, "statement_match")
        final_compile_success = _metric(best_program or {}, "final_compile_success")
        final_no_sorry = _metric(best_program or {}, "final_no_sorry")
        final_statement_match = _metric(best_program or {}, "final_statement_match")
        solved = success_score >= 0.999999 or theorem_solved >= 0.999999

        task_results.append(
            {
                "task_id": task_entry.get("task_id"),
                "benchmark_name": metadata.get("benchmark_name", manifest.get("benchmark_name")),
                "split": metadata.get("split", manifest.get("split")),
                "theorem_name": metadata.get("theorem_name"),
                "theorem_statement": metadata.get("theorem_statement"),
                "task_dir": str(task_dir),
                "config_path": str(config_path),
                "storage_root": str(storage_root),
                "experiment_name": experiment_name,
                "experiment_dir": None if experiment_dir is None else str(experiment_dir),
                "experiment_found": experiment_dir is not None,
                "latest_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
                "best_program_id": None if best_program is None else best_program.get("id"),
                "best_program_path": None if best_program is None else best_program.get("_program_json_path"),
                "num_programs": len(program_payloads),
                "iterations_completed": _iterations_completed(checkpoint, program_payloads),
                "combined_score": combined_score,
                "success_score": success_score,
                "search_score": search_score,
                "theorem_solved": theorem_solved,
                "proof_validity": proof_validity,
                "compile_success": compile_success,
                "no_sorry": no_sorry,
                "statement_match": statement_match,
                "final_compile_success": final_compile_success,
                "final_no_sorry": final_no_sorry,
                "final_statement_match": final_statement_match,
                "validity": _coerce_float((best_program or {}).get("validity"), 0.0),
                "solved": solved,
                "run_returncode": None if batch_entry is None else batch_entry.get("returncode"),
                "run_timeout": None if batch_entry is None else batch_entry.get("timeout"),
                "run_elapsed_seconds": None if batch_entry is None else batch_entry.get("elapsed_seconds"),
                "batch_stdout_path": None if batch_entry is None else batch_entry.get("stdout_path"),
                "batch_stderr_path": None if batch_entry is None else batch_entry.get("stderr_path"),
            }
        )

    num_tasks = len(task_results)
    num_solved = sum(1 for task in task_results if task["solved"])
    mean_success_score = sum(task["success_score"] for task in task_results) / num_tasks if num_tasks else 0.0
    mean_search_score = sum(task["search_score"] for task in task_results) / num_tasks if num_tasks else 0.0
    total_elapsed_seconds = sum(
        _coerce_float(task["run_elapsed_seconds"], 0.0)
        for task in task_results
        if task["run_elapsed_seconds"] is not None
    )

    summary = {
        "benchmark_name": manifest.get("benchmark_name"),
        "split": manifest.get("split"),
        "suite_root": str(suite_root_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "runtime_manifest_path": str(runtime_manifest_file),
        "batch_run_summary_path": None if batch_summary_file is None else str(batch_summary_file),
        "num_tasks": num_tasks,
        "num_solved": num_solved,
        "success_rate": (num_solved / num_tasks) if num_tasks else 0.0,
        "mean_success_score": mean_success_score,
        "mean_search_score": mean_search_score,
        "total_elapsed_seconds": total_elapsed_seconds,
        "tasks": task_results,
    }

    summary_json_path = Path(output_json_path).resolve() if output_json_path else _default_summary_json_path(suite_root_path)
    summary_csv_path = Path(output_csv_path).resolve() if output_csv_path else _default_summary_csv_path(suite_root_path)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_fields = [
        "task_id",
        "benchmark_name",
        "split",
        "theorem_name",
        "theorem_statement",
        "experiment_name",
        "experiment_dir",
        "best_program_id",
        "num_programs",
        "iterations_completed",
        "combined_score",
        "success_score",
        "search_score",
        "theorem_solved",
        "proof_validity",
        "compile_success",
        "no_sorry",
        "statement_match",
        "final_compile_success",
        "final_no_sorry",
        "final_statement_match",
        "validity",
        "solved",
        "run_returncode",
        "run_timeout",
        "run_elapsed_seconds",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for task in task_results:
            writer.writerow({field: task.get(field) for field in csv_fields})

    summary["summary_json_path"] = str(summary_json_path)
    summary["summary_csv_path"] = str(summary_csv_path)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["minif2f", "putnambench"])
    parser.add_argument("--benchmark-root")
    parser.add_argument("--output-root")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-id", action="append", default=None)
    parser.add_argument("--config-template", default=None)
    parser.add_argument("--storage-base-path", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summarize-suite-root", default=None)
    parser.add_argument("--runtime-manifest", default=None)
    parser.add_argument("--batch-summary", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--per-task-timeout", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for batch materializing benchmark tasks."""
    args = _build_arg_parser().parse_args(argv)
    if args.summarize_suite_root or args.runtime_manifest:
        summary = summarize_suite_results(
            suite_root=args.summarize_suite_root,
            runtime_manifest_path=args.runtime_manifest,
            batch_summary_path=args.batch_summary,
            output_json_path=args.summary_json,
            output_csv_path=args.summary_csv,
        )
        print(f"suite_root={summary['suite_root']}")
        print(f"summary_json={summary['summary_json_path']}")
        print(f"summary_csv={summary['summary_csv_path']}")
        print(f"num_tasks={summary['num_tasks']}")
        print(f"num_solved={summary['num_solved']}")
        print(f"success_rate={summary['success_rate']:.4f}")
        return 0

    if not args.benchmark or not args.benchmark_root or not args.output_root:
        raise ValueError(
            "benchmark materialization requires --benchmark, --benchmark-root, and --output-root"
        )
    suite = materialize_benchmark_suite(
        benchmark_name=args.benchmark,
        benchmark_root=args.benchmark_root,
        output_root=args.output_root,
        split=args.split,
        problem_ids=args.problem_id,
        limit=args.limit,
    )
    if args.config_template:
        prepare_suite_runtime(
            suite,
            config_template_path=args.config_template,
            storage_base_path=args.storage_base_path,
        )
    print(f"suite_root={suite.suite_root}")
    print(f"manifest={suite.manifest_path}")
    print(f"num_tasks={len(suite.task_specs)}")
    if args.config_template:
        print(f"runtime_manifest={suite.suite_root / 'runtime_manifest.json'}")
    if args.run:
        if not args.config_template:
            raise ValueError("--run requires --config-template so each task has config/evaluator files")
        summary_path = execute_suite_runtime(
            suite,
            task_ids=args.problem_id,
            per_task_timeout=args.per_task_timeout,
        )
        print(f"batch_run_summary={summary_path}")
    if args.summarize:
        summary = summarize_suite_results(
            suite_root=suite.suite_root,
            batch_summary_path=args.batch_summary or (summary_path if args.run else None),
            output_json_path=args.summary_json,
            output_csv_path=args.summary_csv,
        )
        print(f"summary_json={summary['summary_json_path']}")
        print(f"summary_csv={summary['summary_csv_path']}")
        print(f"num_solved={summary['num_solved']}")
        print(f"success_rate={summary['success_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
