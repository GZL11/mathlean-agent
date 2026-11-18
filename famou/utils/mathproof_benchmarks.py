"""Benchmark adapters for Lean mathproof evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None


THEOREM_START_RE = re.compile(r"^\s*theorem\s+([A-Za-z0-9_']+)\b")
THEOREM_HEADER_RE = re.compile(
    r"^\s*theorem\s+([A-Za-z0-9_']+)\b(.*?)\s*:=\s*",
    re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class BenchmarkProblem:
    """One benchmark theorem that can be materialized into a single Lean scaffold."""

    benchmark_name: str
    problem_id: str
    theorem_name: str
    source_path: Path
    split: Optional[str] = None
    inline_block: Optional[str] = None
    preamble: str = ""

    @property
    def cache_subdir(self) -> str:
        """Stable subdirectory name for cached materialized scaffolds."""
        suffix = f"{self.benchmark_name}_{self.split or 'all'}"
        return suffix.lower()

    @property
    def materialized_filename(self) -> str:
        """Stable file name for the theorem scaffold."""
        safe_id = re.sub(r"[^A-Za-z0-9_]+", "_", self.problem_id).strip("_")
        return f"{safe_id or self.theorem_name}.lean"


@dataclass(frozen=True)
class MathproofTaskSpec:
    """A benchmark theorem packaged in the same shape as current mathproof examples."""

    task_id: str
    benchmark_name: str
    theorem_name: str
    theorem_statement: str
    problem_path: Path
    init_scratch_path: Path
    task_dir: Path
    source_path: Path
    split: Optional[str] = None


def _resolve_dir(path: str | Path) -> Path:
    """Resolve and validate a benchmark root directory."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark root not found: {resolved}")
    return resolved


def _load_text(path: Path) -> str:
    """Read text from a Lean source file."""
    return path.read_text(encoding="utf-8")


def _split_theorem_blocks(source_text: str) -> tuple[str, List[tuple[str, str]]]:
    """
    Split a Lean file into a shared preamble and theorem blocks.

    This adapter intentionally targets benchmark-style files where multiple theorems
    are laid out sequentially with a common import/preamble section.
    """
    lines = source_text.splitlines()
    starts: List[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = THEOREM_START_RE.match(line)
        if match:
            starts.append((index, match.group(1)))

    if not starts:
        raise ValueError("No theorem declarations found in Lean benchmark source")

    preamble = "\n".join(lines[: starts[0][0]]).rstrip()
    blocks: List[tuple[str, str]] = []
    for idx, (start_line, theorem_name) in enumerate(starts):
        end_line = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        block = "\n".join(lines[start_line:end_line]).strip()
        blocks.append((theorem_name, block))
    return preamble, blocks


def _normalize_minif2f_preamble(preamble: str) -> str:
    """
    Normalize Lean3-style miniF2F imports into a Lean4-compatible scaffold prefix.

    The vendored miniF2F Lean assets in this repository use `import minif2f_import`,
    which does not exist in the shared Lean4 verification workspace. For the current
    mathproof runtime we treat that import bundle as a request for the standard
    Mathlib environment and keep the locale directives intact.
    """
    locale_map = {
        "big_operators": "open scoped BigOperators",
    }
    ignored_locales = {
        "nat",
        "real",
        "rat",
        "topological_space",
    }

    normalized_lines: list[str] = []
    for line in preamble.splitlines():
        stripped = line.strip()
        if stripped == "import minif2f_import":
            normalized_lines.append("import Mathlib")
            continue
        if stripped.startswith("open_locale "):
            locale_name = stripped.removeprefix("open_locale ").strip()
            replacement = locale_map.get(locale_name)
            if replacement:
                normalized_lines.append(replacement)
            elif locale_name not in ignored_locales:
                normalized_lines.append(f"-- unsupported Lean3 locale omitted: {locale_name}")
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines).rstrip()


def list_minif2f_problems(
    benchmark_root: str | Path,
    *,
    split: str = "test",
) -> List[BenchmarkProblem]:
    """Enumerate miniF2F Lean problems from the requested split file."""
    root = _resolve_dir(benchmark_root)
    split = split.strip().lower()
    if split not in {"test", "valid"}:
        raise ValueError(f"Unsupported miniF2F split: {split}")

    source_path = root / "lean" / "src" / f"{split}.lean"
    if not source_path.exists():
        raise FileNotFoundError(f"miniF2F Lean split file not found: {source_path}")

    preamble, blocks = _split_theorem_blocks(_load_text(source_path))
    problems: List[BenchmarkProblem] = []
    for theorem_name, block in blocks:
        problems.append(
            BenchmarkProblem(
                benchmark_name="miniF2F",
                problem_id=theorem_name,
                theorem_name=theorem_name,
                source_path=source_path,
                split=split,
                inline_block=block,
                preamble=preamble,
            )
        )
    return problems


def list_putnambench_problems(
    benchmark_root: str | Path,
) -> List[BenchmarkProblem]:
    """Enumerate PutnamBench Lean4 problems, one file per theorem."""
    root = _resolve_dir(benchmark_root)
    source_dir = root / "lean4" / "src"
    if not source_dir.exists():
        raise FileNotFoundError(f"PutnamBench Lean4 source directory not found: {source_dir}")

    problems: List[BenchmarkProblem] = []
    for source_path in sorted(source_dir.glob("*.lean")):
        theorem_name = source_path.stem
        problems.append(
            BenchmarkProblem(
                benchmark_name="PutnamBench",
                problem_id=theorem_name,
                theorem_name=theorem_name,
                source_path=source_path,
            )
        )
    return problems


def materialize_benchmark_problem(
    problem: BenchmarkProblem,
    output_root: str | Path,
) -> Path:
    """
    Materialize one benchmark theorem into a standalone Lean scaffold file.

    - PutnamBench files are already single-theorem scaffolds, so they are returned
      directly.
    - miniF2F theorems are extracted into one-file-per-problem scaffolds under
      `output_root`.
    """
    if problem.inline_block is None:
        return problem.source_path.resolve()

    output_dir = Path(output_root).resolve() / problem.cache_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / problem.materialized_filename

    rendered = "\n\n".join(
        section
        for section in [
            _normalize_minif2f_preamble(problem.preamble).strip()
            if problem.benchmark_name == "miniF2F"
            else problem.preamble.strip(),
            problem.inline_block.strip(),
        ]
        if section
    ).rstrip() + "\n"

    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing == rendered:
            return output_path

    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def _write_text_if_changed(path: Path, content: str) -> None:
    """Avoid rewriting generated files when the rendered content is unchanged."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def extract_theorem_statement(problem_path: str | Path, theorem_name: Optional[str] = None) -> str:
    """Extract the proposition statement from a single-theorem Lean scaffold."""
    text = Path(problem_path).read_text(encoding="utf-8")
    match = THEOREM_HEADER_RE.search(text)
    if match is None:
        raise ValueError(f"Could not extract theorem statement from: {problem_path}")

    found_name, header_body = match.groups()
    if theorem_name is not None and found_name != theorem_name:
        raise ValueError(
            f"Theorem name mismatch while parsing statement: expected {theorem_name}, got {found_name}"
        )

    colon_index: Optional[int] = None
    depth = 0
    openers = {"(": ")", "[": "]", "{": "}"}
    closers = {")", "]", "}"}
    for index, char in enumerate(header_body):
        if char in openers:
            depth += 1
        elif char in closers:
            depth = max(depth - 1, 0)
        elif char == ":" and depth == 0:
            colon_index = index

    if colon_index is None:
        raise ValueError(f"Could not locate theorem statement separator in: {problem_path}")

    statement = header_body[colon_index + 1 :]
    return " ".join(statement.split())


def build_seed_plan_text(
    theorem_statement: str,
    *,
    theorem_name: Optional[str] = None,
    plan_id: str = "seed_plan",
) -> str:
    """Build a minimal initial proof plan for one benchmark theorem."""
    payload = {
        "plan_id": plan_id,
        "theorem_statement": theorem_statement,
        "theorem_name": theorem_name,
        "plan_summary": "Initial benchmark seed plan with one unresolved target subgoal.",
        "strategy_type": "direct",
        "subgoals": [
            {
                "id": "sg_goal",
                "statement": theorem_statement,
                "proof_sketch": "Start from the benchmark theorem statement and refine the final Lean proof iteratively.",
                "status": "in_progress",
            }
        ],
    }
    if yaml is not None:
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    return json.dumps(payload, indent=2, ensure_ascii=True)


def materialize_mathproof_task(
    problem: BenchmarkProblem,
    output_root: str | Path,
) -> MathproofTaskSpec:
    """
    Package one benchmark theorem as a current-style mathproof task directory.

    The output directory contains:
    - `problem.lean`
    - `init.scratch`
    - `metadata.json`
    """
    output_root = Path(output_root).resolve()
    task_dir = output_root / problem.cache_subdir / problem.problem_id
    task_dir.mkdir(parents=True, exist_ok=True)

    problem_path = materialize_benchmark_problem(problem, output_root / "_materialized")
    task_problem_path = task_dir / "problem.lean"
    if problem_path.resolve() != task_problem_path.resolve():
        _write_text_if_changed(task_problem_path, problem_path.read_text(encoding="utf-8"))
    else:
        task_problem_path = problem_path

    theorem_statement = extract_theorem_statement(task_problem_path, theorem_name=problem.theorem_name)
    init_scratch_path = task_dir / "init.scratch"
    _write_text_if_changed(
        init_scratch_path,
        build_seed_plan_text(
            theorem_statement,
            theorem_name=problem.theorem_name,
            plan_id=f"seed_{problem.problem_id}",
        ),
    )

    metadata = {
        "task_id": problem.problem_id,
        "benchmark_name": problem.benchmark_name,
        "split": problem.split,
        "theorem_name": problem.theorem_name,
        "theorem_statement": theorem_statement,
        "source_path": str(problem.source_path.resolve()),
        "problem_path": str(task_problem_path.resolve()),
        "init_scratch_path": str(init_scratch_path.resolve()),
    }
    _write_text_if_changed(
        task_dir / "metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )

    return MathproofTaskSpec(
        task_id=problem.problem_id,
        benchmark_name=problem.benchmark_name,
        theorem_name=problem.theorem_name,
        theorem_statement=theorem_statement,
        problem_path=task_problem_path.resolve(),
        init_scratch_path=init_scratch_path.resolve(),
        task_dir=task_dir.resolve(),
        source_path=problem.source_path.resolve(),
        split=problem.split,
    )


def materialize_mathproof_tasks(
    problems: Sequence[BenchmarkProblem],
    output_root: str | Path,
) -> List[MathproofTaskSpec]:
    """Package a sequence of benchmark problems as current-style mathproof task dirs."""
    return [materialize_mathproof_task(problem, output_root) for problem in problems]


def get_benchmark_problem(
    problems: Sequence[BenchmarkProblem],
    problem_id: str,
) -> BenchmarkProblem:
    """Look up one benchmark problem by problem_id or theorem_name."""
    normalized = problem_id.strip()
    for problem in problems:
        if problem.problem_id == normalized or problem.theorem_name == normalized:
            return problem
    raise KeyError(f"Benchmark problem not found: {problem_id}")


def benchmark_digest(problems: Iterable[BenchmarkProblem]) -> str:
    """Create a short stable digest for a benchmark problem set."""
    digest = hashlib.sha256()
    for problem in problems:
        digest.update(problem.benchmark_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((problem.split or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(problem.problem_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]
