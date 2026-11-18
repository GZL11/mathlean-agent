"""
Bridge verifier that evaluates plan-materialized Lean artifacts inside a Lean project.

The current mathproof pipeline already materializes `_proof`-style artifacts:
- `definition.lean`
- `step_1.lean`, `step_2.lean`, ...
- `final_theorem.lean`

This module turns those artifacts into actual project-context verification calls by:
1. staging the generated files under a Lean project workspace;
2. assembling a wrapper file with the right prefix context;
3. invoking Lean / Lake on the wrapper file;
4. converting the result back into the flat verifier metrics used by Famou.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from famou.core.proof_objects import PlanObject, SubgoalNode


VerifierFn = Callable[[Path, str, PlanObject, Optional[SubgoalNode]], Dict[str, Any]]
RunnerFn = Callable[[List[str], Path, Dict[str, str], int], subprocess.CompletedProcess[str]]
DECL_RE = re.compile(r"\b(?:theorem|lemma|example)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _normalize_text(text: Optional[str]) -> str:
    """Normalize whitespace so statement matching is lenient but deterministic."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _contains_statement(code: str, statement: Optional[str]) -> float:
    """Use exact normalized containment as a lightweight target guard."""
    if not statement:
        return 1.0
    normalized_code = _normalize_text(code)
    normalized_statement = _normalize_text(statement)
    if not normalized_statement:
        return 1.0
    return 1.0 if normalized_statement in normalized_code else 0.0


def _sanitize_name(raw_name: str) -> str:
    """Convert an arbitrary name into a filesystem-safe Lean artifact suffix."""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name.strip())
    cleaned = cleaned.strip("_") or "artifact"
    if cleaned[0].isdigit():
        cleaned = f"artifact_{cleaned}"
    return cleaned


def _split_imports(code: str) -> tuple[List[str], List[str]]:
    """Split Lean source into import lines and body lines."""
    imports: List[str] = []
    body: List[str] = []
    for line in code.splitlines():
        if line.strip().startswith("import "):
            imports.append(line.strip())
        else:
            body.append(line)
    return imports, body


def _extract_decl_names(code: str) -> set[str]:
    """Extract top-level declaration names conservatively from Lean code."""
    return set(DECL_RE.findall(code or ""))


def merge_lean_sources(sources: Iterable[str], banner: Optional[str] = None) -> str:
    """
    Merge multiple Lean snippets into one wrapper file.

    Imports are deduplicated and kept at the top. Remaining bodies are appended in order.
    """
    seen_imports = set()
    ordered_imports: List[str] = []
    bodies: List[str] = []

    for source in sources:
        if not source:
            continue
        imports, body_lines = _split_imports(source)
        for import_line in imports:
            if import_line not in seen_imports:
                seen_imports.add(import_line)
                ordered_imports.append(import_line)
        body = "\n".join(body_lines).strip()
        if body:
            bodies.append(body)

    sections: List[str] = []
    if ordered_imports:
        sections.append("\n".join(ordered_imports))
    if banner:
        sections.append(f"/- {banner} -/")
    sections.extend(bodies)
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def _read_json_file(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a dictionary, returning an empty mapping on failure."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_if_exists(path: Path) -> str:
    """Read a file only when it exists, otherwise return an empty string."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _strip_shared_fragment(code: str, fragment: str) -> str:
    """Remove one shared context fragment from a source snippet."""
    if not code or not fragment:
        return code
    normalized_fragment = fragment.strip()
    if not normalized_fragment:
        return code
    return code.replace(normalized_fragment, "", 1).strip()


def default_command_runner(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Default subprocess-backed command runner."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class ProofRuntimeBridgeVerifier:
    """Project-context Lean verifier built on top of `_proof`-style artifacts."""

    def __init__(
        self,
        project_root: str | Path,
        staging_root: Optional[str | Path] = None,
        lean_command: Optional[List[str]] = None,
        timeout: int = 30,
        command_runner: Optional[RunnerFn] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.staging_root = (
            Path(staging_root).resolve()
            if staging_root is not None
            else (self.project_root / "temp_plan_bridge").resolve()
        )
        self.lean_command = list(lean_command) if lean_command else ["lake", "lean"]
        self.timeout = timeout
        self.command_runner = command_runner or default_command_runner
        self.toolchain = self._read_toolchain()

    def __call__(
        self,
        artifact_path: Path,
        lean_code: str,
        plan: PlanObject,
        node: Optional[SubgoalNode],
    ) -> Dict[str, Any]:
        started_at = time.time()
        source_dir = artifact_path.parent.resolve()
        stage_dir = self._stage_artifacts(source_dir)
        manifest = _read_json_file(stage_dir / "artifact_manifest.json")
        wrapper_path = self._build_wrapper(
            stage_dir=stage_dir,
            artifact_path=artifact_path,
            lean_code=lean_code,
            plan=plan,
            node=node,
            manifest=manifest,
        )

        env = dict(os.environ)
        if self.toolchain and "ELAN_TOOLCHAIN" not in env:
            env["ELAN_TOOLCHAIN"] = self.toolchain

        command = [*self.lean_command, self._to_project_relative_arg(wrapper_path)]

        try:
            completed = self.command_runner(command, self.project_root, env, self.timeout)
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            diagnostics_text = "\n".join(part for part in [stdout, stderr] if part)
            diagnostics = [diagnostics_text] if diagnostics_text else []
            compile_success = 1.0 if completed.returncode == 0 else 0.0
            no_sorry = 0.0 if "sorry" in lean_code else 1.0
            statement = node.statement_formal if node is not None else plan.theorem_statement
            statement_match = _contains_statement(lean_code, statement)
            proof_validity = 1.0 if compile_success and no_sorry and statement_match else 0.0
            return {
                "compile_success": compile_success,
                "no_sorry": no_sorry,
                "statement_match": statement_match,
                "proof_validity": proof_validity,
                "diagnostics": diagnostics,
                "latency_ms": (time.time() - started_at) * 1000.0,
                "command": command,
                "cwd": str(self.project_root),
                "staged_dir": str(stage_dir),
                "wrapper_file": str(wrapper_path),
                "returncode": completed.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "compile_success": 0.0,
                "no_sorry": 0.0 if "sorry" in lean_code else 1.0,
                "statement_match": _contains_statement(
                    lean_code,
                    node.statement_formal if node is not None else plan.theorem_statement,
                ),
                "proof_validity": 0.0,
                "diagnostics": [f"Lean verification timed out after {self.timeout}s."],
                "latency_ms": (time.time() - started_at) * 1000.0,
                "command": command,
                "cwd": str(self.project_root),
                "staged_dir": str(stage_dir),
                "wrapper_file": str(wrapper_path),
                "returncode": None,
            }
        except Exception as exc:
            return {
                "compile_success": 0.0,
                "no_sorry": 0.0 if "sorry" in lean_code else 1.0,
                "statement_match": _contains_statement(
                    lean_code,
                    node.statement_formal if node is not None else plan.theorem_statement,
                ),
                "proof_validity": 0.0,
                "diagnostics": [f"Lean bridge failed before verification: {exc}"],
                "latency_ms": (time.time() - started_at) * 1000.0,
                "command": command,
                "cwd": str(self.project_root),
                "staged_dir": str(stage_dir),
                "wrapper_file": str(wrapper_path),
                "returncode": None,
            }

    def _read_toolchain(self) -> Optional[str]:
        """Read the Lean toolchain version from the project, if declared."""
        toolchain_path = self.project_root / "lean-toolchain"
        if not toolchain_path.exists():
            return None
        content = toolchain_path.read_text(encoding="utf-8").strip()
        return content or None

    def _stage_artifacts(self, source_dir: Path) -> Path:
        """Copy the generated artifacts into the Lean project's temp area."""
        self.staging_root.mkdir(parents=True, exist_ok=True)
        stage_dir = self.staging_root / source_dir.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        for source_file in source_dir.iterdir():
            if not source_file.is_file():
                continue
            if source_file.suffix not in {".lean", ".json", ".yaml", ".yml"}:
                continue
            shutil.copy2(source_file, stage_dir / source_file.name)
        return stage_dir

    def _build_wrapper(
        self,
        stage_dir: Path,
        artifact_path: Path,
        lean_code: str,
        plan: PlanObject,
        node: Optional[SubgoalNode],
        manifest: Dict[str, Any],
    ) -> Path:
        """Build a wrapper that adds the right project-local proof context."""
        source_texts = self._collect_context_sources(
            stage_dir=stage_dir,
            artifact_path=artifact_path,
            lean_code=lean_code,
            plan=plan,
            node=node,
            manifest=manifest,
        )
        wrapper_name = f"_verify_{_sanitize_name(artifact_path.stem)}.lean"
        wrapper_path = stage_dir / wrapper_name
        banner = f"Auto-generated verification wrapper for `{artifact_path.name}`"
        wrapper_path.write_text(merge_lean_sources(source_texts, banner=banner), encoding="utf-8")
        return wrapper_path

    def _collect_context_sources(
        self,
        stage_dir: Path,
        artifact_path: Path,
        lean_code: str,
        plan: PlanObject,
        node: Optional[SubgoalNode],
        manifest: Dict[str, Any],
    ) -> List[str]:
        """
        Collect the Lean source fragments needed for one verification call.

        Node verification uses `definition + step_1 + ... + step_i` so later steps can
        reuse earlier derived lemmas. Final theorem verification sees all steps plus the
        final theorem artifact. Lemmas fall back to `definition + lemma`.
        """
        sources: List[str] = []
        preamble_text = (manifest.get("scaffold_preamble") or "").strip()
        suffix_text = (manifest.get("scaffold_suffix") or "").strip()

        definition_text = _read_text_if_exists(stage_dir / "definition.lean")
        if definition_text:
            sources.append(definition_text)

        steps = manifest.get("steps", [])
        if node is not None and isinstance(steps, list):
            current_index = None
            for step in steps:
                if isinstance(step, dict) and str(step.get("node_id")) == node.node_id:
                    try:
                        current_index = int(step.get("step_index"))
                    except Exception:
                        current_index = None
                    break

            if current_index is not None:
                for step_index in range(1, current_index + 1):
                    step_path = stage_dir / f"step_{step_index}.lean"
                    step_text = _read_text_if_exists(step_path)
                    if step_text:
                        sources.append(step_text)
                if suffix_text:
                    sources.append(suffix_text)
                return sources

        if artifact_path.name == "final_theorem.lean":
            target_decl_names = _extract_decl_names(lean_code)
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    step_path = stage_dir / f"step_{int(step.get('step_index', 0) or 0)}.lean"
                    step_text = _read_text_if_exists(step_path)
                    if step_text and not (_extract_decl_names(step_text) & target_decl_names):
                        sources.append(step_text)
            final_text = _strip_shared_fragment(lean_code, preamble_text) if definition_text else lean_code
            sources.append(final_text)
            if suffix_text and suffix_text not in lean_code:
                sources.append(suffix_text)
            return sources

        leaf_text = _strip_shared_fragment(lean_code, preamble_text) if definition_text else lean_code
        sources.append(leaf_text)
        if suffix_text and suffix_text not in lean_code:
            sources.append(suffix_text)
        return sources

    def _to_project_relative_arg(self, path: Path) -> str:
        """Use a project-relative path when possible so `lake` resolves the file cleanly."""
        try:
            return str(path.resolve().relative_to(self.project_root))
        except Exception:
            return str(path.resolve())


def make_proof_runtime_bridge_verifier(
    project_root: str | Path,
    staging_root: Optional[str | Path] = None,
    lean_command: Optional[List[str]] = None,
    timeout: int = 30,
    command_runner: Optional[RunnerFn] = None,
) -> VerifierFn:
    """Factory that returns a verifier function compatible with PlanLeanEvaluator."""
    verifier = ProofRuntimeBridgeVerifier(
        project_root=project_root,
        staging_root=staging_root,
        lean_command=lean_command,
        timeout=timeout,
        command_runner=command_runner,
    )
    return verifier
