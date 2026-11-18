"""
Plan-aware Lean evaluator and reward aggregation utilities.

This evaluator is designed for the current Famou MVP:
- input is a structured proof plan stored in Program.code
- output follows the standard Famou flat evaluator interface
- Lean verifier integration is pluggable
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from famou.core.proof_objects import PlanObject, SubgoalNode, SubgoalStatus
from famou.utils.plan_runtime import (
    compute_combined_plan_score,
    dump_plan,
    get_terminal_goal_ids,
    parse_plan,
    topological_order,
)


VerifierFn = Callable[[Path, str, PlanObject, Optional[SubgoalNode]], Dict[str, Any]]

THEOREM_RE = re.compile(r"\b(?:theorem|lemma|example)\b", re.IGNORECASE)
SORRY_RE = re.compile(r"\bsorry\b")
DECL_LINE_RE = re.compile(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma|example)\b")
SORRY_LINE_RE = re.compile(r"(?m)^([ \t]*)sorry\b.*$")
SCOPE_OPEN_RE = re.compile(r"^\s*(?:namespace|section)\b")


def _json_safe(value: Any) -> Any:
    """Convert values recursively into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _clip01(value: Any) -> float:
    """Safely coerce arbitrary values into [0, 1]."""
    try:
        numeric = float(value or 0.0)
    except Exception:
        numeric = 0.0
    return min(1.0, max(0.0, numeric))


def _normalize_text(text: Optional[str]) -> str:
    """Normalize whitespace for lenient statement matching."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _sanitize_name(raw_name: str) -> str:
    """Sanitize a string into a Lean-friendly declaration suffix."""
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name.strip())
    normalized = normalized.strip("_") or "goal"
    if normalized[0].isdigit():
        normalized = f"goal_{normalized}"
    return normalized


def _extract_import_lines(code: Optional[str]) -> List[str]:
    """Extract ordered Lean import lines from a source snippet."""
    if not code:
        return []
    imports: List[str] = []
    for line in str(code).splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            imports.append(stripped)
    return imports


def _dedupe_import_lines(import_lines: List[str]) -> List[str]:
    """Deduplicate import lines while preserving order."""
    seen = set()
    deduped: List[str] = []
    for line in import_lines:
        if line and line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped


def _render_import_block(import_lines: List[str]) -> str:
    """Render a normalized Lean import block."""
    return "\n".join(_dedupe_import_lines(import_lines)).strip()


def _prepend_import_block(code: str, import_block: str) -> str:
    """Prepend imports only when the source snippet does not declare any."""
    body = (code or "").strip()
    imports = (import_block or "").strip()
    if not body:
        return imports
    if not imports or _extract_import_lines(body):
        return body
    return "\n\n".join([imports, body])


def _join_nonempty_sections(*sections: str) -> str:
    """Join source sections while skipping empty fragments."""
    return "\n\n".join(section for section in sections if section)


def _normalize_tactic_body_for_sorry_replacement(text: str) -> str:
    """Convert a tactic draft into the body that should replace a scaffold `sorry`."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    if stripped == "by":
        return ""
    if stripped.startswith("by\n"):
        stripped = stripped[3:]
    elif stripped.startswith("by\r\n"):
        stripped = stripped[4:]
    return textwrap.dedent(stripped).strip()


def _normalize_tactic_block(text: str) -> str:
    """Normalize a tactic snippet into a well-indented `by ...` block."""
    stripped = (text or "").strip()
    if not stripped:
        return ""

    if stripped == "by":
        return "by"

    if stripped.startswith("by\n") or stripped.startswith("by\r\n"):
        body = _normalize_tactic_body_for_sorry_replacement(stripped)
    else:
        body = textwrap.dedent(stripped).strip()

    if not body:
        return "by"
    return f"by\n{_indent_block(body, spaces=2)}"


def _replace_first_sorry_with_block(scaffold: str, replacement: str) -> str:
    """Replace the first `sorry` line with a properly indented tactic block."""
    match = SORRY_LINE_RE.search(scaffold or "")
    if match is None:
        return scaffold
    indent = match.group(1)
    replacement_block = "\n".join(
        f"{indent}{line}" if line.strip() else line
        for line in replacement.splitlines()
    )
    return scaffold[:match.start()] + replacement_block + scaffold[match.end():]


def _extract_scaffold_context(scaffold_code: Optional[str]) -> Dict[str, str]:
    """
    Split a theorem scaffold into imports, prelude context, and trailing closers.

    The prelude is the body before the first theorem/lemma/example declaration.
    The suffix is a conservative trailing block of `end` lines and blank/comment
    lines at the end of the file, which is enough for the common
    `namespace/section ... theorem ... end` pattern.
    """
    if not scaffold_code:
        return {
            "imports": "",
            "preamble": "",
            "suffix": "",
        }

    lines = str(scaffold_code).splitlines()
    import_lines = _extract_import_lines(scaffold_code)
    import_set = set(import_lines)

    decl_index = None
    for index, line in enumerate(lines):
        if DECL_LINE_RE.match(line):
            decl_index = index
            break

    if decl_index is None:
        body_lines = [line for line in lines if line.strip() not in import_set]
        return {
            "imports": _render_import_block(import_lines),
            "preamble": "\n".join(body_lines).strip(),
            "suffix": "",
        }

    preamble_lines = []
    for line in lines[:decl_index]:
        if line.strip() in import_set:
            continue
        preamble_lines.append(line)
    while preamble_lines:
        stripped = preamble_lines[-1].strip()
        if (
            not stripped
            or stripped.startswith("/--")
            or stripped.startswith("/-")
            or stripped.startswith("--")
            or stripped.endswith("-/")
        ):
            preamble_lines.pop()
            continue
        break

    has_scope_opener = any(SCOPE_OPEN_RE.match(line or "") for line in preamble_lines)
    suffix_lines: List[str] = []
    if has_scope_opener:
        suffix_start = len(lines)
        for index in range(len(lines) - 1, decl_index - 1, -1):
            stripped = lines[index].strip()
            if (
                not stripped
                or stripped == "end"
                or stripped.startswith("end ")
                or stripped.startswith("--")
                or stripped.startswith("/-")
                or stripped.endswith("-/")
            ):
                suffix_start = index
                continue
            break

        suffix_lines = lines[suffix_start:] if suffix_start < len(lines) else []

    return {
        "imports": _render_import_block(import_lines),
        "preamble": "\n".join(preamble_lines).strip(),
        "suffix": "\n".join(suffix_lines).strip(),
    }


def _find_decl_header_lines(scaffold_code: Optional[str]) -> List[str]:
    """Extract the theorem/lemma/example header lines up to `:=` from a scaffold."""
    if not scaffold_code:
        return []

    lines = str(scaffold_code).splitlines()
    decl_index = None
    for index, line in enumerate(lines):
        if DECL_LINE_RE.match(line):
            decl_index = index
            break
    if decl_index is None:
        return []

    header_lines: List[str] = []
    for line in lines[decl_index:]:
        header_lines.append(line)
        if ":=" in line:
            break
    return header_lines


def _find_top_level_colon(text: str) -> int:
    """Find the first colon that is not nested inside binders or delimiters."""
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    stack: List[str] = []
    for index, ch in enumerate(text):
        if ch in pairs:
            stack.append(pairs[ch])
            depth += 1
        elif ch in closers:
            if stack and ch == stack[-1]:
                stack.pop()
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            return index
    return -1


def _find_top_level_assignment(text: str) -> int:
    """Find the first `:=` that is not nested inside binders or delimiters."""
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    stack: List[str] = []
    index = 0
    while index < len(text) - 1:
        ch = text[index]
        nxt = text[index + 1]
        if ch in pairs:
            stack.append(pairs[ch])
            depth += 1
        elif ch in closers:
            if stack and ch == stack[-1]:
                stack.pop()
            depth = max(0, depth - 1)
        elif ch == ":" and nxt == "=" and depth == 0:
            return index
        index += 1
    return -1


def _extract_scaffold_binders(scaffold_code: Optional[str]) -> str:
    """Extract theorem binders/hypotheses from the scaffold declaration header."""
    header_lines = _find_decl_header_lines(scaffold_code)
    if not header_lines:
        return ""

    header_text = "\n".join(header_lines).strip()
    if ":=" not in header_text:
        return ""
    header_text = header_text.rsplit(":=", 1)[0].rstrip()

    keyword_match = re.match(
        r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma|example)\b",
        header_text,
    )
    if keyword_match is None:
        return ""

    remainder = header_text[keyword_match.end():].lstrip()
    name_match = re.match(r"[A-Za-z_][A-Za-z0-9_']*", remainder)
    if name_match is not None:
        remainder = remainder[name_match.end():]

    colon_index = _find_top_level_colon(remainder)
    if colon_index < 0:
        return ""
    return remainder[:colon_index].strip()


def _extract_decl_tactic_body(code: Optional[str]) -> str:
    """Extract the proof body from a theorem/lemma/example declaration."""
    if not code:
        return ""
    text = str(code).strip()
    assign_index = _find_top_level_assignment(text)
    if assign_index < 0:
        return ""
    body = text[assign_index + 2 :].strip()
    return body


def _extract_scaffold_decl_prefix(scaffold_code: Optional[str]) -> str:
    """Extract the scaffold declaration header up to and including the top-level `:=`."""
    header_lines = _find_decl_header_lines(scaffold_code)
    if not header_lines:
        return ""

    header_text = "\n".join(header_lines).strip()
    assign_index = _find_top_level_assignment(header_text)
    if assign_index < 0:
        return ""
    return header_text[: assign_index + 2].rstrip()


def _rebuild_decl_from_scaffold(scaffold_code: Optional[str], proof_text: str) -> str:
    """Rebuild the scaffold theorem using its original header and a new proof body."""
    decl_prefix = _extract_scaffold_decl_prefix(scaffold_code)
    proof_block = _normalize_tactic_block(proof_text)
    if not decl_prefix or not proof_block:
        return ""
    return f"{decl_prefix} {proof_block}"


def _contains_statement(code: str, statement: Optional[str]) -> float:
    """Use simple normalized containment as a first exact-target safeguard."""
    if not statement:
        return 1.0
    normalized_code = _normalize_text(code)
    normalized_statement = _normalize_text(statement)
    if not normalized_statement:
        return 1.0
    return 1.0 if normalized_statement in normalized_code else 0.0


def _materialize_decl_body(
    draft: Optional[str],
    decl_name: str,
    statement: Optional[str],
    allow_placeholder_sorry: bool = False,
) -> Optional[str]:
    """
    Materialize a local Lean snippet into one declaration body.

    This helper accepts either:
    - a full theorem/lemma/example declaration
    - a bare tactic block such as `by ...`

    It returns declaration-only code with imports stripped so the caller can
    embed it into a larger Lean file safely.
    """
    text = (draft or "").strip()
    if not text:
        return None

    if not allow_placeholder_sorry and SORRY_RE.search(text):
        return None

    if THEOREM_RE.search(text):
        body_lines = [
            line for line in text.splitlines()
            if not line.strip().startswith("import ")
        ]
        embedded_decl = "\n".join(body_lines).strip()
        return embedded_decl or None

    normalized_statement = (statement or "").strip()
    if not normalized_statement or "\n" in normalized_statement:
        return None

    body = _normalize_tactic_block(text)
    if not body:
        return None
    return f"theorem {decl_name} : {normalized_statement} := {body}"


def build_node_lean_artifact(
    node: SubgoalNode,
    plan: PlanObject,
    default_imports: str = "import Mathlib",
    preamble_context: str = "",
    suffix_context: str = "",
    scaffold_code: Optional[str] = None,
    allow_placeholder_sorry: bool = False,
) -> Optional[str]:
    """
    Build a Lean file for one subgoal.

    If the node already contains a declaration, keep it unchanged.
    If it only contains a tactic body and the statement is formal enough, wrap it.
    Otherwise return None and let structural reward drive that node for now.
    """
    draft = (node.local_lean_draft or "").strip()
    if not draft:
        return None

    if THEOREM_RE.search(draft):
        return _join_nonempty_sections(
            default_imports.strip(),
            preamble_context.strip(),
            draft,
            suffix_context.strip(),
        )

    statement = (node.statement_formal or "").strip()
    if not statement or "\n" in statement:
        return None

    if not allow_placeholder_sorry and SORRY_RE.search(draft):
        return None

    theorem_name = _sanitize_name(f"{plan.plan_id}_{node.node_id}")
    body = _normalize_tactic_block(draft)
    binders = _extract_scaffold_binders(scaffold_code)
    theorem_head = f"theorem {theorem_name}"
    if binders:
        theorem_head = f"{theorem_head}\n{binders}"
    return _join_nonempty_sections(
        default_imports.strip(),
        preamble_context.strip(),
        f"{theorem_head} : {statement} := {body}",
        suffix_context.strip(),
    )


def _resolve_path(path_value: Optional[str | Path], project_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve an optional path against project_root and current working directory."""
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if project_root is not None:
        candidate = project_root / path
        if candidate.exists():
            return candidate
    return path.resolve()


def _indent_block(text: str, spaces: int = 2) -> str:
    """Indent a multiline block while preserving blank lines."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def build_full_theorem_artifact(
    plan: PlanObject,
    terminal_nodes: List[SubgoalNode],
    verified_lemma_decls: List[str],
    default_imports: str = "import Mathlib",
    preamble_context: str = "",
    suffix_context: str = "",
    theorem_scaffold_path: Optional[str | Path] = None,
    project_root: Optional[Path] = None,
) -> Optional[str]:
    """
    Build a final theorem artifact for whole-proof verification.

    Priority order:
    1. explicit plan-level full Lean code
    2. terminal node that already contains a theorem/lemma declaration
    3. theorem scaffold with `sorry` replaced by a terminal tactic body
    """
    explicit_keys = [
        "full_lean_code",
        "final_lean_code",
        "theorem_code",
        "full_solution",
    ]
    for key in explicit_keys:
        code = (plan.lean_artifacts.get(key) or "").strip()
        if code:
            return _prepend_import_block(code, default_imports.strip())

    verified_lemmas_text = "\n\n".join(code for code in verified_lemma_decls if code)
    terminal_decl = None
    terminal_body = None
    for node in terminal_nodes:
        draft = (node.local_lean_draft or "").strip()
        if not draft:
            continue
        if THEOREM_RE.search(draft):
            terminal_decl = draft
            break
        if terminal_body is None:
            terminal_body = draft

    scaffold_candidates = [
        theorem_scaffold_path,
        plan.lean_artifacts.get("theorem_scaffold_path"),
        plan.lean_artifacts.get("problem_path"),
        plan.metadata.get("theorem_scaffold_path"),
        plan.metadata.get("problem_path"),
    ]
    scaffold_path = None
    for candidate in scaffold_candidates:
        resolved = _resolve_path(candidate, project_root=project_root)
        if resolved and resolved.exists():
            scaffold_path = resolved
            break

    if scaffold_path is not None:
        scaffold = scaffold_path.read_text(encoding="utf-8")
        if terminal_decl is not None:
            terminal_decl_body = _extract_decl_tactic_body(terminal_decl)
            if terminal_decl_body:
                body = _normalize_tactic_body_for_sorry_replacement(terminal_decl_body)
                replaced = _replace_first_sorry_with_block(scaffold, body)
                if replaced != scaffold:
                    return _join_nonempty_sections(
                        verified_lemmas_text,
                        replaced,
                    )
                rebuilt_decl = _rebuild_decl_from_scaffold(scaffold, terminal_decl_body)
                if rebuilt_decl:
                    return _join_nonempty_sections(
                        default_imports.strip(),
                        preamble_context.strip(),
                        verified_lemmas_text,
                        rebuilt_decl,
                        suffix_context.strip(),
                    )
            return _join_nonempty_sections(
                default_imports.strip(),
                preamble_context.strip(),
                verified_lemmas_text,
                terminal_decl,
                suffix_context.strip(),
            )
        if terminal_body:
            body = _normalize_tactic_body_for_sorry_replacement(terminal_body)
            replaced = _replace_first_sorry_with_block(scaffold, body)
            if replaced != scaffold:
                return _join_nonempty_sections(
                    verified_lemmas_text,
                    replaced,
                )
            rebuilt_decl = _rebuild_decl_from_scaffold(scaffold, terminal_body)
            if rebuilt_decl:
                return _join_nonempty_sections(
                    default_imports.strip(),
                    preamble_context.strip(),
                    verified_lemmas_text,
                    rebuilt_decl,
                    suffix_context.strip(),
                )

    if terminal_decl is not None:
        return _join_nonempty_sections(
            default_imports.strip(),
            preamble_context.strip(),
            verified_lemmas_text,
            terminal_decl,
            suffix_context.strip(),
        )

    return None


def build_definition_artifact(
    plan: PlanObject,
    default_imports: str = "import Mathlib",
    preamble_context: str = "",
) -> str:
    """Build `_proof`-style `definition.lean` content from reusable plan artifacts."""
    sections: List[str] = [default_imports.strip()]

    preamble = preamble_context.strip()
    if preamble:
        sections.append(preamble)

    helper_defs = (plan.lean_artifacts.get("definitions") or "").strip()
    if helper_defs:
        sections.append(helper_defs)

    lemma_codes = []
    for lemma in plan.lemma_pool:
        if lemma.proof_status not in {SubgoalStatus.PROVED, SubgoalStatus.VERIFIED}:
            continue
        code = _materialize_decl_body(
            lemma.lean_code,
            decl_name=_sanitize_name(lemma.lemma_id),
            statement=lemma.statement_formal,
        )
        if code:
            lemma_codes.append(code)

    if lemma_codes:
        sections.append("\n\n".join(lemma_codes))
    else:
        sections.append("-- No additional helper definitions or lemma declarations were materialized.")

    return "\n\n".join(section for section in sections if section)


def build_step_artifacts(
    plan: PlanObject,
    default_imports: str = "import Mathlib",
    preamble_context: str = "",
    suffix_context: str = "",
    scaffold_code: Optional[str] = None,
    allow_placeholder_sorry: bool = False,
) -> List[Dict[str, Any]]:
    """Build ordered `_proof`-style step artifacts from the subgoal DAG."""
    node_map = plan.node_map()
    ordered_nodes = [node_map[node_id] for node_id in topological_order(plan)]
    artifacts: List[Dict[str, Any]] = []

    for index, node in enumerate(ordered_nodes, start=1):
        artifact_text = build_node_lean_artifact(
            node,
            plan,
            default_imports=default_imports,
            preamble_context=preamble_context,
            suffix_context=suffix_context,
            scaffold_code=scaffold_code,
            allow_placeholder_sorry=allow_placeholder_sorry,
        )
        if artifact_text is None:
            statement = (node.statement_formal or "").strip()
            theorem_name = _sanitize_name(f"{plan.plan_id}_{node.node_id}")
            proof_body = "by\n  sorry"
            artifact_text = "\n\n".join(
                section
                for section in [
                    default_imports.strip(),
                    f"/-- Auto-materialized placeholder for plan node `{node.node_id}`. -/",
                    f"theorem {theorem_name} : {statement} := {proof_body}",
                ]
                if section
            )

        dependencies = ", ".join(node.dependencies) if node.dependencies else "none"
        header = (
            f"/- Step {index} for node `{node.node_id}`.\n"
            f"Dependencies: {dependencies}.\n"
            f"Node type: {node.node_type}.\n"
            f"-/\n"
        )
        artifacts.append({
            "node_id": node.node_id,
            "step_index": index,
            "filename": f"step_{index}.lean",
            "content": f"{header}{artifact_text}",
        })

    return artifacts


def run_heuristic_lean_verifier(
    artifact_path: Path,
    lean_code: str,
    plan: PlanObject,
    node: Optional[SubgoalNode],
) -> Dict[str, Any]:
    """
    Cheap offline fallback verifier.

    This is intentionally conservative: it only provides weak compile-like
    signals when no real Lean verifier is available.
    """
    has_decl = bool(THEOREM_RE.search(lean_code))
    no_sorry = 0.0 if SORRY_RE.search(lean_code) else 1.0
    statement_match = _contains_statement(
        lean_code,
        node.statement_formal if node is not None else plan.theorem_statement,
    )
    compile_success = 1.0 if has_decl else 0.0
    diagnostics = [] if compile_success else ["Heuristic verifier could not detect a Lean declaration."]

    return {
        "compile_success": compile_success,
        "no_sorry": no_sorry,
        "statement_match": statement_match,
        "diagnostics": diagnostics,
        "proof_validity": 1.0 if compile_success and no_sorry and statement_match else 0.0,
    }


def make_subprocess_lean_verifier(
    project_root: Optional[str | Path] = None,
    lean_command: Optional[List[str]] = None,
    timeout: int = 30,
) -> VerifierFn:
    """
    Build a verifier that shells out to Lean / lake.

    If `lean_command` is not provided and `project_root` is set, the default is:
    `lake env lean <artifact_path>`.
    """

    project_dir = Path(project_root).resolve() if project_root else None

    def _verifier(
        artifact_path: Path,
        lean_code: str,
        plan: PlanObject,
        node: Optional[SubgoalNode],
    ) -> Dict[str, Any]:
        if lean_command is not None:
            cmd = [*lean_command, str(artifact_path)]
            cwd = project_dir or artifact_path.parent
        elif project_dir is not None:
            cmd = ["lake", "env", "lean", str(artifact_path)]
            cwd = project_dir
        else:
            cmd = ["lean", str(artifact_path)]
            cwd = artifact_path.parent

        started_at = time.time()
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        diagnostics_text = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        diagnostics = [diagnostics_text] if diagnostics_text else []
        compile_success = 1.0 if completed.returncode == 0 else 0.0
        no_sorry = 0.0 if SORRY_RE.search(lean_code) else 1.0
        statement_match = _contains_statement(
            lean_code,
            node.statement_formal if node is not None else plan.theorem_statement,
        )
        proof_validity = 1.0 if compile_success and no_sorry and statement_match else 0.0

        return {
            "compile_success": compile_success,
            "no_sorry": no_sorry,
            "statement_match": statement_match,
            "diagnostics": diagnostics,
            "proof_validity": proof_validity,
            "latency_ms": (time.time() - started_at) * 1000.0,
            "command": cmd,
            "returncode": completed.returncode,
        }

    return _verifier


class PlanLeanEvaluator:
    """Callable evaluator implementing the Famou flat evaluator interface."""

    def __init__(
        self,
        verifier_fn: Optional[VerifierFn] = None,
        project_root: Optional[str | Path] = None,
        artifact_root: Optional[str | Path] = None,
        lean_command: Optional[List[str]] = None,
        verify_all_nodes: bool = True,
        verify_lemmas: bool = True,
        reward_mode: str = "outer",
        default_imports: str = "import Mathlib",
        allow_placeholder_sorry: bool = False,
        cleanup_artifacts: bool = False,
        timeout: int = 30,
        theorem_statement: Optional[str] = None,
        verify_full_theorem: bool = True,
        theorem_scaffold_path: Optional[str | Path] = None,
        materialize_proof_runtime: bool = True,
    ):
        self.verifier_fn = verifier_fn
        self.project_root = Path(project_root).resolve() if project_root else None
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else None
        self.lean_command = lean_command
        self.verify_all_nodes = verify_all_nodes
        self.verify_lemmas = verify_lemmas
        self.reward_mode = reward_mode
        self.default_imports = default_imports
        self.allow_placeholder_sorry = allow_placeholder_sorry
        self.cleanup_artifacts = cleanup_artifacts
        self.timeout = timeout
        self.theorem_statement = theorem_statement
        self.verify_full_theorem = verify_full_theorem
        self.theorem_scaffold_path = theorem_scaffold_path
        self.materialize_proof_runtime = materialize_proof_runtime

    def _resolve_scaffold_path(self, plan: PlanObject) -> Optional[Path]:
        """Resolve the best available Lean scaffold/problem file for this plan."""
        scaffold_candidates = [
            self.theorem_scaffold_path,
            plan.lean_artifacts.get("theorem_scaffold_path"),
            plan.lean_artifacts.get("problem_path"),
            plan.metadata.get("theorem_scaffold_path"),
            plan.metadata.get("problem_path"),
        ]
        for candidate in scaffold_candidates:
            resolved = _resolve_path(candidate, project_root=self.project_root)
            if resolved and resolved.exists():
                return resolved
        return None

    def _resolve_effective_imports(self, plan: PlanObject) -> str:
        """
        Resolve a minimally sufficient import block for generated verification files.

        Preference order:
        1. explicit plan-level import artifacts, if present
        2. imports from scaffold/problem Lean files
        3. imports already present in theorem/step/lemma drafts
        4. evaluator default imports
        """
        explicit_imports = plan.lean_artifacts.get("imports") or plan.metadata.get("imports")
        if isinstance(explicit_imports, str) and explicit_imports.strip():
            explicit_lines = _extract_import_lines(explicit_imports)
            if explicit_lines:
                return _render_import_block(explicit_lines)
        if isinstance(explicit_imports, list):
            explicit_lines = []
            for item in explicit_imports:
                if isinstance(item, str):
                    text = item.strip()
                    if text.startswith("import "):
                        explicit_lines.append(text)
                    elif text:
                        explicit_lines.append(f"import {text}")
            if explicit_lines:
                return _render_import_block(explicit_lines)

        scaffold_path = self._resolve_scaffold_path(plan)
        if scaffold_path is not None:
            scaffold_imports = _extract_import_lines(scaffold_path.read_text(encoding="utf-8"))
            return _render_import_block(scaffold_imports)

        draft_imports: List[str] = []
        plan_level_sources = [
            plan.lean_artifacts.get("full_lean_code"),
            plan.lean_artifacts.get("final_lean_code"),
            plan.lean_artifacts.get("theorem_code"),
            plan.lean_artifacts.get("full_solution"),
            plan.lean_artifacts.get("definitions"),
        ]
        for source in plan_level_sources:
            draft_imports.extend(_extract_import_lines(source))
        for node in plan.subgoals:
            draft_imports.extend(_extract_import_lines(node.local_lean_draft))
        for lemma in plan.lemma_pool:
            draft_imports.extend(_extract_import_lines(lemma.lean_code))
        if draft_imports:
            return _render_import_block(draft_imports)

        return self.default_imports.strip()

    def _resolve_scaffold_context(self, plan: PlanObject) -> Dict[str, str]:
        """Resolve reusable scaffold context around the target theorem declaration."""
        scaffold_path = self._resolve_scaffold_path(plan)
        if scaffold_path is None:
            return {
                "path": "",
                "code": "",
                "imports": "",
                "preamble": "",
                "suffix": "",
            }

        scaffold_code = scaffold_path.read_text(encoding="utf-8")
        context = _extract_scaffold_context(scaffold_code)
        return {
            "path": str(scaffold_path),
            "code": scaffold_code,
            **context,
        }

    def __call__(self, program_path: str) -> Dict[str, Any]:
        plan_text = Path(program_path).read_text(encoding="utf-8")
        plan = parse_plan(plan_text)
        expected_theorem = self.theorem_statement or plan.theorem_statement
        verifier = self._resolve_verifier()
        scaffold_context = self._resolve_scaffold_context(plan)
        effective_imports = scaffold_context["imports"] or self._resolve_effective_imports(plan)
        resolved_scaffold_path = _resolve_path(scaffold_context.get("path"), project_root=self.project_root)

        artifact_dir = self._prepare_artifact_dir(plan.plan_id)
        subgoal_results: List[Dict[str, Any]] = []
        lemma_results: List[Dict[str, Any]] = []
        full_theorem_result: Optional[Dict[str, Any]] = None
        total_latency_ms = 0.0

        try:
            (artifact_dir / "plan.yaml").write_text(dump_plan(plan), encoding="utf-8")

            runtime_artifacts = None
            if self.materialize_proof_runtime:
                runtime_artifacts = self._materialize_proof_runtime_artifacts(
                    plan,
                    artifact_dir,
                    effective_imports=effective_imports,
                    scaffold_path=resolved_scaffold_path,
                    preamble_context=scaffold_context["preamble"],
                    suffix_context=scaffold_context["suffix"],
                )

            selected_nodes = plan.subgoals if self.verify_all_nodes else [
                node for node in plan.subgoals if node.node_id in plan.frontier_nodes
            ]

            for node in selected_nodes:
                node_result = self._evaluate_node(
                    plan,
                    node,
                    artifact_dir,
                    verifier,
                    effective_imports=effective_imports,
                    preamble_context=scaffold_context["preamble"],
                    suffix_context=scaffold_context["suffix"],
                    scaffold_code=scaffold_context.get("code", ""),
                )
                total_latency_ms += float(node_result.get("latency_ms", 0.0) or 0.0)
                subgoal_results.append(node_result)
                self._apply_node_result(node, node_result)

            if self.verify_lemmas:
                for lemma in plan.lemma_pool:
                    lemma_result = self._evaluate_lemma(
                        plan,
                        lemma,
                        artifact_dir,
                        verifier,
                        effective_imports=effective_imports,
                        preamble_context=scaffold_context["preamble"],
                        suffix_context=scaffold_context["suffix"],
                    )
                    total_latency_ms += float(lemma_result.get("latency_ms", 0.0) or 0.0)
                    lemma_results.append(lemma_result)
                    self._apply_lemma_result(lemma, lemma_result)

            if self.verify_full_theorem:
                full_theorem_result = self._evaluate_full_theorem(
                    plan,
                    artifact_dir,
                    verifier,
                    effective_imports=effective_imports,
                    scaffold_path=resolved_scaffold_path,
                    preamble_context=scaffold_context["preamble"],
                    suffix_context=scaffold_context["suffix"],
                )
                total_latency_ms += float(full_theorem_result.get("latency_ms", 0.0) or 0.0)

            attempted_subgoals = sum(1 for result in subgoal_results if result.get("artifact_path"))
            total_subgoals = max(len(plan.subgoals), 1)
            compile_success = sum(_clip01(result.get("compile_success")) for result in subgoal_results) / total_subgoals
            no_sorry = sum(_clip01(result.get("no_sorry")) for result in subgoal_results) / total_subgoals
            theorem_solved = self._compute_theorem_solved(
                plan,
                subgoal_results,
                expected_theorem,
                full_theorem_result,
                require_full_theorem=bool(self.verify_full_theorem and resolved_scaffold_path is not None),
            )
            proof_validity = theorem_solved
            statement_match = 1.0 if _normalize_text(plan.theorem_statement) == _normalize_text(expected_theorem) else 0.0
            final_compile_success = _clip01(full_theorem_result.get("compile_success")) if full_theorem_result else 0.0
            final_no_sorry = _clip01(full_theorem_result.get("no_sorry")) if full_theorem_result else 0.0
            final_statement_match = _clip01(full_theorem_result.get("statement_match")) if full_theorem_result else 0.0

            extra_metrics = {
                "compile_success": compile_success,
                "no_sorry": no_sorry,
                "statement_match": statement_match,
                "proof_validity": proof_validity,
                "theorem_solved": theorem_solved,
                "final_compile_success": final_compile_success,
                "final_no_sorry": final_no_sorry,
                "final_statement_match": final_statement_match,
                "verifier_calls": attempted_subgoals + len(lemma_results) + (1 if full_theorem_result else 0),
                "verifier_budget": max(total_subgoals + max(len(plan.lemma_pool), 1) + 1, 1),
                "latency_ms": total_latency_ms,
                "repair_steps": attempted_subgoals,
                "repair_budget": max(total_subgoals, 1),
            }
            reward = compute_combined_plan_score(
                plan,
                extra_metrics=extra_metrics,
                reward_mode=self.reward_mode,
            )
            plan.combined_score = reward["search_score"]
            plan.metadata["success_score"] = reward["success_score"]
            plan.metadata["search_score"] = reward["search_score"]

            return {
                "validity": 1.0 if attempted_subgoals > 0 or plan.subgoals else 0.0,
                "combined_score": reward["combined_score"],
                "success_score": reward["success_score"],
                "search_score": reward["search_score"],
                "proof_validity": proof_validity,
                "compile_success": compile_success,
                "no_sorry": no_sorry,
                "statement_match": statement_match,
                "final_compile_success": final_compile_success,
                "final_no_sorry": final_no_sorry,
                "final_statement_match": final_statement_match,
                "theorem_solved": theorem_solved,
                "reward_mode": reward["reward_mode"],
                "progress_score": reward["progress_score"],
                "realization_score": reward["realization_score"],
                "terminal_success": reward["terminal_success"],
                "memory_gain_bonus": reward["memory_gain_bonus"],
                "plan_complexity_penalty": reward["plan_complexity_penalty"],
                "quality_score": reward["quality_score"],
                "clean_score": reward["clean_score"],
                "cost_penalty": reward["cost_penalty"],
                "verified_lemmas": reward["verified_lemmas"],
                "frontier_size": reward["frontier_size"],
                "blocked_size": reward["blocked_size"],
                "frontier_nodes": reward["frontier_nodes"],
                "blocked_nodes": reward["blocked_nodes"],
                "subgoal_results": _json_safe(subgoal_results),
                "lemma_results": _json_safe(lemma_results),
                "full_theorem_result": _json_safe(full_theorem_result),
                "proof_runtime_artifacts": _json_safe(runtime_artifacts),
                "plan_object": plan.model_dump(mode="json"),
                "generated_folder": str(artifact_dir),
            }
        except Exception:
            if self.cleanup_artifacts and artifact_dir.exists():
                shutil.rmtree(artifact_dir, ignore_errors=True)
            raise

    def _resolve_verifier(self) -> VerifierFn:
        """Resolve which verifier backend should be used."""
        if self.verifier_fn is not None:
            return self.verifier_fn
        if self.project_root is not None or self.lean_command is not None:
            return make_subprocess_lean_verifier(
                project_root=self.project_root,
                lean_command=self.lean_command,
                timeout=self.timeout,
            )
        return run_heuristic_lean_verifier

    def _prepare_artifact_dir(self, plan_id: str) -> Path:
        """Create a dedicated artifact directory for one evaluation call."""
        base_dir = self.artifact_root
        if base_dir is not None:
            base_dir.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix=f"{_sanitize_name(plan_id)}_", dir=base_dir))
        return Path(tempfile.mkdtemp(prefix=f"{_sanitize_name(plan_id)}_"))

    def _evaluate_node(
        self,
        plan: PlanObject,
        node: SubgoalNode,
        artifact_dir: Path,
        verifier: VerifierFn,
        effective_imports: str,
        preamble_context: str,
        suffix_context: str,
        scaffold_code: str,
    ) -> Dict[str, Any]:
        """Evaluate one subgoal node if it has Lean code."""
        artifact_text = build_node_lean_artifact(
            node,
            plan,
            default_imports=effective_imports,
            preamble_context=preamble_context,
            suffix_context=suffix_context,
            scaffold_code=scaffold_code,
            allow_placeholder_sorry=self.allow_placeholder_sorry,
        )
        result: Dict[str, Any] = {
            "node_id": node.node_id,
            "status_before": node.local_status.value,
            "artifact_path": None,
            "compile_success": 0.0,
            "no_sorry": 0.0,
            "statement_match": 0.0,
            "proof_validity": 0.0,
            "diagnostics": ["No local Lean draft available for verification."],
        }
        if artifact_text is None:
            return result

        artifact_path = artifact_dir / f"{_sanitize_name(node.node_id)}.lean"
        artifact_path.write_text(artifact_text, encoding="utf-8")
        verifier_result = verifier(artifact_path, artifact_text, plan, node)

        result.update({
            "artifact_path": str(artifact_path),
            **_json_safe(verifier_result),
        })
        return result

    def _evaluate_lemma(
        self,
        plan: PlanObject,
        lemma,
        artifact_dir: Path,
        verifier: VerifierFn,
        effective_imports: str,
        preamble_context: str,
        suffix_context: str,
    ) -> Dict[str, Any]:
        """Evaluate a reusable lemma artifact if Lean code is available."""
        result: Dict[str, Any] = {
            "lemma_id": lemma.lemma_id,
            "artifact_path": None,
            "compile_success": 0.0,
            "no_sorry": 0.0,
            "statement_match": 0.0,
            "proof_validity": 0.0,
            "diagnostics": ["No lemma Lean code available for verification."],
        }
        lemma_decl = _materialize_decl_body(
            lemma.lean_code,
            decl_name=_sanitize_name(lemma.lemma_id),
            statement=lemma.statement_formal,
            allow_placeholder_sorry=self.allow_placeholder_sorry,
        )
        if not lemma_decl:
            return result
        lean_code = _join_nonempty_sections(
            effective_imports,
            preamble_context.strip(),
            lemma_decl,
            suffix_context.strip(),
        )

        artifact_path = artifact_dir / f"{_sanitize_name(lemma.lemma_id)}.lean"
        artifact_path.write_text(lean_code, encoding="utf-8")
        verifier_result = verifier(artifact_path, lean_code, plan, None)
        verifier_result["statement_match"] = _contains_statement(lean_code, lemma.statement_formal)
        result.update({
            "artifact_path": str(artifact_path),
            **_json_safe(verifier_result),
        })
        return result

    def _evaluate_full_theorem(
        self,
        plan: PlanObject,
        artifact_dir: Path,
        verifier: VerifierFn,
        effective_imports: str,
        scaffold_path: Optional[Path],
        preamble_context: str,
        suffix_context: str,
    ) -> Dict[str, Any]:
        """Evaluate a full-theorem artifact when enough information is available."""
        terminal_ids = set(get_terminal_goal_ids(plan))
        terminal_nodes = [node for node in plan.subgoals if node.node_id in terminal_ids]
        verified_lemma_decls: List[str] = []
        if not (self.project_root is not None and self.materialize_proof_runtime):
            verified_lemma_decls = [
                _materialize_decl_body(
                    lemma.lean_code,
                    decl_name=_sanitize_name(lemma.lemma_id),
                    statement=lemma.statement_formal,
                )
                for lemma in plan.lemma_pool
                if lemma.proof_status in {SubgoalStatus.PROVED, SubgoalStatus.VERIFIED}
            ]
            verified_lemma_decls = [code for code in verified_lemma_decls if code]

        artifact_text = build_full_theorem_artifact(
            plan,
            terminal_nodes=terminal_nodes,
            verified_lemma_decls=verified_lemma_decls,
            default_imports=effective_imports,
            preamble_context=preamble_context,
            suffix_context=suffix_context,
            theorem_scaffold_path=scaffold_path,
            project_root=self.project_root,
        )
        result: Dict[str, Any] = {
            "artifact_path": None,
            "compile_success": 0.0,
            "no_sorry": 0.0,
            "statement_match": 0.0,
            "proof_validity": 0.0,
            "diagnostics": ["No full theorem artifact could be assembled."],
        }
        if artifact_text is None:
            return result

        artifact_path = artifact_dir / "final_theorem.lean"
        artifact_path.write_text(artifact_text, encoding="utf-8")
        verifier_result = verifier(artifact_path, artifact_text, plan, None)
        verifier_result["statement_match"] = _contains_statement(artifact_text, plan.theorem_statement)
        result.update({
            "artifact_path": str(artifact_path),
            **_json_safe(verifier_result),
        })
        return result

    def _materialize_proof_runtime_artifacts(
        self,
        plan: PlanObject,
        artifact_dir: Path,
        effective_imports: str,
        scaffold_path: Optional[Path],
        preamble_context: str,
        suffix_context: str,
    ) -> Dict[str, Any]:
        """Materialize `_proof`-style files for downstream construct/verify runtimes."""
        definition_content = build_definition_artifact(
            plan,
            default_imports=effective_imports,
            preamble_context=preamble_context,
        )
        definition_path = artifact_dir / "definition.lean"
        definition_path.write_text(definition_content, encoding="utf-8")

        step_artifacts = build_step_artifacts(
            plan,
            default_imports=effective_imports,
            preamble_context=preamble_context,
            suffix_context=suffix_context,
            scaffold_code=scaffold_path.read_text(encoding="utf-8") if scaffold_path and scaffold_path.exists() else "",
            allow_placeholder_sorry=True,
        )
        written_steps = []
        for step in step_artifacts:
            step_path = artifact_dir / step["filename"]
            step_path.write_text(step["content"], encoding="utf-8")
            written_steps.append({
                "node_id": step["node_id"],
                "step_index": step["step_index"],
                "path": str(step_path),
            })

        manifest = {
            "plan_id": plan.plan_id,
            "theorem_statement": plan.theorem_statement,
            "definition_path": str(definition_path),
            "steps": written_steps,
            "theorem_scaffold_path": str(_resolve_path(
                scaffold_path,
                project_root=self.project_root,
            ) or ""),
            "effective_imports": effective_imports,
            "scaffold_preamble": preamble_context,
            "scaffold_suffix": suffix_context,
        }
        manifest_path = artifact_dir / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(_json_safe(manifest), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _apply_node_result(self, node: SubgoalNode, result: Dict[str, Any]) -> None:
        """Fold verifier feedback back into the plan object."""
        compile_success = _clip01(result.get("compile_success"))
        no_sorry = _clip01(result.get("no_sorry"))
        statement_match = _clip01(result.get("statement_match", 1.0))

        if result.get("artifact_path") is None:
            return

        if compile_success >= 1.0 and no_sorry >= 1.0 and statement_match >= 1.0:
            node.local_status = SubgoalStatus.VERIFIED
            node.node_score = max(node.node_score, 1.0)
        elif compile_success >= 1.0:
            node.local_status = SubgoalStatus.PROVED
            node.node_score = max(node.node_score, 0.85)
        else:
            node.local_status = SubgoalStatus.ATTEMPTED
            node.node_score = max(node.node_score, 0.25)

        node.verifier_state = {
            **dict(node.verifier_state),
            **_json_safe(result),
        }

    def _apply_lemma_result(self, lemma, result: Dict[str, Any]) -> None:
        """Fold verifier feedback back into lemma status."""
        compile_success = _clip01(result.get("compile_success"))
        no_sorry = _clip01(result.get("no_sorry"))
        if result.get("artifact_path") is None:
            return
        if compile_success >= 1.0 and no_sorry >= 1.0:
            lemma.proof_status = SubgoalStatus.VERIFIED
            lemma.reusable_score = max(lemma.reusable_score, 1.0)
        elif compile_success >= 1.0:
            lemma.proof_status = SubgoalStatus.PROVED
            lemma.reusable_score = max(lemma.reusable_score, 0.7)
        else:
            lemma.proof_status = SubgoalStatus.ATTEMPTED

    def _compute_theorem_solved(
        self,
        plan: PlanObject,
        subgoal_results: List[Dict[str, Any]],
        expected_theorem: str,
        full_theorem_result: Optional[Dict[str, Any]] = None,
        require_full_theorem: bool = False,
    ) -> float:
        """Determine whether the final theorem has actually been solved."""
        if full_theorem_result and full_theorem_result.get("artifact_path"):
            statement_match = 1.0 if _normalize_text(plan.theorem_statement) == _normalize_text(expected_theorem) else 0.0
            final_success = (
                _clip01(full_theorem_result.get("compile_success")) >= 1.0
                and _clip01(full_theorem_result.get("no_sorry")) >= 1.0
                and _clip01(full_theorem_result.get("statement_match")) >= 1.0
            )
            return 1.0 if final_success and statement_match >= 1.0 else 0.0

        if require_full_theorem:
            return 0.0

        terminal_ids = set(get_terminal_goal_ids(plan))

        terminal_results = [
            result for result in subgoal_results
            if result.get("node_id") in terminal_ids
        ]
        if not terminal_results:
            return 0.0

        statement_match = 1.0 if _normalize_text(plan.theorem_statement) == _normalize_text(expected_theorem) else 0.0
        all_verified = all(
            _clip01(result.get("compile_success")) >= 1.0 and _clip01(result.get("no_sorry")) >= 1.0
            for result in terminal_results
        )
        return 1.0 if all_verified and statement_match >= 1.0 else 0.0


def create_plan_lean_evaluator(**kwargs) -> PlanLeanEvaluator:
    """Convenience factory for compatibility with EvaluateModule injection."""
    return PlanLeanEvaluator(**kwargs)
