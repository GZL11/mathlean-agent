"""
Global memory persistence and retrieval utilities for math-proof search.

This module implements a lightweight Phase-2 memory layer:
- theorem-level experiences distilled from finished rollouts
- subgoal-level repair traces for fixed-plan local search
- verified lemma library for later reuse

The implementation stays intentionally simple and file-based so it can work
without an additional database or embedding service.

Important schema decision:
- theorem and subgoal memories remain distinct
- lemma memory stores both successful and failed lemma attempts
- each lemma row should preserve assumptions / preconditions, proof trajectory,
  and failure localization metadata when available
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from famou.core.data import Program
from famou.utils.plan_runtime import load_experiences, load_lemmas, parse_program_plan


THEOREM_MEMORY_FILE = "theorem_experience.jsonl"
SUBGOAL_MEMORY_FILE = "subgoal_experience.jsonl"
LEMMA_MEMORY_FILE = "verified_lemma_library.jsonl"


def resolve_global_memory_dir(
    config: Optional[Dict[str, Any]] = None,
    default_base_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Resolve the directory used for persistent global memory.

    Resolution order:
    1. explicit `config["global_memory_dir"]`
    2. disabled when `config["enable_global_memory"]` is False
    3. `<default_base_path>/global_memory`
    """
    runtime_config = dict(config or {})
    if runtime_config.get("enable_global_memory", True) is False:
        return None

    explicit_dir = runtime_config.get("global_memory_dir")
    if explicit_dir:
        memory_dir = Path(explicit_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        return memory_dir

    if default_base_path is None:
        return None

    memory_dir = Path(default_base_path) / runtime_config.get("global_memory_subdir", "global_memory")
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


def persist_program_memory(
    program: Program,
    memory_dir: str | Path,
    parent_program: Optional[Program] = None,
) -> Dict[str, int]:
    """Persist theorem/subgoal/lemma memories for one evaluated program."""
    base = Path(memory_dir)
    base.mkdir(parents=True, exist_ok=True)

    theorem_rows = _build_theorem_rows(program)
    subgoal_rows = _build_subgoal_rows(program, parent_program=parent_program)
    lemma_rows = _build_lemma_rows(program)

    theorem_added = _append_jsonl_rows(base / THEOREM_MEMORY_FILE, theorem_rows)
    subgoal_added = _append_jsonl_rows(base / SUBGOAL_MEMORY_FILE, subgoal_rows)
    lemma_added = _append_jsonl_rows(base / LEMMA_MEMORY_FILE, lemma_rows)

    return {
        "theorem_experience": theorem_added,
        "subgoal_experience": subgoal_added,
        "lemma_memory": lemma_added,
        "verified_lemma_library": lemma_added,
    }


def retrieve_theorem_memories(
    memory_dir: str | Path,
    theorem_statement: str,
    limit: int = 3,
    *,
    prefer_solved: bool = True,
    task_constraint: Optional[str] = None,
    task_scaffold: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve theorem-level memories relevant to the current theorem."""
    query = "\n".join(
        part for part in [theorem_statement or "", task_constraint or "", task_scaffold or ""] if part
    )
    rows = _load_jsonl_rows(Path(memory_dir) / THEOREM_MEMORY_FILE)
    return _rank_rows(
        rows,
        query_text=query,
        limit=limit,
        query_fields={
            "theorem_statement": theorem_statement,
            "problem_signature": _build_problem_signature(theorem_statement),
            "prefer_solved": prefer_solved,
            "task_constraint": task_constraint,
            "task_scaffold_signature": _build_scaffold_signature(task_scaffold),
        },
    )


def retrieve_subgoal_memories(
    memory_dir: str | Path,
    statement_formal: str,
    diagnostic: Optional[str] = None,
    limit: int = 3,
    *,
    dependency_ids: Optional[Iterable[str]] = None,
    preferred_action_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve node-local repair memories relevant to one subgoal."""
    parts = [statement_formal or ""]
    if diagnostic:
        parts.append(diagnostic)
    rows = _load_jsonl_rows(Path(memory_dir) / SUBGOAL_MEMORY_FILE)
    return _rank_rows(
        rows,
        query_text="\n".join(parts),
        limit=limit,
        query_fields={
            "statement_formal": statement_formal,
            "diagnostic": diagnostic,
            "dependency_ids": list(dependency_ids or []),
            "preferred_action_types": list(preferred_action_types or []),
        },
    )


def retrieve_lemma_memories(
    memory_dir: str | Path,
    query_text: str,
    limit: int = 3,
    *,
    include_failed: bool = True,
    require_verified: bool = False,
    only_failed: bool = False,
    statement_formal: Optional[str] = None,
    dependency_ids: Optional[Iterable[str]] = None,
    preconditions: Optional[Iterable[str]] = None,
    diagnostic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve lemma memories related to the current theorem or subgoal."""
    rows = _load_jsonl_rows(Path(memory_dir) / LEMMA_MEMORY_FILE)
    if require_verified:
        rows = [
            row for row in rows
            if row.get("proof_outcome") == "successful"
            or row.get("proof_status") in {"proved", "verified"}
        ]
    elif only_failed:
        rows = [row for row in rows if row.get("proof_outcome") == "failed"]
    elif not include_failed:
        rows = [row for row in rows if row.get("proof_outcome") != "failed"]
    return _rank_rows(
        rows,
        query_text=query_text or "",
        limit=limit,
        query_fields={
            "statement_formal": statement_formal,
            "dependency_ids": list(dependency_ids or []),
            "preconditions": list(preconditions or []),
            "diagnostic": diagnostic,
            "require_verified": require_verified,
            "only_failed": only_failed,
        },
    )


def retrieve_verified_lemmas(
    memory_dir: str | Path,
    query_text: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper that only returns verified/proved lemma memories."""
    return retrieve_lemma_memories(
        memory_dir,
        query_text=query_text,
        limit=limit,
        require_verified=True,
        statement_formal=query_text,
    )


def retrieve_failed_lemma_memories(
    memory_dir: str | Path,
    query_text: str,
    limit: int = 3,
    *,
    statement_formal: Optional[str] = None,
    dependency_ids: Optional[Iterable[str]] = None,
    preconditions: Optional[Iterable[str]] = None,
    diagnostic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve failed lemma attempts as negative evidence for repair."""
    return retrieve_lemma_memories(
        memory_dir,
        query_text=query_text,
        limit=limit,
        only_failed=True,
        statement_formal=statement_formal,
        dependency_ids=dependency_ids,
        preconditions=preconditions,
        diagnostic=diagnostic,
    )


def format_global_memory_for_prompt(rows: Iterable[Dict[str, Any]], kind: str) -> str:
    """Format retrieved memory rows into a compact prompt section."""
    items = list(rows)
    if not items:
        return "None"

    sections: List[str] = []
    for index, row in enumerate(items, start=1):
        if kind == "theorem":
            parts = [
                f"- Memory {index}: {row.get('exp_type', 'theorem_memory')}",
                f"  theorem_statement: {row.get('theorem_statement', 'N/A')}",
                f"  outcome: {row.get('outcome', row.get('verifier_outcome', 'unknown'))}",
                f"  success_score: {row.get('success_score', 'N/A')}",
                f"  search_score: {row.get('search_score', 'N/A')}",
            ]
            if row.get("plan_summary"):
                parts.append(f"  plan_summary: {row.get('plan_summary')}")
            if row.get("strategy_type"):
                parts.append(f"  strategy_type: {row.get('strategy_type')}")
            if row.get("subgoal_statements"):
                parts.append(
                    f"  subgoal_statements: {'; '.join(row.get('subgoal_statements', [])[:4])}"
                )
            if row.get("verified_lemma_ids"):
                parts.append(
                    f"  verified_lemma_ids: {', '.join(row.get('verified_lemma_ids', [])[:4])}"
                )
            parts.extend([
                f"  content: {row.get('content', 'N/A')}",
            ])
            failure_tags = row.get("failure_tags") or row.get("failure_patterns") or []
            if failure_tags:
                parts.append(f"  failure_tags: {'; '.join(failure_tags[:4])}")
            if row.get("failure_patterns"):
                parts.append(
                    f"  failure_patterns: {'; '.join(row.get('failure_patterns', [])[:3])}"
                )
            if row.get("repair_hints"):
                parts.append(f"  repair_hints: {'; '.join(row.get('repair_hints', [])[:3])}")
            if row.get("used_lemma_ids"):
                parts.append(f"  used_lemma_ids: {', '.join(row.get('used_lemma_ids', [])[:4])}")
            trajectory = row.get("trajectory_metadata") or {}
            if trajectory:
                parts.append(
                    "  verifier_summary: "
                    f"calls={trajectory.get('verifier_calls')} "
                    f"subgoals={trajectory.get('subgoal_count')} "
                    f"lemmas={trajectory.get('lemma_count')}"
                )
            if row.get("reusable_lemmas"):
                parts.append(
                    f"  reusable_lemmas: {', '.join(row.get('reusable_lemmas', [])[:4])}"
                )
        elif kind == "subgoal":
            parts = [
                f"- Memory {index}: subgoal {row.get('node_id', 'unknown')}",
                f"  statement_formal: {row.get('statement_formal', 'N/A')}",
                f"  status_transition: {row.get('before_status', row.get('parent_status', 'N/A'))} -> {row.get('after_status', row.get('local_status', 'N/A'))}",
            ]
            if row.get("action_type"):
                parts.append(f"  action_type: {row['action_type']}")
            if row.get("diagnostic_before"):
                parts.append(f"  diagnostic_before: {row['diagnostic_before']}")
            if row.get("diagnostic_after"):
                parts.append(f"  diagnostic_after: {row['diagnostic_after']}")
            if row.get("reward_delta") is not None:
                parts.append(f"  reward_delta: {row['reward_delta']}")
            if row.get("search_score_delta") is not None:
                parts.append(f"  search_score_delta: {row['search_score_delta']}")
            if row.get("last_repair_summary"):
                parts.append(f"  repair_summary: {row['last_repair_summary']}")
            if row.get("retrieved_lemma_ids"):
                parts.append(
                    f"  retrieved_lemma_ids: {', '.join(row.get('retrieved_lemma_ids', [])[:4])}"
                )
        elif kind == "lemma":
            parts = [
                f"- Memory {index}: lemma {row.get('lemma_id', 'unknown')}",
                f"  statement_formal: {row.get('statement_formal', 'N/A')}",
                f"  proof_status: {row.get('proof_status', 'N/A')}",
                f"  proof_outcome: {row.get('proof_outcome', 'unknown')}",
            ]
            if row.get("memory_role"):
                parts.append(f"  memory_role: {row.get('memory_role')}")
            assumption_statements = row.get("assumption_statements") or row.get("preconditions") or []
            if assumption_statements:
                parts.append(f"  assumptions: {'; '.join(assumption_statements[:4])}")
            if row.get("reusable_score") is not None:
                parts.append(f"  reusable_score: {row.get('reusable_score')}")
            if row.get("near_success_score") is not None:
                parts.append(f"  near_success_score: {row.get('near_success_score')}")
            if row.get("failure_summary"):
                parts.append(f"  failure_summary: {row.get('failure_summary')}")
            if row.get("failure_category"):
                parts.append(f"  failure_category: {row.get('failure_category')}")
            if row.get("repair_hints"):
                parts.append(f"  repair_hints: {'; '.join(row.get('repair_hints', [])[:3])}")
            diagnostics = row.get("verifier_diagnostics") or []
            if diagnostics:
                parts.append(f"  diagnostics: {'; '.join(diagnostics[:3])}")
            if row.get("source_subgoal_statements"):
                parts.append(
                    f"  source_subgoal_statements: {'; '.join(row.get('source_subgoal_statements', [])[:3])}"
                )
            trajectory = row.get("trajectory_metadata") or {}
            if trajectory:
                source_statuses = trajectory.get("source_node_statuses") or {}
                if source_statuses:
                    parts.append(
                        "  source_node_statuses: "
                        + "; ".join(f"{key}={value}" for key, value in list(source_statuses.items())[:4])
                    )
            if row.get("lean_code"):
                parts.append(f"  lean_code: {row['lean_code']}")
        else:
            parts = [json.dumps(row, ensure_ascii=False)]
        sections.append("\n".join(parts))
    return "\n".join(sections)


def _build_theorem_rows(program: Program) -> List[Dict[str, Any]]:
    plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
    experiences = load_experiences(plan)
    theorem_id = _stable_signature(plan.theorem_statement, prefix="thm")
    problem_signature = _build_problem_signature(plan.theorem_statement)
    outcome = _classify_theorem_outcome(program.metrics)
    failure_tags = _build_theorem_failure_tags(program.metrics)
    used_lemma_ids = _collect_used_lemma_ids(plan)
    subgoal_statements = [
        node.statement_formal for node in plan.subgoals if getattr(node, "statement_formal", None)
    ]
    task_scaffold = _extract_task_scaffold(program)
    trajectory_metadata = _build_theorem_trajectory_metadata(program, plan)
    rows: List[Dict[str, Any]] = []
    for experience in experiences:
        row = {
            "memory_id": experience.exp_id,
            "memory_kind": "theorem_experience",
            "program_id": program.id,
            "plan_id": plan.plan_id,
            "theorem_id": theorem_id,
            "problem_signature": problem_signature,
            "theorem_statement": plan.theorem_statement,
            "plan_summary": plan.plan_summary,
            "strategy_type": plan.strategy_type,
            "outcome": outcome,
            "exp_type": experience.exp_type,
            "content": experience.content,
            "failure_tags": list(failure_tags),
            "failure_patterns": list(experience.failure_patterns),
            "repair_hints": list(experience.repair_hints),
            "reusable_lemmas": list(experience.reusable_lemmas),
            "used_lemma_ids": list(used_lemma_ids),
            "verified_lemma_ids": list(used_lemma_ids),
            "subgoal_statements": list(subgoal_statements),
            "task_scaffold_signature": _build_scaffold_signature(task_scaffold),
            "task_scaffold_excerpt": _truncate(task_scaffold, 400),
            "verifier_outcome": experience.verifier_outcome,
            "theorem_solved": float((program.metrics or {}).get("theorem_solved", 0.0) or 0.0),
            "compile_success": float((program.metrics or {}).get("compile_success", 0.0) or 0.0),
            "no_sorry": float((program.metrics or {}).get("no_sorry", 0.0) or 0.0),
            "statement_match": float((program.metrics or {}).get("statement_match", 0.0) or 0.0),
            "success_score": program.success_score,
            "search_score": program.effective_search_score,
            "trajectory_metadata": trajectory_metadata,
            "generation": program.generation,
            "iteration": program.iteration,
            "created_at": program.created_at,
        }
        row["text"] = " ".join(
            [
                plan.theorem_statement,
                plan.plan_summary or "",
                plan.strategy_type,
                outcome,
                experience.content,
                " ".join(failure_tags),
                " ".join(experience.failure_patterns),
                " ".join(experience.repair_hints),
                " ".join(used_lemma_ids),
                " ".join(subgoal_statements),
                task_scaffold or "",
                " ".join(experience.reusable_lemmas),
            ]
        ).strip()
        rows.append(row)
    return rows


def _build_subgoal_rows(
    program: Program,
    parent_program: Optional[Program] = None,
) -> List[Dict[str, Any]]:
    plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
    theorem_id = _stable_signature(plan.theorem_statement, prefix="thm")
    problem_signature = _build_problem_signature(plan.theorem_statement)
    parent_plan = None
    if parent_program is not None:
        try:
            parent_plan = parse_program_plan(parent_program, fallback_theorem=plan.theorem_statement)
        except Exception:
            parent_plan = None

    parent_map = parent_plan.node_map() if parent_plan is not None else {}
    selected_node_id = (
        program.meta.get("selected_subgoal_id")
        or plan.metadata.get("selected_subgoal_id")
        or plan.metadata.get("last_repaired_node_id")
    )
    patch_payload = dict(program.meta.get("repair_patch", {}) or {})
    selected_candidates = {str(selected_node_id)} if selected_node_id else set()
    if patch_payload.get("node_id"):
        selected_candidates.add(str(patch_payload.get("node_id")))

    rows: List[Dict[str, Any]] = []
    for node in plan.subgoals:
        if selected_candidates and node.node_id not in selected_candidates:
            continue
        has_signal = bool(
            node.attempt_count > 0
            or node.last_diagnostic
            or node.last_repair_summary
            or node.local_lean_draft
            or node.proof_sketch
            or node.local_status.value not in {"pending", "blocked"}
        )
        if not has_signal:
            continue

        parent_status = None
        parent_node = None
        if node.node_id in parent_map:
            parent_node = parent_map[node.node_id]
            parent_status = parent_node.local_status.value

        if not selected_candidates and parent_node is not None:
            changed = (
                parent_node.local_status != node.local_status
                or (parent_node.last_diagnostic or "") != (node.last_diagnostic or "")
                or (parent_node.local_lean_draft or "").strip() != (node.local_lean_draft or "").strip()
                or (parent_node.proof_sketch or "").strip() != (node.proof_sketch or "").strip()
            )
            if not changed:
                continue

        action_type = _infer_subgoal_action_type(patch_payload, parent_node, node)
        before_status = parent_status or "none"
        after_status = node.local_status.value
        before_diagnostic = _short_diagnostic_from_node(parent_node)
        after_diagnostic = _short_diagnostic_from_node(node)
        reward_delta = _score_delta(program.combined_score, parent_program.combined_score if parent_program else None)
        search_score_delta = _score_delta(
            program.effective_search_score,
            parent_program.effective_search_score if parent_program else None,
        )
        node_score_delta = _score_delta(node.node_score, parent_node.node_score if parent_node else None)
        attempt_entry = node.attempt_history[-1] if node.attempt_history else None
        dependency_signature = _stable_signature(
            "|".join(sorted(node.dependencies)) + "::" + node.statement_formal,
            prefix="dep",
        )
        trajectory_metadata = {
            "patch_keys": sorted(patch_payload.keys()),
            "attempt_entry": attempt_entry,
            "local_island_phase": program.meta.get("local_island_phase"),
            "local_island_step": program.meta.get("local_island_step"),
            "selection_reason": program.meta.get("selection_reason") or plan.metadata.get("selection_reason"),
        }

        row = {
            "memory_id": f"{program.id}:{node.node_id}",
            "memory_kind": "subgoal_experience",
            "program_id": program.id,
            "plan_id": plan.plan_id,
            "theorem_id": theorem_id,
            "problem_signature": problem_signature,
            "theorem_statement": plan.theorem_statement,
            "node_id": node.node_id,
            "statement_formal": node.statement_formal,
            "dependencies": list(node.dependencies),
            "dependency_signature": dependency_signature,
            "selected_for_repair": node.node_id in selected_candidates if selected_candidates else False,
            "action_type": action_type,
            "local_status": node.local_status.value,
            "parent_status": parent_status,
            "before_status": before_status,
            "after_status": after_status,
            "attempt_count": node.attempt_count,
            "diagnostic_before": before_diagnostic,
            "diagnostic_after": after_diagnostic,
            "last_diagnostic": node.last_diagnostic,
            "last_repair_summary": node.last_repair_summary,
            "retrieved_lemma_ids": list(node.retrieved_lemma_ids),
            "retrieved_experience_ids": list(node.retrieved_experience_ids),
            "proof_sketch_before": _truncate(parent_node.proof_sketch, 240) if parent_node else None,
            "proof_sketch_after": _truncate(node.proof_sketch, 240),
            "local_lean_before": _truncate(parent_node.local_lean_draft, 400) if parent_node else None,
            "local_lean_after": _truncate(node.local_lean_draft, 400),
            "proof_sketch": _truncate(node.proof_sketch, 240),
            "local_lean_draft": _truncate(node.local_lean_draft, 400),
            "node_score": node.node_score,
            "node_score_delta": node_score_delta,
            "reward_delta": reward_delta,
            "search_score_delta": search_score_delta,
            "is_complete": node.is_complete,
            "trajectory_metadata": trajectory_metadata,
            "generation": program.generation,
            "iteration": program.iteration,
            "created_at": program.created_at,
        }
        row["text"] = " ".join(
            part
            for part in [
                node.statement_formal,
                before_status,
                after_status,
                action_type,
                before_diagnostic or "",
                after_diagnostic or "",
                node.last_diagnostic or "",
                node.last_repair_summary or "",
                node.proof_sketch or "",
                node.local_lean_draft or "",
                " ".join(node.retrieved_lemma_ids),
            ]
            if part
        )
        rows.append(row)
    return rows


def _build_lemma_rows(program: Program) -> List[Dict[str, Any]]:
    plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
    theorem_id = _stable_signature(plan.theorem_statement, prefix="thm")
    problem_signature = _build_problem_signature(plan.theorem_statement)
    node_map = plan.node_map()
    lemma_results = _index_lemma_results(program.metrics)
    rows: List[Dict[str, Any]] = []
    for lemma in load_lemmas(plan):
        lemma_result = lemma_results.get(lemma.lemma_id, {})
        has_signal = bool(
            lemma.lean_code
            or lemma.metadata
            or lemma_result
            or lemma.proof_status.value not in {"pending"}
        )
        if not has_signal:
            continue
        lemma_context = _collect_lemma_context(plan, lemma)
        proof_outcome = _classify_lemma_outcome(lemma, lemma_result)
        diagnostics = list(lemma_result.get("diagnostics", []) or [])
        failure_summary = _summarize_lemma_failure(lemma_result, proof_outcome)
        failure_category = _classify_lemma_failure_category(lemma_result, proof_outcome)
        repair_hints = _build_lemma_repair_hints(lemma_result, proof_outcome, failure_category)
        near_success_score = _compute_lemma_near_success_score(lemma_result, lemma, proof_outcome)
        trajectory_metadata = {
            "generation": program.generation,
            "iteration": program.iteration,
            "source_node_statuses": lemma_context["source_node_statuses"],
            "local_island_summary": _compact_local_island_summary(program.meta.get("local_island_summary")),
            "verifier_trace": {
                "artifact_path": lemma_result.get("artifact_path"),
                "returncode": lemma_result.get("returncode"),
                "wrapper_file": lemma_result.get("wrapper_file"),
                "staged_dir": lemma_result.get("staged_dir"),
                "latency_ms": lemma_result.get("latency_ms"),
            },
        }
        row = {
            "memory_id": f"{program.id}:{lemma.lemma_id}",
            "memory_kind": "lemma_memory",
            "program_id": program.id,
            "plan_id": plan.plan_id,
            "theorem_id": theorem_id,
            "problem_signature": problem_signature,
            "theorem_statement": plan.theorem_statement,
            "lemma_id": lemma.lemma_id,
            "statement_formal": lemma.statement_formal,
            "statement_nl": lemma.statement_nl,
            "proof_status": lemma.proof_status.value,
            "proof_outcome": proof_outcome,
            "memory_role": _lemma_memory_role(proof_outcome),
            "reusable_score": lemma.reusable_score,
            "near_success_score": near_success_score,
            "source_plan_id": lemma.source_plan_id,
            "source_node_ids": list(lemma.source_node_ids),
            "source_subgoal_statements": lemma_context["source_subgoal_statements"],
            "dependency_ids": lemma_context["dependency_ids"],
            "dependency_signature": lemma_context["dependency_signature"],
            "assumption_statements": lemma_context["assumption_statements"],
            "preconditions": lemma_context["preconditions"],
            "lean_code": _truncate(lemma.lean_code, 400),
            "artifact_path": lemma_result.get("artifact_path"),
            "statement_match": lemma_result.get("statement_match"),
            "compile_success": lemma_result.get("compile_success"),
            "no_sorry": lemma_result.get("no_sorry"),
            "proof_validity": lemma_result.get("proof_validity"),
            "verifier_diagnostics": diagnostics,
            "failure_summary": failure_summary,
            "failure_category": failure_category,
            "repair_hints": repair_hints,
            "failure_stage": "lemma_verifier" if lemma_result else None,
            "trajectory_metadata": trajectory_metadata,
            "lemma_metadata": dict(lemma.metadata or {}),
            "created_at": program.created_at,
        }
        row["text"] = " ".join(
            part
            for part in [
                lemma.statement_formal,
                plan.theorem_statement,
                " ".join(lemma_context["assumption_statements"]),
                " ".join(lemma_context["source_subgoal_statements"]),
                lemma.lean_code or "",
                failure_summary or "",
                failure_category or "",
                " ".join(repair_hints),
                " ".join(diagnostics),
            ]
            if part
        )
        rows.append(row)
    return rows


def _index_lemma_results(metrics: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index evaluator lemma results by lemma_id."""
    indexed: Dict[str, Dict[str, Any]] = {}
    if not isinstance(metrics, dict):
        return indexed
    for row in metrics.get("lemma_results", []) or []:
        if not isinstance(row, dict):
            continue
        lemma_id = row.get("lemma_id")
        if lemma_id:
            indexed[str(lemma_id)] = dict(row)
    return indexed


def _collect_lemma_context(plan, lemma) -> Dict[str, Any]:
    """Collect assumptions and dependency context for one lemma memory row."""
    node_map = plan.node_map()
    source_nodes = [node_map[node_id] for node_id in lemma.source_node_ids if node_id in node_map]
    dependency_ids: List[str] = []
    source_subgoal_statements: List[str] = []
    assumption_statements: List[str] = []
    source_node_statuses: Dict[str, str] = {}

    for node in source_nodes:
        source_subgoal_statements.append(node.statement_formal)
        source_node_statuses[node.node_id] = node.local_status.value
        for dependency_id in node.dependencies:
            if dependency_id not in dependency_ids:
                dependency_ids.append(dependency_id)
            dependency_node = node_map.get(dependency_id)
            if dependency_node is not None and dependency_node.statement_formal not in assumption_statements:
                assumption_statements.append(dependency_node.statement_formal)

    metadata_preconditions = list((lemma.metadata or {}).get("preconditions", []) or [])
    preconditions = []
    for item in [*metadata_preconditions, *assumption_statements]:
        if item and item not in preconditions:
            preconditions.append(item)

    return {
        "dependency_ids": dependency_ids,
        "dependency_signature": _stable_signature(
            "|".join(sorted(dependency_ids)) + "::" + "|".join(sorted(preconditions)),
            prefix="dep",
        ),
        "source_subgoal_statements": source_subgoal_statements,
        "assumption_statements": assumption_statements,
        "preconditions": preconditions,
        "source_node_statuses": source_node_statuses,
    }


def _score_delta(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Compute a rounded score delta when both ends are available."""
    if current is None or previous is None:
        return None
    return round(float(current) - float(previous), 6)


def _classify_theorem_outcome(metrics: Optional[Dict[str, Any]]) -> str:
    """Map evaluator metrics to a stable theorem-run outcome label."""
    if not isinstance(metrics, dict):
        return "unknown"
    if float(metrics.get("theorem_solved", 0.0) or 0.0) >= 1.0:
        return "solved"
    if bool(metrics.get("error")):
        return "invalid"
    return "unsolved"


def _build_theorem_failure_tags(metrics: Optional[Dict[str, Any]]) -> List[str]:
    """Summarize theorem-level failure tags for retrieval and ablation."""
    if not isinstance(metrics, dict):
        return []
    tags: List[str] = []
    if float(metrics.get("compile_success", 1.0) or 1.0) < 1.0:
        tags.append("compile_error")
    if float(metrics.get("no_sorry", 1.0) or 1.0) < 1.0:
        tags.append("contains_sorry")
    if float(metrics.get("statement_match", 1.0) or 1.0) < 1.0:
        tags.append("statement_drift")
    if list(metrics.get("blocked_nodes", []) or []):
        tags.append("blocked_frontier")
    if list(metrics.get("frontier_nodes", []) or []):
        tags.append("unfinished_frontier")
    if bool(metrics.get("error")):
        tags.append("runtime_error")
    return tags


def _collect_used_lemma_ids(plan) -> List[str]:
    """Collect lemma ids that reached a reusable state inside the plan."""
    used: List[str] = []
    for lemma in plan.lemma_pool:
        if lemma.proof_status.value in {"proved", "verified"} and lemma.lemma_id not in used:
            used.append(lemma.lemma_id)
    return used


def _build_theorem_trajectory_metadata(program: Program, plan) -> Dict[str, Any]:
    """Build a compact theorem-run trajectory summary."""
    metrics = program.metrics or {}
    return {
        "subgoal_count": len(plan.subgoals),
        "lemma_count": len(plan.lemma_pool),
        "frontier_nodes": list(metrics.get("frontier_nodes", []) or []),
        "blocked_nodes": list(metrics.get("blocked_nodes", []) or []),
        "verifier_calls": metrics.get("verifier_calls"),
        "verifier_budget": metrics.get("verifier_budget"),
        "latency_ms": metrics.get("latency_ms"),
        "reward_mode": metrics.get("reward_mode"),
        "local_island_summary": _compact_local_island_summary(program.meta.get("local_island_summary")),
    }


def _short_diagnostic_from_node(node: Any) -> Optional[str]:
    """Extract a compact node diagnostic from explicit and verifier-state fields."""
    if node is None:
        return None
    if getattr(node, "last_diagnostic", None):
        return str(node.last_diagnostic)
    verifier_state = getattr(node, "verifier_state", {}) or {}
    diagnostics = verifier_state.get("diagnostics", [])
    if isinstance(diagnostics, list):
        for item in diagnostics:
            if item:
                return str(item)
    return None


def _infer_subgoal_action_type(
    patch_payload: Dict[str, Any],
    parent_node: Any,
    child_node: Any,
) -> str:
    """Infer a coarse repair action type for subgoal attempt memory."""
    if patch_payload.get("introduced_lemmas"):
        return "introduce_lemma"
    parent_draft = ((parent_node.local_lean_draft if parent_node else None) or "").strip()
    child_draft = ((child_node.local_lean_draft if child_node else None) or "").strip()
    parent_sketch = ((parent_node.proof_sketch if parent_node else None) or "").strip()
    child_sketch = ((child_node.proof_sketch if child_node else None) or "").strip()
    if not parent_draft and child_draft:
        return "draft"
    if parent_draft and child_draft and parent_draft != child_draft:
        return "repair"
    if parent_sketch != child_sketch:
        return "rewrite"
    if patch_payload.get("verifier_state_update"):
        return "dependency_fix"
    return "repair"


def _classify_lemma_outcome(lemma, lemma_result: Dict[str, Any]) -> str:
    """Map lemma status plus verifier result into a stable outcome label."""
    compile_success = float(lemma_result.get("compile_success", 0.0) or 0.0)
    no_sorry = float(lemma_result.get("no_sorry", 0.0) or 0.0)
    if compile_success >= 1.0 and no_sorry >= 1.0 and lemma.proof_status.value in {"proved", "verified"}:
        return "successful"
    if lemma_result and compile_success < 1.0:
        return "failed"
    if lemma_result and no_sorry < 1.0:
        return "failed"
    if lemma.proof_status.value in {"attempted", "failed", "blocked"}:
        return "failed"
    if lemma.proof_status.value in {"in_progress", "pending"}:
        return "draft"
    return "partial"


def _lemma_memory_role(proof_outcome: str) -> str:
    """Assign a coarse role so prompts can separate reusable assets from debug traces."""
    if proof_outcome == "successful":
        return "reusable_knowledge"
    if proof_outcome == "failed":
        return "failure_experience"
    if proof_outcome == "draft":
        return "draft_candidate"
    return "partial_attempt"


def _classify_lemma_failure_category(
    lemma_result: Dict[str, Any],
    proof_outcome: str,
) -> Optional[str]:
    """Group failed lemma attempts into a small set of repair-relevant categories."""
    if proof_outcome != "failed":
        return None

    diagnostics = " ".join(str(item) for item in (lemma_result.get("diagnostics", []) or []))
    diagnostics_lower = diagnostics.lower()
    if "unknown identifier" in diagnostics_lower:
        return "unknown_identifier"
    if "type mismatch" in diagnostics_lower:
        return "type_mismatch"
    if "application type mismatch" in diagnostics_lower:
        return "application_mismatch"
    if "unsolved goals" in diagnostics_lower or "remaining goals" in diagnostics_lower:
        return "unsolved_goals"
    if float(lemma_result.get("statement_match", 1.0) or 1.0) < 1.0:
        return "statement_drift"
    if float(lemma_result.get("no_sorry", 1.0) or 1.0) < 1.0:
        return "contains_sorry"
    if float(lemma_result.get("compile_success", 1.0) or 1.0) < 1.0:
        return "compile_error"
    return "failed_without_structured_diagnostic"


def _build_lemma_repair_hints(
    lemma_result: Dict[str, Any],
    proof_outcome: str,
    failure_category: Optional[str],
) -> List[str]:
    """Derive compact repair hints from failed lemma diagnostics."""
    if proof_outcome != "failed":
        return []

    hints: List[str] = []
    if failure_category == "unknown_identifier":
        hints.append("check whether all hypotheses are introduced and named consistently")
    elif failure_category == "type_mismatch":
        hints.append("align the goal and intermediate terms before applying the lemma")
    elif failure_category == "application_mismatch":
        hints.append("check argument order and implicit arguments for the applied lemma")
    elif failure_category == "unsolved_goals":
        hints.append("inspect the remaining goals and split the proof into smaller local steps")
    elif failure_category == "statement_drift":
        hints.append("keep the helper lemma statement aligned with the intended subgoal")
    elif failure_category == "contains_sorry":
        hints.append("replace sorry with a concrete local Lean draft before reuse")
    elif failure_category == "compile_error":
        hints.append("repair the Lean syntax or imported names before retrying this lemma")

    diagnostics = list(lemma_result.get("diagnostics", []) or [])
    if diagnostics:
        hints.append(diagnostics[0][:160])

    deduped: List[str] = []
    for hint in hints:
        normalized = hint.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:3]


def _compute_lemma_near_success_score(
    lemma_result: Dict[str, Any],
    lemma,
    proof_outcome: str,
) -> Optional[float]:
    """Estimate how close a lemma attempt is to being reusable."""
    if proof_outcome == "successful":
        return 1.0

    metrics: List[float] = []
    for key in ("compile_success", "no_sorry", "statement_match", "proof_validity"):
        value = lemma_result.get(key)
        if value is not None:
            metrics.append(float(value))

    if not metrics:
        status_progress = {
            "verified": 1.0,
            "proved": 0.85,
            "in_progress": 0.5,
            "attempted": 0.25,
            "failed": 0.0,
            "blocked": 0.0,
            "pending": 0.0,
        }
        return status_progress.get(lemma.proof_status.value, 0.0)

    return round(sum(metrics) / len(metrics), 6)


def _summarize_lemma_failure(lemma_result: Dict[str, Any], proof_outcome: str) -> Optional[str]:
    """Build a compact failure summary for failed lemma attempts."""
    if proof_outcome != "failed":
        return None
    diagnostics = list(lemma_result.get("diagnostics", []) or [])
    if diagnostics:
        return diagnostics[0][:240]
    if float(lemma_result.get("compile_success", 0.0) or 0.0) < 1.0:
        return "lemma failed to compile in Lean"
    if float(lemma_result.get("no_sorry", 0.0) or 0.0) < 1.0:
        return "lemma still contains sorry"
    if float(lemma_result.get("statement_match", 1.0) or 1.0) < 1.0:
        return "lemma statement does not match its declared conclusion"
    return "lemma attempt failed without a structured diagnostic"


def _compact_local_island_summary(summary: Any) -> Optional[Dict[str, Any]]:
    """Keep only the most useful local-island metadata inside lemma trajectory."""
    if not isinstance(summary, dict):
        return None
    return {
        "budget": summary.get("budget"),
        "steps_executed": summary.get("steps_executed"),
        "best_score": summary.get("best_score"),
        "history": list(summary.get("history", []) or [])[-3:],
    }


def _append_jsonl_rows(path: Path, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    existing_rows = _load_jsonl_rows(path)
    existing_ids = {row.get("memory_id") for row in existing_rows}
    existing_keys = {_row_uniqueness_key(row) for row in existing_rows}
    added = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            memory_id = row.get("memory_id")
            uniqueness_key = _row_uniqueness_key(row)
            if memory_id in existing_ids or uniqueness_key in existing_keys:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing_ids.add(memory_id)
            existing_keys.add(uniqueness_key)
            added += 1
    return added


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _row_uniqueness_key(row: Dict[str, Any]) -> tuple:
    """Build a stable semantic key to avoid cross-run duplicate memories."""
    kind = row.get("memory_kind")
    if kind == "theorem_experience":
        return (
            kind,
            row.get("theorem_statement"),
            row.get("exp_type"),
            row.get("content"),
        )
    if kind == "subgoal_experience":
        return (
            kind,
            row.get("theorem_statement"),
            row.get("node_id"),
            row.get("statement_formal"),
            row.get("last_repair_summary"),
            row.get("local_lean_draft"),
            row.get("local_status"),
        )
    if kind == "lemma_memory":
        return (
            kind,
            row.get("theorem_statement"),
            row.get("lemma_id"),
            row.get("statement_formal"),
            row.get("lean_code"),
            row.get("proof_outcome"),
            row.get("failure_summary"),
        )
    return (kind, row.get("memory_id"))


def _rank_rows(
    rows: List[Dict[str, Any]],
    query_text: str,
    limit: int,
    *,
    query_fields: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not rows or limit <= 0:
        return []

    query_tokens = _tokenize(query_text)
    query_fields = dict(query_fields or {})
    scored: List[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        score = _row_retrieval_score(row, query_text, query_tokens, query_fields)
        if score <= 0.0:
            continue
        scored.append((score, row))

    if not scored:
        return []

    scored.sort(
        key=lambda item: (
            item[0],
            _proof_outcome_priority(item[1]),
            float(item[1].get("reusable_score", 0.0) or 0.0),
            float(item[1].get("search_score", 0.0) or 0.0),
            float(item[1].get("success_score", 0.0) or 0.0),
            float(item[1].get("created_at", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return [row for _, row in scored[:limit]]


def _row_retrieval_score(
    row: Dict[str, Any],
    query_text: str,
    query_tokens: set[str],
    query_fields: Dict[str, Any],
) -> float:
    kind = str(row.get("memory_kind") or "")
    if kind == "theorem_experience":
        return _theorem_retrieval_score(row, query_text, query_tokens, query_fields)
    if kind == "subgoal_experience":
        return _subgoal_retrieval_score(row, query_text, query_tokens, query_fields)
    if kind == "lemma_memory":
        return _lemma_retrieval_score(row, query_text, query_tokens, query_fields)
    return _base_text_retrieval_score(row, query_text, query_tokens)


def _theorem_retrieval_score(
    row: Dict[str, Any],
    query_text: str,
    query_tokens: set[str],
    query_fields: Dict[str, Any],
) -> float:
    score = _base_text_retrieval_score(row, query_text, query_tokens)

    theorem_statement = _normalize_statement(row.get("theorem_statement") or "").lower()
    query_statement = _normalize_statement(query_fields.get("theorem_statement") or query_text).lower()
    if theorem_statement and query_statement:
        if theorem_statement == query_statement:
            score += 1.2
        elif theorem_statement in query_statement or query_statement in theorem_statement:
            score += 0.4

    if row.get("problem_signature") and row.get("problem_signature") == query_fields.get("problem_signature"):
        score += 0.8

    query_scaffold_signature = query_fields.get("task_scaffold_signature")
    if (
        query_scaffold_signature
        and row.get("task_scaffold_signature")
        and row.get("task_scaffold_signature") == query_scaffold_signature
    ):
        score += 0.9

    score += 0.2 * _text_match_score(
        query_fields.get("task_constraint") or "",
        " ".join(
            part
            for part in [
                row.get("plan_summary"),
                " ".join(row.get("subgoal_statements", []) or []),
                " ".join(row.get("verified_lemma_ids", []) or []),
                row.get("task_scaffold_excerpt"),
            ]
            if part
        ),
    )

    prefer_solved = bool(query_fields.get("prefer_solved", True))
    outcome = row.get("outcome")
    if prefer_solved and outcome == "solved":
        score += 0.35
    elif prefer_solved and outcome == "unsolved":
        score += 0.08

    score += 0.15 * float(row.get("success_score", 0.0) or 0.0)
    score += 0.1 * float(row.get("search_score", 0.0) or 0.0)
    return score


def _subgoal_retrieval_score(
    row: Dict[str, Any],
    query_text: str,
    query_tokens: set[str],
    query_fields: Dict[str, Any],
) -> float:
    score = _base_text_retrieval_score(row, query_text, query_tokens)

    statement = _normalize_statement(row.get("statement_formal") or "").lower()
    query_statement = _normalize_statement(query_fields.get("statement_formal") or query_text).lower()
    if statement and query_statement:
        if statement == query_statement:
            score += 1.1
        elif statement in query_statement or query_statement in statement:
            score += 0.35

    query_dependencies = {str(item) for item in (query_fields.get("dependency_ids") or []) if item}
    row_dependencies = {str(item) for item in (row.get("dependencies") or []) if item}
    score += 0.55 * _set_match_score(query_dependencies, row_dependencies)

    preferred_action_types = {
        str(item) for item in (query_fields.get("preferred_action_types") or []) if item
    }
    if preferred_action_types and str(row.get("action_type") or "") in preferred_action_types:
        score += 0.25

    diagnostic_overlap = _text_match_score(
        query_fields.get("diagnostic") or "",
        " ".join(
            part
            for part in [
                row.get("diagnostic_before"),
                row.get("diagnostic_after"),
                row.get("last_diagnostic"),
            ]
            if part
        ),
    )
    score += 0.35 * diagnostic_overlap

    reward_delta = row.get("reward_delta")
    if reward_delta is not None:
        reward_delta = float(reward_delta)
        if reward_delta > 0:
            score += min(0.25, reward_delta)
        elif reward_delta < 0:
            score -= min(0.12, abs(reward_delta))

    if row.get("is_complete"):
        score += 0.08
    return score


def _lemma_retrieval_score(
    row: Dict[str, Any],
    query_text: str,
    query_tokens: set[str],
    query_fields: Dict[str, Any],
) -> float:
    score = _base_text_retrieval_score(row, query_text, query_tokens)

    statement = _normalize_statement(row.get("statement_formal") or "").lower()
    query_statement = _normalize_statement(query_fields.get("statement_formal") or "").lower()
    if statement and query_statement:
        if statement == query_statement:
            score += 1.0
        elif statement in query_statement or query_statement in statement:
            score += 0.35

    query_dependencies = {str(item) for item in (query_fields.get("dependency_ids") or []) if item}
    row_dependencies = {str(item) for item in (row.get("dependency_ids") or []) if item}
    score += 0.45 * _set_match_score(query_dependencies, row_dependencies)

    query_preconditions = _normalized_set(query_fields.get("preconditions") or [])
    row_preconditions = _normalized_set(
        (row.get("preconditions") or []) or (row.get("assumption_statements") or [])
    )
    score += 0.8 * _set_match_score(query_preconditions, row_preconditions)

    diagnostic_match = _text_match_score(
        query_fields.get("diagnostic") or "",
        " ".join(
            part
            for part in [
                row.get("failure_summary"),
                " ".join(row.get("verifier_diagnostics", []) or []),
            ]
            if part
        ),
    )

    proof_outcome = row.get("proof_outcome")
    if proof_outcome == "successful":
        score += 0.32 + 0.18 * float(row.get("reusable_score", 0.0) or 0.0)
    elif proof_outcome == "failed":
        score -= 0.08
        score += 0.28 * diagnostic_match
        score += 0.1 * float(row.get("near_success_score", 0.0) or 0.0)
    elif row.get("proof_status") in {"proved", "verified"}:
        score += 0.15

    if query_fields.get("require_verified") and proof_outcome != "successful":
        score -= 0.2
    if query_fields.get("only_failed"):
        if proof_outcome == "failed":
            score += 0.18 + 0.12 * diagnostic_match
        else:
            score -= 0.25

    return score


def _base_text_retrieval_score(
    row: Dict[str, Any],
    query_text: str,
    query_tokens: set[str],
) -> float:
    candidate_text = str(row.get("text") or "")
    if not candidate_text.strip():
        candidate_text = " ".join(
            str(part)
            for part in [
                row.get("theorem_statement"),
                row.get("statement_formal"),
                row.get("content"),
            ]
            if part
        )
    if not candidate_text.strip():
        return 0.0

    if not query_tokens:
        return 0.1

    candidate_tokens = _tokenize(candidate_text)
    overlap = len(query_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(query_tokens))


def _set_match_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _text_match_score(query_text: str, candidate_text: str) -> float:
    query_tokens = _tokenize(query_text)
    candidate_tokens = _tokenize(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(query_tokens))


def _normalized_set(items: Iterable[str]) -> set[str]:
    return {_normalize_statement(str(item)).lower() for item in items if str(item).strip()}


def _proof_outcome_priority(row: Dict[str, Any]) -> int:
    proof_outcome = row.get("proof_outcome")
    if proof_outcome == "successful":
        return 3
    if row.get("outcome") == "solved":
        return 2
    if row.get("proof_status") in {"proved", "verified"}:
        return 1
    if proof_outcome == "failed":
        return -1
    return 0


def _normalize_statement(text: str) -> str:
    return " ".join((text or "").split())


def _stable_signature(text: str, prefix: str) -> str:
    normalized = _normalize_statement(text)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _build_problem_signature(theorem_statement: str) -> str:
    return _stable_signature(theorem_statement, prefix="prob")


def _build_scaffold_signature(scaffold_text: Optional[str]) -> Optional[str]:
    normalized = _normalize_statement(scaffold_text or "")
    if not normalized:
        return None
    return _stable_signature(normalized, prefix="scaf")


def _extract_task_scaffold(program: Program) -> str:
    """
    Recover the best available theorem scaffold snapshot for theorem memory.

    Preferred order:
    1. the exact system prompt used for generation
    2. explicit scaffold text stored in program metadata
    3. scaffold/problem files referenced from metadata or parsed plan artifacts
    4. a prompt fallback when it embeds the theorem scaffold
    5. the theorem statement itself as the weakest non-empty fallback
    """
    scaffold = str(getattr(program, "system_prompt", "") or "").strip()
    if scaffold:
        return scaffold

    for key in ("task_scaffold", "task_description", "theorem_scaffold"):
        candidate = str(program.meta.get(key, "") or "").strip()
        if candidate:
            return candidate

    plan = None
    try:
        plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
    except Exception:
        plan = None

    for path_text in _collect_scaffold_candidate_paths(program, plan):
        text = _read_text_if_exists(path_text)
        if text:
            return text

    prompt = str(getattr(program, "prompt", "") or "")
    if "Target theorem:" in prompt:
        return prompt.strip()

    theorem_statement = str(
        program.meta.get("theorem_statement")
        or getattr(plan, "theorem_statement", "")
        or ""
    ).strip()
    return theorem_statement


def _collect_scaffold_candidate_paths(program: Program, plan: Any) -> List[str]:
    """Collect possible scaffold/problem file paths from program and parsed plan metadata."""
    candidates: List[str] = []

    for key in ("theorem_scaffold_path", "problem_path"):
        value = program.meta.get(key)
        if value:
            candidates.append(str(value))

    if plan is not None:
        for key in ("theorem_scaffold_path", "problem_path"):
            value = plan.lean_artifacts.get(key) or plan.metadata.get(key)
            if value:
                candidates.append(str(value))

    # Preserve order while removing duplicates.
    deduped: List[str] = []
    seen = set()
    for item in candidates:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _read_text_if_exists(path_text: str) -> str:
    """Best-effort file read helper used by scaffold extraction."""
    try:
        path = Path(path_text)
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return ""


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_']+", text or "")}


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."
