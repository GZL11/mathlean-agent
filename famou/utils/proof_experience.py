"""
Experience extraction utilities for plan-centric mathproof search.

This module keeps the first version intentionally lightweight:
- distill verifier / local-island outcomes into a small number of structured experiences
- attach those experiences back to the generated plan object
- format the memory so later plan mutations can condition on it
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from famou.core.data import Program
from famou.core.proof_objects import ExperienceObject
from famou.utils.plan_runtime import (
    dump_plan,
    load_experiences,
    parse_program_plan,
    summarize_plan_delta,
)


def _clip01(value: Any) -> float:
    """Safely clamp an arbitrary numeric value into [0, 1]."""
    try:
        numeric = float(value or 0.0)
    except Exception:
        numeric = 0.0
    return min(1.0, max(0.0, numeric))


def _first_diagnostic(metrics: Dict[str, Any]) -> Optional[str]:
    """Pull out a short representative verifier diagnostic when available."""
    collections = [
        metrics.get("subgoal_results", []),
        metrics.get("lemma_results", []),
    ]
    full_theorem_result = metrics.get("full_theorem_result")
    if isinstance(full_theorem_result, dict):
        collections.append([full_theorem_result])

    for collection in collections:
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            diagnostics = item.get("diagnostics", [])
            if isinstance(diagnostics, list):
                for diagnostic in diagnostics:
                    if diagnostic and str(diagnostic).strip():
                        return str(diagnostic).strip()
    return None


def _verified_lemma_ids(plan) -> List[str]:
    """Collect reusable verified lemmas from a plan."""
    lemma_ids: List[str] = []
    for lemma in plan.lemma_pool:
        if lemma.is_verified:
            lemma_ids.append(lemma.lemma_id)
    return lemma_ids


def _failure_patterns(metrics: Dict[str, Any]) -> List[str]:
    """Summarize coarse failure signatures from evaluator metrics."""
    patterns: List[str] = []
    if _clip01(metrics.get("compile_success")) < 1.0:
        patterns.append("some proof artifacts failed to compile in Lean")
    if _clip01(metrics.get("no_sorry")) < 1.0:
        patterns.append("remaining sorry placeholders still exist in the proof plan")
    if _clip01(metrics.get("statement_match", 1.0)) < 1.0:
        patterns.append("the generated theorem statement drifted away from the target")

    blocked_nodes = metrics.get("blocked_nodes", [])
    if blocked_nodes:
        patterns.append(f"blocked frontier remained at nodes: {', '.join(map(str, blocked_nodes[:4]))}")

    diagnostic = _first_diagnostic(metrics)
    if diagnostic:
        patterns.append(f"verifier diagnostic: {diagnostic[:240]}")
    return patterns[:4]


def _repair_hints(
    child_plan,
    parent_plan=None,
    metrics: Optional[Dict[str, Any]] = None,
    local_summary: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Turn successful deltas and island traces into short reusable hints."""
    hints: List[str] = []
    if parent_plan is not None:
        delta = summarize_plan_delta(parent_plan, child_plan)
        if delta["status_changes"]:
            hints.append(f"status improvements: {'; '.join(delta['status_changes'][:3])}")
        if delta["added_nodes"]:
            hints.append(f"new subgoals introduced: {', '.join(delta['added_nodes'][:4])}")
        if delta["sketch_updates"]:
            hints.append(f"updated proof sketches at: {', '.join(delta['sketch_updates'][:4])}")

    if metrics:
        frontier_nodes = metrics.get("frontier_nodes", [])
        if frontier_nodes:
            hints.append(f"next repair should focus on frontier nodes: {', '.join(map(str, frontier_nodes[:4]))}")

    if local_summary:
        history = local_summary.get("history", [])
        if history:
            best_step = max(
                history,
                key=lambda item: float(item.get("score", float("-inf")) or float("-inf")),
            )
            hints.append(
                "local-island best step "
                f"{best_step.get('local_step')} reached score {best_step.get('score')}"
            )
    return hints[:4]


def extract_experiences_from_program(
    program: Program,
    parent_program: Optional[Program] = None,
    max_items: int = 4,
) -> List[ExperienceObject]:
    """Distill a generated program into a compact experience set."""
    child_plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
    parent_plan = None
    if parent_program is not None:
        try:
            parent_plan = parse_program_plan(parent_program, fallback_theorem=child_plan.theorem_statement)
        except Exception:
            parent_plan = None

    metrics = program.metrics or {}
    theorem_solved = _clip01(metrics.get("theorem_solved"))
    local_summary = program.meta.get("local_island_summary")
    verified_lemmas = _verified_lemma_ids(child_plan)

    outcome = "solved" if theorem_solved >= 1.0 else "unsolved"
    outcome_content = (
        f"Plan `{child_plan.plan_id}` for theorem `{child_plan.theorem_statement}` finished as {outcome}. "
        f"success_score={program.combined_score}, "
        f"search_score={program.effective_search_score}, "
        f"compile_success={metrics.get('compile_success')}, "
        f"no_sorry={metrics.get('no_sorry')}, "
        f"statement_match={metrics.get('statement_match')}"
    )

    experiences: List[ExperienceObject] = [
        ExperienceObject(
            exp_id=f"{program.id}_outcome",
            exp_type="success_summary" if theorem_solved >= 1.0 else "failure_pattern",
            content=outcome_content,
            source_problem=child_plan.theorem_statement,
            source_plan=child_plan.plan_id,
            verifier_outcome=outcome,
            failure_patterns=[] if theorem_solved >= 1.0 else _failure_patterns(metrics),
            repair_hints=_repair_hints(
                child_plan,
                parent_plan=parent_plan,
                metrics=metrics,
                local_summary=local_summary if isinstance(local_summary, dict) else None,
            ),
            reusable_lemmas=verified_lemmas,
            metadata={
                "success_score": program.combined_score,
                "search_score": program.effective_search_score,
                "theorem_solved": theorem_solved,
            },
        )
    ]

    if verified_lemmas:
        experiences.append(
            ExperienceObject(
                exp_id=f"{program.id}_lemmas",
                exp_type="lemma_memory",
                content=f"Verified reusable lemmas: {', '.join(verified_lemmas)}",
                source_problem=child_plan.theorem_statement,
                source_plan=child_plan.plan_id,
                verifier_outcome=outcome,
                reusable_lemmas=verified_lemmas,
                repair_hints=["consider reusing the verified lemmas before replanning"],
                metadata={"lemma_count": len(verified_lemmas)},
            )
        )

    unique: List[ExperienceObject] = []
    seen_content = set()
    for experience in experiences:
        if experience.content in seen_content:
            continue
        seen_content.add(experience.content)
        unique.append(experience)
        if len(unique) >= max_items:
            break
    return unique


def attach_experiences_to_program(
    program: Program,
    experiences: List[ExperienceObject],
    *,
    max_items: int = 8,
) -> Program:
    """Write experiences back into both Program.meta and the normalized plan object."""
    if not experiences:
        return program

    plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
    existing = load_experiences(plan)

    merged: List[ExperienceObject] = []
    seen_ids = set()
    seen_content = set()
    for experience in [*existing, *experiences]:
        if experience.exp_id in seen_ids or experience.content in seen_content:
            continue
        seen_ids.add(experience.exp_id)
        seen_content.add(experience.content)
        merged.append(experience)

    merged = merged[-max_items:]
    serialized = [experience.model_dump(mode="json") for experience in merged]
    plan.metadata["experiences"] = serialized
    plan.experience_refs = [experience.exp_id for experience in merged]

    program.meta["plan_object"] = plan.model_dump(mode="json")
    program.meta["experience_pool"] = serialized
    program.meta["experience_refs"] = plan.experience_refs
    program.code = dump_plan(plan)
    return program


def extract_and_attach_experiences(
    program: Program,
    parent_program: Optional[Program] = None,
    *,
    max_items: int = 8,
) -> Program:
    """Convenience helper that distills a program and writes the memory back in place."""
    experiences = extract_experiences_from_program(
        program,
        parent_program=parent_program,
        max_items=max_items,
    )
    return attach_experiences_to_program(program, experiences, max_items=max_items)


def format_experiences_for_prompt(program: Optional[Program], max_items: int = 4) -> str:
    """Serialize attached experience memory into a compact prompt section."""
    if program is None:
        return "None"

    try:
        plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
        experiences = load_experiences(plan)
    except Exception:
        return "None"

    if not experiences:
        return "None"

    sections: List[str] = []
    for index, experience in enumerate(experiences[-max_items:], start=1):
        parts = [
            f"- Experience {index}: {experience.exp_type}",
            f"  content: {experience.content}",
        ]
        if experience.failure_patterns:
            parts.append(f"  failure_patterns: {'; '.join(experience.failure_patterns[:3])}")
        if experience.repair_hints:
            parts.append(f"  repair_hints: {'; '.join(experience.repair_hints[:3])}")
        if experience.reusable_lemmas:
            parts.append(f"  reusable_lemmas: {', '.join(experience.reusable_lemmas[:4])}")
        sections.append("\n".join(parts))
    return "\n".join(sections)


def build_repair_agenda(program: Optional[Program]) -> Dict[str, Any]:
    """
    Build a small mutation policy from the current parent plan state.

    This converts verifier metrics and attached experiences into an explicit
    next-step control signal for the mutation model.
    """
    if program is None:
        return {
            "mutation_mode": "refine_plan",
            "primary_targets": [],
            "avoid_patterns": [],
            "repair_hints": [],
            "reusable_lemmas": [],
        }

    try:
        plan = parse_program_plan(program, fallback_theorem=program.meta.get("theorem_statement"))
        experiences = load_experiences(plan)
    except Exception:
        return {
            "mutation_mode": "refine_plan",
            "primary_targets": [],
            "avoid_patterns": [],
            "repair_hints": [],
            "reusable_lemmas": [],
        }

    metrics = program.metrics or {}
    blocked_nodes = list(metrics.get("blocked_nodes", []) or [])
    frontier_nodes = list(metrics.get("frontier_nodes", []) or [])
    reusable_lemmas = _verified_lemma_ids(plan)

    avoid_patterns: List[str] = []
    repair_hints: List[str] = []
    for experience in experiences[-4:]:
        avoid_patterns.extend(experience.failure_patterns)
        repair_hints.extend(experience.repair_hints)

    if _clip01(metrics.get("statement_match", 1.0)) < 1.0:
        mutation_mode = "recover_statement"
    elif _clip01(metrics.get("no_sorry", 1.0)) < 1.0:
        mutation_mode = "eliminate_sorry"
    elif _clip01(metrics.get("compile_success", 1.0)) < 1.0:
        mutation_mode = "repair_compile"
    elif blocked_nodes:
        mutation_mode = "unblock_dependencies"
    elif reusable_lemmas and _clip01(metrics.get("theorem_solved", 0.0)) < 1.0:
        mutation_mode = "reuse_verified_lemmas"
    elif frontier_nodes:
        mutation_mode = "advance_frontier"
    else:
        mutation_mode = "refine_plan"

    primary_targets: List[str] = []
    if mutation_mode == "recover_statement":
        primary_targets.append("restore the exact target theorem statement before any further expansion")
    if mutation_mode == "eliminate_sorry":
        primary_targets.append("replace sorry in the highest-value frontier proof first")
    if mutation_mode == "repair_compile":
        primary_targets.append("fix Lean-compilation errors before changing plan topology")
    if blocked_nodes:
        primary_targets.append(f"unblock nodes: {', '.join(map(str, blocked_nodes[:4]))}")
    if frontier_nodes:
        primary_targets.append(f"push frontier nodes: {', '.join(map(str, frontier_nodes[:4]))}")
    if reusable_lemmas:
        primary_targets.append(f"reuse verified lemmas: {', '.join(reusable_lemmas[:4])}")

    if not primary_targets:
        primary_targets.append("make one concrete local improvement without expanding the plan unnecessarily")

    return {
        "mutation_mode": mutation_mode,
        "primary_targets": primary_targets[:5],
        "avoid_patterns": avoid_patterns[:4],
        "repair_hints": repair_hints[:4],
        "reusable_lemmas": reusable_lemmas[:4],
    }


def format_repair_agenda_for_prompt(program: Optional[Program]) -> str:
    """Serialize the derived mutation policy into a prompt-friendly section."""
    agenda = build_repair_agenda(program)
    lines = [f"- mutation_mode: {agenda['mutation_mode']}"]

    primary_targets = agenda.get("primary_targets", [])
    if primary_targets:
        lines.append(f"- primary_targets: {'; '.join(primary_targets)}")

    avoid_patterns = agenda.get("avoid_patterns", [])
    if avoid_patterns:
        lines.append(f"- avoid_patterns: {'; '.join(avoid_patterns)}")

    repair_hints = agenda.get("repair_hints", [])
    if repair_hints:
        lines.append(f"- repair_hints: {'; '.join(repair_hints)}")

    reusable_lemmas = agenda.get("reusable_lemmas", [])
    if reusable_lemmas:
        lines.append(f"- reusable_lemmas: {', '.join(reusable_lemmas)}")

    return "\n".join(lines)
