"""
Runtime utilities for plan-centric math-proof evolution.

This module keeps the first MVP lightweight:
- parse structured plan text from LLM output or stored Program.code
- validate the subgoal DAG
- derive frontier / topology information
- compute PRM-style reward breakdown and combined plan score
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from famou.core.data import Program
from famou.core.proof_objects import (
    ExperienceObject,
    LemmaObject,
    PlanObject,
    SubgoalNode,
    SubgoalStatus,
)

try:
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None


PLAN_BLOCK_RE = re.compile(r"```(?:yaml|yml|json|plan)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
YAML_SCALAR_LINE_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):\s+(.*)$")
YAML_QUOTED_KEYS = {
    "theorem_statement",
    "theorem_name",
    "plan_summary",
    "strategy_type",
    "sketch",
    "statement",
    "statement_formal",
    "statement_nl",
    "proof_sketch",
    "last_diagnostic",
    "last_repair_summary",
    "local_status",
    "status",
    "node_type",
    "lemma_id",
    "exp_type",
    "content",
}


def extract_plan_text(raw_text: str) -> str:
    """Extract the structured plan block from markdown, or return the raw text."""
    if not raw_text:
        return ""
    match = PLAN_BLOCK_RE.search(raw_text)
    if match:
        return match.group(1).strip()

    stripped = raw_text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    first_line = lines[0].strip().lower()
    if not first_line.startswith("```"):
        return stripped

    body_lines = lines[1:]
    closing_index = None
    for index in range(len(body_lines) - 1, -1, -1):
        if body_lines[index].strip() == "```":
            closing_index = index
            break

    if closing_index is not None:
        body_lines = body_lines[:closing_index]

    return "\n".join(body_lines).strip()


def _yaml_needs_quoted_scalar(key: str, value: str) -> bool:
    """Heuristically detect YAML scalar lines that should be quoted."""
    stripped = value.strip()
    if not stripped:
        return False
    if stripped in {"|", ">"}:
        return False
    if stripped[0] in "\"'[{":
        return False
    if stripped.startswith("- "):
        return False
    if key not in YAML_QUOTED_KEYS:
        return False
    return ": " in stripped


def _quote_yaml_scalar(value: str) -> str:
    """Quote a YAML scalar conservatively using JSON string escaping."""
    return json.dumps(value.strip(), ensure_ascii=False)


def _sanitize_yaml_scalars(plan_text: str) -> str:
    """Quote common plain scalars that contain colons and break YAML parsing."""
    sanitized_lines: List[str] = []
    for line in plan_text.splitlines():
        match = YAML_SCALAR_LINE_RE.match(line)
        if not match:
            sanitized_lines.append(line)
            continue
        indent, key, value = match.groups()
        if _yaml_needs_quoted_scalar(key, value):
            sanitized_lines.append(f"{indent}{key}: {_quote_yaml_scalar(value)}")
        else:
            sanitized_lines.append(line)
    return "\n".join(sanitized_lines)


def load_plan_payload(raw_text: str) -> Dict[str, Any]:
    """Parse JSON / YAML plan text into a dictionary."""
    plan_text = extract_plan_text(raw_text)
    if not plan_text:
        raise ValueError("Plan text is empty")

    stripped = plan_text.lstrip()
    if stripped.startswith("{"):
        payload = json.loads(plan_text)
    elif yaml is not None:
        try:
            payload = yaml.safe_load(plan_text)
        except Exception:
            sanitized_plan_text = _sanitize_yaml_scalars(plan_text)
            payload = yaml.safe_load(sanitized_plan_text)
    else:
        raise ValueError("Plan text is not JSON and PyYAML is unavailable")

    if not isinstance(payload, dict):
        raise ValueError("Structured plan must decode into a dictionary")
    return payload


def _normalize_subgoal(raw_node: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Normalize aliases and defaults for one subgoal entry."""
    if not isinstance(raw_node, dict):
        raise ValueError(f"Subgoal at index {index} must be a mapping, got {type(raw_node)}")

    dependencies = raw_node.get("dependencies", raw_node.get("deps", []))
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    dependencies = list(dependencies or [])

    dependents = raw_node.get("dependents", [])
    if isinstance(dependents, str):
        dependents = [dependents]

    statement_formal = (
        raw_node.get("statement_formal")
        or raw_node.get("statement")
        or raw_node.get("formal_statement")
        or raw_node.get("goal")
    )
    if not statement_formal:
        raise ValueError(f"Subgoal at index {index} is missing statement_formal/statement/goal")

    node_id = raw_node.get("node_id") or raw_node.get("id") or raw_node.get("name") or f"sg_{index}"
    raw_status = raw_node.get("local_status", raw_node.get("status", "pending"))

    normalized: Dict[str, Any] = {
        "node_id": str(node_id),
        "node_type": raw_node.get("node_type", raw_node.get("type", "subgoal")),
        "statement_nl": raw_node.get("statement_nl", raw_node.get("description")),
        "statement_formal": str(statement_formal),
        "dependencies": [str(dep) for dep in dependencies],
        "dependents": [str(dep) for dep in dependents],
        "proof_sketch": raw_node.get("proof_sketch", raw_node.get("sketch")),
        "local_lean_draft": raw_node.get("local_lean_draft", raw_node.get("lean_code")),
        "local_status": raw_status,
        "verifier_state": dict(raw_node.get("verifier_state", {})),
        "attempt_count": int(raw_node.get("attempt_count", 0) or 0),
        "attempt_history": list(raw_node.get("attempt_history", [])),
        "last_diagnostic": raw_node.get("last_diagnostic"),
        "retrieved_lemma_ids": list(raw_node.get("retrieved_lemma_ids", [])),
        "retrieved_experience_ids": list(raw_node.get("retrieved_experience_ids", [])),
        "last_repair_summary": raw_node.get("last_repair_summary"),
        "node_score": float(raw_node.get("node_score", raw_node.get("score", 0.0)) or 0.0),
        "importance_weight": float(raw_node.get("importance_weight", 1.0) or 1.0),
        "tags": list(raw_node.get("tags", [])),
        "metadata": dict(raw_node.get("metadata", {})),
    }
    return normalized


def _normalize_lemma(raw_lemma: Dict[str, Any], index: int, default_plan_id: str) -> Dict[str, Any]:
    """Normalize lemma aliases and defaults."""
    if not isinstance(raw_lemma, dict):
        raise ValueError(f"Lemma at index {index} must be a mapping, got {type(raw_lemma)}")

    lemma_id = raw_lemma.get("lemma_id") or raw_lemma.get("id") or raw_lemma.get("name") or f"lemma_{index}"
    statement = (
        raw_lemma.get("statement_formal")
        or raw_lemma.get("statement")
        or raw_lemma.get("formal_statement")
    )
    if not statement:
        raise ValueError(f"Lemma at index {index} is missing statement_formal/statement")

    source_node_ids = raw_lemma.get("source_node_ids", raw_lemma.get("source_subgoal_ids", []))
    if isinstance(source_node_ids, str):
        source_node_ids = [source_node_ids]

    return {
        "lemma_id": str(lemma_id),
        "statement_formal": str(statement),
        "statement_nl": raw_lemma.get("statement_nl"),
        "proof_status": raw_lemma.get("proof_status", raw_lemma.get("status", "pending")),
        "reusable_score": float(raw_lemma.get("reusable_score", raw_lemma.get("score", 0.0)) or 0.0),
        "source_plan_id": raw_lemma.get("source_plan_id", default_plan_id),
        "source_node_ids": [str(node_id) for node_id in source_node_ids or []],
        "lean_code": raw_lemma.get("lean_code"),
        "used_by": list(raw_lemma.get("used_by", [])),
        "metadata": dict(raw_lemma.get("metadata", {})),
    }


def _normalize_experience(raw_exp: Dict[str, Any], index: int, default_plan_id: str) -> Dict[str, Any]:
    """Normalize experience object aliases and defaults."""
    if not isinstance(raw_exp, dict):
        raise ValueError(f"Experience at index {index} must be a mapping, got {type(raw_exp)}")

    exp_id = raw_exp.get("exp_id") or raw_exp.get("id") or f"exp_{index}"
    content = raw_exp.get("content") or raw_exp.get("summary")
    if not content:
        raise ValueError(f"Experience at index {index} is missing content/summary")

    return {
        "exp_id": str(exp_id),
        "exp_type": raw_exp.get("exp_type", raw_exp.get("type", "plan_feedback")),
        "content": str(content),
        "source_problem": raw_exp.get("source_problem"),
        "source_plan": raw_exp.get("source_plan", default_plan_id),
        "verifier_outcome": raw_exp.get("verifier_outcome"),
        "failure_patterns": list(raw_exp.get("failure_patterns", [])),
        "repair_hints": list(raw_exp.get("repair_hints", [])),
        "reusable_lemmas": list(raw_exp.get("reusable_lemmas", [])),
        "metadata": dict(raw_exp.get("metadata", {})),
    }


def normalize_plan_payload(payload: Dict[str, Any], default_plan_id: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a raw structured payload into PlanObject-compatible fields."""
    plan_id = payload.get("plan_id") or payload.get("id") or default_plan_id or "plan_0"
    theorem_statement = (
        payload.get("theorem_statement")
        or payload.get("statement")
        or payload.get("goal")
        or payload.get("target")
    )
    if not theorem_statement:
        raise ValueError("Plan payload is missing theorem_statement/statement/goal")

    normalized_subgoals = [
        _normalize_subgoal(raw_node, index + 1)
        for index, raw_node in enumerate(payload.get("subgoals", []))
    ]

    normalized_lemmas = [
        _normalize_lemma(raw_lemma, index + 1, str(plan_id))
        for index, raw_lemma in enumerate(payload.get("lemma_pool", payload.get("lemmas", [])))
    ]

    normalized_experience = [
        _normalize_experience(raw_exp, index + 1, str(plan_id))
        for index, raw_exp in enumerate(payload.get("experience_pool", payload.get("experiences", [])))
    ]

    return {
        "plan_id": str(plan_id),
        "theorem_id": payload.get("theorem_id"),
        "theorem_statement": str(theorem_statement),
        "theorem_name": payload.get("theorem_name"),
        "plan_summary": payload.get("plan_summary", payload.get("summary")),
        "strategy_type": payload.get("strategy_type", "decomposition-first"),
        "sketch": payload.get("sketch"),
        "subgoals": normalized_subgoals,
        "lemma_pool": normalized_lemmas,
        "frontier_nodes": list(payload.get("frontier_nodes", [])),
        "verifier_feedback": dict(payload.get("verifier_feedback", {})),
        "combined_score": float(payload.get("combined_score", 0.0) or 0.0),
        "parent_plan_ids": list(payload.get("parent_plan_ids", payload.get("parents", []))),
        "experience_refs": list(payload.get("experience_refs", payload.get("experience_ids", []))),
        "status": payload.get("status", "draft"),
        "metadata": {
            **dict(payload.get("metadata", {})),
            "experiences": normalized_experience,
        },
        "lean_artifacts": dict(payload.get("lean_artifacts", {})),
    }


def parse_plan(raw_text: str, default_plan_id: Optional[str] = None) -> PlanObject:
    """Parse raw structured text into a validated PlanObject."""
    payload = load_plan_payload(raw_text)
    plan = PlanObject.model_validate(normalize_plan_payload(payload, default_plan_id=default_plan_id))
    validate_plan_graph(plan)
    plan.frontier_nodes = get_frontier_node_ids(plan)
    return plan


def parse_program_plan(program: Program, fallback_theorem: Optional[str] = None) -> PlanObject:
    """
    Parse Program.code/meta into a PlanObject.

    If the program does not already contain structured plan text, degrade
    gracefully by wrapping the code into a one-node seed plan.
    """
    plan_payload = program.meta.get("plan_object")
    if isinstance(plan_payload, dict):
        plan = PlanObject.model_validate(plan_payload)
        validate_plan_graph(plan)
        plan.frontier_nodes = get_frontier_node_ids(plan)
        return plan

    try:
        return parse_plan(program.code, default_plan_id=program.id)
    except Exception:
        theorem_statement = fallback_theorem or program.code.strip() or "Unspecified theorem"
        seed = PlanObject(
            plan_id=program.id,
            theorem_statement=theorem_statement,
            plan_summary="Auto-wrapped seed plan from unstructured program content.",
            subgoals=[
                SubgoalNode(
                    node_id="sg_1",
                    statement_nl="Initial seed goal",
                    statement_formal=theorem_statement,
                    proof_sketch=program.code.strip() or None,
                )
            ],
        )
        seed.frontier_nodes = get_frontier_node_ids(seed)
        return seed


def validate_plan_graph(plan: PlanObject) -> None:
    """Validate subgoal references and ensure the graph is acyclic."""
    node_map = plan.node_map()

    for node in plan.subgoals:
        for dep in node.dependencies:
            if dep not in node_map:
                raise ValueError(f"Node '{node.node_id}' depends on unknown node '{dep}'")
            if dep == node.node_id:
                raise ValueError(f"Node '{node.node_id}' cannot depend on itself")

    state: Dict[str, int] = {}

    def dfs(node_id: str) -> None:
        state[node_id] = 1
        node = node_map[node_id]
        for dep_id in node.dependencies:
            dep_state = state.get(dep_id, 0)
            if dep_state == 1:
                raise ValueError(f"Cycle detected involving '{node_id}' and '{dep_id}'")
            if dep_state == 0:
                dfs(dep_id)
        state[node_id] = 2

    for node_id in node_map:
        if state.get(node_id, 0) == 0:
            dfs(node_id)


def build_dependents(plan: PlanObject) -> Dict[str, List[str]]:
    """Construct reverse edges from dependency declarations."""
    dependents: Dict[str, List[str]] = defaultdict(list)
    for node in plan.subgoals:
        for dep in node.dependencies:
            dependents[dep].append(node.node_id)
    return {key: sorted(value) for key, value in dependents.items()}


def topological_order(plan: PlanObject) -> List[str]:
    """Return a dependency-respecting node order."""
    validate_plan_graph(plan)
    indegree = {node.node_id: len(node.dependencies) for node in plan.subgoals}
    dependents = build_dependents(plan)
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: List[str] = []

    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for child_id in dependents.get(node_id, []):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)

    if len(order) != len(plan.subgoals):
        raise ValueError("Failed to compute topological order due to a graph inconsistency")
    return order


def dependency_importance(plan: PlanObject) -> Dict[str, float]:
    """
    Estimate topology-aware importance.

    Nodes that unlock more downstream structure receive higher weight.
    """
    dependents = build_dependents(plan)
    order = topological_order(plan)
    descendants: Dict[str, int] = {node.node_id: 0 for node in plan.subgoals}

    for node_id in reversed(order):
        child_ids = dependents.get(node_id, [])
        if not child_ids:
            descendants[node_id] = 0
            continue
        seen: set[str] = set()
        queue = deque(child_ids)
        while queue:
            child_id = queue.popleft()
            if child_id in seen:
                continue
            seen.add(child_id)
            queue.extend(dependents.get(child_id, []))
        descendants[node_id] = len(seen)

    total_nodes = max(len(plan.subgoals), 1)
    return {
        node.node_id: node.importance_weight * (1.0 + descendants[node.node_id] / total_nodes)
        for node in plan.subgoals
    }


def get_frontier_nodes(plan: PlanObject) -> List[SubgoalNode]:
    """Return currently actionable nodes whose dependencies are already complete."""
    node_map = plan.node_map()
    frontier: List[SubgoalNode] = []
    for node in plan.subgoals:
        if node.is_complete:
            continue
        if all(node_map[dep].is_complete for dep in node.dependencies):
            frontier.append(node)
    return frontier


def get_frontier_node_ids(plan: PlanObject) -> List[str]:
    """Helper returning frontier node identifiers."""
    return [node.node_id for node in get_frontier_nodes(plan)]


def get_terminal_goal_ids(plan: PlanObject) -> List[str]:
    """Return top-level goals that are not prerequisites of another node."""
    dependents = build_dependents(plan)
    return sorted(node.node_id for node in plan.subgoals if not dependents.get(node.node_id))


def get_blocked_node_ids(plan: PlanObject) -> List[str]:
    """Return nodes that are not complete and still waiting on prerequisites."""
    node_map = plan.node_map()
    blocked = []
    for node in plan.subgoals:
        if node.is_complete:
            continue
        if any(not node_map[dep].is_complete for dep in node.dependencies):
            blocked.append(node.node_id)
    return sorted(blocked)


def _node_progress_value(node: SubgoalNode) -> float:
    """Combine discrete node status with optional fine-grained local score."""
    return max(node.local_status.progress_value(), node.node_score)


def compute_progress_score(plan: PlanObject) -> Dict[str, Any]:
    """Compute topology-aware progress statistics for one plan."""
    if not plan.subgoals:
        return {
            "progress_score": 0.0,
            "completed_subgoals": 0,
            "total_subgoals": 0,
            "frontier_size": 0,
            "blocked_size": 0,
            "terminal_success": 0.0,
            "node_scores": {},
        }

    weights = dependency_importance(plan)
    weight_sum = sum(weights.values()) or 1.0
    node_scores = {node.node_id: _node_progress_value(node) for node in plan.subgoals}
    weighted_progress = sum(node_scores[node.node_id] * weights[node.node_id] for node in plan.subgoals)

    terminal_ids = get_terminal_goal_ids(plan)
    terminal_success = (
        sum(node_scores[node_id] for node_id in terminal_ids) / max(len(terminal_ids), 1)
        if terminal_ids
        else 0.0
    )

    completed = sum(1 for node in plan.subgoals if node.is_complete)
    frontier_ids = get_frontier_node_ids(plan)
    blocked_ids = get_blocked_node_ids(plan)

    return {
        "progress_score": weighted_progress / weight_sum,
        "completed_subgoals": completed,
        "total_subgoals": len(plan.subgoals),
        "frontier_size": len(frontier_ids),
        "frontier_nodes": frontier_ids,
        "blocked_size": len(blocked_ids),
        "blocked_nodes": blocked_ids,
        "terminal_success": terminal_success,
        "node_scores": node_scores,
    }


def compute_reusable_lemma_bonus(plan: PlanObject) -> Dict[str, Any]:
    """Compute a small bonus for verified and reusable lemma objects."""
    if not plan.lemma_pool:
        return {
            "reusable_lemma_bonus": 0.0,
            "verified_lemmas": 0,
            "total_lemmas": 0,
        }

    verified = [lemma for lemma in plan.lemma_pool if lemma.is_verified]
    total = len(plan.lemma_pool)
    if not verified:
        return {
            "reusable_lemma_bonus": 0.0,
            "verified_lemmas": 0,
            "total_lemmas": total,
        }

    avg_reuse = sum(lemma.reusable_score for lemma in verified) / len(verified)
    density = len(verified) / total
    return {
        "reusable_lemma_bonus": min(1.0, avg_reuse * density),
        "verified_lemmas": len(verified),
        "total_lemmas": total,
    }


def _clip01(value: Any) -> float:
    """Safely coerce arbitrary numeric-like values into [0, 1]."""
    try:
        numeric = float(value or 0.0)
    except Exception:
        numeric = 0.0
    return min(1.0, max(0.0, numeric))


def _normalized_cost_ratio(metrics: Dict[str, Any], used_keys: List[str], budget_keys: List[str]) -> float:
    """Normalize a cost term if both used and budget values are available."""
    used = None
    for key in used_keys:
        if key in metrics:
            used = metrics[key]
            break
    if used is None:
        return 0.0

    budget = None
    for key in budget_keys:
        if key in metrics:
            budget = metrics[key]
            break
    if budget in (None, 0):
        return 0.0

    try:
        return min(1.0, max(0.0, float(used) / float(budget)))
    except Exception:
        return 0.0


def compute_clean_score(plan: PlanObject, extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute how clean and complete the current proof object is."""
    metrics = dict(extra_metrics or {})
    total_nodes = max(len(plan.subgoals), 1)
    blocked_ratio = len(get_blocked_node_ids(plan)) / total_nodes
    dead_node_ratio = _clip01(metrics.get("dead_node_ratio", 0.0))

    sorry_density = metrics.get("sorry_density")
    if sorry_density is None:
        no_sorry = metrics.get("no_sorry")
        sorry_density = 1.0 - _clip01(no_sorry) if no_sorry is not None else 0.0
    sorry_density = _clip01(sorry_density)

    clean_score = 1.0 - (0.6 * sorry_density + 0.25 * blocked_ratio + 0.15 * dead_node_ratio)
    clean_score = min(1.0, max(0.0, clean_score))

    return {
        "clean_score": clean_score,
        "sorry_density": sorry_density,
        "blocked_ratio": blocked_ratio,
        "dead_node_ratio": dead_node_ratio,
    }


def compute_cost_penalty(extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute a small normalized penalty for excessive search cost."""
    metrics = dict(extra_metrics or {})

    verifier_ratio = _normalized_cost_ratio(
        metrics,
        ["verifier_calls", "verifier_call_count"],
        ["verifier_budget", "max_verifier_calls"],
    )
    token_ratio = _normalized_cost_ratio(
        metrics,
        ["token_usage", "tokens_used"],
        ["token_budget", "max_tokens"],
    )
    repair_ratio = _normalized_cost_ratio(
        metrics,
        ["repair_steps", "repair_count", "local_iterations"],
        ["repair_budget", "max_repair_steps", "inner_loop_budget"],
    )

    cost_penalty = min(0.2, 0.03 * verifier_ratio + 0.02 * token_ratio + 0.02 * repair_ratio)

    return {
        "cost_penalty": cost_penalty,
        "verifier_cost_ratio": verifier_ratio,
        "token_cost_ratio": token_ratio,
        "repair_cost_ratio": repair_ratio,
    }


def compute_memory_gain_bonus(
    plan: PlanObject,
    lemma_bonus: Dict[str, Any],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Estimate how much reusable memory this plan contributes back to search.

    The current implementation uses verified lemma signal as the main proxy and
    allows callers to inject an explicit experience gain override later.
    """
    metrics = dict(extra_metrics or {})
    explicit_bonus = metrics.get("memory_gain_bonus")
    if explicit_bonus is not None:
        return {"memory_gain_bonus": _clip01(explicit_bonus)}

    verified_lemmas = int(lemma_bonus.get("verified_lemmas", 0) or 0)
    reusable_bonus = _clip01(lemma_bonus.get("reusable_lemma_bonus", 0.0))
    experience_gain = _clip01(metrics.get("experience_gain", 0.0))
    verified_lemma_ratio = min(1.0, verified_lemmas / 2.0)

    memory_gain_bonus = min(
        1.0,
        0.55 * reusable_bonus + 0.35 * verified_lemma_ratio + 0.10 * experience_gain,
    )
    return {"memory_gain_bonus": memory_gain_bonus}


def compute_plan_complexity_penalty(plan: PlanObject) -> Dict[str, float]:
    """Apply a small penalty once plans grow beyond the intended compact regime."""
    total_subgoals = len(plan.subgoals)
    total_lemmas = len(load_lemmas(plan))

    excess_subgoals = max(0, total_subgoals - 4)
    excess_lemmas = max(0, total_lemmas - 2)
    penalty = min(0.12, 0.02 * excess_subgoals + 0.015 * excess_lemmas)

    return {
        "plan_complexity_penalty": penalty,
        "subgoal_complexity_excess": float(excess_subgoals),
        "lemma_complexity_excess": float(excess_lemmas),
    }


def compute_reward_breakdown(
    plan: PlanObject,
    extra_metrics: Optional[Dict[str, Any]] = None,
    reward_mode: str = "outer",
) -> Dict[str, Any]:
    """Compute the full reward breakdown used by math-proof search."""
    progress = compute_progress_score(plan)
    lemma_bonus = compute_reusable_lemma_bonus(plan)
    metrics = dict(extra_metrics or {})
    clean = compute_clean_score(plan, metrics)
    cost = compute_cost_penalty(metrics)
    memory_gain = compute_memory_gain_bonus(plan, lemma_bonus, metrics)
    complexity = compute_plan_complexity_penalty(plan)

    compile_success = _clip01(metrics.get("compile_success", metrics.get("compilation_success", 0.0)))
    no_sorry = _clip01(metrics.get("no_sorry", 0.0))
    statement_match = _clip01(metrics.get("statement_match", 1.0))

    proof_validity = metrics.get("proof_validity", metrics.get("validity"))
    if proof_validity is None:
        proof_validity = 1.0 if (
            progress["terminal_success"] >= 1.0 and statement_match >= 1.0 and no_sorry >= 1.0
        ) else 0.0
    proof_validity = _clip01(proof_validity)

    theorem_solved = metrics.get("theorem_solved")
    if theorem_solved is None:
        theorem_solved = 1.0 if (
            proof_validity >= 1.0 and statement_match >= 1.0 and no_sorry >= 1.0
        ) else 0.0
    theorem_solved = _clip01(theorem_solved)

    if reward_mode == "inner":
        weights = {
            "final": 0.15,
            "node_progress": 0.55,
            "terminal": 0.10,
            "lemma": 0.10,
            "clean": 0.10,
        }
    else:
        weights = {
            "final": 0.55,
            "node_progress": 0.25,
            "terminal": 0.10,
            "lemma": 0.05,
            "clean": 0.05,
        }

    final_component = theorem_solved
    node_progress_component = progress["progress_score"]
    terminal_component = progress["terminal_success"]
    lemma_component = lemma_bonus["reusable_lemma_bonus"]
    clean_component = clean["clean_score"]
    realization_score = progress["progress_score"]
    memory_gain_bonus = memory_gain["memory_gain_bonus"]
    plan_complexity_penalty = complexity["plan_complexity_penalty"]
    quality_score = 0.5 * lemma_component + 0.5 * clean_component

    search_score = (
        weights["final"] * final_component
        + weights["node_progress"] * node_progress_component
        + weights["terminal"] * terminal_component
        + weights["lemma"] * lemma_component
        + weights["clean"] * clean_component
        - cost["cost_penalty"]
    )
    if reward_mode == "outer":
        search_score += 0.05 * memory_gain_bonus
        search_score -= plan_complexity_penalty
    search_score = min(1.0, max(0.0, search_score))
    success_score = theorem_solved
    combined_score = success_score

    verifier_signal = max(proof_validity, compile_success, no_sorry)

    return {
        "combined_score": combined_score,
        "success_score": success_score,
        "search_score": search_score,
        "reward_mode": reward_mode,
        "theorem_solved": theorem_solved,
        "proof_validity": proof_validity,
        "compile_success": compile_success,
        "no_sorry": no_sorry,
        "statement_match": statement_match,
        "final_component": final_component,
        "node_progress_component": node_progress_component,
        "terminal_component": terminal_component,
        "lemma_component": lemma_component,
        "clean_component": clean_component,
        "realization_score": realization_score,
        "memory_gain_bonus": memory_gain_bonus,
        "plan_complexity_penalty": plan_complexity_penalty,
        "quality_score": quality_score,
        "cost_penalty": cost["cost_penalty"],
        "verifier_cost_ratio": cost["verifier_cost_ratio"],
        "token_cost_ratio": cost["token_cost_ratio"],
        "repair_cost_ratio": cost["repair_cost_ratio"],
        "progress_score": progress["progress_score"],
        "terminal_success": progress["terminal_success"],
        "frontier_size": progress["frontier_size"],
        "blocked_size": progress["blocked_size"],
        "completed_subgoals": progress["completed_subgoals"],
        "total_subgoals": progress["total_subgoals"],
        "verified_lemmas": lemma_bonus["verified_lemmas"],
        "total_lemmas": lemma_bonus["total_lemmas"],
        "reusable_lemma_bonus": lemma_component,
        "clean_score": clean["clean_score"],
        "sorry_density": clean["sorry_density"],
        "blocked_ratio": clean["blocked_ratio"],
        "dead_node_ratio": clean["dead_node_ratio"],
        "subgoal_complexity_excess": complexity["subgoal_complexity_excess"],
        "lemma_complexity_excess": complexity["lemma_complexity_excess"],
        "verifier_signal": verifier_signal,
        "frontier_nodes": progress["frontier_nodes"],
        "blocked_nodes": progress["blocked_nodes"],
        "node_scores": progress["node_scores"],
    }


def compute_combined_plan_score(
    plan: PlanObject,
    extra_metrics: Optional[Dict[str, Any]] = None,
    reward_mode: str = "outer",
) -> Dict[str, Any]:
    """Backward-compatible wrapper around the full reward breakdown."""
    return compute_reward_breakdown(plan, extra_metrics=extra_metrics, reward_mode=reward_mode)


def plan_metrics_for_program(program: Program, fallback_theorem: Optional[str] = None) -> Dict[str, Any]:
    """Parse a program as a plan and return a flat metric dictionary."""
    plan = parse_program_plan(program, fallback_theorem=fallback_theorem)
    plan_score = compute_combined_plan_score(plan, extra_metrics=program.metrics)
    return {
        "plan_id": plan.plan_id,
        "plan_object": plan.model_dump(mode="json"),
        **plan_score,
    }


def dump_plan(plan: PlanObject) -> str:
    """Serialize a normalized plan for storage back into Program.code."""
    payload = plan.model_dump(mode="json")
    experiences = payload.get("metadata", {}).pop("experiences", [])
    if experiences:
        payload["experiences"] = experiences

    if yaml is not None:
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    return json.dumps(payload, ensure_ascii=True, indent=2)


def summarize_plan_delta(parent: PlanObject, child: PlanObject) -> Dict[str, Any]:
    """Summarize what changed between two plans for judge feedback."""
    parent_nodes = parent.node_map()
    child_nodes = child.node_map()

    added_nodes = sorted(set(child_nodes) - set(parent_nodes))
    removed_nodes = sorted(set(parent_nodes) - set(child_nodes))
    status_changes = []
    sketch_updates = []

    for node_id in sorted(set(child_nodes) & set(parent_nodes)):
        parent_node = parent_nodes[node_id]
        child_node = child_nodes[node_id]
        if parent_node.local_status != child_node.local_status:
            status_changes.append(
                f"{node_id}: {parent_node.local_status.value} -> {child_node.local_status.value}"
            )
        if (parent_node.proof_sketch or "").strip() != (child_node.proof_sketch or "").strip():
            sketch_updates.append(node_id)

    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "status_changes": status_changes,
        "sketch_updates": sketch_updates,
    }


def load_experiences(plan: PlanObject) -> List[ExperienceObject]:
    """Read normalized experiences from PlanObject.metadata."""
    raw_experiences = plan.metadata.get("experiences", [])
    return [ExperienceObject.model_validate(exp) for exp in raw_experiences]


def load_lemmas(plan: PlanObject) -> List[LemmaObject]:
    """Convenience wrapper kept for symmetry with experience loading."""
    return [LemmaObject.model_validate(lemma.model_dump(mode="json")) for lemma in plan.lemma_pool]
