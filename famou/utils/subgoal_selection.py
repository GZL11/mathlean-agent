"""Deterministic subgoal selection for fixed-plan inner repair."""

from __future__ import annotations

from typing import Any, Dict, List

from famou.core.proof_objects import PlanObject, SubgoalNode, SubgoalStatus
from famou.utils.plan_runtime import dependency_importance, get_blocked_node_ids, get_frontier_node_ids


def select_subgoal_for_repair(plan: PlanObject) -> Dict[str, Any]:
    """
    Select one unresolved subgoal for the next inner repair step.

    Priority order for the first Phase 1 implementation:
    1. Nodes explicitly marked as blocked in verifier feedback
    2. Frontier nodes whose dependencies are already complete
    3. Remaining unresolved nodes
    Within each bucket, rank by dependency importance, diagnostic severity,
    attempt count, and node status in a deterministic way.
    """

    node_map = plan.node_map()
    feedback_blocked = [
        node_id
        for node_id in _feedback_blocked_node_ids(plan)
        if node_id in node_map and not node_map[node_id].is_complete
    ]
    frontier_ids = [
        node_id
        for node_id in get_frontier_node_ids(plan)
        if node_id not in feedback_blocked and not node_map[node_id].is_complete
    ]
    unresolved_ids = [
        node.node_id
        for node in plan.subgoals
        if not node.is_complete and node.node_id not in feedback_blocked and node.node_id not in frontier_ids
    ]

    ordered_ids = feedback_blocked + frontier_ids + sorted(unresolved_ids)
    if not ordered_ids:
        raise ValueError("No unresolved subgoal is available for repair")

    importance = dependency_importance(plan)
    ranked = sorted(
        ordered_ids,
        key=lambda node_id: _candidate_sort_key(
            node_map[node_id],
            blocked_priority=0 if node_id in feedback_blocked else (1 if node_id in frontier_ids else 2),
            importance=importance.get(node_id, 0.0),
        ),
    )

    sticky_candidate = _sticky_repair_candidate(plan, node_map)
    if sticky_candidate and sticky_candidate["node_id"] in ranked:
        selected_node_id = sticky_candidate["node_id"]
        ranked = [selected_node_id] + [node_id for node_id in ranked if node_id != selected_node_id]
        selection_reason = sticky_candidate["reason"]
    else:
        selected_node_id = ranked[0]
        selection_reason = (
            "blocked_by_verifier_feedback"
            if selected_node_id in feedback_blocked
            else "frontier_ready"
            if selected_node_id in frontier_ids
            else "highest_remaining_priority"
        )
        sticky_candidate = None

    selected_node = node_map[selected_node_id]
    repeated_failure_count = _repeated_failure_count(selected_node)

    return {
        "selected_node_id": selected_node_id,
        "selection_reason": selection_reason,
        "candidate_node_ids": ranked,
        "frontier_node_ids": frontier_ids,
        "blocked_node_ids": feedback_blocked,
        "sticky_candidate_id": sticky_candidate["node_id"] if sticky_candidate else None,
        "sticky_mode": sticky_candidate["reason"] if sticky_candidate else None,
        "recent_attempt_count": selected_node.attempt_count,
        "repeated_failure_count": repeated_failure_count,
    }


def _feedback_blocked_node_ids(plan: PlanObject) -> List[str]:
    """Prefer verifier-provided blocked nodes when available, otherwise topology-blocked nodes."""
    verifier_blocked = plan.verifier_feedback.get("blocked_nodes", [])
    if verifier_blocked:
        return sorted(str(node_id) for node_id in verifier_blocked)
    metadata_blocked = plan.metadata.get("blocked_nodes", [])
    if metadata_blocked:
        return sorted(str(node_id) for node_id in metadata_blocked)
    return get_blocked_node_ids(plan)


def _candidate_sort_key(node: SubgoalNode, blocked_priority: int, importance: float) -> tuple:
    """Deterministic ordering key for repair candidates."""
    diagnostic_severity = _diagnostic_severity(node)
    status_priority = _status_priority(node.local_status)
    return (
        blocked_priority,
        -importance,
        -diagnostic_severity,
        node.attempt_count,
        status_priority,
        node.node_id,
    )


def _sticky_repair_candidate(
    plan: PlanObject,
    node_map: Dict[str, SubgoalNode],
) -> Dict[str, Any] | None:
    """Prefer continuing the most recent unresolved repair when it has useful local momentum."""
    recent_node_id = (
        plan.metadata.get("last_repaired_node_id")
        or plan.metadata.get("selected_subgoal_id")
        or plan.metadata.get("last_selected_subgoal_id")
    )
    if not recent_node_id:
        return None
    node = node_map.get(str(recent_node_id))
    if node is None or node.is_complete or node.attempt_count <= 0:
        return None

    if _has_recent_progress(node):
        return {"node_id": node.node_id, "reason": "sticky_recent_progress"}

    repeated_failure_count = _repeated_failure_count(node)
    if repeated_failure_count >= 2:
        return {"node_id": node.node_id, "reason": "sticky_repeated_failure"}
    return None


def _has_recent_progress(node: SubgoalNode) -> bool:
    """Approximate whether the last repair created usable local proof state worth continuing."""
    if node.local_status in {SubgoalStatus.IN_PROGRESS, SubgoalStatus.PROVED}:
        return True
    if not node.attempt_history:
        return False
    last_attempt = node.attempt_history[-1]
    if str(last_attempt.get("new_status") or "") in {"in_progress", "proved", "verified"}:
        return True
    return bool(
        int(last_attempt.get("introduced_lemma_count", 0) or 0) > 0
    )


def _repeated_failure_count(node: SubgoalNode) -> int:
    """Count recent unresolved attempts so repeated-failure repairs can stay local."""
    if node.is_complete or node.attempt_count < 2:
        return 0
    unresolved_statuses = {"attempted", "failed", "blocked", "in_progress"}
    statuses = [
        str(entry.get("new_status") or "")
        for entry in node.attempt_history[-3:]
        if isinstance(entry, dict)
    ]
    repeated = len([status for status in statuses if status in unresolved_statuses])
    if repeated >= 2:
        return repeated
    if node.local_status.value in unresolved_statuses and node.last_diagnostic:
        return node.attempt_count
    return 0


def _diagnostic_severity(node: SubgoalNode) -> int:
    """Approximate severity using latest diagnostic and stored diagnostics length."""
    diagnostics = node.verifier_state.get("diagnostics", [])
    count = len(diagnostics) if isinstance(diagnostics, list) else 0
    if node.last_diagnostic:
        count += 1
    return count


def _status_priority(status: SubgoalStatus) -> int:
    """Prefer nodes that have already received some repair context before untouched nodes."""
    order = {
        SubgoalStatus.BLOCKED: 0,
        SubgoalStatus.IN_PROGRESS: 1,
        SubgoalStatus.ATTEMPTED: 2,
        SubgoalStatus.PENDING: 3,
        SubgoalStatus.FAILED: 4,
        SubgoalStatus.PROVED: 5,
        SubgoalStatus.VERIFIED: 6,
    }
    return order[status]
