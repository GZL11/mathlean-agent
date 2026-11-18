"""
Utilities for fixed-plan subgoal repair.

Phase 1 uses this module to enforce the critical invariant that inner-loop
repairs may update node-local proof state, but may not change the plan's
topology or target theorem.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from famou.core.proof_objects import LemmaObject, PlanObject, SubgoalStatus
from famou.utils.plan_runtime import get_frontier_node_ids


class SubgoalRepairPatch(BaseModel):
    """Structured patch returned by a subgoal repair generator."""

    node_id: str
    updated_proof_sketch: Optional[str] = None
    updated_local_lean_draft: Optional[str] = None
    updated_status: Optional[SubgoalStatus] = None
    verifier_state_update: Dict[str, Any] = Field(default_factory=dict)
    introduced_lemmas: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_lemma_ids: List[str] = Field(default_factory=list)
    retrieved_experience_ids: List[str] = Field(default_factory=list)
    last_diagnostic: Optional[str] = None
    repair_summary: Optional[str] = None

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def compute_plan_topology_signature(plan: PlanObject) -> str:
    """
    Compute a stable signature for plan structure.

    The signature intentionally ignores local proof-state fields such as drafts,
    verifier diagnostics, node scores, and attempt history. It only captures the
    invariants that must remain fixed inside a fixed-plan inner rollout.
    """

    nodes = [
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "statement_formal": node.statement_formal,
            "dependencies": sorted(node.dependencies),
        }
        for node in plan.subgoals
    ]
    nodes.sort(key=lambda item: item["node_id"])
    topology = {
        "theorem_statement": plan.theorem_statement,
        "strategy_type": plan.strategy_type,
        "nodes": nodes,
    }
    payload = json.dumps(topology, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_plan_topology_unchanged(parent_plan: PlanObject, child_plan: PlanObject) -> None:
    """Raise when a supposed fixed-plan repair changed structural invariants."""

    if parent_plan.theorem_statement != child_plan.theorem_statement:
        raise ValueError("Fixed-plan repair cannot change theorem_statement")
    if parent_plan.strategy_type != child_plan.strategy_type:
        raise ValueError("Fixed-plan repair cannot change strategy_type")

    parent_node_ids = {node.node_id for node in parent_plan.subgoals}
    child_node_ids = {node.node_id for node in child_plan.subgoals}
    if parent_node_ids != child_node_ids:
        raise ValueError("Fixed-plan repair cannot add, remove, or rename subgoals")

    parent_node_map = parent_plan.node_map()
    child_node_map = child_plan.node_map()
    for node_id in sorted(parent_node_ids):
        parent_node = parent_node_map[node_id]
        child_node = child_node_map[node_id]
        if parent_node.statement_formal != child_node.statement_formal:
            raise ValueError(f"Fixed-plan repair cannot change statement_formal for node '{node_id}'")
        if parent_node.node_type != child_node.node_type:
            raise ValueError(f"Fixed-plan repair cannot change node_type for node '{node_id}'")
        if sorted(parent_node.dependencies) != sorted(child_node.dependencies):
            raise ValueError(f"Fixed-plan repair cannot change dependencies for node '{node_id}'")

    if compute_plan_topology_signature(parent_plan) != compute_plan_topology_signature(child_plan):
        raise ValueError("Fixed-plan repair changed the topology signature unexpectedly")


def merge_subgoal_repair_patch(
    parent_plan: PlanObject,
    patch: SubgoalRepairPatch | Dict[str, Any],
) -> PlanObject:
    """
    Apply a node-local repair patch while preserving plan topology.

    The returned child plan is a deep copy of the parent plan with only the
    selected node's local proof state updated. New local lemmas may be appended
    or refreshed in the plan lemma pool.
    """

    validated_patch = (
        patch if isinstance(patch, SubgoalRepairPatch) else SubgoalRepairPatch.model_validate(patch)
    )
    child_plan = parent_plan.model_copy(deep=True)
    node_map = child_plan.node_map()
    if validated_patch.node_id not in node_map:
        raise ValueError(f"Unknown subgoal node_id '{validated_patch.node_id}' in repair patch")

    node = node_map[validated_patch.node_id]
    previous_status = node.local_status

    if validated_patch.updated_proof_sketch is not None:
        node.proof_sketch = validated_patch.updated_proof_sketch
    if validated_patch.updated_local_lean_draft is not None:
        node.local_lean_draft = validated_patch.updated_local_lean_draft
    if validated_patch.updated_status is not None:
        node.local_status = validated_patch.updated_status
    if validated_patch.verifier_state_update:
        node.verifier_state = {
            **dict(node.verifier_state),
            **dict(validated_patch.verifier_state_update),
        }
    if validated_patch.last_diagnostic is not None:
        node.last_diagnostic = validated_patch.last_diagnostic
    if validated_patch.repair_summary is not None:
        node.last_repair_summary = validated_patch.repair_summary

    node.retrieved_lemma_ids = _merge_unique(
        node.retrieved_lemma_ids,
        validated_patch.retrieved_lemma_ids,
    )
    node.retrieved_experience_ids = _merge_unique(
        node.retrieved_experience_ids,
        validated_patch.retrieved_experience_ids,
    )
    node.attempt_count += 1
    node.attempt_history.append(
        {
            "attempt_index": node.attempt_count,
            "previous_status": previous_status.value,
            "new_status": node.local_status.value,
            "has_proof_sketch": bool((node.proof_sketch or "").strip()),
            "has_local_lean_draft": bool((node.local_lean_draft or "").strip()),
            "introduced_lemma_count": len(validated_patch.introduced_lemmas),
            "retrieved_lemma_ids": list(validated_patch.retrieved_lemma_ids),
            "retrieved_experience_ids": list(validated_patch.retrieved_experience_ids),
        }
    )

    if validated_patch.introduced_lemmas:
        child_plan.lemma_pool = _merge_lemmas(child_plan.lemma_pool, validated_patch.introduced_lemmas)

    child_plan.frontier_nodes = get_frontier_node_ids(child_plan)
    child_plan.metadata["fixed_plan_topology_signature"] = compute_plan_topology_signature(child_plan)
    child_plan.metadata["last_repaired_node_id"] = validated_patch.node_id
    child_plan.metadata["inner_repair_mode"] = "fixed_plan"

    assert_plan_topology_unchanged(parent_plan, child_plan)
    return child_plan


def _merge_unique(existing: List[str], additions: List[str]) -> List[str]:
    """Append unique identifiers while preserving order."""
    merged = list(existing)
    seen = set(existing)
    for item in additions:
        text = str(item)
        if text in seen:
            continue
        merged.append(text)
        seen.add(text)
    return merged


def _merge_lemmas(existing: List[LemmaObject], introduced: List[Dict[str, Any]]) -> List[LemmaObject]:
    """Merge introduced lemmas by lemma_id."""
    merged = [lemma.model_copy(deep=True) for lemma in existing]
    index = {lemma.lemma_id: pos for pos, lemma in enumerate(merged)}
    for raw_lemma in introduced:
        lemma = LemmaObject.model_validate(raw_lemma)
        position = index.get(lemma.lemma_id)
        if position is None:
            merged.append(lemma)
            index[lemma.lemma_id] = len(merged) - 1
            continue
        current = merged[position]
        merged[position] = current.model_copy(
            update={
                "statement_formal": lemma.statement_formal,
                "statement_nl": lemma.statement_nl or current.statement_nl,
                "proof_status": lemma.proof_status,
                "reusable_score": max(current.reusable_score, lemma.reusable_score),
                "source_plan_id": lemma.source_plan_id or current.source_plan_id,
                "source_node_ids": _merge_unique(current.source_node_ids, lemma.source_node_ids),
                "lean_code": lemma.lean_code or current.lean_code,
                "used_by": _merge_unique(current.used_by, lemma.used_by),
                "metadata": {
                    **dict(current.metadata),
                    **dict(lemma.metadata),
                },
            },
            deep=True,
        )
    return merged
