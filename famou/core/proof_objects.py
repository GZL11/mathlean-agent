"""
Math-proof specific structured proof objects.

These models provide a typed representation for plan-centric formal proof
search while remaining lightweight enough to fit the current Famou MVP.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanStatus(str, Enum):
    """High-level lifecycle status for a proof plan."""

    DRAFT = "draft"
    RUNNING = "running"
    SOLVED = "solved"
    STALLED = "stalled"
    EXHAUSTED = "exhausted"
    DISCARDED = "discarded"


class SubgoalStatus(str, Enum):
    """Execution / verification status for a subgoal node."""

    PENDING = "pending"
    ATTEMPTED = "attempted"
    IN_PROGRESS = "in_progress"
    PROVED = "proved"
    VERIFIED = "verified"
    FAILED = "failed"
    BLOCKED = "blocked"

    def progress_value(self) -> float:
        """Map a symbolic node status to a coarse PRM-style progress value."""
        return {
            SubgoalStatus.PENDING: 0.0,
            SubgoalStatus.BLOCKED: 0.05,
            SubgoalStatus.ATTEMPTED: 0.25,
            SubgoalStatus.IN_PROGRESS: 0.6,
            SubgoalStatus.PROVED: 0.85,
            SubgoalStatus.VERIFIED: 1.0,
            SubgoalStatus.FAILED: 0.0,
        }[self]


class SubgoalNode(BaseModel):
    """A single node in a proof-plan DAG."""

    node_id: str = Field(description="Unique node identifier within one plan")
    node_type: str = Field(default="subgoal", description="theorem/lemma/decomposition/repair")
    statement_nl: Optional[str] = Field(default=None, description="Natural language subgoal")
    statement_formal: str = Field(description="Formal or semi-formal target statement")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisite node ids")
    dependents: List[str] = Field(default_factory=list, description="Reverse edges, optional")
    proof_sketch: Optional[str] = Field(default=None, description="Local proof idea")
    local_lean_draft: Optional[str] = Field(default=None, description="Local Lean draft")
    local_status: SubgoalStatus = Field(default=SubgoalStatus.PENDING)
    verifier_state: Dict[str, Any] = Field(default_factory=dict)
    attempt_count: int = Field(default=0, ge=0, description="How many repair attempts have targeted this node")
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list, description="Compact per-attempt summaries")
    last_diagnostic: Optional[str] = Field(default=None, description="Most recent short verifier diagnostic")
    retrieved_lemma_ids: List[str] = Field(default_factory=list, description="Lemma ids retrieved for the latest repairs")
    retrieved_experience_ids: List[str] = Field(default_factory=list, description="Experience ids retrieved for the latest repairs")
    last_repair_summary: Optional[str] = Field(default=None, description="Short natural-language summary of the latest repair action")
    node_score: float = Field(default=0.0, ge=0.0, le=1.0)
    importance_weight: float = Field(default=1.0, gt=0.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)

    @property
    def is_complete(self) -> bool:
        """Whether this node is already established strongly enough to unblock dependents."""
        return self.local_status in {SubgoalStatus.PROVED, SubgoalStatus.VERIFIED}


class LemmaObject(BaseModel):
    """Reusable lemma candidate discovered inside one or more plans."""

    lemma_id: str
    statement_formal: str
    statement_nl: Optional[str] = None
    proof_status: SubgoalStatus = Field(default=SubgoalStatus.PENDING)
    reusable_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_plan_id: Optional[str] = None
    source_node_ids: List[str] = Field(default_factory=list)
    lean_code: Optional[str] = None
    used_by: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)

    @property
    def is_verified(self) -> bool:
        """Whether the lemma is established strongly enough to be reused."""
        return self.proof_status in {SubgoalStatus.PROVED, SubgoalStatus.VERIFIED}


class ExperienceObject(BaseModel):
    """Condensed reusable experience distilled from prior plan attempts."""

    exp_id: str
    exp_type: str = Field(default="plan_feedback", description="e.g. failure_pattern / repair_hint")
    content: str
    source_problem: Optional[str] = None
    source_plan: Optional[str] = None
    verifier_outcome: Optional[str] = None
    failure_patterns: List[str] = Field(default_factory=list)
    repair_hints: List[str] = Field(default_factory=list)
    reusable_lemmas: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)


class PlanObject(BaseModel):
    """Top-level population individual for math-proof evolution."""

    plan_id: str
    theorem_id: Optional[str] = None
    theorem_statement: str
    theorem_name: Optional[str] = None
    plan_summary: Optional[str] = None
    strategy_type: str = Field(default="decomposition-first")
    sketch: Optional[str] = None
    subgoals: List[SubgoalNode] = Field(default_factory=list)
    lemma_pool: List[LemmaObject] = Field(default_factory=list)
    lean_artifacts: Dict[str, Any] = Field(default_factory=dict)
    frontier_nodes: List[str] = Field(default_factory=list)
    verifier_feedback: Dict[str, Any] = Field(default_factory=dict)
    combined_score: float = Field(default=0.0, ge=0.0, le=1.0)
    parent_plan_ids: List[str] = Field(default_factory=list)
    experience_refs: List[str] = Field(default_factory=list)
    status: PlanStatus = Field(default=PlanStatus.DRAFT)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def validate_unique_node_ids(self) -> "PlanObject":
        """Enforce local identifier uniqueness before graph validation runs elsewhere."""
        node_ids = [node.node_id for node in self.subgoals]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("PlanObject.subgoals contains duplicate node_id values")
        return self

    def node_map(self) -> Dict[str, SubgoalNode]:
        """Convenience mapping for node lookup."""
        return {node.node_id: node for node in self.subgoals}
