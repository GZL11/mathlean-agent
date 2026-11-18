"""Fixed-plan subgoal repair generation module for math-proof search."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.utils.id_gen import generate_program_id
from famou.utils.plan_runtime import (
    compute_combined_plan_score,
    dump_plan,
    parse_program_plan,
)
from famou.utils.proof_experience import (
    build_repair_agenda,
    format_experiences_for_prompt,
)
from famou.utils.global_memory_store import (
    format_global_memory_for_prompt,
    retrieve_failed_lemma_memories,
    retrieve_lemma_memories,
    retrieve_subgoal_memories,
)
from famou.utils.subgoal_repair_runtime import (
    SubgoalRepairPatch,
    compute_plan_topology_signature,
    merge_subgoal_repair_patch,
)
from famou.utils.subgoal_selection import select_subgoal_for_repair


class SubgoalRepairGenerate(GenerateModule):
    """
    Generate a node-local repair patch for a fixed proof plan.

    Unlike plan mutation, this module must preserve theorem statement, subgoal ids,
    and dependency graph. The LLM returns a structured JSON patch for one selected
    subgoal, and the runtime merges that patch back into the parent plan.
    """

    def build_prompt(
        self,
        parent: Program,
        context: Context,
        result: RolloutResult,
        *,
        selection: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a fixed-plan repair prompt for a selected subgoal."""
        parent_plan = parse_program_plan(parent, fallback_theorem=context.task_description)
        selection = selection or select_subgoal_for_repair(parent_plan)
        node = parent_plan.node_map()[selection["selected_node_id"]]
        dependency_nodes = self._dependency_nodes(parent_plan, node.dependencies)
        dependency_context = self._format_dependency_context(dependency_nodes)
        dependency_preconditions = [
            dependency_node.statement_formal
            for dependency_node in dependency_nodes
            if dependency_node.statement_formal
        ]
        local_lemmas = self._format_local_lemmas(parent_plan)
        parent_experiences = format_experiences_for_prompt(parent)
        repair_agenda = build_repair_agenda(parent)
        recent_attempt_history = self._format_recent_attempt_history(node)
        sticky_repair_note = self._build_sticky_repair_note(selection, node)
        repeated_failure_note = self._build_repeated_failure_note(selection, node)
        repeated_failure_count = int(selection.get("repeated_failure_count", 0) or 0)
        preferred_action_types = self._preferred_action_types(node, repeated_failure_count)
        global_memory_dir = context.config.get("global_memory_dir")
        subgoal_memories = []
        reusable_lemma_memories = []
        failed_lemma_memories = []
        lemma_query = self._build_lemma_query(
            theorem_statement=parent_plan.theorem_statement,
            node=node,
            dependency_preconditions=dependency_preconditions,
        )
        if global_memory_dir:
            subgoal_memories = retrieve_subgoal_memories(
                global_memory_dir,
                statement_formal=node.statement_formal,
                diagnostic=node.last_diagnostic,
                limit=5 if repeated_failure_count >= 2 else 3,
                dependency_ids=node.dependencies,
                preferred_action_types=preferred_action_types,
            )
            reusable_lemma_memories = retrieve_lemma_memories(
                global_memory_dir,
                query_text=lemma_query,
                limit=6 if repeated_failure_count >= 2 else 4,
                require_verified=True,
                statement_formal=node.statement_formal,
                dependency_ids=node.dependencies,
                preconditions=dependency_preconditions,
                diagnostic=node.last_diagnostic,
            )
            failed_lemma_memories = retrieve_failed_lemma_memories(
                global_memory_dir,
                query_text=lemma_query,
                limit=4 if repeated_failure_count >= 2 else 2,
                statement_formal=node.statement_formal,
                dependency_ids=node.dependencies,
                preconditions=dependency_preconditions,
                diagnostic=node.last_diagnostic,
            )

        result.stats["selected_subgoal_id"] = node.node_id
        result.stats["selected_subgoal_reason"] = selection["selection_reason"]
        result.stats["candidate_subgoal_ids"] = selection["candidate_node_ids"]
        result.stats["selected_subgoal_attempt_count"] = node.attempt_count
        result.stats["selected_subgoal_repeated_failure_count"] = repeated_failure_count
        result.stats["selected_subgoal_sticky_mode"] = selection.get("sticky_mode")
        result.stats["retrieved_global_subgoal_memory_ids"] = [
            row.get("memory_id") for row in subgoal_memories
        ]
        result.stats["retrieved_global_reusable_lemma_memory_ids"] = [
            row.get("memory_id") for row in reusable_lemma_memories
        ]
        result.stats["retrieved_global_failed_lemma_memory_ids"] = [
            row.get("memory_id") for row in failed_lemma_memories
        ]
        result.stats["retrieved_global_lemma_memory_ids"] = [
            row.get("memory_id")
            for row in [*reusable_lemma_memories, *failed_lemma_memories]
        ]

        template_name = self.config.get("template_generate", "generation/subgoal_repair.txt")
        return prompt_registry.get(
            template_name,
            theorem_statement=parent_plan.theorem_statement,
            task_constraint=context.task_constraint,
            frozen_plan=dump_plan(parent_plan),
            topology_signature=compute_plan_topology_signature(parent_plan),
            selected_node_id=node.node_id,
            selected_statement=node.statement_formal,
            selected_status=node.local_status.value,
            selected_proof_sketch=node.proof_sketch or "None",
            selected_local_lean_draft=node.local_lean_draft or "None",
            selected_last_diagnostic=node.last_diagnostic or "None",
            recent_attempt_history=recent_attempt_history,
            sticky_repair_note=sticky_repair_note,
            repeated_failure_note=repeated_failure_note,
            dependency_context=dependency_context,
            local_lemmas=local_lemmas,
            frontier_nodes=selection["frontier_node_ids"],
            blocked_nodes=selection["blocked_node_ids"],
            repair_agenda=repair_agenda,
            parent_experiences=parent_experiences,
            global_subgoal_memories=format_global_memory_for_prompt(subgoal_memories, kind="subgoal"),
            global_lemma_memories=self._format_lemma_memory_sections(
                reusable_lemma_memories,
                failed_lemma_memories,
            ),
        )

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Generate one node-local repair and merge it into a child fixed-plan program."""
        parent = context.get_program_by_id(result.selected_parent_id)
        if parent is None:
            raise ValueError(
                f"Parent program {result.selected_parent_id} not found in population. "
                "Cannot generate without a valid parent."
            )

        parent_plan = parse_program_plan(parent, fallback_theorem=context.task_description)
        selection = select_subgoal_for_repair(parent_plan)
        selected_node_id = selection["selected_node_id"]
        prompt = self.build_prompt(parent, context, result, selection=selection)

        llm_kwargs = {}
        if "temperature" in self.config:
            llm_kwargs["temperature"] = self.config["temperature"]
        if "max_tokens" in self.config:
            llm_kwargs["max_tokens"] = self.config["max_tokens"]

        response = self.llm_client.generate(
            prompt=prompt,
            system=context.task_description,
            **llm_kwargs,
        )

        patch_payload = extract_json(response.text)
        if not patch_payload:
            raise ValueError("Subgoal repair generation did not return a valid JSON patch")
        patch_payload = self._normalize_patch_payload(patch_payload)
        patch_payload.setdefault("node_id", selected_node_id)
        if patch_payload["node_id"] != selected_node_id:
            raise ValueError(
                f"Repair patch targeted node '{patch_payload['node_id']}', "
                f"expected selected node '{selected_node_id}'"
            )
        patch = SubgoalRepairPatch.model_validate(patch_payload)

        new_program_id = generate_program_id(
            generation=parent.generation + 1,
            iteration=result.iteration,
        )
        child_plan = merge_subgoal_repair_patch(parent_plan, patch)
        if not child_plan.parent_plan_ids:
            child_plan.parent_plan_ids = [parent.id]
        if not child_plan.experience_refs:
            child_plan.experience_refs = list(parent_plan.experience_refs)

        plan_metrics = compute_combined_plan_score(child_plan, reward_mode="inner")
        child_plan.combined_score = plan_metrics["search_score"]
        child_plan.metadata["success_score"] = plan_metrics["success_score"]
        child_plan.metadata["search_score"] = plan_metrics["search_score"]
        child_plan.metadata["selected_subgoal_id"] = selected_node_id
        child_plan.metadata["selection_reason"] = selection["selection_reason"]
        child_plan.metadata["candidate_subgoal_ids"] = selection["candidate_node_ids"]

        result.generated_program = Program(
            id=new_program_id,
            code=dump_plan(child_plan),
            generation=parent.generation + 1,
            iteration=result.iteration,
            language=self.config.get("language", context.language),
            parent_ids=[parent.id],
            system_prompt=context.task_description,
            prompt=prompt,
            response=response.text,
            thinking=response.thinking,
            meta={
                "experiment_id": context.experiment_id,
                "parent_score": parent.effective_search_score,
                "parent_success_score": parent.success_score,
                "plan_object": child_plan.model_dump(mode="json"),
                "plan_metrics": plan_metrics,
                "theorem_statement": child_plan.theorem_statement,
                "strategy_type": child_plan.strategy_type,
                "experience_refs": child_plan.experience_refs,
                "selected_subgoal_id": selected_node_id,
                "selection_reason": selection["selection_reason"],
                "candidate_subgoal_ids": selection["candidate_node_ids"],
                "fixed_plan_topology_signature": compute_plan_topology_signature(child_plan),
                "repair_patch": patch.model_dump(mode="json"),
            },
            created_at=time.time(),
        )
        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that generation produced a fixed-plan child program."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a fixed-plan repair program. "
                "Check LLM client and prompt configuration."
            )
        child_plan = parse_program_plan(
            result.generated_program,
            fallback_theorem=context.task_description,
        )
        if "fixed_plan_topology_signature" not in child_plan.metadata:
            raise ValueError(f"{self.name}: Child plan is missing fixed-plan topology metadata")

    def _dependency_nodes(self, plan, dependency_ids: List[str]) -> List[Any]:
        """Collect dependency nodes for one selected subgoal."""
        if not dependency_ids:
            return []
        node_map = plan.node_map()
        nodes = []
        for dependency_id in dependency_ids:
            node = node_map.get(dependency_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def _format_dependency_context(self, dependency_nodes: List[Any]) -> str:
        """Serialize completed dependency nodes that provide context for repair."""
        if not dependency_nodes:
            return "None"
        sections = []
        for node in dependency_nodes:
            sections.append(
                "\n".join(
                    [
                        f"- node_id: {node.node_id}",
                        f"  statement_formal: {node.statement_formal}",
                        f"  status: {node.local_status.value}",
                        f"  proof_sketch: {node.proof_sketch or 'None'}",
                        f"  local_lean_draft: {node.local_lean_draft or 'None'}",
                    ]
                )
            )
        return "\n".join(sections) if sections else "None"

    def _format_local_lemmas(self, plan) -> str:
        """Serialize locally available lemmas for prompt grounding."""
        if not plan.lemma_pool:
            return "None"
        sections = []
        for lemma in plan.lemma_pool:
            sections.append(
                "\n".join(
                    [
                        f"- lemma_id: {lemma.lemma_id}",
                        f"  statement_formal: {lemma.statement_formal}",
                        f"  proof_status: {lemma.proof_status.value}",
                        f"  reusable_score: {lemma.reusable_score}",
                    ]
                )
            )
        return "\n".join(sections)

    def _format_lemma_memory_sections(
        self,
        reusable_rows: List[Dict[str, Any]],
        failed_rows: List[Dict[str, Any]],
    ) -> str:
        """Separate reusable lemma assets from failed debug traces inside one prompt block."""
        return "\n".join(
            [
                "Reusable lemma memories:",
                format_global_memory_for_prompt(reusable_rows, kind="lemma"),
                "",
                "Failed lemma attempts:",
                format_global_memory_for_prompt(failed_rows, kind="lemma"),
            ]
        )

    def _normalize_patch_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize common LLM alias values before strict patch validation."""
        normalized = dict(payload)
        status_aliases = {
            "complete": "verified",
            "completed": "verified",
            "done": "verified",
            "solved": "verified",
            "successful": "verified",
            "proof_complete": "verified",
            "proof_completed": "verified",
            "working": "in_progress",
            "drafted": "attempted",
            "partial": "attempted",
            "partially_verified": "proved",
            "compilable": "proved",
        }
        updated_status = normalized.get("updated_status")
        if isinstance(updated_status, str):
            canonical = status_aliases.get(updated_status.strip().lower())
            if canonical is not None:
                normalized["updated_status"] = canonical
        return normalized

    def _format_recent_attempt_history(self, node) -> str:
        """Summarize the most recent local repair attempts for prompt grounding."""
        if not node.attempt_history:
            return "None"
        sections = []
        for entry in node.attempt_history[-3:]:
            sections.append(
                "\n".join(
                    [
                        f"- attempt_index: {entry.get('attempt_index', 'N/A')}",
                        f"  previous_status: {entry.get('previous_status', 'N/A')}",
                        f"  new_status: {entry.get('new_status', 'N/A')}",
                        f"  has_proof_sketch: {entry.get('has_proof_sketch', False)}",
                        f"  has_local_lean_draft: {entry.get('has_local_lean_draft', False)}",
                        f"  introduced_lemma_count: {entry.get('introduced_lemma_count', 0)}",
                        f"  retrieved_lemma_ids: {entry.get('retrieved_lemma_ids', [])}",
                        f"  retrieved_experience_ids: {entry.get('retrieved_experience_ids', [])}",
                    ]
                )
            )
        return "\n".join(sections)

    def _build_sticky_repair_note(self, selection: Dict[str, Any], node) -> str:
        """Explain why this node was kept for another inner-step repair."""
        sticky_mode = selection.get("sticky_mode")
        if not sticky_mode:
            return "None"
        if sticky_mode == "sticky_recent_progress":
            return (
                f"Continue repairing node {node.node_id} because the latest local attempt "
                "already produced useful proof state; prefer extending that draft instead of switching nodes."
            )
        if sticky_mode == "sticky_repeated_failure":
            return (
                f"Continue repairing node {node.node_id} locally because repeated unresolved attempts "
                "suggest the node needs diagnosis-driven repair before exploring other nodes."
            )
        return f"Continue repairing node {node.node_id} under sticky mode: {sticky_mode}."

    def _build_repeated_failure_note(self, selection: Dict[str, Any], node) -> str:
        """Surface repeated-failure state so the prompt can change tactics explicitly."""
        repeated_failure_count = int(selection.get("repeated_failure_count", 0) or 0)
        if repeated_failure_count < 2:
            return "None"
        return (
            f"This node has {repeated_failure_count} recent unresolved attempts. "
            f"Latest diagnostic: {node.last_diagnostic or 'None'}. "
            "Do not restate the same broken draft; change tactic, reuse better matching lemmas, "
            "or repair the dependency/interface mismatch explicitly."
        )

    def _preferred_action_types(self, node, repeated_failure_count: int) -> List[str]:
        """Bias subgoal retrieval toward actions that fit the current repair regime."""
        if repeated_failure_count >= 2:
            return ["repair", "dependency_fix", "introduce_lemma", "rewrite"]
        if node.attempt_count <= 0:
            return ["draft", "rewrite"]
        if node.local_status.value in {"attempted", "failed", "blocked"}:
            return ["repair", "dependency_fix", "introduce_lemma"]
        return ["repair", "rewrite", "introduce_lemma", "draft"]

    def _build_lemma_query(
        self,
        *,
        theorem_statement: str,
        node,
        dependency_preconditions: List[str],
    ) -> str:
        """Build a slightly richer lemma retrieval query for repeated local repair."""
        parts = [
            theorem_statement,
            node.statement_formal,
            node.proof_sketch or "",
            node.local_lean_draft or "",
            node.last_diagnostic or "",
            " ".join(dependency_preconditions),
        ]
        return "\n".join(part for part in parts if part)
