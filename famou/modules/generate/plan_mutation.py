"""Plan-centric mutation module for math-proof search."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
from famou.utils.proof_experience import (
    build_repair_agenda,
    format_experiences_for_prompt,
    format_repair_agenda_for_prompt,
)
from famou.utils.global_memory_store import (
    format_global_memory_for_prompt,
    retrieve_lemma_memories,
    retrieve_theorem_memories,
)
from famou.utils.plan_runtime import (
    compute_combined_plan_score,
    dump_plan,
    load_lemmas,
    parse_plan,
    parse_program_plan,
)


class PlanMutationGenerate(GenerateModule):
    """
    Mutate a structured proof plan instead of raw executable code.

    This is the compatibility bridge for the current Famou pipeline:
    - `Program.code` stores the normalized structured plan text
    - `Program.meta["plan_object"]` stores the parsed PlanObject payload
    - later evaluator / judge stages can stay on the existing Program API
    """

    def _build_island_specialization_directive(self, context: Context) -> Dict[str, Any]:
        """Read the island-level outer-search role from runtime config."""
        specialization = dict(context.config.get("island_specialization", {}) or {})
        if not specialization:
            return {
                "mode": "generalist",
                "reason": "single_island_or_unspecialized_runtime",
                "instruction": "No island-specific bias is active; follow verifier evidence normally.",
                "enabled": False,
                "island_id": context.island_id,
            }

        return {
            "mode": str(specialization.get("mode") or "generalist"),
            "reason": str(specialization.get("reason") or "unspecified"),
            "instruction": str(
                specialization.get("instruction")
                or "Apply the island's outer-search bias only as a soft prior."
            ),
            "enabled": True,
            "island_id": int(specialization.get("island_id", context.island_id) or context.island_id),
            "num_islands": int(specialization.get("num_islands", 0) or 0),
        }

    def _format_island_specialization_directive(self, directive: Dict[str, Any]) -> str:
        """Serialize island-level search specialization into the prompt."""
        lines = [
            f"- island_specialization_mode: {directive.get('mode', 'generalist')}",
            f"- island_specialization_reason: {directive.get('reason', 'unspecified')}",
            f"- island_id: {directive.get('island_id', 0)}",
        ]
        if directive.get("num_islands"):
            lines.append(f"- num_islands: {directive.get('num_islands')}")
        lines.append(f"- enabled: {bool(directive.get('enabled', False))}")
        lines.append(f"- instruction: {directive.get('instruction', 'follow verifier evidence')}")
        return "\n".join(lines)

    def _build_replanning_directive(
        self,
        parent: Program,
        theorem_memories: List[Dict[str, Any]],
        lemma_memories: List[Dict[str, Any]],
        island_specialization: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decide whether the next outer mutation should refine or replan from memory."""
        metrics = parent.metrics or {}
        theorem_solved = float(metrics.get("theorem_solved", parent.success_score) or 0.0)
        success_score = parent.success_score
        search_score = parent.effective_search_score
        progress_score = float(metrics.get("progress_score", search_score) or search_score or 0.0)
        frontier_size = float(metrics.get("frontier_size", 0.0) or 0.0)
        blocked_size = float(metrics.get("blocked_size", 0.0) or 0.0)

        local_summary = dict(parent.meta.get("local_island_summary", {}) or {})
        budget = int(parent.meta.get("local_island_budget") or local_summary.get("budget") or 0)
        steps_executed = int(local_summary.get("steps_executed") or 0)
        history = list(local_summary.get("history", []) or [])
        best_local_score = None
        if history:
            best_local_score = max(
                float(item.get("score", float("-inf")) or float("-inf"))
                for item in history
            )

        theorem_memory_ids = [
            str(row.get("memory_id"))
            for row in theorem_memories
            if row.get("memory_id")
        ]
        lemma_memory_ids = [
            str(row.get("memory_id"))
            for row in lemma_memories
            if row.get("memory_id")
        ]

        evidence: List[str] = []
        reasons: List[str] = []
        if budget > 0 and steps_executed >= budget and theorem_solved < 1.0:
            reasons.append("local_budget_exhausted")
            evidence.append(f"local island exhausted its {budget}-step budget without solving the theorem")
        if blocked_size > 0.0 and frontier_size <= 0.0 and theorem_solved < 1.0:
            reasons.append("blocked_without_frontier")
            evidence.append("the current plan is blocked and has no remaining frontier nodes to advance")
        if (
            best_local_score is not None
            and best_local_score <= search_score + 1e-9
            and progress_score < 0.75
            and theorem_solved < 1.0
        ):
            reasons.append("local_search_stagnated")
            evidence.append("local island search did not improve the parent's search score")

        if theorem_solved >= 1.0 or success_score >= 1.0:
            mode = "refine_parent"
            reason = "already_solved_parent"
        elif not theorem_memory_ids and not lemma_memory_ids:
            mode = "refine_parent"
            reason = "no_matching_global_memory"
        elif reasons:
            mode = "replan_from_memory"
            reason = reasons[0]
        else:
            mode = "refine_parent"
            reason = "parent_still_promising"

        specialization_mode = str((island_specialization or {}).get("mode", "") or "")
        if (
            specialization_mode == "memory_guided_replanning"
            and theorem_solved < 1.0
            and success_score < 1.0
            and (theorem_memory_ids or lemma_memory_ids)
            and mode == "refine_parent"
            and (blocked_size > 0.0 or frontier_size <= 1.0 or progress_score < 0.55)
        ):
            mode = "replan_from_memory"
            reason = "island_prefers_memory_guided_replanning"
            evidence.append(
                "the island specialization prefers memory-backed restructuring when the parent shows weak progress"
            )
        elif (
            specialization_mode == "conservative_refinement"
            and mode == "replan_from_memory"
            and reason == "local_search_stagnated"
            and frontier_size > 0.0
            and progress_score >= 0.35
        ):
            mode = "refine_parent"
            reason = "island_prefers_conservative_refinement"
            evidence.append(
                "the island specialization keeps the current plan because the verifier still exposes an actionable frontier"
            )

        if mode == "replan_from_memory":
            evidence.append("use retrieved theorem-level outcomes and lemma memories to redesign the plan")
        else:
            evidence.append("keep the parent plan as the primary anchor and prefer local structural refinement")

        return {
            "mode": mode,
            "reason": reason,
            "evidence": evidence[:4],
            "budget": budget,
            "steps_executed": steps_executed,
            "success_score": success_score,
            "search_score": search_score,
            "progress_score": progress_score,
            "frontier_size": frontier_size,
            "blocked_size": blocked_size,
            "retrieved_theorem_memory_ids": theorem_memory_ids[:4],
            "retrieved_lemma_memory_ids": lemma_memory_ids[:6],
        }

    def _format_replanning_directive(self, directive: Dict[str, Any]) -> str:
        """Serialize the replanning decision into a prompt-friendly block."""
        lines = [
            f"- replanning_mode: {directive.get('mode', 'refine_parent')}",
            f"- replanning_reason: {directive.get('reason', 'unknown')}",
        ]
        evidence = list(directive.get("evidence", []) or [])
        if evidence:
            lines.append(f"- evidence: {'; '.join(evidence)}")
        theorem_memory_ids = list(directive.get("retrieved_theorem_memory_ids", []) or [])
        if theorem_memory_ids:
            lines.append(f"- retrieved_theorem_memory_ids: {', '.join(theorem_memory_ids)}")
        lemma_memory_ids = list(directive.get("retrieved_lemma_memory_ids", []) or [])
        if lemma_memory_ids:
            lines.append(f"- retrieved_lemma_memory_ids: {', '.join(lemma_memory_ids)}")
        if directive.get("mode") == "replan_from_memory":
            lines.append(
                "- instruction: you may replace the parent's subgoal decomposition with a fresh plan, "
                "but keep the target theorem unchanged and reuse only matching memories"
            )
        else:
            lines.append(
                "- instruction: treat the parent plan as the main anchor and prefer minimal structural changes"
            )
        return "\n".join(lines)

    def _build_crossover_directive(
        self,
        parent: Program,
        context: Context,
        inspiration_ids: List[str],
        replanning_directive: Dict[str, Any],
        island_specialization: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether to do plain mutation or explicit plan crossover.

        The controller remains lightweight: we still emit one child plan, but
        the prompt now explicitly asks the model to merge compatible structure
        from selected inspirations when they offer complementary signal.
        """
        if self.config.get("enable_plan_crossover", True) is False:
            return {
                "mode": "mutation_only",
                "reason": "plan_crossover_disabled",
                "selected_inspiration_ids": [],
                "evidence": [],
            }

        if not inspiration_ids:
            return {
                "mode": "mutation_only",
                "reason": "no_inspirations_available",
                "selected_inspiration_ids": [],
                "evidence": [],
            }

        parent_plan = parse_program_plan(parent, fallback_theorem=context.task_description)
        parent_statements = {
            str(node.statement_formal).strip()
            for node in parent_plan.subgoals
            if getattr(node, "statement_formal", None)
        }
        parent_lemmas = {
            str(lemma.lemma_id)
            for lemma in load_lemmas(parent_plan)
            if getattr(lemma, "is_verified", False)
        }
        parent_strategy = str(parent_plan.strategy_type or "").strip().lower()
        parent_search = float(parent.effective_search_score or 0.0)
        parent_success = float(parent.success_score or 0.0)
        parent_blocked = float((parent.metrics or {}).get("blocked_size", 0.0) or 0.0)
        replanning_mode = str(replanning_directive.get("mode", "refine_parent"))

        candidates: List[Dict[str, Any]] = []
        for inspiration_id in inspiration_ids:
            inspiration = context.get_program_by_id(inspiration_id)
            if inspiration is None:
                continue
            inspiration_plan = parse_program_plan(
                inspiration,
                fallback_theorem=context.task_description,
            )
            inspiration_strategy = str(inspiration_plan.strategy_type or "").strip().lower()
            inspiration_statements = {
                str(node.statement_formal).strip()
                for node in inspiration_plan.subgoals
                if getattr(node, "statement_formal", None)
            }
            inspiration_lemmas = {
                str(lemma.lemma_id)
                for lemma in load_lemmas(inspiration_plan)
                if getattr(lemma, "is_verified", False)
            }

            novel_statement_count = len(inspiration_statements - parent_statements)
            novel_lemma_count = len(inspiration_lemmas - parent_lemmas)
            strategy_diverse = inspiration_strategy and inspiration_strategy != parent_strategy
            stronger_score = float(inspiration.effective_search_score or 0.0) > parent_search + 1e-6
            solved_bonus = float(inspiration.success_score or 0.0) >= 1.0 > parent_success

            complementarity = 0.0
            if strategy_diverse:
                complementarity += 1.0
            if novel_statement_count > 0:
                complementarity += 1.0
            if novel_lemma_count > 0:
                complementarity += 1.0
            if solved_bonus:
                complementarity += 1.0

            evidence: List[str] = []
            if strategy_diverse:
                evidence.append(f"strategy differs ({parent_strategy or 'none'} -> {inspiration_strategy})")
            if novel_statement_count > 0:
                evidence.append(f"adds {novel_statement_count} novel subgoal statements")
            if novel_lemma_count > 0:
                evidence.append(f"adds {novel_lemma_count} verified lemmas")
            if solved_bonus:
                evidence.append("provides a solved proof lineage for the same theorem")
            if stronger_score:
                evidence.append("has a stronger search score than the parent")

            if complementarity <= 0.0 and not stronger_score:
                continue

            candidates.append({
                "program": inspiration,
                "program_id": inspiration.id,
                "evidence": evidence[:4],
                "complementarity": complementarity,
                "utility": (
                    0.45 * float(inspiration.effective_search_score or 0.0)
                    + 0.35 * float(inspiration.success_score or 0.0)
                    + 0.20 * min(1.0, complementarity / 2.0)
                ),
            })

        if not candidates:
            return {
                "mode": "mutation_only",
                "reason": "no_complementary_inspirations",
                "selected_inspiration_ids": [],
                "evidence": [],
            }

        candidates.sort(
            key=lambda item: (
                float(item["utility"]),
                float(item["complementarity"]),
                float(item["program"].ranking_score),
            ),
            reverse=True,
        )
        selected = candidates[: min(2, len(candidates))]
        selected_ids = [str(item["program_id"]) for item in selected]

        if replanning_mode == "replan_from_memory":
            mode = "crossover_with_inspirations"
            reason = "memory_guided_plan_crossover"
        elif parent_blocked > 0.0 and selected:
            mode = "crossover_with_inspirations"
            reason = "blocked_parent_with_complementary_inspirations"
        elif any(float(item["program"].effective_search_score or 0.0) > parent_search + 0.05 for item in selected):
            mode = "crossover_with_inspirations"
            reason = "stronger_inspiration_merge"
        else:
            mode = "mutation_only"
            reason = "parent_refinement_preferred"

        specialization_mode = str((island_specialization or {}).get("mode", "") or "")
        if (
            specialization_mode == "crossover_recombination"
            and selected
            and parent_success < 1.0
            and mode == "mutation_only"
        ):
            mode = "crossover_with_inspirations"
            reason = "island_prefers_crossover_recombination"
        elif (
            specialization_mode == "conservative_refinement"
            and mode == "crossover_with_inspirations"
            and reason == "stronger_inspiration_merge"
            and parent_blocked <= 0.0
            and replanning_mode != "replan_from_memory"
        ):
            mode = "mutation_only"
            reason = "island_prefers_parent_refinement"

        flattened_evidence: List[str] = []
        for item in selected:
            flattened_evidence.append(f"{item['program_id']}: " + "; ".join(item["evidence"]))
        if specialization_mode and specialization_mode != "generalist":
            flattened_evidence.append(
                f"island specialization bias: {specialization_mode}"
            )

        return {
            "mode": mode,
            "reason": reason,
            "selected_inspiration_ids": selected_ids,
            "evidence": flattened_evidence[:4],
        }

    def _format_crossover_directive(self, directive: Dict[str, Any]) -> str:
        """Serialize crossover policy into a compact prompt block."""
        lines = [
            f"- crossover_mode: {directive.get('mode', 'mutation_only')}",
            f"- crossover_reason: {directive.get('reason', 'none')}",
        ]
        selected_ids = list(directive.get("selected_inspiration_ids", []) or [])
        if selected_ids:
            lines.append(f"- selected_inspiration_ids: {', '.join(selected_ids)}")
        evidence = list(directive.get("evidence", []) or [])
        if evidence:
            lines.append(f"- evidence: {' | '.join(evidence)}")
        if directive.get("mode") == "crossover_with_inspirations":
            lines.append(
                "- instruction: merge only compatible subgoal structure, lemmas, and proof ideas from the selected inspirations; "
                "do not blindly union incompatible DAG fragments"
            )
        else:
            lines.append(
                "- instruction: treat inspirations as weak context only; do not explicitly cross over plan structure"
            )
        return "\n".join(lines)

    def build_prompt(self, parent: Program, context: Context, result: RolloutResult) -> str:
        """Build a plan-mutation prompt with parent state and inspirations."""
        template_name = self.config.get("template_generate", "generation/plan_mutation.txt")
        parent_plan = parse_program_plan(parent, fallback_theorem=context.task_description)
        parent_plan_text = dump_plan(parent_plan)
        inspirations_text = self._format_inspirations(context, result.inspiration_ids)
        repair_agenda = format_repair_agenda_for_prompt(parent)
        parent_experiences = format_experiences_for_prompt(parent)
        inspiration_experiences = self._format_inspiration_experiences(context, result.inspiration_ids)
        parent_feedback = parent.llm_feedback or {}
        island_specialization = self._build_island_specialization_directive(context)
        global_memory_dir = context.config.get("global_memory_dir")
        theorem_memories = []
        lemma_memories = []
        theorem_query_text = "\n".join(
            part
            for part in [
                parent_plan.theorem_statement,
                context.task_constraint,
                context.task_description,
            ]
            if part
        )
        if global_memory_dir:
            theorem_memories = retrieve_theorem_memories(
                global_memory_dir,
                theorem_statement=parent_plan.theorem_statement,
                limit=3,
                task_constraint=context.task_constraint,
                task_scaffold=context.task_description,
            )
            lemma_memories = retrieve_lemma_memories(
                global_memory_dir,
                query_text=theorem_query_text,
                limit=4,
                statement_formal=parent_plan.theorem_statement,
            )
        replanning_directive = self._build_replanning_directive(
            parent,
            theorem_memories=theorem_memories,
            lemma_memories=lemma_memories,
            island_specialization=island_specialization,
        )
        result.stats["replanning_mode"] = replanning_directive["mode"]
        result.stats["replanning_reason"] = replanning_directive["reason"]
        result.stats["retrieved_theorem_memory_ids"] = list(
            replanning_directive.get("retrieved_theorem_memory_ids", [])
        )
        result.stats["retrieved_lemma_memory_ids"] = list(
            replanning_directive.get("retrieved_lemma_memory_ids", [])
        )
        result.stats["replanning_directive"] = dict(replanning_directive)
        result.stats["island_specialization_mode"] = island_specialization["mode"]
        result.stats["island_specialization_reason"] = island_specialization["reason"]
        result.stats["island_specialization_directive"] = dict(island_specialization)
        crossover_directive = self._build_crossover_directive(
            parent,
            context,
            result.inspiration_ids,
            replanning_directive,
            island_specialization=island_specialization,
        )
        result.stats["crossover_mode"] = crossover_directive["mode"]
        result.stats["crossover_reason"] = crossover_directive["reason"]
        result.stats["selected_crossover_inspiration_ids"] = list(
            crossover_directive.get("selected_inspiration_ids", [])
        )
        result.stats["crossover_directive"] = dict(crossover_directive)

        return prompt_registry.get(
            template_name,
            theorem_statement=parent_plan.theorem_statement,
            task_constraint=context.task_constraint,
            parent_plan=parent_plan_text,
            parent_score=parent.effective_search_score,
            parent_success_score=parent.success_score,
            parent_metrics=parent.metrics,
            parent_feedback=parent_feedback,
            repair_agenda=repair_agenda,
            parent_experiences=parent_experiences,
            island_specialization_directive=self._format_island_specialization_directive(
                island_specialization
            ),
            replanning_directive=self._format_replanning_directive(replanning_directive),
            crossover_directive=self._format_crossover_directive(crossover_directive),
            inspirations=inspirations_text,
            inspiration_experiences=inspiration_experiences,
            global_theorem_memories=format_global_memory_for_prompt(theorem_memories, kind="theorem"),
            global_lemma_memories=format_global_memory_for_prompt(lemma_memories, kind="lemma"),
        )

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Generate one mutated plan and store it as a Program."""
        parent = context.get_program_by_id(result.selected_parent_id)
        if parent is None:
            raise ValueError(
                f"Parent program {result.selected_parent_id} not found in population. "
                "Cannot generate without a valid parent."
            )
        parent_plan = parse_program_plan(parent, fallback_theorem=context.task_description)

        prompt = self.build_prompt(parent, context, result)

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

        new_program_id = generate_program_id(
            generation=parent.generation + 1,
            iteration=result.iteration,
        )

        child_plan = parse_plan(response.text, default_plan_id=new_program_id)
        if not child_plan.parent_plan_ids:
            child_plan.parent_plan_ids = [parent.id]
        if not child_plan.experience_refs:
            child_plan.experience_refs = list(parent_plan.experience_refs)
        repair_agenda = build_repair_agenda(parent)
        replanning_directive = dict(result.stats.get("replanning_directive", {}) or {})
        crossover_directive = dict(result.stats.get("crossover_directive", {}) or {})
        island_specialization = dict(result.stats.get("island_specialization_directive", {}) or {})

        plan_metrics = compute_combined_plan_score(child_plan)
        child_plan.combined_score = plan_metrics["search_score"]
        child_plan.metadata["success_score"] = plan_metrics["success_score"]
        child_plan.metadata["search_score"] = plan_metrics["search_score"]
        child_plan.metadata["mutation_mode_from_parent"] = repair_agenda["mutation_mode"]
        child_plan.metadata["parent_repair_targets"] = repair_agenda["primary_targets"]
        child_plan.metadata["island_specialization_mode"] = island_specialization.get("mode", "generalist")
        child_plan.metadata["island_specialization_reason"] = island_specialization.get("reason")
        child_plan.metadata["island_specialization_instruction"] = island_specialization.get("instruction")
        child_plan.metadata["replanning_mode"] = replanning_directive.get("mode", "refine_parent")
        child_plan.metadata["replanning_reason"] = replanning_directive.get("reason")
        child_plan.metadata["replanning_evidence"] = list(replanning_directive.get("evidence", []) or [])
        child_plan.metadata["retrieved_theorem_memory_ids"] = list(
            replanning_directive.get("retrieved_theorem_memory_ids", []) or []
        )
        child_plan.metadata["retrieved_lemma_memory_ids"] = list(
            replanning_directive.get("retrieved_lemma_memory_ids", []) or []
        )
        child_plan.metadata["crossover_mode"] = crossover_directive.get("mode", "mutation_only")
        child_plan.metadata["crossover_reason"] = crossover_directive.get("reason")
        child_plan.metadata["selected_crossover_inspiration_ids"] = list(
            crossover_directive.get("selected_inspiration_ids", []) or []
        )
        child_plan.metadata["crossover_evidence"] = list(crossover_directive.get("evidence", []) or [])

        crossover_inspirations = list(crossover_directive.get("selected_inspiration_ids", []) or [])
        if crossover_inspirations:
            merged_parent_ids: List[str] = []
            for program_id in [*child_plan.parent_plan_ids, *crossover_inspirations]:
                if program_id and program_id not in merged_parent_ids:
                    merged_parent_ids.append(program_id)
            child_plan.parent_plan_ids = merged_parent_ids

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
                "parent_repair_agenda": repair_agenda,
                "island_specialization_mode": island_specialization.get("mode", "generalist"),
                "island_specialization_reason": island_specialization.get("reason"),
                "island_specialization_instruction": island_specialization.get("instruction"),
                "replanning_mode": replanning_directive.get("mode", "refine_parent"),
                "replanning_reason": replanning_directive.get("reason"),
                "replanning_evidence": list(replanning_directive.get("evidence", []) or []),
                "retrieved_theorem_memory_ids": list(
                    replanning_directive.get("retrieved_theorem_memory_ids", []) or []
                ),
                "retrieved_lemma_memory_ids": list(
                    replanning_directive.get("retrieved_lemma_memory_ids", []) or []
                ),
                "crossover_mode": crossover_directive.get("mode", "mutation_only"),
                "crossover_reason": crossover_directive.get("reason"),
                "selected_crossover_inspiration_ids": list(
                    crossover_directive.get("selected_inspiration_ids", []) or []
                ),
                "crossover_evidence": list(crossover_directive.get("evidence", []) or []),
            },
            created_at=time.time(),
        )
        return result

    def _format_inspirations(self, context: Context, inspiration_ids: List[str]) -> str:
        """Serialize inspiration plans into a compact prompt section."""
        if not inspiration_ids:
            return "None"

        sections = []
        for idx, program_id in enumerate(inspiration_ids, start=1):
            inspiration = context.get_program_by_id(program_id)
            if inspiration is None:
                continue
            inspiration_plan = parse_program_plan(
                inspiration,
                fallback_theorem=context.task_description,
            )
            sections.append(
                "\n".join(
                    [
                        f"## Inspiration {idx}",
                        f"- Program ID: {inspiration.id}",
                        f"- Success score: {inspiration.success_score}",
                        f"- Search score: {inspiration.effective_search_score}",
                        dump_plan(inspiration_plan),
                    ]
                )
            )
        return "\n\n".join(sections) if sections else "None"

    def _format_inspiration_experiences(self, context: Context, inspiration_ids: List[str]) -> str:
        """Serialize experience memory carried by inspiration programs."""
        if not inspiration_ids:
            return "None"

        sections = []
        for idx, program_id in enumerate(inspiration_ids, start=1):
            inspiration = context.get_program_by_id(program_id)
            if inspiration is None:
                continue
            sections.append(
                "\n".join(
                    [
                        f"## Inspiration Experience {idx}",
                        f"- Program ID: {inspiration.id}",
                        format_experiences_for_prompt(inspiration),
                    ]
                )
            )
        return "\n\n".join(sections) if sections else "None"

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that generation produced a structured plan program."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a plan program. "
                "Check LLM client and prompt configuration."
            )
        parse_plan(result.generated_program.code, default_plan_id=result.generated_program.id)
