"""Plan-centric judge for math-proof evolution."""

from __future__ import annotations

from typing import Any, Dict, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.modules.judge.base import JudgeModule
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.utils.plan_runtime import (
    compute_combined_plan_score,
    dump_plan,
    parse_program_plan,
    summarize_plan_delta,
)


class PlanJudge(JudgeModule):
    """
    Judge a proof plan using structural heuristics, with optional LLM refinement.

    The fallback heuristic is intentionally strong so that the module remains
    usable in tests and offline settings.
    """

    _RESERVED_METRIC_KEYS = {
        "combined_score",
        "success_score",
        "search_score",
        "validity",
        "error_info",
    }

    def judge(self, context: Context, result: RolloutResult) -> Dict[str, Any]:
        """Attach plan-centric feedback to the generated program."""
        program = result.generated_program
        if program is None:
            raise ValueError(f"{self.name}: No generated program available for judging")

        parent = self._resolve_parent(context, program)
        child_plan = parse_program_plan(program, fallback_theorem=context.task_description)
        parent_plan = (
            parse_program_plan(parent, fallback_theorem=context.task_description)
            if parent is not None
            else None
        )

        heuristic_feedback = self._build_heuristic_feedback(
            child_plan,
            parent_plan,
            existing_metrics=program.metrics,
        )
        llm_feedback = self._try_llm_feedback(context, program, child_plan, parent_plan)

        feedback = {**heuristic_feedback}
        if llm_feedback:
            feedback["llm_analysis"] = llm_feedback

        updated_metrics = dict(program.metrics)
        for key, value in feedback["score_breakdown"].items():
            if key not in updated_metrics and key not in self._RESERVED_METRIC_KEYS:
                updated_metrics[key] = value
        program.meta["plan_object"] = child_plan.model_dump(mode="json")
        program.meta["plan_metrics"] = feedback["score_breakdown"]
        program.meta["frontier_nodes"] = feedback["frontier_nodes"]
        program.meta["blocked_nodes"] = feedback["blocked_nodes"]
        program.metrics = updated_metrics

        return feedback

    def _resolve_parent(self, context: Context, program: Program) -> Optional[Program]:
        """Resolve the first parent program if available."""
        if not program.parent_ids:
            return None
        return context.get_program_by_id(program.parent_ids[0])

    def _build_heuristic_feedback(
        self,
        child_plan,
        parent_plan=None,
        existing_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce deterministic structural feedback."""
        score_breakdown = compute_combined_plan_score(
            child_plan,
            extra_metrics=existing_metrics,
        )
        delta = summarize_plan_delta(parent_plan, child_plan) if parent_plan is not None else None

        frontier_nodes = score_breakdown["frontier_nodes"]
        blocked_nodes = score_breakdown["blocked_nodes"]
        improvement_summary = []

        if delta:
            if delta["added_nodes"]:
                improvement_summary.append(f"Added nodes: {', '.join(delta['added_nodes'])}")
            if delta["status_changes"]:
                improvement_summary.append(f"Status changes: {'; '.join(delta['status_changes'])}")
            if delta["sketch_updates"]:
                improvement_summary.append(
                    f"Updated proof sketches: {', '.join(delta['sketch_updates'])}"
                )
            if not improvement_summary:
                improvement_summary.append("Plan structure is largely unchanged from the parent.")
        else:
            improvement_summary.append("Initial plan draft prepared for evaluation.")

        suggestions = []
        if frontier_nodes:
            suggestions.append(
                f"Prioritize frontier nodes: {', '.join(frontier_nodes[:3])}"
            )
        if blocked_nodes:
            suggestions.append(
                f"Unblock dependencies for: {', '.join(blocked_nodes[:3])}"
            )
        if score_breakdown["terminal_success"] < 1.0:
            suggestions.append("Strengthen the top-level proof route before expanding more branches.")
        if score_breakdown["verified_lemmas"] == 0 and child_plan.lemma_pool:
            suggestions.append("Promote at least one reusable lemma to a verified state.")

        return {
            "change_summary": " ".join(improvement_summary),
            "frontier_nodes": frontier_nodes,
            "blocked_nodes": blocked_nodes,
            "score_breakdown": score_breakdown,
            "next_actions": suggestions,
        }

    def _try_llm_feedback(
        self,
        context: Context,
        program: Program,
        child_plan,
        parent_plan=None,
    ) -> Optional[Dict[str, Any]]:
        """Optionally refine judge output with an LLM JSON response."""
        if not self.config.get("use_llm", True):
            return None
        if getattr(self, "llm_client", None) is None:
            return None

        prompt = prompt_registry.get(
            self.config.get("template_judge", "evaluation/plan_judge.txt"),
            task_constraint=context.task_constraint,
            current_plan=dump_plan(child_plan),
            parent_plan=dump_plan(parent_plan) if parent_plan is not None else "None",
            evaluation_metrics=program.metrics,
            error_info=program.error_info or "None",
        )

        llm_kwargs = {}
        if "temperature" in self.config:
            llm_kwargs["temperature"] = self.config["temperature"]
        if "max_tokens" in self.config:
            llm_kwargs["max_tokens"] = self.config["max_tokens"]

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system=context.task_description,
                **llm_kwargs,
            )
            parsed = extract_json(response.text)
            return parsed or None
        except Exception as exc:
            self.log_warning(f"LLM plan judge fallback activated: {exc}")
            return None
