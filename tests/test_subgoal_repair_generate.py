from __future__ import annotations

import unittest

from famou.core.data import Context, Program, RolloutResult
from famou.core.types import Language
from famou.infrastructure.llm.mock_client import MockLLMClient
from famou.modules.generate.subgoal_repair import SubgoalRepairGenerate
from famou.modules.select.topk import TopKSelect
from famou.strategies import StrategyRegistry
from famou.utils.plan_runtime import parse_program_plan
from famou.utils.subgoal_repair_runtime import compute_plan_topology_signature


SIMPLE_PLAN = """
plan_id: seed_plan
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Transitivity proof"
strategy_type: direct
subgoals:
  - id: sg_assumptions
    statement: "Introduce hypotheses hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Apply le_trans hab hbc."
    dependencies: [sg_assumptions]
    status: attempted
    verifier_state:
      diagnostics:
        - "type mismatch at final exact step"
lemmas:
  - id: lemma_le_trans
    statement: "le_trans hab hbc"
    status: verified
    reusable_score: 0.9
"""


REPEATED_FAILURE_PLAN = """
plan_id: retry_plan
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Retry the transitivity node locally."
strategy_type: direct
metadata:
  last_repaired_node_id: sg_transitivity
subgoals:
  - id: sg_assumptions
    statement: "Introduce hypotheses hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Apply le_trans hab hbc."
    dependencies: [sg_assumptions]
    status: attempted
    attempt_count: 2
    proof_sketch: "Use transitivity."
    local_lean_draft: |
      by
        exact missing_hyp
    last_diagnostic: "unknown identifier 'missing_hyp'"
    attempt_history:
      - attempt_index: 1
        previous_status: attempted
        new_status: attempted
        has_proof_sketch: true
        has_local_lean_draft: true
        introduced_lemma_count: 0
        retrieved_lemma_ids: [lemma_le_trans]
        retrieved_experience_ids: [exp_transitivity]
      - attempt_index: 2
        previous_status: attempted
        new_status: attempted
        has_proof_sketch: true
        has_local_lean_draft: true
        introduced_lemma_count: 0
        retrieved_lemma_ids: [lemma_le_trans]
        retrieved_experience_ids: [exp_transitivity]
lemmas:
  - id: lemma_le_trans
    statement: "le_trans hab hbc"
    status: verified
    reusable_score: 0.9
"""


PATCH_RESPONSE = """
```json
{
  "node_id": "sg_transitivity",
  "updated_proof_sketch": "Use the transitivity lemma directly on the two assumptions.",
  "updated_local_lean_draft": "by\\n  exact le_trans hab hbc",
  "updated_status": "in_progress",
  "verifier_state_update": {
    "repair_mode": "local_draft"
  },
  "introduced_lemmas": [
    {
      "lemma_id": "lemma_local_trans",
      "statement_formal": "a <= b -> b <= c -> a <= c",
      "proof_status": "attempted",
      "source_node_ids": ["sg_transitivity"]
    }
  ],
  "retrieved_lemma_ids": ["lemma_le_trans"],
  "retrieved_experience_ids": ["exp_transitivity"],
  "last_diagnostic": "main subgoal now has a direct draft",
  "repair_summary": "Added a direct Lean draft for the transitivity step."
}
```
"""


PATCH_RESPONSE_WITH_STATUS_ALIAS = """
```json
{
  "node_id": "sg_transitivity",
  "updated_proof_sketch": "Use the transitivity lemma directly on the assumptions.",
  "updated_local_lean_draft": "by\\n  exact le_trans hab hbc",
  "updated_status": "completed",
  "repair_summary": "Alias status should normalize to verified."
}
```
"""


class TestSubgoalRepairGenerate(unittest.TestCase):
    def setUp(self):
        self.parent_program = Program(
            id="0_0_seed",
            code=SIMPLE_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.25,
            metrics={"plan_progress": 0.2},
        )
        self.context = Context(
            experiment_id="exp_mathproof",
            task_description="Repair one selected subgoal inside a structured Lean proof plan.",
            task_constraint="Keep the theorem and subgoal DAG fixed.",
            population={"population": [self.parent_program]},
            config={"language": "lean"},
        )
        self.result = RolloutResult(rollout_id="rollout_1", iteration=1)

    def test_subgoal_repair_generation_preserves_topology(self):
        selector = TopKSelect(k=1, num_inspirations=0)
        generator = SubgoalRepairGenerate(language="lean", temperature=0.0, max_tokens=1200)
        generator.llm_client = MockLLMClient(response=PATCH_RESPONSE)

        self.result = selector(self.context, self.result)
        self.result = generator(self.context, self.result)

        program = self.result.generated_program
        self.assertIsNotNone(program)
        self.assertIn("selected_subgoal_id", program.meta)
        self.assertEqual(program.meta["selected_subgoal_id"], "sg_transitivity")
        self.assertIn("fixed_plan_topology_signature", program.meta)
        self.assertIn("sg_transitivity", program.prompt)

        parent_plan = parse_program_plan(self.parent_program, fallback_theorem=self.context.task_description)
        child_plan = parse_program_plan(program, fallback_theorem=self.context.task_description)
        self.assertEqual(
            compute_plan_topology_signature(parent_plan),
            compute_plan_topology_signature(child_plan),
        )

        repaired_node = child_plan.node_map()["sg_transitivity"]
        self.assertEqual(repaired_node.local_status.value, "in_progress")
        self.assertEqual(repaired_node.attempt_count, 1)
        self.assertIn("le_trans hab hbc", repaired_node.local_lean_draft)
        self.assertEqual(repaired_node.retrieved_lemma_ids, ["lemma_le_trans"])
        self.assertEqual(repaired_node.retrieved_experience_ids, ["exp_transitivity"])
        self.assertEqual(len(child_plan.lemma_pool), 2)

    def test_strategy_registry_loads_fixed_plan_repair_strategy(self):
        strategy = StrategyRegistry.get(
            "mathproof_fixed_plan_repair",
            evaluate_fn=lambda _: {"combined_score": 0.0, "validity": 1.0},
        )
        strategy.rollout.validate()
        module_names = [module.__class__.__name__ for module in strategy.rollout.modules]
        self.assertIn("SubgoalRepairGenerate", module_names)
        self.assertIn("PlanJudge", module_names)

    def test_subgoal_repair_normalizes_status_aliases(self):
        selector = TopKSelect(k=1, num_inspirations=0)
        generator = SubgoalRepairGenerate(language="lean", temperature=0.0, max_tokens=1200)
        generator.llm_client = MockLLMClient(response=PATCH_RESPONSE_WITH_STATUS_ALIAS)

        self.result = selector(self.context, self.result)
        self.result = generator(self.context, self.result)

        child_plan = parse_program_plan(
            self.result.generated_program,
            fallback_theorem=self.context.task_description,
        )
        self.assertEqual(child_plan.node_map()["sg_transitivity"].local_status.value, "verified")

    def test_build_prompt_includes_recent_retry_context(self):
        parent_program = Program(
            id="0_0_retry_seed",
            code=REPEATED_FAILURE_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.3,
            metrics={"plan_progress": 0.3},
        )
        context = self.context.model_copy(
            update={"population": {"population": [parent_program]}},
            deep=True,
        )
        result = RolloutResult(rollout_id="rollout_retry", iteration=1)
        generator = SubgoalRepairGenerate(language="lean", temperature=0.0, max_tokens=1200)

        prompt = generator.build_prompt(parent_program, context, result)

        self.assertIn("Recent attempt history:", prompt)
        self.assertIn("Repeated failure note:", prompt)
        self.assertIn("missing_hyp", prompt)
        self.assertEqual(result.stats["selected_subgoal_reason"], "sticky_repeated_failure")
        self.assertEqual(result.stats["selected_subgoal_repeated_failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
