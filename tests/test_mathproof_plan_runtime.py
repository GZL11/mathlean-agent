from __future__ import annotations

import unittest
from pathlib import Path

from famou.config.settings import ExperimentConfig, IslandConfig
from famou.controller.evolver import Evolver
from famou.core.data import Context, Program, RolloutResult
from famou.core.types import Language
from famou.infrastructure.env.local_env import LocalEnv
from famou.infrastructure.llm.mock_client import MockLLMClient
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.plan_mutation import PlanMutationGenerate
from famou.modules.judge.plan_judge import PlanJudge
from famou.modules.population.topk import TopKPopulation
from famou.modules.select.topk import TopKSelect
from famou.strategies import StrategyRegistry
from famou.utils.plan_runtime import (
    compute_combined_plan_score,
    get_frontier_node_ids,
    parse_plan,
    parse_program_plan,
)


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
    status: pending
lemmas:
  - id: lemma_le_trans
    statement: "le_trans hab hbc"
    status: verified
    reusable_score: 0.9
"""


INSPIRATION_PLAN = """
plan_id: inspiration_plan
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Alternative decomposition with an explicit final transitivity node"
strategy_type: decomposition-first
subgoals:
  - id: sg_intro
    statement: "Introduce hypotheses hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_reduce
    statement: "Reduce the target to a direct transitivity application."
    dependencies: [sg_intro]
    status: verified
  - id: sg_finish
    statement: "Conclude a <= c using le_trans hab hbc."
    dependencies: [sg_reduce]
    status: pending
lemmas:
  - id: lemma_le_trans_alt
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
    reusable_score: 1.0
"""


MILD_INSPIRATION_PLAN = """
plan_id: mild_inspiration_plan
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Same high-level route, but with a slightly different intermediate bridge."
strategy_type: direct
subgoals:
  - id: sg_assumptions
    statement: "Introduce hypotheses hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_bridge
    statement: "Reduce the goal to a single direct transitivity application."
    dependencies: [sg_assumptions]
    status: pending
lemmas:
  - id: lemma_bridge
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
    reusable_score: 0.8
"""


CHILD_PLAN_RESPONSE = """
```yaml
plan_id: child_plan
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Transitivity proof refined with explicit final node"
strategy_type: direct
sketch: "First introduce hypotheses, then discharge the final theorem by transitivity."
subgoals:
  - id: sg_assumptions
    statement: "Introduce hypotheses hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Apply le_trans hab hbc."
    dependencies: [sg_assumptions]
    proof_sketch: "Use the library transitivity lemma on the two assumptions."
    status: in_progress
  - id: sg_finalize
    statement: "Conclude the theorem from sg_transitivity."
    dependencies: [sg_transitivity]
    status: pending
lemmas:
  - id: lemma_le_trans
    statement: "le_trans hab hbc"
    status: verified
    reusable_score: 0.9
```
"""


CROSSOVER_CHILD_RESPONSE = """
```yaml
plan_id: crossover_child
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Merged the parent's simple skeleton with the inspiration's explicit final node."
strategy_type: decomposition-first
subgoals:
  - id: sg_assumptions
    statement: "Introduce hypotheses hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Conclude a <= c using le_trans hab hbc."
    dependencies: [sg_assumptions]
    proof_sketch: "Use the library transitivity lemma directly."
    status: pending
lemmas:
  - id: lemma_le_trans
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
    reusable_score: 1.0
```
"""


CHILD_PLAN_WITH_UNQUOTED_COLON = """
```yaml
plan_id: child_plan_bad_yaml
theorem_statement: "P ∧ Q → P"
plan_summary: "And-left projection"
strategy_type: direct
subgoals:
  - id: sg_unpack
    statement: "P ∧ Q → P"
    proof_sketch: Introduce hypothesis h : P ∧ Q and return h.left.
    status: in_progress
    local_lean_draft: |
      by
        intro h
        exact h.1
```
"""


CHILD_PLAN_WITH_UNCLOSED_FENCE = """```yaml
plan_id: child_plan_unclosed
theorem_statement: "P ∧ Q → P"
plan_summary: "And-left projection without a closing fence"
strategy_type: direct
subgoals:
  - id: sg_goal
    statement: "P ∧ Q → P"
    status: pending
"""


CHILD_PLAN_WITH_UNCLOSED_FENCE = """
```yaml
plan_id: child_plan_unclosed
theorem_statement: "P ∧ Q → P"
plan_summary: "Unclosed fenced response"
strategy_type: direct
subgoals:
  - id: sg_goal
    statement: "P ∧ Q → P"
    proof_sketch: "Introduce h and return h.1."
    local_lean_draft: |
      by
        intro h
        exact h.1
    status: verified
lemmas: []
"""


def plan_evaluator(program_path: str):
    plan_text = Path(program_path).read_text(encoding="utf-8")
    plan = parse_plan(plan_text)
    score = compute_combined_plan_score(
        plan,
        extra_metrics={
            "compile_success": 1.0,
            "no_sorry": 1.0,
            "validity": 1.0,
        },
    )
    return {
        "combined_score": score["combined_score"],
        "search_score": score["search_score"],
        "validity": 1.0,
        "compile_success": 1.0,
        "no_sorry": 1.0,
        "plan_progress": score["progress_score"],
        "terminal_success": score["terminal_success"],
    }


class TestPlanRuntime(unittest.TestCase):
    def test_parse_and_score_plan(self):
        plan = parse_plan(SIMPLE_PLAN)
        self.assertEqual(plan.plan_id, "seed_plan")
        self.assertEqual(get_frontier_node_ids(plan), ["sg_transitivity"])

        score = compute_combined_plan_score(plan)
        self.assertEqual(score["completed_subgoals"], 1)
        self.assertEqual(score["total_subgoals"], 2)
        self.assertGreater(score["reusable_lemma_bonus"], 0.0)
        self.assertGreater(score["progress_score"], 0.0)

    def test_cycle_detection(self):
        bad_plan = """
theorem_statement: bad
subgoals:
  - id: a
    statement: A
    dependencies: [b]
  - id: b
    statement: B
    dependencies: [a]
"""
        with self.assertRaises(ValueError):
            parse_plan(bad_plan)

    def test_parse_plan_recovers_unquoted_colon_scalars(self):
        plan = parse_plan(CHILD_PLAN_WITH_UNQUOTED_COLON)
        self.assertEqual(plan.plan_id, "child_plan_bad_yaml")
        self.assertEqual(plan.subgoals[0].local_status.value, "in_progress")
        self.assertIn("h : P ∧ Q", plan.subgoals[0].proof_sketch)

    def test_parse_plan_recovers_unclosed_fenced_yaml(self):
        plan = parse_plan(CHILD_PLAN_WITH_UNCLOSED_FENCE)
        self.assertEqual(plan.plan_id, "child_plan_unclosed")
        self.assertEqual(plan.theorem_statement, "P ∧ Q → P")
        self.assertEqual(plan.subgoals[0].node_id, "sg_goal")

    def test_parse_plan_recovers_unclosed_fenced_yaml(self):
        plan = parse_plan(CHILD_PLAN_WITH_UNCLOSED_FENCE)
        self.assertEqual(plan.plan_id, "child_plan_unclosed")
        self.assertEqual(plan.subgoals[0].local_status.value, "verified")
        self.assertIn("exact h.1", plan.subgoals[0].local_lean_draft)


class TestPlanModules(unittest.TestCase):
    def setUp(self):
        self.parent_program = Program(
            id="0_0_seed",
            code=SIMPLE_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.25,
            metrics={"plan_progress": 0.2, "blocked_size": 1.0, "frontier_size": 0.0},
            meta={
                "local_island_budget": 2,
                "local_island_summary": {
                    "budget": 2,
                    "steps_executed": 2,
                    "history": [
                        {"local_step": 0, "score": 0.25},
                        {"local_step": 1, "score": 0.25},
                    ],
                },
            },
        )
        self.inspiration_program = Program(
            id="0_0_inspiration",
            code=INSPIRATION_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.72,
            metrics={"plan_progress": 0.7, "frontier_size": 1.0, "blocked_size": 0.0},
        )
        self.context = Context(
            experiment_id="exp_mathproof",
            task_description="Evolve a structured Lean proof plan for an order-theory theorem.",
            task_constraint="Keep the theorem unchanged and use an acyclic subgoal DAG.",
            population={"population": [self.parent_program, self.inspiration_program]},
            config={"language": "lean"},
        )
        self.result = RolloutResult(rollout_id="rollout_1", iteration=1)

    def test_generation_evaluation_and_judge_pipeline(self):
        selector = TopKSelect(k=1, num_inspirations=0)
        generator = PlanMutationGenerate(language="lean", temperature=0.0, max_tokens=1200)
        generator.llm_client = MockLLMClient(response=CHILD_PLAN_RESPONSE)
        evaluator = EvaluateModule(evaluate_fn=plan_evaluator, language="lean")
        evaluator.env = LocalEnv()
        judge = PlanJudge(use_llm=False)

        self.result = selector(self.context, self.result)
        self.result = generator(self.context, self.result)
        self.result = evaluator(self.context, self.result)
        self.result = judge(self.context, self.result)

        program = self.result.generated_program
        self.assertIsNotNone(program)
        self.assertIsNotNone(program.combined_score)
        self.assertIsNotNone(program.search_score)
        self.assertGreater(program.search_score, 0.0)
        self.assertIn("plan_object", program.meta)
        self.assertIn("plan_metrics", program.meta)
        self.assertIn("parent_repair_agenda", program.meta)
        self.assertIn("mutation_mode", program.prompt)
        self.assertIn("next_actions", program.llm_feedback)
        self.assertIn("frontier_nodes", program.llm_feedback)
        self.assertNotIn("combined_score", program.metrics)
        self.assertNotIn("search_score", program.metrics)
        self.assertIn("combined_score", program.meta["plan_metrics"])
        self.assertIn("search_score", program.meta["plan_metrics"])

    def test_iteration_metrics_keep_evaluator_score(self):
        program = Program(
            id="1_1_child",
            code=CHILD_PLAN_RESPONSE,
            generation=1,
            iteration=1,
            language=Language.LEAN,
            combined_score=1.0,
            search_score=0.91,
            validity=1.0,
            metrics={
                "combined_score": 0.60,
                "search_score": 0.40,
                "validity": 0.40,
                "error_info": "judge-only",
                "plan_progress": 0.75,
            },
        )

        metrics = Evolver._build_iteration_metrics(program)

        self.assertEqual(metrics["combined_score"], 1.0)
        self.assertEqual(metrics["success_score"], 1.0)
        self.assertEqual(metrics["search_score"], 0.91)
        self.assertEqual(metrics["validity"], 1.0)
        self.assertEqual(metrics["plan_progress"], 0.75)
        self.assertNotIn("error_info", metrics)

    def test_strategy_registry_loads_mathproof_strategy(self):
        strategy = StrategyRegistry.get(
            "mathproof_plan_island",
            evaluate_fn=plan_evaluator,
        )
        strategy.rollout.validate()
        module_names = [module.__class__.__name__ for module in strategy.rollout.modules]
        self.assertIn("PlanMutationGenerate", module_names)
        self.assertIn("PlanJudge", module_names)

    def test_runtime_config_exposes_multi_island_specialization(self):
        config = ExperimentConfig(
            name="mathproof-multi-island-specialization",
            task_description="Evolve structured Lean proof plans across islands.",
            task_constraint="Keep theorem fixed.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=2,
            max_workers=1,
            island=IslandConfig(num_islands=3, island_size=3),
        )
        evolver = Evolver(
            config=config,
            population_module=TopKPopulation(),
            initial_programs=[self.parent_program],
            llm_client=MockLLMClient(responses=[]),
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )

        runtime_config = evolver._build_runtime_config(island_id=2)
        specialization = runtime_config["island_specialization"]

        self.assertEqual(runtime_config["island_id"], 2)
        self.assertEqual(specialization["mode"], "crossover_recombination")
        self.assertEqual(specialization["island_id"], 2)
        self.assertEqual(specialization["num_islands"], 3)

    def test_plan_mutation_can_switch_into_crossover_mode(self):
        result = RolloutResult(
            rollout_id="rollout_crossover",
            iteration=1,
            selected_parent_id=self.parent_program.id,
            inspiration_ids=[self.inspiration_program.id],
        )
        generator = PlanMutationGenerate(language="lean", temperature=0.0, max_tokens=1200)
        generator.llm_client = MockLLMClient(response=CROSSOVER_CHILD_RESPONSE)

        prompt = generator.build_prompt(self.parent_program, self.context, result)
        self.assertIn("Crossover directive", prompt)
        self.assertIn("crossover_with_inspirations", prompt)
        self.assertEqual(result.stats["crossover_mode"], "crossover_with_inspirations")
        self.assertEqual(
            result.stats["selected_crossover_inspiration_ids"],
            [self.inspiration_program.id],
        )

        result = generator.execute(self.context, result)
        program = result.generated_program
        self.assertEqual(program.meta["crossover_mode"], "crossover_with_inspirations")
        self.assertEqual(
            program.meta["selected_crossover_inspiration_ids"],
            [self.inspiration_program.id],
        )

        child_plan = parse_program_plan(program, fallback_theorem=self.context.task_description)
        self.assertEqual(child_plan.metadata["crossover_mode"], "crossover_with_inspirations")
        self.assertIn(self.parent_program.id, child_plan.parent_plan_ids)
        self.assertIn(self.inspiration_program.id, child_plan.parent_plan_ids)

    def test_island_specialization_can_force_crossover_bias(self):
        parent = Program(
            id="0_0_parent_bias",
            code=SIMPLE_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.30,
            metrics={"plan_progress": 0.30, "blocked_size": 0.0, "frontier_size": 2.0},
        )
        mild_inspiration = Program(
            id="0_0_mild_inspiration",
            code=MILD_INSPIRATION_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.32,
            metrics={"plan_progress": 0.35, "frontier_size": 1.0, "blocked_size": 0.0},
        )
        specialized_context = Context(
            experiment_id="exp_mathproof_specialized",
            task_description="Evolve a structured Lean proof plan for an order-theory theorem.",
            task_constraint="Keep the theorem unchanged and use an acyclic subgoal DAG.",
            population={"population": [parent, mild_inspiration]},
            island_id=2,
            config={
                "language": "lean",
                "island_specialization": {
                    "mode": "crossover_recombination",
                    "reason": "merge_complementary_plans",
                    "instruction": "Prefer compatible crossover when inspirations add useful structure.",
                    "island_id": 2,
                    "num_islands": 3,
                },
            },
        )
        result = RolloutResult(
            rollout_id="rollout_specialized_crossover",
            iteration=1,
            selected_parent_id=parent.id,
            inspiration_ids=[mild_inspiration.id],
        )
        generator = PlanMutationGenerate(language="lean", temperature=0.0, max_tokens=1200)
        generator.llm_client = MockLLMClient(response=CROSSOVER_CHILD_RESPONSE)

        prompt = generator.build_prompt(parent, specialized_context, result)
        self.assertIn("Island specialization directive", prompt)
        self.assertIn("crossover_recombination", prompt)
        self.assertEqual(result.stats["island_specialization_mode"], "crossover_recombination")
        self.assertEqual(result.stats["crossover_mode"], "crossover_with_inspirations")
        self.assertEqual(result.stats["crossover_reason"], "island_prefers_crossover_recombination")

        result = generator.execute(specialized_context, result)
        program = result.generated_program
        self.assertEqual(program.meta["island_specialization_mode"], "crossover_recombination")
        self.assertEqual(program.meta["crossover_mode"], "crossover_with_inspirations")
        child_plan = parse_program_plan(program, fallback_theorem=specialized_context.task_description)
        self.assertEqual(
            child_plan.metadata["island_specialization_mode"],
            "crossover_recombination",
        )


if __name__ == "__main__":
    unittest.main()
