from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from famou.config.settings import ExperimentConfig, IslandConfig, ModulesConfig
from famou.controller.evolver import Evolver
from famou.core.data import Context, Program
from famou.core.protocol import Rollout
from famou.core.types import Language
from famou.infrastructure.env.local_env import LocalEnv
from famou.infrastructure.llm.mock_client import MockLLMClient
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.plan_mutation import PlanMutationGenerate
from famou.modules.judge.plan_judge import PlanJudge
from famou.modules.population.topk import TopKPopulation
from famou.modules.select.topk import TopKSelect
from famou.strategies.mathproof_plan_island import create_strategy
from famou.utils.plan_lean_evaluator import create_plan_lean_evaluator


INITIAL_PLAN = """
plan_id: seed_plan
theorem_statement: "True"
plan_summary: "Seed plan"
strategy_type: direct
subgoals:
  - id: sg_goal
    statement: "True"
    status: attempted
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        sorry
"""


LOCAL_CHILD_1 = """
```yaml
plan_id: local_child_1
theorem_statement: "True"
plan_summary: "First local repair attempt"
strategy_type: direct
subgoals:
  - id: sg_goal
    statement: "True"
    status: attempted
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        sorry
```
"""


LOCAL_CHILD_2 = """
```yaml
plan_id: local_child_2
theorem_statement: "True"
plan_summary: "Second local repair attempt solves the theorem"
strategy_type: direct
subgoals:
  - id: sg_goal
    statement: "True"
    status: verified
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        exact True.intro
lemmas:
  - id: lemma_true
    statement: "True"
    status: verified
    reusable_score: 1.0
    lean_code: |
      import Mathlib

      theorem lemma_true : True := by
        trivial
```
"""


OUTER_MUTATED_CHILD = """
```yaml
plan_id: local_outer_child
theorem_statement: "True"
plan_summary: "Outer mutation keeps the same topology but rewrites the sketch."
strategy_type: direct
subgoals:
  - id: sg_goal
    statement: "True"
    status: attempted
    proof_sketch: "The theorem is trivial; the inner loop should provide the final draft."
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        sorry
```
"""


INNER_REPAIR_PATCH = """
```json
{
  "node_id": "sg_goal",
  "updated_proof_sketch": "Finish the trivial theorem directly.",
  "updated_local_lean_draft": "import Mathlib\\n\\ntheorem sg_goal : True := by\\n  exact True.intro",
  "updated_status": "verified",
  "verifier_state_update": {
    "repair_mode": "fixed_plan"
  },
  "retrieved_lemma_ids": ["lemma_true"],
  "retrieved_experience_ids": ["exp_true"],
  "last_diagnostic": "proof is complete",
  "repair_summary": "Inner repair removed sorry and completed the theorem."
}
```
"""


def mock_verifier(artifact_path: Path, lean_code: str, plan, node):
    has_sorry = "sorry" in lean_code
    return {
        "compile_success": 1.0,
        "no_sorry": 0.0 if has_sorry else 1.0,
        "statement_match": 1.0,
        "proof_validity": 0.0 if has_sorry else 1.0,
        "diagnostics": [] if not has_sorry else ["contains sorry"],
    }


class TestMathproofLocalIsland(unittest.TestCase):
    def test_mathproof_strategy_disables_embedding_by_default(self):
        strategy = create_strategy(evaluate_fn=lambda _: {"combined_score": 0.0, "validity": 1.0})
        module_names = [module.name for module in strategy["rollout"].modules]
        self.assertNotIn("EmbeddingFeature", module_names)
        inner_rollout = strategy["rollout"].execution_config["local_island_inner_rollout"]
        inner_module_names = [module.__class__.__name__ for module in inner_rollout.modules]
        self.assertIn("SubgoalRepairGenerate", inner_module_names)

    def test_mathproof_strategy_passes_adaptive_budget_config(self):
        strategy = create_strategy(
            evaluate_fn=lambda _: {"combined_score": 0.0, "validity": 1.0},
            params=ModulesConfig(
                generate={
                    "local_island_budget": 4,
                    "local_island_budget_mode": "adaptive",
                    "local_island_budget_min": 1,
                    "local_island_budget_max": 4,
                }
            ),
        )

        execution_config = strategy["rollout"].execution_config
        self.assertEqual(execution_config["local_island_budget"], 4)
        self.assertEqual(execution_config["local_island_budget_mode"], "adaptive")
        self.assertEqual(execution_config["local_island_budget_min"], 1)
        self.assertEqual(execution_config["local_island_budget_max"], 4)

    def test_allocate_local_island_budget_adapts_to_parent_quality(self):
        promising_program = Program(
            id="promising_parent",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.9,
            metrics={
                "progress_score": 0.85,
                "cost_penalty": 0.05,
                "frontier_size": 2,
                "blocked_size": 0,
                "theorem_solved": 0.0,
            },
        )
        weak_program = Program(
            id="weak_parent",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.1,
            metrics={
                "progress_score": 0.1,
                "cost_penalty": 0.6,
                "frontier_size": 0,
                "blocked_size": 2,
                "theorem_solved": 0.0,
            },
        )

        config = ExperimentConfig(
            name="mathproof-adaptive-budget",
            task_description="Allocate local budget adaptively.",
            task_constraint="Keep theorem fixed.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=1,
            max_workers=1,
            island=IslandConfig(num_islands=1, island_size=3),
        )
        evolver = Evolver(
            config=config,
            population_module=TopKPopulation(),
            initial_programs=[promising_program],
            llm_client=MockLLMClient(responses=[]),
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )

        promising_context = Context(
            experiment_id="adaptive-budget-exp",
            task_description=config.task_description,
            task_constraint=config.task_constraint,
            population={"main": [promising_program]},
            experiment_config=config,
            iteration=1,
        )
        weak_context = Context(
            experiment_id="adaptive-budget-exp",
            task_description=config.task_description,
            task_constraint=config.task_constraint,
            population={"main": [weak_program]},
            experiment_config=config,
            iteration=1,
        )

        execution_config = {
            "local_island_budget": 4,
            "local_island_budget_mode": "adaptive",
            "local_island_budget_min": 1,
            "local_island_budget_max": 4,
        }
        promising_budget, promising_summary = evolver._allocate_local_island_budget(
            promising_context,
            execution_config,
        )
        weak_budget, weak_summary = evolver._allocate_local_island_budget(
            weak_context,
            execution_config,
        )

        self.assertGreater(promising_budget, weak_budget)
        self.assertEqual(promising_budget, 4)
        self.assertEqual(weak_budget, 1)
        self.assertEqual(promising_summary["anchor_program_id"], "promising_parent")
        self.assertEqual(weak_summary["anchor_program_id"], "weak_parent")
        self.assertEqual(promising_summary["mode"], "adaptive")

    def test_allocate_local_island_budget_auto_boosts_fixed_single_step_with_inner_rollout(self):
        anchor_program = Program(
            id="seed_parent",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.22,
            metrics={
                "progress_score": 0.6,
                "frontier_size": 1,
                "blocked_size": 0,
                "theorem_solved": 0.0,
            },
        )

        config = ExperimentConfig(
            name="mathproof-fixed-budget-auto-boost",
            task_description="Allocate local budget for fixed mode.",
            task_constraint="Keep theorem fixed.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=1,
            max_workers=1,
            island=IslandConfig(num_islands=1, island_size=3),
        )
        evolver = Evolver(
            config=config,
            population_module=TopKPopulation(),
            initial_programs=[anchor_program],
            llm_client=MockLLMClient(responses=[]),
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )

        context = Context(
            experiment_id="fixed-budget-auto-boost-exp",
            task_description=config.task_description,
            task_constraint=config.task_constraint,
            population={"main": [anchor_program]},
            experiment_config=config,
            iteration=1,
        )

        execution_config = {
            "local_island_budget": 1,
            "local_island_budget_mode": "fixed",
            "local_island_inner_rollout": Rollout([], name="inner-rollout-placeholder"),
        }
        budget, summary = evolver._allocate_local_island_budget(
            context,
            execution_config,
        )

        self.assertEqual(budget, 2)
        self.assertEqual(summary["reason"], "auto_boost_for_inner_repair")
        self.assertEqual(summary["anchor_program_id"], "seed_parent")

    def test_topk_select_breaks_seed_lock_for_unsolved_lean_iteration(self):
        selector = TopKSelect(k=1, num_inspirations=0)
        seed = Program(
            id="seed_0",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.22,
        )
        child = Program(
            id="1_1_child",
            code=INITIAL_PLAN,
            generation=1,
            iteration=1,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.09,
        )
        context = Context(
            experiment_id="seed-lock-break",
            task_description="Select parent for Lean benchmark.",
            task_constraint="",
            population={"main": [seed, child]},
            experiment_config=ExperimentConfig(
                name="seed-lock-break",
                task_description="Select parent for Lean benchmark.",
                task_constraint="",
                language=Language.LEAN,
                strategy="mathproof_plan_island",
                max_iterations=2,
                max_workers=1,
                island=IslandConfig(num_islands=1, island_size=3),
            ),
            iteration=2,
            language=Language.LEAN,
            config={},
        )

        selected_id = selector.select_parent(context, [seed, child])
        self.assertEqual(selected_id, "1_1_child")

    def test_topk_select_honors_preferred_parent_id(self):
        selector = TopKSelect(k=1, num_inspirations=0)
        seed = Program(
            id="seed_0",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.22,
        )
        child = Program(
            id="1_1_child",
            code=INITIAL_PLAN,
            generation=1,
            iteration=1,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.09,
        )
        context = Context(
            experiment_id="preferred-parent",
            task_description="Select forced parent.",
            task_constraint="",
            population={"main": [seed, child]},
            experiment_config=ExperimentConfig(
                name="preferred-parent",
                task_description="Select forced parent.",
                task_constraint="",
                language=Language.LEAN,
                strategy="mathproof_plan_island",
                max_iterations=2,
                max_workers=1,
                island=IslandConfig(num_islands=1, island_size=3),
            ),
            iteration=10,
            language=Language.LEAN,
            config={"preferred_parent_id": "1_1_child"},
        )

        selected_id = selector.select_parent(context, [seed, child])
        self.assertEqual(selected_id, "1_1_child")

    def test_evolver_runs_local_island_budget(self):
        evaluator_fn = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=True,
        )

        rollout = Rollout(
            [
                TopKSelect(k=1, num_inspirations=0),
                PlanMutationGenerate(language="lean", temperature=0.0, max_tokens=1200),
                EvaluateModule(evaluate_fn=evaluator_fn, language="lean"),
                PlanJudge(use_llm=False),
            ],
            name="mathproof_local_island_test",
            execution_config={
                "local_island_budget": 3,
                "local_island_return_mode": "best",
                "local_island_stop_on_solved": True,
            },
        )

        llm_client = MockLLMClient(responses=[LOCAL_CHILD_1, LOCAL_CHILD_2, LOCAL_CHILD_2])
        initial_program = Program(
            id="0_0_seed",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
        )

        config = ExperimentConfig(
            name="mathproof-local-island",
            task_description="Evolve a plan-local Lean proof.",
            task_constraint="Keep theorem fixed and avoid sorry in final proof.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=1,
            max_workers=1,
            island=IslandConfig(num_islands=1, island_size=3),
        )

        evolver = Evolver(
            config=config,
            population_module=TopKPopulation(),
            initial_programs=[initial_program],
            llm_client=llm_client,
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )

        experiment = evolver.run(rollout)

        self.assertEqual(experiment.current_iteration, 1)
        self.assertEqual(len(experiment.rollout_history), 1)

        final_result = experiment.rollout_history[0]
        self.assertEqual(final_result.status, "success")
        self.assertEqual(final_result.generated_program.metrics["theorem_solved"], 1.0)
        self.assertEqual(final_result.stats["local_island_steps_executed"], 2)
        self.assertEqual(final_result.stats["local_island_budget"], 3)
        self.assertEqual(len(final_result.stats["local_island_history"]), 2)
        self.assertEqual(final_result.generated_program.meta["local_island_summary"]["steps_executed"], 2)
        self.assertEqual(final_result.generated_program.meta["local_island_step"], 1)
        self.assertIn("experience_pool", final_result.generated_program.meta)
        self.assertGreaterEqual(len(final_result.generated_program.meta["experience_pool"]), 1)
        self.assertEqual(final_result.generated_program.combined_score, 1.0)
        self.assertGreater(final_result.generated_program.search_score, 0.8)

    def test_evolver_stops_global_iterations_after_solution(self):
        evaluator_fn = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=True,
        )

        rollout = Rollout(
            [
                TopKSelect(k=1, num_inspirations=0),
                PlanMutationGenerate(language="lean", temperature=0.0, max_tokens=1200),
                EvaluateModule(evaluate_fn=evaluator_fn, language="lean"),
                PlanJudge(use_llm=False),
            ],
            name="mathproof_global_stop_test",
            execution_config={
                "local_island_budget": 1,
                "local_island_return_mode": "best",
                "local_island_stop_on_solved": True,
                "global_stop_on_solved": True,
            },
        )

        llm_client = MockLLMClient(responses=[LOCAL_CHILD_2, LOCAL_CHILD_2, LOCAL_CHILD_2])
        initial_program = Program(
            id="0_0_seed",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
        )

        config = ExperimentConfig(
            name="mathproof-global-stop",
            task_description="Stop once the Lean theorem is solved.",
            task_constraint="Keep theorem fixed and avoid sorry in final proof.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=3,
            max_workers=1,
            island=IslandConfig(num_islands=1, island_size=3),
        )

        evolver = Evolver(
            config=config,
            population_module=TopKPopulation(),
            initial_programs=[initial_program],
            llm_client=llm_client,
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )

        experiment = evolver.run(rollout)

        self.assertEqual(experiment.current_iteration, 1)
        self.assertEqual(len(experiment.rollout_history), 1)
        self.assertEqual(experiment.rollout_history[0].generated_program.metrics["theorem_solved"], 1.0)
        self.assertEqual(experiment.rollout_history[0].generated_program.success_score, 1.0)

    def test_local_island_runs_outer_plan_then_inner_repair_rollout(self):
        evaluator_fn = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=True,
        )

        strategy = create_strategy(
            evaluate_fn=evaluator_fn,
            params=ModulesConfig(
                generate={
                    "local_island_budget": 2,
                    "local_island_return_mode": "best",
                    "local_island_stop_on_solved": True,
                    "global_stop_on_solved": True,
                },
                judge={"use_llm": False},
            ),
        )

        llm_client = MockLLMClient(responses=[OUTER_MUTATED_CHILD, INNER_REPAIR_PATCH])
        initial_program = Program(
            id="0_0_seed",
            code=INITIAL_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
        )

        config = ExperimentConfig(
            name="mathproof-outer-inner-local-island",
            task_description="First mutate the plan, then repair one selected subgoal.",
            task_constraint="Keep theorem fixed and avoid sorry in final proof.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=1,
            max_workers=1,
            island=IslandConfig(num_islands=1, island_size=3),
        )

        evolver = Evolver(
            config=config,
            population_module=strategy["population_module"],
            initial_programs=[initial_program],
            llm_client=llm_client,
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )

        experiment = evolver.run(strategy["rollout"])
        final_result = experiment.rollout_history[0]

        self.assertEqual(final_result.status, "success")
        self.assertEqual(final_result.generated_program.metrics["theorem_solved"], 1.0)
        self.assertEqual(final_result.stats["local_island_steps_executed"], 2)
        self.assertEqual(final_result.stats["local_island_history"][0]["phase"], "outer_mutation")
        self.assertEqual(final_result.stats["local_island_history"][1]["phase"], "inner_repair")
        self.assertEqual(
            final_result.stats["local_island_history"][1]["rollout_name"],
            "mathproof_fixed_plan_inner_repair_rollout",
        )
        self.assertEqual(final_result.generated_program.meta["local_island_phase"], "inner_repair")
        self.assertEqual(final_result.generated_program.meta["selected_subgoal_id"], "sg_goal")
        self.assertEqual(final_result.generated_program.combined_score, 1.0)


if __name__ == "__main__":
    unittest.main()
