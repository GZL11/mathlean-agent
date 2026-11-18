from __future__ import annotations

import unittest

from famou.config.settings import ExperimentConfig, IslandConfig
from famou.controller.evolver import Evolver
from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Rollout
from famou.core.types import Language
from famou.infrastructure.env.local_env import LocalEnv
from famou.modules.evaluate import EvaluateModule
from famou.modules.feature.embedding import EmbeddingFeature
from famou.modules.population.topk import TopKPopulation


def solved_plan_evaluator(program_path: str):
    return {
        "combined_score": 1.0,
        "search_score": 1.0,
        "validity": 1.0,
        "theorem_solved": 1.0,
        "proof_validity": 1.0,
        "compile_success": 1.0,
        "no_sorry": 1.0,
    }


class TestMathproofRuntimeStability(unittest.TestCase):
    def test_embedding_feature_falls_back_when_key_features_missing(self):
        module = EmbeddingFeature()
        program = Program(
            id="seed_0",
            code="plain program text",
            generation=0,
            iteration=0,
            language=Language.LEAN,
        )
        program.llm_feedback = {"next_actions": ["do something"]}
        result = RolloutResult(rollout_id="r1", iteration=0, generated_program=program)
        context = Context(
            experiment_id="exp",
            task_description="task",
            task_constraint="constraint",
            population={},
            config={},
        )

        vector = module.extract_features(context, result)

        self.assertEqual(len(vector), 1536)
        self.assertTrue(all(value == 0.0 for value in vector))

    def test_evolver_stops_after_solved_initial_enrichment(self):
        initial_program = Program(
            id="seed_0",
            code="plan_id: seed\ntheorem_statement: 'P -> P'\nsubgoals: []\n",
            generation=0,
            iteration=0,
            language=Language.LEAN,
        )
        config = ExperimentConfig(
            name="mathproof-stop-on-seed",
            task_description="P -> P",
            task_constraint="Keep theorem fixed.",
            language=Language.LEAN,
            strategy="mathproof_plan_island",
            max_iterations=3,
            max_workers=1,
            island=IslandConfig(num_islands=1, island_size=2),
        )
        evolver = Evolver(
            config=config,
            population_module=TopKPopulation(),
            initial_programs=[initial_program],
            llm_client=None,
            env=LocalEnv(),
            logger=None,
            data_service=None,
        )
        rollout = Rollout(
            [
                EvaluateModule(evaluate_fn=solved_plan_evaluator, language="lean"),
            ],
            name="seed-enrichment-only",
        )

        experiment = evolver.run(rollout)

        self.assertEqual(experiment.current_iteration, 0)
        self.assertEqual(len(experiment.rollout_history), 0)
        self.assertEqual(len(experiment.archive), 1)
        self.assertEqual(experiment.archive[0].metrics["theorem_solved"], 1.0)


if __name__ == "__main__":
    unittest.main()
