from __future__ import annotations

import unittest

from famou.core.data import Context, Program, RolloutResult
from famou.core.types import Language
from famou.modules.generate.plan_mutation import PlanMutationGenerate
from famou.utils.plan_runtime import load_experiences, parse_program_plan
from famou.utils.proof_experience import (
    build_repair_agenda,
    extract_and_attach_experiences,
    format_experiences_for_prompt,
)


SOLVED_PLAN = """
plan_id: solved_plan
theorem_statement: "True"
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
    lean_code: |
      import Mathlib

      theorem lemma_true : True := by
        trivial
"""


PLAN_WITH_INVALID_EXPERIENCE = """
plan_id: invalid_experience_plan
theorem_statement: "True"
subgoals:
  - id: sg_goal
    statement: "True"
    status: pending
experiences:
  - exp_id: exp_good
    exp_type: failure_pattern
    content: "Keep the theorem statement fixed."
  - exp_id: exp_bad
    exp_type: failure_pattern
"""


class TestProofExperience(unittest.TestCase):
    def test_extract_and_attach_experiences_updates_program_and_plan(self):
        program = Program(
            id="1_0_child",
            code=SOLVED_PLAN,
            generation=1,
            iteration=1,
            language=Language.LEAN,
            combined_score=1.0,
            search_score=0.95,
            metrics={
                "theorem_solved": 1.0,
                "compile_success": 1.0,
                "no_sorry": 1.0,
                "statement_match": 1.0,
            },
            meta={"theorem_statement": "True"},
        )

        extract_and_attach_experiences(program)

        self.assertIn("experience_pool", program.meta)
        self.assertGreaterEqual(len(program.meta["experience_pool"]), 1)
        self.assertIn("experience_refs", program.meta)

        plan = parse_program_plan(program, fallback_theorem="True")
        experiences = load_experiences(plan)
        self.assertGreaterEqual(len(experiences), 1)
        self.assertTrue(any(exp.exp_type in {"success_summary", "lemma_memory"} for exp in experiences))

    def test_plan_mutation_prompt_includes_parent_experiences(self):
        parent = Program(
            id="0_0_parent",
            code=SOLVED_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=1.0,
            search_score=0.9,
            metrics={"theorem_solved": 1.0},
            meta={"theorem_statement": "True"},
        )
        extract_and_attach_experiences(parent)

        context = Context(
            experiment_id="exp_test",
            task_description="Prove True in Lean",
            task_constraint="Keep theorem fixed.",
            population={"main": [parent]},
        )
        result = RolloutResult(
            rollout_id="rollout_1",
            iteration=1,
            selected_parent_id=parent.id,
            inspiration_ids=[],
        )

        generator = PlanMutationGenerate(language="lean", temperature=0.0, max_tokens=512)
        prompt = generator.build_prompt(parent, context, result)

        self.assertIn("Recommended mutation agenda", prompt)
        self.assertIn("mutation_mode", prompt)
        self.assertIn("Parent experience memory", prompt)
        self.assertIn("Verified reusable lemmas", prompt)
        self.assertIn("lemma_true", prompt)

    def test_repair_agenda_prefers_eliminate_sorry(self):
        parent = Program(
            id="0_0_unsolved",
            code=SOLVED_PLAN,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.3,
            metrics={
                "theorem_solved": 0.0,
                "compile_success": 1.0,
                "no_sorry": 0.0,
                "statement_match": 1.0,
                "frontier_nodes": ["sg_goal"],
                "blocked_nodes": [],
            },
            meta={"theorem_statement": "True"},
        )
        extract_and_attach_experiences(parent)

        agenda = build_repair_agenda(parent)

        self.assertEqual(agenda["mutation_mode"], "eliminate_sorry")
        self.assertTrue(any("replace sorry" in item for item in agenda["primary_targets"]))
        self.assertTrue(any("sg_goal" in item for item in agenda["primary_targets"]))

    def test_invalid_experiences_are_skipped_instead_of_crashing(self):
        program = Program(
            id="0_0_invalid_exp",
            code=PLAN_WITH_INVALID_EXPERIENCE,
            generation=0,
            iteration=0,
            language=Language.LEAN,
            combined_score=0.0,
            search_score=0.1,
            meta={"theorem_statement": "True"},
        )

        plan = parse_program_plan(program, fallback_theorem="True")
        experiences = load_experiences(plan)

        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0].exp_id, "exp_good")
        self.assertEqual(plan.metadata.get("invalid_experience_count"), 1)
        prompt_text = format_experiences_for_prompt(program)
        self.assertIn("Keep the theorem statement fixed.", prompt_text)


if __name__ == "__main__":
    unittest.main()
