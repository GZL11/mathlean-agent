from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from famou.utils.plan_lean_evaluator import (
    build_full_theorem_artifact,
    build_node_lean_artifact,
    create_plan_lean_evaluator,
)
from famou.utils.plan_runtime import compute_combined_plan_score, parse_plan


PLAN_WITH_DRAFTS = """
plan_id: solved_plan
theorem_statement: "True"
plan_summary: "A tiny proof plan"
strategy_type: direct
subgoals:
  - id: sg_intro
    statement: "True"
    status: verified
    local_lean_draft: |
      import Mathlib

      theorem sg_intro : True := by
        trivial
  - id: sg_goal
    statement: "True"
    dependencies: [sg_intro]
    status: in_progress
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        exact True.intro
lemmas:
  - id: lemma_true
    statement: "True"
    status: pending
    reusable_score: 0.5
    lean_code: |
      import Mathlib

      theorem lemma_true : True := by
        trivial
"""


PLAN_WITH_SORRY = """
plan_id: sorry_plan
theorem_statement: "True"
subgoals:
  - id: sg_goal
    statement: "True"
    status: attempted
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        sorry
"""


PLAN_WITH_SCAFFOLD = """
plan_id: scaffold_plan
theorem_statement: "True"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      by
        exact True.intro
"""


PLAN_WITH_PRECISE_IMPORTS = """
plan_id: precise_import_plan
theorem_statement: "True"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      theorem sg_goal : True := by
        trivial
lemmas:
  - id: helper_true
    statement: "True"
    status: pending
    lean_code: |
      theorem helper_true : True := by
        trivial
"""


PLAN_WITH_ZERO_IMPORT_SCAFFOLD = """
plan_id: zero_import_plan
theorem_statement: "True"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      theorem sg_goal : True := by
        trivial
"""


PLAN_WITH_NAMESPACE_SCAFFOLD = """
plan_id: namespace_scaffold_plan
theorem_statement: "P → P"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "P → P"
    status: in_progress
    local_lean_draft: |
      theorem local_goal (P : Prop) : P → P := by
        intro h
        exact h
"""


PLAN_WITH_BEGIN_END_SCAFFOLD = """
plan_id: begin_end_scaffold_plan
theorem_statement: "True"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      theorem final_goal : True := by
        trivial
"""


PLAN_WITH_BINDER_SCAFFOLD = """
plan_id: binder_scaffold_plan
theorem_statement: "x = 42"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "x = 42"
    status: in_progress
    local_lean_draft: |
      have hx : x = 42 := by
        linarith [h₀, h₁]
      exact hx
"""


PLAN_WITH_BINDER_SCAFFOLD_DECL = """
plan_id: binder_scaffold_decl_plan
theorem_statement: "x = 42"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "x = 42"
    status: in_progress
    local_lean_draft: |
      theorem local_goal : x = 42 := by
        have hx : x = 42 := by
          linarith [h₀, h₁]
        exact hx
"""


PLAN_WITH_EXISTING_PROOF_SCAFFOLD = """
plan_id: existing_proof_scaffold_plan
theorem_statement: "True"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      exact True.intro
"""


PLAN_WITH_MIXED_BARE_LEMMA_BODIES = """
plan_id: bare_lemma_body_plan
theorem_statement: "True"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      theorem sg_goal : True := by
        exact True.intro
lemmas:
  - id: lemma_true_good
    statement: "True"
    status: verified
    reusable_score: 0.9
    lean_code: |
      by
        exact True.intro
  - id: lemma_true_bad
    statement: "True"
    status: attempted
    reusable_score: 0.1
    lean_code: |
      by
        exact missing_hyp
"""


PLAN_WITH_COMPLEXITY_PRESSURE = """
plan_id: complexity_plan
theorem_statement: "True"
subgoals:
  - id: sg1
    statement: "True"
    status: verified
  - id: sg2
    statement: "True"
    dependencies: [sg1]
    status: verified
  - id: sg3
    statement: "True"
    dependencies: [sg2]
    status: pending
  - id: sg4
    statement: "True"
    dependencies: [sg3]
    status: pending
  - id: sg5
    statement: "True"
    dependencies: [sg4]
    status: pending
lemmas:
  - id: lemma_a
    statement: "True"
    status: verified
    reusable_score: 0.6
  - id: lemma_b
    statement: "True"
    status: verified
    reusable_score: 0.6
  - id: lemma_c
    statement: "True"
    status: attempted
    reusable_score: 0.1
"""


PLAN_WITH_MALFORMED_DOC_COMMENT = """
plan_id: broken_comment_plan
theorem_statement: "True"
lean_artifacts:
  theorem_scaffold_path: "__SCAFFOLD__"
subgoals:
  - id: sg_goal
    statement: "True"
    status: in_progress
    local_lean_draft: |
      /--
      Broken generated comment that never closes

      theorem broken_comment_plan_sg_goal : True := by
        exact True.intro
"""


def mock_verifier(artifact_path: Path, lean_code: str, plan, node):
    has_sorry = "sorry" in lean_code
    return {
        "compile_success": 1.0,
        "no_sorry": 0.0 if has_sorry else 1.0,
        "statement_match": 1.0,
        "proof_validity": 0.0 if has_sorry else 1.0,
        "diagnostics": [] if not has_sorry else ["contains sorry"],
        "checked_file": str(artifact_path),
    }


class TestRewardBreakdown(unittest.TestCase):
    def test_outer_reward_with_cost_penalty(self):
        plan = parse_plan(PLAN_WITH_DRAFTS)
        score = compute_combined_plan_score(
            plan,
            extra_metrics={
                "theorem_solved": 1.0,
                "proof_validity": 1.0,
                "compile_success": 1.0,
                "no_sorry": 1.0,
                "verifier_calls": 4,
                "verifier_budget": 8,
            },
            reward_mode="outer",
        )
        self.assertEqual(score["reward_mode"], "outer")
        self.assertEqual(score["theorem_solved"], 1.0)
        self.assertEqual(score["combined_score"], 1.0)
        self.assertGreater(score["search_score"], 0.7)
        self.assertGreater(score["cost_penalty"], 0.0)
        self.assertLess(score["cost_penalty"], 0.2)

    def test_inner_reward_emphasizes_progress(self):
        plan = parse_plan(PLAN_WITH_SORRY)
        score = compute_combined_plan_score(
            plan,
            extra_metrics={
                "theorem_solved": 0.0,
                "proof_validity": 0.0,
                "compile_success": 1.0,
                "no_sorry": 0.0,
            },
            reward_mode="inner",
        )
        self.assertEqual(score["reward_mode"], "inner")
        self.assertGreaterEqual(score["node_progress_component"], 0.25)
        self.assertEqual(score["combined_score"], 0.0)
        self.assertGreater(score["search_score"], 0.0)

    def test_outer_reward_exposes_memory_gain_and_complexity_penalty(self):
        plan = parse_plan(PLAN_WITH_COMPLEXITY_PRESSURE)
        score = compute_combined_plan_score(
            plan,
            extra_metrics={
                "theorem_solved": 0.0,
                "proof_validity": 0.0,
                "compile_success": 0.8,
                "no_sorry": 0.8,
                "experience_gain": 0.5,
            },
            reward_mode="outer",
        )
        self.assertIn("realization_score", score)
        self.assertIn("memory_gain_bonus", score)
        self.assertIn("plan_complexity_penalty", score)
        self.assertGreater(score["memory_gain_bonus"], 0.0)
        self.assertGreater(score["plan_complexity_penalty"], 0.0)
        self.assertGreater(score["subgoal_complexity_excess"], 0.0)
        self.assertGreater(score["lemma_complexity_excess"], 0.0)


class TestPlanLeanEvaluator(unittest.TestCase):
    def test_build_node_artifact_normalizes_multiline_tactic_block(self):
        plan = parse_plan(
            """
plan_id: multiline_block_plan
theorem_statement: "x = 42"
subgoals:
  - id: sg_goal
    statement: "x = 42"
    status: in_progress
    local_lean_draft: |
      have hx : x = 42 := by
        linarith [h₀, h₁]
      exact hx
"""
        )
        artifact = build_node_lean_artifact(
            plan.subgoals[0],
            plan,
            default_imports="import Mathlib",
        )

        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertIn("theorem multiline_block_plan_sg_goal : x = 42 := by", artifact)
        self.assertIn("  have hx : x = 42 := by", artifact)
        self.assertIn("    linarith [h₀, h₁]", artifact)
        self.assertIn("  exact hx", artifact)

    def test_evaluator_updates_reward_and_artifacts(self):
        evaluator = create_plan_lean_evaluator(verifier_fn=mock_verifier, cleanup_artifacts=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(PLAN_WITH_DRAFTS, encoding="utf-8")

            result = evaluator(str(plan_path))

            self.assertEqual(result["theorem_solved"], 1.0)
            self.assertEqual(result["proof_validity"], 1.0)
            self.assertEqual(result["combined_score"], 1.0)
            self.assertGreater(result["search_score"], 0.8)
            self.assertEqual(result["verified_lemmas"], 1)
            self.assertIn("realization_score", result)
            self.assertIn("memory_gain_bonus", result)
            self.assertIn("plan_complexity_penalty", result)
            self.assertTrue(Path(result["generated_folder"]).exists())
            self.assertEqual(len(result["subgoal_results"]), 2)
            self.assertEqual(len(result["lemma_results"]), 1)
            self.assertIn("plan_object", result)
            self.assertIn("proof_runtime_artifacts", result)
            runtime = result["proof_runtime_artifacts"]
            self.assertTrue(Path(runtime["definition_path"]).exists())
            self.assertEqual(len(runtime["steps"]), 2)
            self.assertTrue(Path(runtime["steps"][0]["path"]).exists())
            self.assertTrue(Path(runtime["manifest_path"]).exists())
            self.assertEqual(result["plan_object"]["combined_score"], result["combined_score"])

    def test_evaluator_blocks_statement_mismatch(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            theorem_statement="False",
            cleanup_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(PLAN_WITH_DRAFTS, encoding="utf-8")

            result = evaluator(str(plan_path))

            self.assertEqual(result["statement_match"], 0.0)
            self.assertEqual(result["theorem_solved"], 0.0)
            self.assertEqual(result["combined_score"], 0.0)
            self.assertLess(result["search_score"], 0.8)

    def test_evaluator_builds_full_theorem_from_scaffold(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                "import Mathlib\n\ntheorem final_goal : True := by\n  sorry\n",
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))

            self.assertIsNotNone(result["full_theorem_result"])
            self.assertEqual(result["full_theorem_result"]["compile_success"], 1.0)
            self.assertEqual(result["full_theorem_result"]["no_sorry"], 1.0)
            self.assertEqual(result["theorem_solved"], 1.0)
            self.assertEqual(result["final_no_sorry"], 1.0)

    def test_evaluator_uses_single_node_fast_path_for_scaffold_theorem(self):
        seen_artifacts = []

        def counting_verifier(artifact_path: Path, lean_code: str, plan, node):
            seen_artifacts.append(artifact_path.name)
            return {
                "compile_success": 1.0,
                "no_sorry": 1.0,
                "statement_match": 1.0,
                "proof_validity": 1.0,
                "diagnostics": [],
                "checked_file": str(artifact_path),
            }

        evaluator = create_plan_lean_evaluator(
            verifier_fn=counting_verifier,
            cleanup_artifacts=False,
            verify_lemmas=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                "import Mathlib\n\ntheorem final_goal : True := by\n  sorry\n",
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))

            self.assertEqual(seen_artifacts, ["final_theorem.lean"])
            self.assertEqual(result["verifier_calls"], 1)
            self.assertEqual(len(result["subgoal_results"]), 1)
            self.assertIn("single_node_fast_path", " ".join(result["subgoal_results"][0]["diagnostics"]))
            self.assertEqual(
                result["subgoal_results"][0]["artifact_path"],
                result["full_theorem_result"]["artifact_path"],
            )

    def test_evaluator_rebuilds_full_theorem_from_scaffold_without_sorry(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                "import Mathlib\n\ntheorem final_goal : True := by\n  trivial\n",
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_EXISTING_PROOF_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))

            self.assertIsNotNone(result["full_theorem_result"])
            self.assertEqual(result["full_theorem_result"]["compile_success"], 1.0)
            self.assertEqual(result["full_theorem_result"]["no_sorry"], 1.0)
            self.assertEqual(result["theorem_solved"], 1.0)

            final_theorem_text = Path(result["full_theorem_result"]["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("theorem final_goal : True := by", final_theorem_text)
            self.assertNotIn("theorem existing_proof_scaffold_plan_sg_goal", final_theorem_text)
            self.assertIn("exact True.intro", final_theorem_text)

    def test_evaluator_materializes_bare_lemma_bodies_as_declarations(self):
        def strict_verifier(artifact_path: Path, lean_code: str, plan, node):
            has_bad_name = "missing_hyp" in lean_code
            has_sorry = "sorry" in lean_code
            compile_success = 0.0 if has_bad_name else 1.0
            return {
                "compile_success": compile_success,
                "no_sorry": 0.0 if has_sorry else 1.0,
                "statement_match": 1.0,
                "proof_validity": 1.0 if compile_success and not has_sorry else 0.0,
                "diagnostics": [] if compile_success else ["unknown identifier 'missing_hyp'"],
                "checked_file": str(artifact_path),
            }

        evaluator = create_plan_lean_evaluator(
            verifier_fn=strict_verifier,
            cleanup_artifacts=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(PLAN_WITH_MIXED_BARE_LEMMA_BODIES, encoding="utf-8")

            result = evaluator(str(plan_path))

            runtime = result["proof_runtime_artifacts"]
            definition_text = Path(runtime["definition_path"]).read_text(encoding="utf-8")
            self.assertIn("theorem lemma_true_good : True := by", definition_text)
            self.assertNotIn("\nby\n  exact True.intro", definition_text)
            self.assertNotIn("missing_hyp", definition_text)

            lemma_text = Path(result["lemma_results"][0]["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("theorem lemma_true_good : True := by", lemma_text)

            final_theorem_text = Path(result["full_theorem_result"]["artifact_path"]).read_text(encoding="utf-8")
            self.assertTrue(final_theorem_text.startswith("import Mathlib"))
            self.assertIn("theorem lemma_true_good : True := by", final_theorem_text)
            self.assertNotIn("missing_hyp", final_theorem_text)
            self.assertNotIn("\nby\n  exact True.intro\n\n", final_theorem_text)

    def test_full_theorem_artifact_strips_malformed_generated_doc_comments(self):
        plan = parse_plan(
            PLAN_WITH_MALFORMED_DOC_COMMENT.replace("__SCAFFOLD__", "/tmp/nonexistent_problem.lean")
        )

        artifact = build_full_theorem_artifact(
            plan,
            terminal_nodes=plan.subgoals,
            verified_lemma_decls=[],
            default_imports="import Mathlib",
        )

        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertIn("theorem broken_comment_plan_sg_goal : True := by", artifact)
        self.assertNotIn("/--", artifact)

    def test_evaluator_prefers_scaffold_imports_for_generated_artifacts(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                "import Lean\nimport Std\n\ntheorem final_goal : True := by\n  sorry\n",
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_PRECISE_IMPORTS.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))

            runtime = result["proof_runtime_artifacts"]
            definition_text = Path(runtime["definition_path"]).read_text(encoding="utf-8")
            self.assertIn("import Lean", definition_text)
            self.assertIn("import Std", definition_text)
            self.assertNotIn("import Mathlib\n\n-- No additional", definition_text)
            self.assertEqual(runtime["effective_imports"], "import Lean\nimport Std")

            node_artifact = Path(result["subgoal_results"][0]["artifact_path"]).read_text(encoding="utf-8")
            self.assertTrue(node_artifact.startswith("import Lean\nimport Std"))

            lemma_artifact = Path(result["lemma_results"][0]["artifact_path"]).read_text(encoding="utf-8")
            self.assertTrue(lemma_artifact.startswith("import Lean\nimport Std"))

    def test_evaluator_respects_zero_import_scaffold(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                "theorem final_goal : True := by\n  sorry\n",
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_ZERO_IMPORT_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))
            runtime = result["proof_runtime_artifacts"]
            self.assertEqual(runtime["effective_imports"], "")

            node_artifact = Path(result["subgoal_results"][0]["artifact_path"]).read_text(encoding="utf-8")
            self.assertFalse(node_artifact.startswith("import Mathlib"))

    def test_evaluator_carries_scaffold_preamble_and_suffix(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                (
                    "namespace Demo\n"
                    "variable (P : Prop)\n\n"
                    "/-- Example theorem doc comment. -/\n"
                    "theorem final_goal : P → P := by\n"
                    "  sorry\n\n"
                    "end Demo\n"
                ),
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_NAMESPACE_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))
            runtime = result["proof_runtime_artifacts"]

            self.assertIn("namespace Demo", runtime["scaffold_preamble"])
            self.assertIn("variable (P : Prop)", runtime["scaffold_preamble"])
            self.assertNotIn("Example theorem doc comment", runtime["scaffold_preamble"])
            self.assertEqual(runtime["scaffold_suffix"], "end Demo")

            definition_text = Path(runtime["definition_path"]).read_text(encoding="utf-8")
            self.assertIn("namespace Demo", definition_text)
            self.assertNotIn("Example theorem doc comment", definition_text)

            node_artifact = Path(result["subgoal_results"][0]["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("namespace Demo", node_artifact)
            self.assertTrue(node_artifact.strip().endswith("end Demo"))

            final_theorem_text = Path(result["full_theorem_result"]["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("theorem final_goal", final_theorem_text)
            self.assertNotIn("theorem local_goal", final_theorem_text)
            self.assertNotIn(":= by\n  by", final_theorem_text)
            self.assertIn(":= by\n  intro h\n  exact h", final_theorem_text)

    def test_evaluator_strips_multiline_doc_comment_from_scaffold_preamble(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                (
                    "import Mathlib\n\n"
                    "open MeasureTheory\n\n"
                    "/--\n"
                    "Given five points in a plane.\n"
                    "-/\n"
                    "theorem final_goal : True := by\n"
                    "  sorry\n"
                ),
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))
            runtime = result["proof_runtime_artifacts"]

            self.assertEqual(runtime["scaffold_preamble"].strip(), "open MeasureTheory")

            definition_text = Path(runtime["definition_path"]).read_text(encoding="utf-8")
            self.assertIn("open MeasureTheory", definition_text)
            self.assertNotIn("/--", definition_text)
            self.assertNotIn("Given five points in a plane.", definition_text)

            final_theorem_text = Path(result["full_theorem_result"]["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("theorem final_goal : True := by", final_theorem_text)

    def test_evaluator_drops_bare_end_suffix_without_scope_opener(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                (
                    "import Mathlib\n\n"
                    "open scoped BigOperators\n\n"
                    "theorem final_goal : True :=\n"
                    "begin\n"
                    "  trivial\n"
                    "end\n"
                ),
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_BEGIN_END_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))
            runtime = result["proof_runtime_artifacts"]

            self.assertEqual(runtime["scaffold_suffix"], "")
            node_artifact = Path(result["subgoal_results"][0]["artifact_path"]).read_text(encoding="utf-8")
            self.assertFalse(node_artifact.strip().endswith("end"))

    def test_evaluator_materializes_step_artifacts_with_scaffold_binders(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                (
                    "import Mathlib\n\n"
                    "theorem final_goal\n"
                    "  (n x : ℝ)\n"
                    "  (h₀ : n + x = 97)\n"
                    "  (h₁ : n + 5 * x = 265) :\n"
                    "  x = 42 := by\n"
                    "  sorry\n"
                ),
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_BINDER_SCAFFOLD.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))

            runtime = result["proof_runtime_artifacts"]
            step_text = Path(runtime["steps"][0]["path"]).read_text(encoding="utf-8")
            self.assertIn("(n x : ℝ)", step_text)
            self.assertIn("(h₀ : n + x = 97)", step_text)
            self.assertIn("(h₁ : n + 5 * x = 265)", step_text)
            self.assertIn("    linarith [h₀, h₁]", step_text)

    def test_evaluator_materializes_full_theorem_with_scaffold_binders_from_decl(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
            default_imports="import Mathlib",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scaffold_path = Path(tmpdir) / "problem.lean"
            scaffold_path.write_text(
                (
                    "import Mathlib\n\n"
                    "theorem final_goal\n"
                    "  (n x : ℝ)\n"
                    "  (h₀ : n + x = 97)\n"
                    "  (h₁ : n + 5 * x = 265) :\n"
                    "  x = 42 := by\n"
                    "  sorry\n"
                ),
                encoding="utf-8",
            )
            plan_path = Path(tmpdir) / "plan.yaml"
            plan_path.write_text(
                PLAN_WITH_BINDER_SCAFFOLD_DECL.replace("__SCAFFOLD__", str(scaffold_path)),
                encoding="utf-8",
            )

            result = evaluator(str(plan_path))

            final_theorem_text = Path(result["full_theorem_result"]["artifact_path"]).read_text(encoding="utf-8")
            self.assertIn("(n x : ℝ)", final_theorem_text)
            self.assertIn("(h₀ : n + x = 97)", final_theorem_text)
            self.assertIn("(h₁ : n + 5 * x = 265)", final_theorem_text)
            self.assertNotIn("theorem local_goal", final_theorem_text)
            self.assertIn("linarith [h₀, h₁]", final_theorem_text)
            self.assertEqual(result["full_theorem_result"]["compile_success"], 1.0)

    def test_compute_theorem_solved_requires_full_artifact_when_requested(self):
        evaluator = create_plan_lean_evaluator(
            verifier_fn=mock_verifier,
            cleanup_artifacts=False,
        )
        plan = parse_plan(PLAN_WITH_SCAFFOLD)
        subgoal_results = [
            {
                "node_id": "sg_goal",
                "compile_success": 1.0,
                "no_sorry": 1.0,
                "statement_match": 1.0,
            }
        ]

        solved = evaluator._compute_theorem_solved(
            plan,
            subgoal_results,
            expected_theorem="True",
            full_theorem_result={
                "artifact_path": None,
                "compile_success": 0.0,
                "no_sorry": 0.0,
                "statement_match": 0.0,
            },
            require_full_theorem=True,
        )

        self.assertEqual(solved, 0.0)


if __name__ == "__main__":
    unittest.main()
