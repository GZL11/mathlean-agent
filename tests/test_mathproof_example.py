from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from famou.utils.mathproof_example import create_mathproof_example_evaluator


PLAN_TEXT = """
plan_id: example_seed
theorem_statement: "P → P"
subgoals:
  - id: sg_goal
    statement: "P → P"
    status: in_progress
    local_lean_draft: |
      by
        intro h
        exact h
"""


def mock_verifier(artifact_path: Path, lean_code: str, plan, node):
    return {
        "compile_success": 1.0,
        "no_sorry": 1.0,
        "statement_match": 1.0,
        "proof_validity": 1.0,
        "diagnostics": [],
        "checked_file": str(artifact_path),
    }


class TestMathproofExampleSupport(unittest.TestCase):
    def test_example_evaluator_uses_higher_default_lean_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            example_dir = repo_root / "examples" / "demo"
            project_root = repo_root / "proof_runtime" / "project"
            example_dir.mkdir(parents=True)
            project_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (example_dir / "problem.lean").write_text(
                "namespace Demo\nvariable (P : Prop)\n\ntheorem target : P → P := by\n  sorry\n\nend Demo\n",
                encoding="utf-8",
            )
            (example_dir / "init.scratch").write_text(PLAN_TEXT, encoding="utf-8")

            old_timeout = os.environ.pop("FAMOU_LEAN_VERIFY_TIMEOUT", None)
            try:
                evaluator = create_mathproof_example_evaluator(
                    example_dir=example_dir,
                    verify_all_nodes=True,
                    verify_lemmas=False,
                )
            finally:
                if old_timeout is not None:
                    os.environ["FAMOU_LEAN_VERIFY_TIMEOUT"] = old_timeout

            self.assertEqual(evaluator.verifier_fn.timeout, 240)

    def test_example_evaluator_allows_timeout_env_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            example_dir = repo_root / "examples" / "demo"
            project_root = repo_root / "proof_runtime" / "project"
            example_dir.mkdir(parents=True)
            project_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (example_dir / "problem.lean").write_text(
                "namespace Demo\nvariable (P : Prop)\n\ntheorem target : P → P := by\n  sorry\n\nend Demo\n",
                encoding="utf-8",
            )
            (example_dir / "init.scratch").write_text(PLAN_TEXT, encoding="utf-8")

            old_timeout = os.environ.get("FAMOU_LEAN_VERIFY_TIMEOUT")
            os.environ["FAMOU_LEAN_VERIFY_TIMEOUT"] = "90"
            try:
                evaluator = create_mathproof_example_evaluator(
                    example_dir=example_dir,
                    verify_all_nodes=True,
                    verify_lemmas=False,
                )
            finally:
                if old_timeout is None:
                    os.environ.pop("FAMOU_LEAN_VERIFY_TIMEOUT", None)
                else:
                    os.environ["FAMOU_LEAN_VERIFY_TIMEOUT"] = old_timeout

            self.assertEqual(evaluator.verifier_fn.timeout, 90)

    def test_example_evaluator_resolves_problem_scaffold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            example_dir = repo_root / "examples" / "demo"
            project_root = repo_root / "_proof" / "project"
            example_dir.mkdir(parents=True)
            project_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (example_dir / "problem.lean").write_text(
                "namespace Demo\nvariable (P : Prop)\n\ntheorem target : P → P := by\n  sorry\n\nend Demo\n",
                encoding="utf-8",
            )
            plan_path = example_dir / "init.scratch"
            plan_path.write_text(PLAN_TEXT, encoding="utf-8")

            evaluator = create_mathproof_example_evaluator(
                example_dir=example_dir,
                project_root=project_root,
                artifact_root=project_root / "temp_plan_eval",
                staging_root=project_root / "temp" / "plan_bridge",
                verify_all_nodes=True,
                verify_lemmas=False,
            )
            evaluator.verifier_fn = mock_verifier

            result = evaluator(str(plan_path))
            runtime = result["proof_runtime_artifacts"]

            self.assertEqual(result["theorem_solved"], 1.0)
            self.assertEqual(Path(runtime["theorem_scaffold_path"]), example_dir / "problem.lean")
            self.assertIn("namespace Demo", runtime["scaffold_preamble"])
            self.assertEqual(runtime["scaffold_suffix"], "end Demo")

    def test_example_evaluator_supports_proof_runtime_workspace_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            example_dir = repo_root / "examples" / "demo"
            project_root = repo_root / "proof_runtime" / "project"
            example_dir.mkdir(parents=True)
            project_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (example_dir / "problem.lean").write_text(
                "namespace Demo\nvariable (P : Prop)\n\ntheorem target : P → P := by\n  sorry\n\nend Demo\n",
                encoding="utf-8",
            )
            plan_path = example_dir / "init.scratch"
            plan_path.write_text(PLAN_TEXT, encoding="utf-8")

            evaluator = create_mathproof_example_evaluator(
                example_dir=example_dir,
                verify_all_nodes=True,
                verify_lemmas=False,
            )
            evaluator.verifier_fn = mock_verifier

            result = evaluator(str(plan_path))
            runtime = result["proof_runtime_artifacts"]

            self.assertEqual(result["theorem_solved"], 1.0)
            self.assertEqual(Path(runtime["theorem_scaffold_path"]), example_dir / "problem.lean")
            generated_folder = Path(result["generated_folder"]).resolve()
            self.assertTrue(str(generated_folder).startswith(str((project_root / "temp_plan_eval").resolve())))

    def test_example_evaluator_supports_external_materialized_task_dirs(self):
        with tempfile.TemporaryDirectory() as repo_tmpdir, tempfile.TemporaryDirectory() as task_tmpdir:
            repo_root = Path(repo_tmpdir)
            example_dir = Path(task_tmpdir) / "materialized_task"
            project_root = repo_root / "proof_runtime" / "project"
            example_dir.mkdir(parents=True)
            project_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (example_dir / "problem.lean").write_text(
                "namespace Demo\nvariable (P : Prop)\n\ntheorem target : P → P := by\n  sorry\n\nend Demo\n",
                encoding="utf-8",
            )
            plan_path = example_dir / "init.scratch"
            plan_path.write_text(PLAN_TEXT, encoding="utf-8")

            old_repo_root = os.environ.get("FAMOU_REPO_ROOT")
            os.environ["FAMOU_REPO_ROOT"] = str(repo_root)
            try:
                evaluator = create_mathproof_example_evaluator(
                    example_dir=example_dir,
                    verify_all_nodes=True,
                    verify_lemmas=False,
                )
            finally:
                if old_repo_root is None:
                    os.environ.pop("FAMOU_REPO_ROOT", None)
                else:
                    os.environ["FAMOU_REPO_ROOT"] = old_repo_root

            evaluator.verifier_fn = mock_verifier
            result = evaluator(str(plan_path))

            self.assertEqual(result["theorem_solved"], 1.0)
            generated_folder = Path(result["generated_folder"]).resolve()
            self.assertTrue(str(generated_folder).startswith(str((project_root / "temp_plan_eval").resolve())))


if __name__ == "__main__":
    unittest.main()
