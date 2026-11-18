from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from famou.utils.plan_runtime import parse_plan
from famou.utils.proof_runtime_bridge import (
    make_proof_runtime_bridge_verifier,
    merge_lean_sources,
)


PLAN_WITH_STEPS = """
plan_id: bridge_plan
theorem_statement: "True"
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
    status: attempted
    local_lean_draft: |
      import Mathlib

      theorem sg_goal : True := by
        exact sg_intro
"""


class TestProofRuntimeBridge(unittest.TestCase):
    def test_merge_lean_sources_deduplicates_imports(self):
        merged = merge_lean_sources(
            [
                "import Mathlib\n\nlemma helper_true : True := by\n  trivial\n",
                "import Mathlib\nimport Project.Basic\n\ntheorem goal : True := by\n  exact helper_true\n",
            ],
            banner="bridge test",
        )

        self.assertEqual(merged.count("import Mathlib"), 1)
        self.assertIn("import Project.Basic", merged)
        self.assertIn("lemma helper_true : True := by", merged)
        self.assertIn("theorem goal : True := by", merged)

    def test_bridge_verifier_uses_step_prefix_context(self):
        plan = parse_plan(PLAN_WITH_STEPS)
        node = plan.node_map()["sg_goal"]

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            artifact_root = Path(tmpdir) / "artifacts"
            project_root.mkdir(parents=True)
            artifact_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (artifact_root / "definition.lean").write_text(
                "import Project.Basic\n\nlemma helper_true : True := by\n  trivial\n",
                encoding="utf-8",
            )
            (artifact_root / "step_1.lean").write_text(
                "import Mathlib\n\ntheorem sg_intro : True := by\n  exact helper_true\n",
                encoding="utf-8",
            )
            (artifact_root / "step_2.lean").write_text(
                "import Mathlib\n\ntheorem sg_goal : True := by\n  exact sg_intro\n",
                encoding="utf-8",
            )
            (artifact_root / "sg_goal.lean").write_text(
                "import Mathlib\n\ntheorem sg_goal : True := by\n  exact sg_intro\n",
                encoding="utf-8",
            )
            (artifact_root / "artifact_manifest.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {"node_id": "sg_intro", "step_index": 1, "path": str(artifact_root / "step_1.lean")},
                            {"node_id": "sg_goal", "step_index": 2, "path": str(artifact_root / "step_2.lean")},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            captured = {}

            def fake_runner(command, cwd, env, timeout):
                captured["command"] = command
                captured["cwd"] = cwd
                wrapper_path = Path(cwd) / command[-1]
                captured["wrapper_text"] = wrapper_path.read_text(encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            verifier = make_proof_runtime_bridge_verifier(
                project_root=project_root,
                staging_root=project_root / "temp_plan_bridge",
                lean_command=["fake-lean"],
                command_runner=fake_runner,
            )
            result = verifier(
                artifact_root / "sg_goal.lean",
                (artifact_root / "sg_goal.lean").read_text(encoding="utf-8"),
                plan,
                node,
            )

            self.assertEqual(result["compile_success"], 1.0)
            self.assertEqual(result["no_sorry"], 1.0)
            self.assertEqual(result["proof_validity"], 1.0)
            self.assertEqual(captured["command"][:-1], ["fake-lean"])
            self.assertEqual(captured["cwd"], project_root)
            self.assertIn("lemma helper_true : True := by", captured["wrapper_text"])
            self.assertIn("theorem sg_intro : True := by", captured["wrapper_text"])
            self.assertIn("theorem sg_goal : True := by", captured["wrapper_text"])
            self.assertTrue((Path(result["staged_dir"]) / "step_2.lean").exists())

    def test_bridge_final_theorem_avoids_duplicate_terminal_declaration(self):
        plan = parse_plan(PLAN_WITH_STEPS)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            artifact_root = Path(tmpdir) / "artifacts"
            project_root.mkdir(parents=True)
            artifact_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (artifact_root / "definition.lean").write_text("import Project.Basic\n", encoding="utf-8")
            (artifact_root / "step_1.lean").write_text(
                "import Mathlib\n\ntheorem sg_intro : True := by\n  trivial\n",
                encoding="utf-8",
            )
            (artifact_root / "step_2.lean").write_text(
                "import Mathlib\n\ntheorem sg_goal : True := by\n  exact sg_intro\n",
                encoding="utf-8",
            )
            final_code = "import Mathlib\n\ntheorem sg_goal : True := by\n  exact sg_intro\n"
            (artifact_root / "final_theorem.lean").write_text(final_code, encoding="utf-8")
            (artifact_root / "artifact_manifest.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {"node_id": "sg_intro", "step_index": 1, "path": str(artifact_root / "step_1.lean")},
                            {"node_id": "sg_goal", "step_index": 2, "path": str(artifact_root / "step_2.lean")},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            captured = {}

            def fake_runner(command, cwd, env, timeout):
                wrapper_path = Path(cwd) / command[-1]
                captured["wrapper_text"] = wrapper_path.read_text(encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            verifier = make_proof_runtime_bridge_verifier(
                project_root=project_root,
                staging_root=project_root / "temp_plan_bridge",
                lean_command=["fake-lean"],
                command_runner=fake_runner,
            )
            result = verifier(artifact_root / "final_theorem.lean", final_code, plan, None)

            self.assertEqual(result["compile_success"], 1.0)
            self.assertIn("theorem sg_intro : True := by", captured["wrapper_text"])
            self.assertEqual(captured["wrapper_text"].count("theorem sg_goal : True := by"), 1)

    def test_bridge_verifier_propagates_failure_diagnostics(self):
        plan = parse_plan(PLAN_WITH_STEPS)
        node = plan.node_map()["sg_goal"]

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            artifact_root = Path(tmpdir) / "artifacts"
            project_root.mkdir(parents=True)
            artifact_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            lean_code = "import Mathlib\n\ntheorem sg_goal : True := by\n  sorry\n"
            (artifact_root / "sg_goal.lean").write_text(lean_code, encoding="utf-8")

            def fake_runner(command, cwd, env, timeout):
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="synthetic lean failure")

            verifier = make_proof_runtime_bridge_verifier(
                project_root=project_root,
                staging_root=project_root / "temp_plan_bridge",
                lean_command=["fake-lean"],
                command_runner=fake_runner,
            )
            result = verifier(artifact_root / "sg_goal.lean", lean_code, plan, node)

            self.assertEqual(result["compile_success"], 0.0)
            self.assertEqual(result["no_sorry"], 0.0)
            self.assertEqual(result["proof_validity"], 0.0)
            self.assertTrue(result["diagnostics"])
            self.assertIn("synthetic lean failure", result["diagnostics"][0])

    def test_bridge_appends_scaffold_suffix_to_wrapper(self):
        plan = parse_plan(PLAN_WITH_STEPS)
        node = plan.node_map()["sg_goal"]

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            artifact_root = Path(tmpdir) / "artifacts"
            project_root.mkdir(parents=True)
            artifact_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            (artifact_root / "definition.lean").write_text(
                "namespace Demo\n\nlemma helper_true : True := by\n  trivial\n",
                encoding="utf-8",
            )
            lean_code = "theorem sg_goal : True := by\n  exact helper_true\n"
            (artifact_root / "sg_goal.lean").write_text(lean_code, encoding="utf-8")
            (artifact_root / "artifact_manifest.json").write_text(
                json.dumps(
                    {
                        "steps": [],
                        "scaffold_preamble": "namespace Demo",
                        "scaffold_suffix": "end Demo",
                    }
                ),
                encoding="utf-8",
            )

            captured = {}

            def fake_runner(command, cwd, env, timeout):
                wrapper_path = Path(cwd) / command[-1]
                captured["wrapper_text"] = wrapper_path.read_text(encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            verifier = make_proof_runtime_bridge_verifier(
                project_root=project_root,
                staging_root=project_root / "temp_plan_bridge",
                lean_command=["fake-lean"],
                command_runner=fake_runner,
            )
            result = verifier(artifact_root / "sg_goal.lean", lean_code, plan, node)

            self.assertEqual(result["compile_success"], 1.0)
            self.assertEqual(captured["wrapper_text"].count("namespace Demo"), 1)
            self.assertTrue(captured["wrapper_text"].strip().endswith("end Demo"))

    def test_bridge_retries_with_lake_env_lean_after_timeout(self):
        plan = parse_plan(PLAN_WITH_STEPS)
        node = plan.node_map()["sg_goal"]

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            artifact_root = Path(tmpdir) / "artifacts"
            project_root.mkdir(parents=True)
            artifact_root.mkdir(parents=True)

            (project_root / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")
            lean_code = "import Mathlib\n\ntheorem sg_goal : True := by\n  trivial\n"
            (artifact_root / "sg_goal.lean").write_text(lean_code, encoding="utf-8")

            seen_commands = []

            def fake_runner(command, cwd, env, timeout):
                seen_commands.append(command)
                if command[:2] == ["lake", "lean"]:
                    raise subprocess.TimeoutExpired(command, timeout)
                return subprocess.CompletedProcess(command, 0, stdout="fallback ok", stderr="")

            verifier = make_proof_runtime_bridge_verifier(
                project_root=project_root,
                staging_root=project_root / "temp_plan_bridge",
                lean_command=["lake", "lean"],
                command_runner=fake_runner,
                timeout=7,
            )
            result = verifier(artifact_root / "sg_goal.lean", lean_code, plan, node)

            self.assertEqual(result["compile_success"], 1.0)
            self.assertEqual(seen_commands[0][:2], ["lake", "lean"])
            self.assertEqual(seen_commands[1][:3], ["lake", "env", "lean"])
            self.assertEqual(result["command"][:3], ["lake", "env", "lean"])
            self.assertEqual(result["command_attempts"][1][:3], ["lake", "env", "lean"])
            self.assertTrue(any("retrying with fallback command" in item for item in result["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
