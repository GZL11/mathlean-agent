from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from famou.utils.mathproof_benchmarks import (
    benchmark_digest,
    build_seed_plan_text,
    extract_theorem_statement,
    get_benchmark_problem,
    list_minif2f_problems,
    list_putnambench_problems,
    materialize_benchmark_problem,
    materialize_mathproof_task,
)
from famou.utils.mathproof_benchmark_runner import materialize_benchmark_suite
from famou.utils.mathproof_benchmark_runner import prepare_suite_runtime
from famou.utils.mathproof_benchmark_runner import summarize_suite_results


MINIF2F_TEST_TEXT = """
import minif2f_import

open_locale real

theorem demo_problem_1
  (x : ℝ) :
  x = x :=
begin
  ring
end

theorem demo_problem_2
  (n : ℕ) :
  n = n :=
begin
  rfl
end
"""


PUTNAM_TEXT = """
import Mathlib

theorem putnam_demo
  (x : ℕ) :
  x = x :=
by
  rfl
"""


class TestMathproofBenchmarks(unittest.TestCase):
    def _write_program_payload(
        self,
        exp_dir: Path,
        program_id: str,
        *,
        iteration: int,
        generation: int,
        success_score: float,
        search_score: float,
        combined_score: float,
        theorem_solved: float,
    ) -> None:
        programs_dir = exp_dir / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": program_id,
            "iteration": iteration,
            "generation": generation,
            "combined_score": combined_score,
            "search_score": search_score,
            "validity": 1.0,
            "metrics": {
                "success_score": success_score,
                "search_score": search_score,
                "combined_score": combined_score,
                "theorem_solved": theorem_solved,
                "proof_validity": theorem_solved,
                "compile_success": theorem_solved,
                "no_sorry": theorem_solved,
                "statement_match": 1.0,
                "final_compile_success": theorem_solved,
                "final_no_sorry": theorem_solved,
                "final_statement_match": 1.0,
            },
        }
        (programs_dir / f"{program_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_list_minif2f_problems_splits_aggregate_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "miniF2F"
            lean_src = root / "lean" / "src"
            lean_src.mkdir(parents=True)
            (lean_src / "test.lean").write_text(MINIF2F_TEST_TEXT, encoding="utf-8")

            problems = list_minif2f_problems(root, split="test")

            self.assertEqual(len(problems), 2)
            self.assertEqual(problems[0].problem_id, "demo_problem_1")
            self.assertEqual(problems[1].problem_id, "demo_problem_2")
            self.assertEqual(problems[0].benchmark_name, "miniF2F")
            self.assertEqual(problems[0].split, "test")
            self.assertIn("open_locale real", problems[0].preamble)

    def test_materialize_minif2f_problem_creates_single_theorem_scaffold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "miniF2F"
            lean_src = root / "lean" / "src"
            cache_root = Path(tmpdir) / "cache"
            lean_src.mkdir(parents=True)
            (lean_src / "test.lean").write_text(MINIF2F_TEST_TEXT, encoding="utf-8")

            problems = list_minif2f_problems(root, split="test")
            selected = get_benchmark_problem(problems, "demo_problem_2")
            materialized = materialize_benchmark_problem(selected, cache_root)
            text = materialized.read_text(encoding="utf-8")

            self.assertTrue(materialized.exists())
            self.assertIn("import Mathlib", text)
            self.assertNotIn("import minif2f_import", text)
            self.assertNotIn("open_locale real", text)
            self.assertIn("theorem demo_problem_2", text)
            self.assertNotIn("theorem demo_problem_1", text)

    def test_list_putnambench_problems_reads_one_file_per_theorem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PutnamBench"
            lean_src = root / "lean4" / "src"
            lean_src.mkdir(parents=True)
            (lean_src / "putnam_demo.lean").write_text(PUTNAM_TEXT, encoding="utf-8")

            problems = list_putnambench_problems(root)

            self.assertEqual(len(problems), 1)
            self.assertEqual(problems[0].problem_id, "putnam_demo")
            self.assertEqual(problems[0].theorem_name, "putnam_demo")
            self.assertEqual(materialize_benchmark_problem(problems[0], root), problems[0].source_path.resolve())

    def test_extract_theorem_statement_handles_sorry_style_scaffold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theorem_path = Path(tmpdir) / "putnam_demo.lean"
            theorem_path.write_text(PUTNAM_TEXT.replace("by\n  rfl", "sorry"), encoding="utf-8")

            statement = extract_theorem_statement(theorem_path, theorem_name="putnam_demo")

            self.assertEqual(statement, "x = x")

    def test_materialize_mathproof_task_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "miniF2F"
            lean_src = root / "lean" / "src"
            output_root = Path(tmpdir) / "tasks"
            lean_src.mkdir(parents=True)
            (lean_src / "test.lean").write_text(MINIF2F_TEST_TEXT, encoding="utf-8")

            problem = get_benchmark_problem(list_minif2f_problems(root, split="test"), "demo_problem_1")
            task = materialize_mathproof_task(problem, output_root)

            self.assertTrue(task.problem_path.exists())
            self.assertTrue(task.init_scratch_path.exists())
            self.assertTrue((task.task_dir / "metadata.json").exists())
            self.assertIn("theorem demo_problem_1", task.problem_path.read_text(encoding="utf-8"))
            self.assertEqual(task.theorem_statement, "x = x")

            plan_text = task.init_scratch_path.read_text(encoding="utf-8")
            self.assertIn("plan_id: seed_demo_problem_1", plan_text)
            self.assertIn("theorem_statement: x = x", plan_text)

            metadata = json.loads((task.task_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["task_id"], "demo_problem_1")
            self.assertEqual(metadata["theorem_name"], "demo_problem_1")
            self.assertEqual(metadata["theorem_statement"], "x = x")

    def test_build_seed_plan_text_falls_back_when_yaml_missing(self):
        import famou.utils.mathproof_benchmarks as mathproof_benchmarks

        original_yaml = mathproof_benchmarks.yaml
        try:
            mathproof_benchmarks.yaml = None
            rendered = build_seed_plan_text("P -> P", theorem_name="target")
        finally:
            mathproof_benchmarks.yaml = original_yaml

        payload = json.loads(rendered)
        self.assertEqual(payload["theorem_name"], "target")
        self.assertEqual(payload["subgoals"][0]["statement"], "P -> P")

    def test_materialize_benchmark_suite_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PutnamBench"
            lean_src = root / "lean4" / "src"
            output_root = Path(tmpdir) / "suite_out"
            lean_src.mkdir(parents=True)
            (lean_src / "putnam_demo.lean").write_text(PUTNAM_TEXT, encoding="utf-8")

            suite = materialize_benchmark_suite(
                benchmark_name="putnambench",
                benchmark_root=root,
                output_root=output_root,
            )

            self.assertTrue(suite.manifest_path.exists())
            manifest = json.loads(suite.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_name"], "PutnamBench")
            self.assertEqual(manifest["num_tasks"], 1)
            self.assertEqual(manifest["tasks"][0]["task_id"], "putnam_demo")
            self.assertTrue(Path(manifest["tasks"][0]["task_dir"]).exists())

    def test_prepare_suite_runtime_writes_config_evaluator_and_run_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PutnamBench"
            lean_src = root / "lean4" / "src"
            output_root = Path(tmpdir) / "suite_out"
            template_path = Path(tmpdir) / "template.yaml"
            lean_src.mkdir(parents=True)
            (lean_src / "putnam_demo.lean").write_text(PUTNAM_TEXT, encoding="utf-8")
            template_path.write_text(
                """
experiment:
  name: template_exp
  task_description: examples/lean_identity_namespace/problem.lean
  task_constraint: keep theorem unchanged
  language: lean
  strategy: mathproof_plan_island
  max_iterations: 2
  max_workers: 1
  checkpoint_interval: 2
  seed: 42
  island:
    num_islands: 1
    island_size: 2
    migration_interval: 50
    migration_size: 1
    migration_topology: ring
infrastructure:
  llm:
    provider: openai
    api_base: https://example.com/v1
    model: demo-model
    api_key: demo-key
    temperature: 0.0
    max_tokens: 512
    timeout: 60
  storage:
    type: local
    base_path: ./famou_data
  logger:
    type: local
    level: INFO
    file_enabled: true
  env:
    type: local
    timeout: 30
modules:
  generate:
    language: lean
  evaluate:
    language: lean
  judge:
    use_llm: false
problem: {}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            suite = materialize_benchmark_suite(
                benchmark_name="putnambench",
                benchmark_root=root,
                output_root=output_root,
            )
            prepare_suite_runtime(suite, template_path, storage_base_path="./suite_data")

            task_dir = suite.task_specs[0].task_dir
            self.assertTrue((task_dir / "config.yaml").exists())
            self.assertTrue((task_dir / "evaluator.py").exists())
            self.assertTrue((task_dir / "run.sh").exists())
            runtime_manifest = json.loads((suite.suite_root / "runtime_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime_manifest["tasks"][0]["task_id"], "putnam_demo")
            generated_config = (task_dir / "config.yaml").read_text(encoding="utf-8")
            self.assertIn("name: putnambench_all_putnam_demo", generated_config)
            self.assertIn("task_description: problem.lean", generated_config)
            self.assertIn("base_path: ./suite_data", generated_config)

    def test_benchmark_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "miniF2F"
            lean_src = root / "lean" / "src"
            lean_src.mkdir(parents=True)
            (lean_src / "test.lean").write_text(MINIF2F_TEST_TEXT, encoding="utf-8")

            problems = list_minif2f_problems(root, split="test")
            digest_a = benchmark_digest(problems)
            digest_b = benchmark_digest(problems)

            self.assertEqual(digest_a, digest_b)
            self.assertEqual(len(digest_a), 12)

    def test_summarize_suite_results_prefers_checkpoint_best_program(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PutnamBench"
            lean_src = root / "lean4" / "src"
            output_root = Path(tmpdir) / "suite_out"
            template_path = Path(tmpdir) / "template.yaml"
            lean_src.mkdir(parents=True)
            (lean_src / "putnam_demo.lean").write_text(PUTNAM_TEXT, encoding="utf-8")
            template_path.write_text(
                """
experiment:
  name: template_exp
  task_description: examples/lean_identity_namespace/problem.lean
  task_constraint: keep theorem unchanged
  language: lean
  strategy: mathproof_plan_island
  max_iterations: 2
  max_workers: 1
  checkpoint_interval: 2
  seed: 42
  island:
    num_islands: 1
    island_size: 2
    migration_interval: 50
    migration_size: 1
    migration_topology: ring
infrastructure:
  llm:
    provider: openai
    api_base: https://example.com/v1
    model: demo-model
    api_key: demo-key
    temperature: 0.0
    max_tokens: 512
    timeout: 60
  storage:
    type: local
    base_path: ./suite_data
  logger:
    type: local
    level: INFO
    file_enabled: true
  env:
    type: local
    timeout: 30
modules:
  generate:
    language: lean
  evaluate:
    language: lean
  judge:
    use_llm: false
problem: {}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            suite = materialize_benchmark_suite(
                benchmark_name="putnambench",
                benchmark_root=root,
                output_root=output_root,
            )
            prepare_suite_runtime(suite, template_path, storage_base_path="./suite_data")

            task_dir = suite.task_specs[0].task_dir
            exp_dir = task_dir / "suite_data" / "putnambench_all_putnam_demo_run01"
            self._write_program_payload(
                exp_dir,
                "seed_0",
                iteration=0,
                generation=0,
                success_score=0.0,
                search_score=0.18,
                combined_score=0.18,
                theorem_solved=0.0,
            )
            self._write_program_payload(
                exp_dir,
                "2_2_solved",
                iteration=2,
                generation=2,
                success_score=1.0,
                search_score=0.93,
                combined_score=1.0,
                theorem_solved=1.0,
            )
            (exp_dir / "experiment_checkpoint_2.json").write_text(
                json.dumps(
                    {
                        "current_iteration": 2,
                        "best_program_id": "2_2_solved",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            batch_summary_path = suite.suite_root / "batch_run_summary.json"
            batch_summary_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "putnam_demo",
                                "returncode": 0,
                                "timeout": False,
                                "elapsed_seconds": 12.5,
                                "stdout_path": str(task_dir / "batch_stdout.log"),
                                "stderr_path": str(task_dir / "batch_stderr.log"),
                            }
                        ]
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_suite_results(
                suite_root=suite.suite_root,
                batch_summary_path=batch_summary_path,
            )

            self.assertEqual(summary["num_tasks"], 1)
            self.assertEqual(summary["num_solved"], 1)
            self.assertTrue(Path(summary["summary_json_path"]).exists())
            self.assertTrue(Path(summary["summary_csv_path"]).exists())
            task_summary = summary["tasks"][0]
            self.assertEqual(task_summary["best_program_id"], "2_2_solved")
            self.assertEqual(task_summary["iterations_completed"], 2)
            self.assertEqual(task_summary["num_programs"], 2)
            self.assertAlmostEqual(task_summary["success_score"], 1.0)
            self.assertAlmostEqual(task_summary["search_score"], 0.93)
            self.assertEqual(task_summary["run_returncode"], 0)
            self.assertFalse(task_summary["run_timeout"])

    def test_summarize_suite_results_handles_seed_only_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PutnamBench"
            lean_src = root / "lean4" / "src"
            output_root = Path(tmpdir) / "suite_out"
            template_path = Path(tmpdir) / "template.yaml"
            lean_src.mkdir(parents=True)
            (lean_src / "putnam_demo.lean").write_text(PUTNAM_TEXT, encoding="utf-8")
            template_path.write_text(
                """
experiment:
  name: template_exp
  task_description: examples/lean_identity_namespace/problem.lean
  task_constraint: keep theorem unchanged
  language: lean
  strategy: mathproof_plan_island
  max_iterations: 1
  max_workers: 1
  checkpoint_interval: 1
  seed: 42
  island:
    num_islands: 1
    island_size: 1
    migration_interval: 50
    migration_size: 1
    migration_topology: ring
infrastructure:
  llm:
    provider: openai
    api_base: https://example.com/v1
    model: demo-model
    api_key: demo-key
    temperature: 0.0
    max_tokens: 512
    timeout: 60
  storage:
    type: local
    base_path: ./suite_data
  logger:
    type: local
    level: INFO
    file_enabled: true
  env:
    type: local
    timeout: 30
modules:
  generate:
    language: lean
  evaluate:
    language: lean
  judge:
    use_llm: false
problem: {}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            suite = materialize_benchmark_suite(
                benchmark_name="putnambench",
                benchmark_root=root,
                output_root=output_root,
            )
            prepare_suite_runtime(suite, template_path, storage_base_path="./suite_data")

            task_dir = suite.task_specs[0].task_dir
            exp_dir = task_dir / "suite_data" / "putnambench_all_putnam_demo_seedpass"
            self._write_program_payload(
                exp_dir,
                "seed_0",
                iteration=0,
                generation=0,
                success_score=1.0,
                search_score=0.88,
                combined_score=1.0,
                theorem_solved=1.0,
            )

            summary = summarize_suite_results(suite_root=suite.suite_root)

            task_summary = summary["tasks"][0]
            self.assertEqual(task_summary["best_program_id"], "seed_0")
            self.assertIsNone(task_summary["latest_checkpoint_path"])
            self.assertEqual(task_summary["iterations_completed"], 0)
            self.assertTrue(task_summary["solved"])


if __name__ == "__main__":
    unittest.main()
