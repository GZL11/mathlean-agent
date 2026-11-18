from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from famou.core.data import Context, Program, RolloutResult
from famou.core.types import Language
from famou.infrastructure.llm.mock_client import MockLLMClient
from famou.modules.generate.plan_mutation import PlanMutationGenerate
from famou.modules.generate.subgoal_repair import SubgoalRepairGenerate
from famou.utils.global_memory_store import (
    LEMMA_MEMORY_FILE,
    SUBGOAL_MEMORY_FILE,
    THEOREM_MEMORY_FILE,
    persist_program_memory,
    retrieve_failed_lemma_memories,
    retrieve_lemma_memories,
    retrieve_subgoal_memories,
    retrieve_theorem_memories,
    retrieve_verified_lemmas,
)
from famou.utils.plan_runtime import parse_program_plan
from famou.utils.proof_experience import extract_and_attach_experiences


SOLVED_MEMORY_PLAN = """
plan_id: solved_memory_plan
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Prove transitivity using le_trans."
strategy_type: direct
subgoals:
  - id: sg_assumptions
    statement: "Introduce hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Conclude a <= c using le_trans hab hbc."
    dependencies: [sg_assumptions]
    status: verified
    attempt_count: 1
    proof_sketch: "Apply transitivity directly."
    local_lean_draft: |
      by
        exact le_trans hab hbc
    last_diagnostic: "previous draft had a type mismatch before exact le_trans hab hbc"
    last_repair_summary: "Switch to the direct transitivity lemma."
    retrieved_lemma_ids: [lemma_le_trans]
lemmas:
  - id: lemma_le_trans
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
    reusable_score: 1.0
    lean_code: |
      by
        intro hab hbc
        exact le_trans hab hbc
    metadata:
      preconditions:
        - "hab : a <= b"
        - "hbc : b <= c"
  - id: lemma_bad_trans
    statement: "a <= c"
    status: attempted
    reusable_score: 0.1
    source_node_ids: [sg_transitivity]
    lean_code: |
      by
        sorry
    metadata:
      preconditions:
        - "hab : a <= b"
        - "hbc : b <= c"
"""


UNSOLVED_PARENT_PLAN = """
plan_id: unsolved_parent
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Parent plan still needs local repair."
strategy_type: direct
subgoals:
  - id: sg_assumptions
    statement: "Introduce hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Conclude a <= c using le_trans hab hbc."
    dependencies: [sg_assumptions]
    status: attempted
    local_lean_draft: |
      by
        sorry
    last_diagnostic: "contains sorry"
lemmas:
  - id: lemma_le_trans
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
    reusable_score: 0.9
"""


REPAIRED_CHILD_PLAN = """
plan_id: repaired_child
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Child plan repaired the transitivity node."
strategy_type: direct
subgoals:
  - id: sg_assumptions
    statement: "Introduce hab : a <= b and hbc : b <= c."
    status: verified
  - id: sg_transitivity
    statement: "Conclude a <= c using le_trans hab hbc."
    dependencies: [sg_assumptions]
    status: in_progress
    attempt_count: 1
    proof_sketch: "Apply the transitivity lemma directly."
    local_lean_draft: |
      by
        exact le_trans hab hbc
    last_diagnostic: "main subgoal now has a direct draft"
    last_repair_summary: "Added a direct Lean draft for the transitivity step."
    retrieved_lemma_ids: [lemma_le_trans]
    retrieved_experience_ids: [exp_transitivity]
    attempt_history:
      - attempt_index: 1
        previous_status: attempted
        new_status: in_progress
        has_proof_sketch: true
        has_local_lean_draft: true
        introduced_lemma_count: 0
        retrieved_lemma_ids: [lemma_le_trans]
        retrieved_experience_ids: [exp_transitivity]
lemmas:
  - id: lemma_le_trans
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
    reusable_score: 1.0
"""


LEMMA_RESULTS = [
    {
        "lemma_id": "lemma_le_trans",
        "artifact_path": "/tmp/lemma_le_trans.lean",
        "compile_success": 1.0,
        "no_sorry": 1.0,
        "statement_match": 1.0,
        "proof_validity": 1.0,
        "diagnostics": [],
        "returncode": 0,
        "wrapper_file": "/tmp/_verify_lemma_le_trans.lean",
    },
    {
        "lemma_id": "lemma_bad_trans",
        "artifact_path": "/tmp/lemma_bad_trans.lean",
        "compile_success": 0.0,
        "no_sorry": 0.0,
        "statement_match": 1.0,
        "proof_validity": 0.0,
        "diagnostics": ["unknown identifier 'missing_hyp'"],
        "returncode": 1,
        "wrapper_file": "/tmp/_verify_lemma_bad_trans.lean",
    },
]


REPLANNED_CHILD_PLAN = """
```yaml
plan_id: replanned_child
theorem_statement: "For real numbers a b c, if a <= b and b <= c then a <= c."
plan_summary: "Fresh decomposition that isolates hypothesis introduction and the final transitivity step."
strategy_type: decomposition-first
subgoals:
  - id: sg_intro
    statement: "Introduce hab : a <= b and hbc : b <= c."
    status: pending
  - id: sg_apply
    statement: "Apply le_trans hab hbc to conclude a <= c."
    dependencies: [sg_intro]
    status: pending
lemmas:
  - id: lemma_le_trans
    statement: "a <= b -> b <= c -> a <= c"
    status: verified
```
"""


class TestGlobalMemoryStore(unittest.TestCase):
    def test_persist_and_retrieve_global_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            program = Program(
                id="1_0_memory_source",
                code=SOLVED_MEMORY_PLAN,
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
                    "lemma_results": LEMMA_RESULTS,
                },
                meta={"theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c."},
            )
            extract_and_attach_experiences(program)

            counts = persist_program_memory(program, tmpdir)

            self.assertGreaterEqual(counts["theorem_experience"], 1)
            self.assertGreaterEqual(counts["subgoal_experience"], 1)
            self.assertGreaterEqual(counts["verified_lemma_library"], 2)
            self.assertTrue((Path(tmpdir) / THEOREM_MEMORY_FILE).exists())
            self.assertTrue((Path(tmpdir) / SUBGOAL_MEMORY_FILE).exists())
            self.assertTrue((Path(tmpdir) / LEMMA_MEMORY_FILE).exists())

            theorem_rows = retrieve_theorem_memories(
                tmpdir,
                "For real numbers a b c, if a <= b and b <= c then a <= c.",
                limit=3,
            )
            subgoal_rows = retrieve_subgoal_memories(
                tmpdir,
                "Conclude a <= c using le_trans hab hbc.",
                diagnostic="type mismatch before exact le_trans hab hbc",
                limit=3,
            )
            lemma_rows = retrieve_verified_lemmas(
                tmpdir,
                "Conclude a <= c using le_trans hab hbc.",
                limit=3,
            )
            lemma_memory_rows = retrieve_lemma_memories(
                tmpdir,
                "Conclude a <= c using le_trans hab hbc missing_hyp",
                limit=4,
            )
            failed_lemma_rows = retrieve_failed_lemma_memories(
                tmpdir,
                "missing_hyp",
                diagnostic="missing_hyp",
                limit=2,
            )

            self.assertTrue(any("transitivity" in row["content"].lower() or "finished as solved" in row["content"].lower() for row in theorem_rows))
            solved_theorem_row = next(row for row in theorem_rows if row["outcome"] == "solved")
            self.assertEqual(solved_theorem_row["plan_summary"], "Prove transitivity using le_trans.")
            self.assertEqual(solved_theorem_row["strategy_type"], "direct")
            self.assertEqual(solved_theorem_row["theorem_solved"], 1.0)
            self.assertIn("lemma_le_trans", solved_theorem_row["used_lemma_ids"])
            self.assertIn("lemma_le_trans", solved_theorem_row["verified_lemma_ids"])
            self.assertTrue(any("Conclude a <= c using le_trans hab hbc." in item for item in solved_theorem_row["subgoal_statements"]))
            self.assertIsNotNone(solved_theorem_row["task_scaffold_signature"])
            self.assertEqual(solved_theorem_row["trajectory_metadata"]["subgoal_count"], 2)
            self.assertTrue(any(row["node_id"] == "sg_transitivity" for row in subgoal_rows))
            self.assertTrue(any(row["lemma_id"] == "lemma_le_trans" for row in lemma_rows))
            self.assertTrue(any(row["lemma_id"] == "lemma_bad_trans" and row["proof_outcome"] == "failed" for row in lemma_memory_rows))
            self.assertTrue(any("hab : a <= b" in ";".join(row.get("preconditions", [])) for row in lemma_memory_rows))
            failed_row = next(row for row in lemma_memory_rows if row["lemma_id"] == "lemma_bad_trans")
            self.assertEqual(failed_row["memory_role"], "failure_experience")
            self.assertEqual(failed_row["failure_category"], "unknown_identifier")
            self.assertGreater(failed_row["near_success_score"], 0.0)
            self.assertIn("check whether all hypotheses are introduced", failed_row["repair_hints"][0])
            self.assertEqual(failed_lemma_rows[0]["lemma_id"], "lemma_bad_trans")

    def test_subgoal_memory_records_repair_step_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Program(
                id="0_0_parent",
                code=UNSOLVED_PARENT_PLAN,
                generation=0,
                iteration=0,
                language=Language.LEAN,
                combined_score=0.0,
                search_score=0.2,
                metrics={
                    "theorem_solved": 0.0,
                    "compile_success": 1.0,
                    "no_sorry": 0.0,
                    "statement_match": 1.0,
                },
                meta={"theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c."},
            )
            child = Program(
                id="1_0_child",
                code=REPAIRED_CHILD_PLAN,
                generation=1,
                iteration=1,
                language=Language.LEAN,
                combined_score=0.3,
                search_score=0.62,
                metrics={
                    "theorem_solved": 0.0,
                    "compile_success": 1.0,
                    "no_sorry": 1.0,
                    "statement_match": 1.0,
                },
                meta={
                    "theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c.",
                    "selected_subgoal_id": "sg_transitivity",
                    "selection_reason": "frontier_ready",
                    "local_island_phase": "inner_repair",
                    "local_island_step": 1,
                    "repair_patch": {
                        "node_id": "sg_transitivity",
                        "updated_local_lean_draft": "by\n  exact le_trans hab hbc",
                        "updated_status": "in_progress",
                        "retrieved_lemma_ids": ["lemma_le_trans"],
                    },
                },
            )

            counts = persist_program_memory(child, tmpdir, parent_program=parent)
            self.assertGreaterEqual(counts["subgoal_experience"], 1)

            subgoal_rows = retrieve_subgoal_memories(
                tmpdir,
                "Conclude a <= c using le_trans hab hbc.",
                diagnostic="contains sorry",
                limit=3,
            )
            target_row = next(row for row in subgoal_rows if row["node_id"] == "sg_transitivity")
            self.assertEqual(target_row["before_status"], "attempted")
            self.assertEqual(target_row["after_status"], "in_progress")
            self.assertEqual(target_row["action_type"], "repair")
            self.assertEqual(target_row["diagnostic_before"], "contains sorry")
            self.assertEqual(target_row["diagnostic_after"], "main subgoal now has a direct draft")
            self.assertEqual(target_row["reward_delta"], 0.3)
            self.assertEqual(target_row["search_score_delta"], 0.42)
            self.assertTrue(target_row["selected_for_repair"])
            self.assertEqual(target_row["trajectory_metadata"]["local_island_phase"], "inner_repair")

    def test_persist_deduplicates_semantically_identical_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            program = Program(
                id="1_0_memory_source",
                code=SOLVED_MEMORY_PLAN,
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
                    "lemma_results": LEMMA_RESULTS,
                },
                meta={"theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c."},
            )
            extract_and_attach_experiences(program)

            first_counts = persist_program_memory(program, tmpdir)
            duplicate_program = program.model_copy(deep=True)
            duplicate_program.id = "1_1_duplicate_source"
            duplicate_program.created_at = program.created_at + 1.0
            second_counts = persist_program_memory(duplicate_program, tmpdir)

            self.assertGreaterEqual(first_counts["theorem_experience"], 1)
            self.assertEqual(second_counts["theorem_experience"], 0)
            self.assertEqual(second_counts["subgoal_experience"], 0)
            self.assertEqual(second_counts["verified_lemma_library"], 0)

    def test_retrieval_reranks_subgoal_rows_by_dependency_and_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "memory_id": "subgoal_good",
                    "memory_kind": "subgoal_experience",
                    "node_id": "sg_transitivity",
                    "statement_formal": "Conclude a <= c using le_trans hab hbc.",
                    "dependencies": ["sg_assumptions"],
                    "action_type": "repair",
                    "reward_delta": 0.3,
                    "diagnostic_before": "contains sorry",
                    "diagnostic_after": "draft now applies le_trans",
                    "text": "Conclude a <= c using le_trans hab hbc contains sorry repair",
                },
                {
                    "memory_id": "subgoal_weaker",
                    "memory_kind": "subgoal_experience",
                    "node_id": "sg_transitivity_alt",
                    "statement_formal": "Conclude a <= c using le_trans hab hbc.",
                    "dependencies": ["sg_other"],
                    "action_type": "rewrite",
                    "reward_delta": 0.0,
                    "diagnostic_before": "contains sorry",
                    "diagnostic_after": "restated the sketch only",
                    "text": "Conclude a <= c using le_trans hab hbc contains sorry rewrite",
                },
            ]
            with (Path(tmpdir) / SUBGOAL_MEMORY_FILE).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            ranked = retrieve_subgoal_memories(
                tmpdir,
                "Conclude a <= c using le_trans hab hbc.",
                diagnostic="contains sorry",
                dependency_ids=["sg_assumptions"],
                preferred_action_types=["repair"],
                limit=2,
            )

            self.assertEqual(ranked[0]["memory_id"], "subgoal_good")

    def test_retrieval_reranks_theorem_rows_by_matching_scaffold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theorem_statement = "P ∧ Q → P"
            scaffold_text = "namespace Demo\nvariable (P Q : Prop)\n\ntheorem and_left : P ∧ Q → P := by"
            problem_signature = f"prob_{hashlib.sha1(theorem_statement.encode('utf-8')).hexdigest()[:12]}"
            scaffold_signature = f"scaf_{hashlib.sha1(' '.join(scaffold_text.split()).encode('utf-8')).hexdigest()[:12]}"
            rows = [
                {
                    "memory_id": "theorem_wrong_scaffold",
                    "memory_kind": "theorem_experience",
                    "theorem_statement": theorem_statement,
                    "problem_signature": problem_signature,
                    "task_scaffold_signature": "scaf_other",
                    "outcome": "solved",
                    "success_score": 1.0,
                    "search_score": 0.9,
                    "text": "P ∧ Q → P direct conjunction elimination",
                },
                {
                    "memory_id": "theorem_right_scaffold",
                    "memory_kind": "theorem_experience",
                    "theorem_statement": theorem_statement,
                    "problem_signature": problem_signature,
                    "task_scaffold_signature": scaffold_signature,
                    "outcome": "solved",
                    "success_score": 1.0,
                    "search_score": 0.9,
                    "text": "P ∧ Q → P namespace Demo variable P Q conjunction elimination",
                },
            ]
            with (Path(tmpdir) / THEOREM_MEMORY_FILE).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            ranked = retrieve_theorem_memories(
                tmpdir,
                theorem_statement,
                task_constraint="Keep theorem target unchanged.",
                task_scaffold=scaffold_text,
                limit=2,
            )

            self.assertEqual(ranked[0]["memory_id"], "theorem_right_scaffold")

    def test_retrieval_reranks_lemma_rows_by_reuse_vs_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "memory_id": "lemma_success",
                    "memory_kind": "lemma_memory",
                    "lemma_id": "lemma_le_trans_success",
                    "statement_formal": "a <= b -> b <= c -> a <= c",
                    "preconditions": ["hab : a <= b", "hbc : b <= c"],
                    "dependency_ids": ["sg_assumptions"],
                    "proof_status": "verified",
                    "proof_outcome": "successful",
                    "reusable_score": 1.0,
                    "text": "a <= b -> b <= c -> a <= c hab : a <= b hbc : b <= c exact le_trans hab hbc",
                },
                {
                    "memory_id": "lemma_failed",
                    "memory_kind": "lemma_memory",
                    "lemma_id": "lemma_le_trans_failed",
                    "statement_formal": "a <= b -> b <= c -> a <= c",
                    "preconditions": ["hab : a <= b", "hbc : b <= c"],
                    "dependency_ids": ["sg_assumptions"],
                    "proof_status": "attempted",
                    "proof_outcome": "failed",
                    "failure_summary": "unknown identifier 'missing_hyp'",
                    "verifier_diagnostics": ["unknown identifier 'missing_hyp'"],
                    "text": "a <= b -> b <= c -> a <= c hab : a <= b hbc : b <= c missing_hyp",
                },
            ]
            with (Path(tmpdir) / LEMMA_MEMORY_FILE).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            reuse_ranked = retrieve_lemma_memories(
                tmpdir,
                query_text="Conclude a <= c using le_trans hab hbc.",
                statement_formal="a <= b -> b <= c -> a <= c",
                dependency_ids=["sg_assumptions"],
                preconditions=["hab : a <= b", "hbc : b <= c"],
                limit=2,
            )
            diagnostic_ranked = retrieve_lemma_memories(
                tmpdir,
                query_text="missing_hyp",
                diagnostic="missing_hyp",
                include_failed=True,
                limit=2,
            )

            self.assertEqual(reuse_ranked[0]["memory_id"], "lemma_success")
            self.assertEqual(diagnostic_ranked[0]["memory_id"], "lemma_failed")

    def test_prompts_include_retrieved_global_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_source = Program(
                id="1_0_memory_source",
                code=SOLVED_MEMORY_PLAN,
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
                    "lemma_results": LEMMA_RESULTS,
                },
                meta={"theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c."},
            )
            extract_and_attach_experiences(memory_source)
            persist_program_memory(memory_source, tmpdir)

            parent = Program(
                id="0_0_parent",
                code=UNSOLVED_PARENT_PLAN,
                generation=0,
                iteration=0,
                language=Language.LEAN,
                combined_score=0.0,
                search_score=0.2,
                metrics={
                    "theorem_solved": 0.0,
                    "compile_success": 1.0,
                    "no_sorry": 0.0,
                    "statement_match": 1.0,
                },
                meta={"theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c."},
            )
            extract_and_attach_experiences(parent)

            context = Context(
                experiment_id="exp_global_memory",
                task_description="Prove the target theorem in Lean.",
                task_constraint="Keep theorem fixed and remove sorry from the final proof.",
                population={"main": [parent]},
                config={"language": "lean", "global_memory_dir": tmpdir},
            )

            plan_result = RolloutResult(
                rollout_id="rollout_plan",
                iteration=1,
                selected_parent_id=parent.id,
                inspiration_ids=[],
            )
            plan_prompt = PlanMutationGenerate(
                language="lean",
                temperature=0.0,
                max_tokens=512,
            ).build_prompt(parent, context, plan_result)

            self.assertIn("Global theorem memories", plan_prompt)
            self.assertIn("Global lemma memories", plan_prompt)
            self.assertIn("lemma_le_trans", plan_prompt)
            self.assertIn("lemma_bad_trans", plan_prompt)
            self.assertIn("proof_outcome: failed", plan_prompt)
            self.assertIn("outcome: solved", plan_prompt)
            self.assertIn("plan_summary: Prove transitivity using le_trans.", plan_prompt)
            self.assertIn("subgoal_statements:", plan_prompt)
            self.assertIn("verified_lemma_ids: lemma_le_trans", plan_prompt)

            repair_result = RolloutResult(
                rollout_id="rollout_repair",
                iteration=1,
                selected_parent_id=parent.id,
            )
            repair_prompt = SubgoalRepairGenerate(
                language="lean",
                temperature=0.0,
                max_tokens=512,
            ).build_prompt(parent, context, repair_result)

            self.assertIn("Global subgoal memories", repair_prompt)
            self.assertIn("Global lemma memories", repair_prompt)
            self.assertIn("sg_transitivity", repair_prompt)
            self.assertIn("unknown identifier 'missing_hyp'", repair_prompt)

    def test_plan_mutation_switches_to_memory_guided_replanning_when_parent_exhausted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_source = Program(
                id="1_0_memory_source",
                code=SOLVED_MEMORY_PLAN,
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
                    "lemma_results": LEMMA_RESULTS,
                },
                meta={"theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c."},
            )
            extract_and_attach_experiences(memory_source)
            persist_program_memory(memory_source, tmpdir)

            parent = Program(
                id="0_0_stalled_parent",
                code=UNSOLVED_PARENT_PLAN,
                generation=0,
                iteration=0,
                language=Language.LEAN,
                combined_score=0.0,
                search_score=0.18,
                metrics={
                    "theorem_solved": 0.0,
                    "compile_success": 1.0,
                    "no_sorry": 1.0,
                    "statement_match": 1.0,
                    "progress_score": 0.2,
                    "frontier_size": 0.0,
                    "blocked_size": 2.0,
                },
                meta={
                    "theorem_statement": "For real numbers a b c, if a <= b and b <= c then a <= c.",
                    "local_island_budget": 3,
                    "local_island_summary": {
                        "budget": 3,
                        "steps_executed": 3,
                        "history": [
                            {"local_step": 0, "score": 0.18},
                            {"local_step": 1, "score": 0.18},
                            {"local_step": 2, "score": 0.17},
                        ],
                    },
                },
            )
            extract_and_attach_experiences(parent)

            context = Context(
                experiment_id="exp_memory_guided_replan",
                task_description="Prove the target theorem in Lean.",
                task_constraint="Keep theorem fixed and produce an acyclic proof plan.",
                population={"main": [parent]},
                config={"language": "lean", "global_memory_dir": tmpdir},
            )
            result = RolloutResult(
                rollout_id="rollout_replan",
                iteration=1,
                selected_parent_id=parent.id,
                inspiration_ids=[],
            )
            generator = PlanMutationGenerate(
                language="lean",
                temperature=0.0,
                max_tokens=512,
            )
            generator.llm_client = MockLLMClient(response=REPLANNED_CHILD_PLAN)

            prompt = generator.build_prompt(parent, context, result)

            self.assertIn("Replanning directive", prompt)
            self.assertIn("replan_from_memory", prompt)
            self.assertIn("local_budget_exhausted", prompt)
            self.assertIn("blocked and has no remaining frontier nodes", prompt)
            self.assertEqual(result.stats["replanning_mode"], "replan_from_memory")
            self.assertEqual(result.stats["replanning_reason"], "local_budget_exhausted")
            self.assertIn("1_0_memory_source_outcome", result.stats["retrieved_theorem_memory_ids"])

            result = generator.execute(context, result)
            program = result.generated_program
            self.assertIsNotNone(program)
            self.assertEqual(program.meta["replanning_mode"], "replan_from_memory")
            self.assertIn("local_budget_exhausted", program.meta["replanning_reason"])
            self.assertIn("1_0_memory_source_outcome", program.meta["retrieved_theorem_memory_ids"])

            child_plan = parse_program_plan(
                program,
                fallback_theorem="For real numbers a b c, if a <= b and b <= c then a <= c.",
            )
            self.assertEqual(child_plan.metadata["replanning_mode"], "replan_from_memory")
            self.assertIn("1_0_memory_source_outcome", child_plan.metadata["retrieved_theorem_memory_ids"])


if __name__ == "__main__":
    unittest.main()
