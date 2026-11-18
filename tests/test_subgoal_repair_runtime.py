from __future__ import annotations

import unittest

from famou.core.proof_objects import PlanObject, SubgoalStatus
from famou.utils.subgoal_repair_runtime import (
    SubgoalRepairPatch,
    assert_plan_topology_unchanged,
    compute_plan_topology_signature,
    merge_subgoal_repair_patch,
)
from famou.utils.subgoal_selection import select_subgoal_for_repair


class TestSubgoalRepairRuntime(unittest.TestCase):
    def setUp(self):
        self.plan = PlanObject.model_validate(
            {
                "plan_id": "plan_fixed",
                "theorem_statement": "A -> C",
                "strategy_type": "decomposition-first",
                "subgoals": [
                    {
                        "node_id": "sg_a",
                        "statement_formal": "A",
                        "local_status": "verified",
                    },
                    {
                        "node_id": "sg_b",
                        "statement_formal": "B",
                        "dependencies": ["sg_a"],
                        "local_status": "attempted",
                        "verifier_state": {"diagnostics": ["type mismatch"]},
                    },
                    {
                        "node_id": "sg_c",
                        "statement_formal": "C",
                        "dependencies": ["sg_b"],
                        "local_status": "pending",
                    },
                ],
                "verifier_feedback": {"blocked_nodes": ["sg_b"]},
            }
        )

    def test_topology_signature_ignores_local_repair_fields(self):
        parent_signature = compute_plan_topology_signature(self.plan)
        child = self.plan.model_copy(deep=True)
        child.node_map()["sg_b"].proof_sketch = "Refine B using A."
        child.node_map()["sg_b"].local_lean_draft = "by exact hA"
        child.node_map()["sg_b"].attempt_count = 3
        child.node_map()["sg_b"].last_diagnostic = "goal mismatch"

        self.assertEqual(parent_signature, compute_plan_topology_signature(child))
        assert_plan_topology_unchanged(self.plan, child)

    def test_topology_signature_detects_dependency_change(self):
        child = self.plan.model_copy(deep=True)
        child.node_map()["sg_c"].dependencies = ["sg_a"]

        with self.assertRaises(ValueError):
            assert_plan_topology_unchanged(self.plan, child)

    def test_merge_patch_updates_only_local_node_state(self):
        patch = SubgoalRepairPatch(
            node_id="sg_b",
            updated_proof_sketch="Use the verified assumption to build B.",
            updated_local_lean_draft="by exact hA",
            updated_status=SubgoalStatus.IN_PROGRESS,
            verifier_state_update={"selected": True},
            introduced_lemmas=[
                {
                    "lemma_id": "lemma_local_b",
                    "statement_formal": "A -> B",
                    "proof_status": "attempted",
                    "source_node_ids": ["sg_b"],
                }
            ],
            retrieved_lemma_ids=["lemma_global_1"],
            retrieved_experience_ids=["exp_1"],
            last_diagnostic="still one goal remaining",
            repair_summary="Filled the main draft for sg_b.",
        )

        child = merge_subgoal_repair_patch(self.plan, patch)
        node = child.node_map()["sg_b"]

        self.assertEqual(node.local_status, SubgoalStatus.IN_PROGRESS)
        self.assertEqual(node.attempt_count, 1)
        self.assertEqual(node.last_diagnostic, "still one goal remaining")
        self.assertEqual(node.last_repair_summary, "Filled the main draft for sg_b.")
        self.assertIn("selected", node.verifier_state)
        self.assertEqual(node.retrieved_lemma_ids, ["lemma_global_1"])
        self.assertEqual(node.retrieved_experience_ids, ["exp_1"])
        self.assertEqual(len(node.attempt_history), 1)
        self.assertEqual(child.frontier_nodes, ["sg_b"])
        self.assertEqual(len(child.lemma_pool), 1)
        self.assertEqual(child.lemma_pool[0].lemma_id, "lemma_local_b")
        self.assertEqual(
            compute_plan_topology_signature(self.plan),
            compute_plan_topology_signature(child),
        )

    def test_select_subgoal_prefers_blocked_feedback_then_frontier(self):
        selected = select_subgoal_for_repair(self.plan)
        self.assertEqual(selected["selected_node_id"], "sg_b")
        self.assertEqual(selected["selection_reason"], "blocked_by_verifier_feedback")
        self.assertIn("sg_b", selected["candidate_node_ids"])

    def test_select_subgoal_sticks_to_recent_progress(self):
        sticky_plan = PlanObject.model_validate(
            {
                "plan_id": "plan_sticky",
                "theorem_statement": "A -> B /\\ C",
                "strategy_type": "decomposition-first",
                "subgoals": [
                    {
                        "node_id": "sg_a",
                        "statement_formal": "A",
                        "local_status": "verified",
                    },
                    {
                        "node_id": "sg_b",
                        "statement_formal": "B",
                        "dependencies": ["sg_a"],
                        "local_status": "in_progress",
                        "attempt_count": 1,
                        "local_lean_draft": "by exact hA",
                        "attempt_history": [
                            {
                                "attempt_index": 1,
                                "previous_status": "attempted",
                                "new_status": "in_progress",
                                "has_proof_sketch": False,
                                "has_local_lean_draft": True,
                                "introduced_lemma_count": 0,
                            }
                        ],
                    },
                    {
                        "node_id": "sg_c",
                        "statement_formal": "C",
                        "local_status": "pending",
                    },
                ],
                "metadata": {"last_repaired_node_id": "sg_b"},
            }
        )

        selected = select_subgoal_for_repair(sticky_plan)
        self.assertEqual(selected["selected_node_id"], "sg_b")
        self.assertEqual(selected["selection_reason"], "sticky_recent_progress")
        self.assertEqual(selected["sticky_candidate_id"], "sg_b")

    def test_select_subgoal_prefers_frontier_over_topology_blocked(self):
        plan = PlanObject.model_validate(
            {
                "plan_id": "plan_frontier_priority",
                "theorem_statement": "A -> C",
                "strategy_type": "decomposition-first",
                "subgoals": [
                    {
                        "node_id": "sg_a",
                        "statement_formal": "A",
                        "local_status": "verified",
                    },
                    {
                        "node_id": "sg_b",
                        "statement_formal": "B",
                        "dependencies": ["sg_a"],
                        "local_status": "pending",
                    },
                    {
                        "node_id": "sg_c",
                        "statement_formal": "C",
                        "dependencies": ["sg_b"],
                        "local_status": "pending",
                    },
                ],
            }
        )

        selected = select_subgoal_for_repair(plan)
        self.assertEqual(selected["selected_node_id"], "sg_b")
        self.assertEqual(selected["selection_reason"], "frontier_ready")
        self.assertEqual(selected["blocked_node_ids"], [])
        self.assertIn("sg_c", selected["dependency_blocked_node_ids"])

    def test_sticky_does_not_pin_dependency_blocked_node(self):
        plan = PlanObject.model_validate(
            {
                "plan_id": "plan_no_sticky_blocked",
                "theorem_statement": "A -> C",
                "strategy_type": "decomposition-first",
                "metadata": {"last_repaired_node_id": "sg_c"},
                "subgoals": [
                    {
                        "node_id": "sg_a",
                        "statement_formal": "A",
                        "local_status": "verified",
                    },
                    {
                        "node_id": "sg_b",
                        "statement_formal": "B",
                        "dependencies": ["sg_a"],
                        "local_status": "in_progress",
                        "attempt_count": 1,
                        "local_lean_draft": "by exact hA",
                    },
                    {
                        "node_id": "sg_c",
                        "statement_formal": "C",
                        "dependencies": ["sg_b"],
                        "local_status": "attempted",
                        "attempt_count": 3,
                        "last_diagnostic": "dependency not yet solved",
                        "attempt_history": [
                            {"new_status": "attempted"},
                            {"new_status": "attempted"},
                            {"new_status": "attempted"},
                        ],
                    },
                ],
            }
        )

        selected = select_subgoal_for_repair(plan)
        self.assertEqual(selected["selected_node_id"], "sg_b")
        self.assertEqual(selected["selection_reason"], "frontier_ready")
        self.assertIsNone(selected["sticky_candidate_id"])


if __name__ == "__main__":
    unittest.main()
