from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.lean_and_left.evaluator import evaluate as evaluate_and_left


class TestMathproofExamplesRealistic(unittest.TestCase):
    def test_lean_and_left_seed_is_not_already_solved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "init.scratch"
            source = Path("examples/lean_and_left/init.scratch").read_text(encoding="utf-8")
            plan_path.write_text(source, encoding="utf-8")

            result = evaluate_and_left(str(plan_path))

            self.assertEqual(result["theorem_solved"], 0.0)
            self.assertEqual(result["final_compile_success"], 0.0)
            self.assertEqual(result["validity"], 1.0)


if __name__ == "__main__":
    unittest.main()
