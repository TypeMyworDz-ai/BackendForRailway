"""Pure, production-data-free tests for Human Work pay rates."""
import ast
import math
import re
import unittest
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class HumanRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        wanted = {"_validate_human_transcriber_rate", "human_credit_quote"}
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
        ]
        cls.namespace = {
            "math": math,
            "re": re,
            "HUMAN_MIN_TRANSCRIBER_RATE_KES": 1,
            "HUMAN_MAX_TRANSCRIBER_RATE_KES": 500,
            "HUMAN_STANDARD_PAYOUT_KES": 30,
            "HUMAN_LEGACY_STANDARD_PAYOUT_KES": 25,
            "HUMAN_RUSH_PAYOUT_KES": 38,
            "HUMAN_PROOFREADING_PAYOUT_KES": 10,
            "HUMAN_STANDARD_CREDITS_PER_MINUTE": 40,
            "HUMAN_RUSH_CREDITS_PER_MINUTE": 55,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), cls.namespace)
        cls.tree = tree

    def test_standard_quote_uses_30_kes_default_and_admin_rate(self):
        quote = self.namespace["human_credit_quote"]
        self.assertEqual(quote(60)["transcriber_payout_kes_per_minute"], 30)
        self.assertEqual(quote(60, transcriber_rate_kes=37)["transcriber_payout_kes_per_minute"], 37)

    def test_rush_rate_and_proofreading_rate_remain_separate(self):
        quote = self.namespace["human_credit_quote"]
        self.assertEqual(quote(60, turnaround="rush", transcriber_rate_kes=37)["transcriber_payout_kes_per_minute"], 38)
        self.assertEqual(self.namespace["HUMAN_PROOFREADING_PAYOUT_KES"], 10)

    def test_rate_validation_requires_whole_kes_within_safe_bounds(self):
        validate = self.namespace["_validate_human_transcriber_rate"]
        self.assertEqual(validate("30"), 30)
        self.assertEqual(validate(500), 500)
        for invalid in (0, -1, 501, 30.5, "30.5", True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate(invalid)

    def test_admin_rate_routes_require_general_admin_guard(self):
        routes = {
            node.name: node for node in self.tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"human_admin_get_transcriber_rate", "human_admin_update_transcriber_rate"}
        }
        self.assertEqual(set(routes), {"human_admin_get_transcriber_rate", "human_admin_update_transcriber_rate"})
        for name, node in routes.items():
            called = {
                item.func.id for item in ast.walk(node)
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
            }
            self.assertIn("_require_admin", called, name)
            self.assertNotIn("_require_human_job_admin", called, name)

    def test_legacy_quotes_keep_previous_rate_as_fallback(self):
        self.assertEqual(self.namespace["HUMAN_LEGACY_STANDARD_PAYOUT_KES"], 25)
        self.assertEqual(self.namespace["HUMAN_STANDARD_PAYOUT_KES"], 30)


if __name__ == "__main__":
    unittest.main()
