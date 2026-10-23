import ast
import os
import unittest
from datetime import datetime

SRC = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
TREE = ast.parse(SRC)


def load(names, extra=None):
    ns = {"datetime": datetime, "HUMAN_SPEED_TEST_MISS_STREAK": 3, "HUMAN_SPEED_TEST_SECONDS": 30, "HUMAN_SPEED_TEST_MIN_WPM": 60.0}
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "HUMAN_SPEED_TEST_PASSAGE" for t in node.targets):
            exec(compile(ast.Module([node], []), "m", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "m", "exec"), ns)
    ns.update(extra or {})
    return ns


class SpeedTestTests(unittest.TestCase):
    def test_streak_requires_test_only_at_three_for_approved_workers(self):
        ns = load({"_human_speed_test_streak_updates"})
        f = ns["_human_speed_test_streak_updates"]
        self.assertNotIn("speed_test_required", f({"workerApproved": True, "deadline_miss_streak": 1}, 1))
        self.assertTrue(f({"workerApproved": True, "deadline_miss_streak": 2}, 1)["speed_test_required"])
        self.assertNotIn("speed_test_required", f({"workerApproved": False, "deadline_miss_streak": 5}, 1))

    def test_score_uses_server_passage_and_60_wpm_means_150_characters(self):
        ns = load({"_human_speed_test_score"})
        passage = ns["HUMAN_SPEED_TEST_PASSAGE"]
        correct, wpm = ns["_human_speed_test_score"](passage[:150], 30)
        self.assertEqual((correct, wpm), (150, 60.0))
        self.assertLess(ns["_human_speed_test_score"](passage[:149], 30)[1], 60)
        self.assertEqual(ns["_human_speed_test_score"]("~" * 500, 30)[0], 0)

    def test_gates_and_reset_are_wired(self):
        self.assertIn("_human_assert_no_speed_test(actor)", SRC)
        self.assertIn('"speed_test_required": True', SRC)
        self.assertIn('{"deadline_miss_streak": 0}', SRC)


class RatingRuleTests(unittest.TestCase):
    def test_prompt_has_only_the_four_criteria_and_forbids_paragraphing(self):
        ns = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in {"RATING_AGENT_DEFAULT_INSTRUCTIONS", "RATING_AGENT_HARD_RULES"} for t in node.targets):
                exec(compile(ast.Module([node], []), "m", "exec"), ns)
        text = ns["RATING_AGENT_DEFAULT_INSTRUCTIONS"] + ns["RATING_AGENT_HARD_RULES"]
        for word in ("Guideline adherence", "Research", "Accuracy", "Double spaces"):
            self.assertIn(word, text)
        self.assertIn("NEVER rate or deduct for paragraphing", text)
        self.assertIn("[dictation ends here]", text)
        self.assertIn("never rate paragraphing", ns["RATING_AGENT_HARD_RULES"])

    def test_ratings_apply_automatically_and_can_be_disputed(self):
        self.assertIn('"Rating Agent")', SRC)
        self.assertIn("/human-transcription/worker/ratings/{rating_id}/dispute", SRC)
        self.assertIn("/api/admin/rating-agent-instructions", SRC)


class SttRoutingTests(unittest.TestCase):
    def test_xai_tester_super_admin_and_default_order(self):
        self.assertIn('"tier_2": None, "tier_3": None, "tier_4": None, "reason": "dedicated_xai_stt_tester"', SRC)
        self.assertIn('"tier_1": "xai", "tier_2": "assemblyai"', SRC)
        self.assertIn('order = ["assemblyai", "xai", "voxtral", "openai_whisper", "deepgram"]', SRC)
        self.assertIn("XIAOMI_STT_TESTER_EMAILS = set()", SRC)

    def test_human_work_can_route_to_xai_and_xiaomi(self):
        self.assertIn('HUMAN_WORK_STT_PROVIDERS = ("assemblyai", "deepgram", "voxtral", "openai", "xai", "xiaomi")', SRC)
        self.assertIn("result = await transcribe_with_xiaomi(audio_path", SRC)


if __name__ == "__main__":
    unittest.main()
