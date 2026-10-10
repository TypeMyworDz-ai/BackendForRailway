import ast
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
TREE = ast.parse(SRC)


def _fn(name):
    return next(node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def _namespace():
    ns = {"datetime": datetime, "timezone": timezone}
    for name in ("_as_dt", "_human_ai_review_run_active"):
        exec(compile(ast.Module(body=[_fn(name)], type_ignores=[]), "main.py", "exec"), ns)
    ns["_AI_REVIEW_RUN_STALE_SECONDS"] = 30 * 60
    return ns


class BackgroundProofreadState(unittest.TestCase):
    def test_active_only_when_recent_and_not_finished(self):
        active = _namespace()["_human_ai_review_run_active"]
        now = datetime.now(timezone.utc)
        self.assertTrue(active({"ai_review_run": {"status": "processing", "startedAt": now.isoformat()}}))
        self.assertFalse(active({"ai_review_run": {"status": "completed", "startedAt": now.isoformat()}}))
        self.assertFalse(active({"ai_review_run": {"status": "processing", "startedAt": (now - timedelta(minutes=45)).isoformat()}}))
        self.assertFalse(active({}))

    def test_start_route_schedules_background_work_and_worker_views_hide_run(self):
        start = ast.unparse(_fn("human_admin_ai_review_start"))
        self.assertIn("background_tasks.add_task", start)
        self.assertIn("_human_require_ai_call_credits", start)
        self.assertNotIn("_human_charge_ai_call", start)
        self.assertGreaterEqual(SRC.count('"ai_review_run"'), 3)

    def test_core_review_charges_once_and_reports_progress(self):
        core = ast.unparse(_fn("human_admin_ai_review"))
        self.assertEqual(core.count("_human_charge_ai_call("), 1)
        self.assertIn("progress", core)

    def test_forced_research_searches_unconfirmed_terms(self):
        research = ast.unparse(_fn("_human_ai_agent_research"))
        self.assertIn("_human_review_candidate_terms", research)
        self.assertIn("_RESEARCH_SCOPE_RULES", research)
        self.assertIn("_human_research_blocking", research)


if __name__ == "__main__":
    unittest.main()
