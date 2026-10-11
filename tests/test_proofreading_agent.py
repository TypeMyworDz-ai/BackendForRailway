"""Checks for the Proofreading Agent (no paid AI or transcription calls)."""
import ast
import unittest
from datetime import datetime, timezone
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"
SRC = MAIN.read_text(encoding="utf-8")


def _ns():
    tree = ast.parse(SRC)
    wanted = {"_proofreading_agent_scope", "_proofreading_agent_blocker", "_human_is_split_job", "_proofreading_run_active", "_as_dt"}
    consts = {"_PROOFREADING_RUN_STALE_SECONDS"}
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in wanted:
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in n.targets):
            nodes.append(n)
    ns = {"datetime": datetime, "timezone": timezone}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main", "exec"), ns)
    return ns


class ProofreadingAgentTests(unittest.TestCase):
    def test_scope(self):
        f = _ns()["_proofreading_agent_scope"]
        self.assertEqual(f({"job_type": "general_job"}), "audio")
        self.assertEqual(f({"job_type": "pdf_job", "pdf_review": {}}), "")
        self.assertEqual(f({"job_type": "pdf_job", "pdf_review": {"page_texts": ["a"]}}), "image")
        self.assertEqual(f({"job_type": "letter_job"}), "")

    def test_blocker_needs_every_part_submitted(self):
        ns = _ns()
        ns["_human_is_split_job"] = lambda job: bool(job.get("segments"))
        b = ns["_proofreading_agent_blocker"]
        done = {"status": "submitted", "transcript": "x"}
        self.assertEqual(b({"job_type": "general_job", "segments": [done, dict(done)]}), "")
        self.assertTrue(b({"job_type": "general_job", "segments": [done, {"status": "in_progress"}]}))
        self.assertTrue(b({"job_type": "general_job", "segments": [done], "proofreader_status": "in_progress"}))

    def test_routing_and_privacy_wiring(self):
        self.assertIn('"proofreading_agent": ("Proofreading Agent"', SRC)
        self.assertEqual(SRC.count('"proofreading_run"'), 2 + SRC.count('"proofreading_run"') - 2)
        self.assertIn('"ai_review_run", "proofreading_run"', SRC)
        self.assertIn("/proofreading-agent", SRC)

    def test_stale_run_not_active(self):
        f = _ns()["_proofreading_run_active"]
        self.assertTrue(f({"proofreading_run": {"status": "processing", "startedAt": datetime.now(timezone.utc).isoformat()}}))
        self.assertFalse(f({"proofreading_run": {"status": "processing", "startedAt": "2020-01-01T00:00:00+00:00"}}))
        self.assertFalse(f({"proofreading_run": {"status": "failed"}}))


if __name__ == "__main__":
    unittest.main()
