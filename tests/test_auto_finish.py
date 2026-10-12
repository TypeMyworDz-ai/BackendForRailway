import ast
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
NAMES = {"_human_job_is_internal", "_human_job_proofread_done", "_human_submitted_at", "_human_auto_finish_due",
         "_human_admin_finish_job_eligible", "_as_dt"}


def _ns():
    tree = ast.parse(SRC)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "HUMAN_AUTO_FINISH_HOURS" for t in n.targets)] + body
    ns = {"datetime": datetime, "timedelta": timedelta, "timezone": timezone}
    exec(compile(ast.Module(body=body, type_ignores=[]), "main", "exec"), ns)
    return ns


NOW = datetime(2026, 10, 12, 12, 0, 0)


def hours_ago(h):
    return NOW - timedelta(hours=h)


class AutoFinishTests(unittest.TestCase):
    def setUp(self):
        self.due = _ns()["_human_auto_finish_due"]
        self.job = {"status": "submitted", "admin_uploaded": True, "transcript": "Done.", "reviewer_status": "completed",
                    "submittedAt": hours_ago(13)}

    def test_proofread_internal_job_finishes_after_twelve_hours(self):
        self.assertTrue(self.due(self.job, now=NOW))
        self.assertFalse(self.due({**self.job, "submittedAt": hours_ago(11)}, now=NOW))

    def test_job_that_still_needs_proofreading_never_auto_finishes(self):
        job = {**self.job}
        job.pop("reviewer_status")
        self.assertFalse(self.due(job, now=NOW))
        self.assertTrue(self.due({**job, "admin_proofread_at": hours_ago(13)}, now=NOW))

    def test_latest_proofreading_time_restarts_the_clock(self):
        self.assertFalse(self.due({**self.job, "submittedAt": hours_ago(30), "ai_review_appliedAt": hours_ago(1)}, now=NOW))

    def test_client_jobs_and_active_assignments_are_left_alone(self):
        self.assertFalse(self.due({**self.job, "admin_uploaded": False, "job_type": "general_job"}, now=NOW))
        self.assertFalse(self.due({**self.job, "status": "proofreading_available"}, now=NOW))
        self.assertFalse(self.due({**self.job, "proofreader_status": "in_progress"}, now=NOW))
        self.assertFalse(self.due({**self.job, "transcript": ""}, now=NOW))

    def test_monitor_is_started_and_releases_through_a_transaction(self):
        self.assertIn("auto_finish_task = asyncio.create_task(_human_auto_finish_monitor())", SRC)
        body = SRC[SRC.index("async def _human_auto_finish_submitted_jobs"):SRC.index("async def _human_auto_finish_monitor")]
        self.assertIn('"status": "released"', body)
        self.assertIn("firestore.transactional", body)
        self.assertIn("_human_subadmin_accrue_job_earnings", body)


if __name__ == "__main__":
    unittest.main()
