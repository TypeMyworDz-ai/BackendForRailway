import ast
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
TREE = ast.parse(SRC)


def _fn(name):
    return next(node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def _ns():
    ns = {"re": re, "HUMAN_AI_AGENTS": {"agent_a": {}, "agent_b": {}}}
    ns["is_admin_user"] = lambda email: str(email).lower() == "typemywordz@gmail.com"
    for name in ("_human_job_owner_of", "_human_admin_can_see_job", "_human_anonymize_node", "_human_collect_worker_uids"):
        exec(compile(ast.Module(body=[_fn(name)], type_ignores=[]), "main.py", "exec"), ns)
    return ns


ADMIN = {"uid": "main1", "email": "typemywordz@gmail.com"}
INFO = {"uid": "info1", "email": "info@typemywordz.ai"}
GRACE = {"uid": "grace1", "email": "gracenyaitara@gmail.com"}


class Visibility(unittest.TestCase):
    def test_each_admin_sees_only_own_uploads(self):
        see = _ns()["_human_admin_can_see_job"]
        own_info = {"created_by_uid": "info1", "created_by_email": "info@typemywordz.ai"}
        self.assertTrue(see(INFO, own_info))
        self.assertFalse(see(GRACE, own_info))
        self.assertFalse(see(ADMIN, own_info))

    def test_main_admin_sees_everything_only_in_dashboard_scope(self):
        see = _ns()["_human_admin_can_see_job"]
        own_info = {"created_by_uid": "info1"}
        self.assertTrue(see(ADMIN, own_info, owner_all=True))
        self.assertFalse(see(GRACE, own_info, owner_all=True))

    def test_ownerless_jobs_belong_to_main_admin(self):
        see = _ns()["_human_admin_can_see_job"]
        self.assertTrue(see(ADMIN, {}))
        self.assertFalse(see(INFO, {}))

    def test_email_fallback_when_no_uid(self):
        see = _ns()["_human_admin_can_see_job"]
        self.assertTrue(see(GRACE, {"created_by_email": "GraceNyaitara@gmail.com"}))
        self.assertFalse(see(INFO, {"created_by_email": "gracenyaitara@gmail.com"}))


class Anonymity(unittest.TestCase):
    def test_workers_and_agents_are_relabelled(self):
        ns = _ns()
        data = {"jobs": [{
            "worker_uid": "w1", "worker_name": "Jane Doe", "worker_email": "jane@example.com",
            "note": "Claimed by Jane Doe (jane@example.com)",
            "segments": [{"id": "s1", "ai_agent_id": "agent_b", "worker_name": "Claude"}],
        }], "workers": [{"uid": "w1", "online": True, "email": "jane@example.com", "name": "Jane Doe", "phone": "1"}]}
        out = ns["_human_anonymize_node"](data, {"w1": "Human Worker 3"}, [("jane doe", "Human Worker 3"), ("jane@example.com", "Human Worker 3")])
        job = out["jobs"][0]
        self.assertEqual(job["worker_name"], "Human Worker 3")
        self.assertEqual(job["worker_email"], "")
        self.assertNotIn("Jane", job["note"])
        self.assertNotIn("jane@", job["note"])
        self.assertEqual(job["segments"][0]["worker_name"], "Agent Worker 2")
        self.assertEqual(out["workers"][0]["name"], "Human Worker 3")
        self.assertEqual(out["workers"][0]["email"], "")


class Wiring(unittest.TestCase):
    def test_grace_is_a_subadmin_everywhere(self):
        for const in ("HUMAN_JOB_ADMIN_EMAILS", "PDF_JOB_ADMIN_EMAILS", "LETTER_JOB_ADMIN_EMAILS"):
            line = next(l for l in SRC.splitlines() if l.startswith(const + " ="))
            self.assertIn("gracenyaitara@gmail.com", line)
        self.assertNotIn("gracenyaitara@gmail.com", next(l for l in SRC.splitlines() if l.startswith("ADMIN_EMAILS")))

    def test_filters_and_middleware_are_in_place(self):
        self.assertIn("_human_subadmin_scope_middleware", SRC)
        self.assertIn("_human_job_owner_lookup(job_id)", ast.unparse(_fn("_notify_human_admins")))
        for name in ("human_list_jobs", "human_admin_list_pdf_jobs", "human_admin_list_letter_jobs", "human_admin_recent_pdf_batches", "human_workflow_notifications"):
            self.assertIn("_human_admin_can_see_job", ast.unparse(_fn(name)), name)
        self.assertIn("owner_all", ast.unparse(_fn("human_list_jobs")))


if __name__ == "__main__":
    unittest.main()
