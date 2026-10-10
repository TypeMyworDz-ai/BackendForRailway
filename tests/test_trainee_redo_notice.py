"""Redo invites must reach the trainee in the app and by email."""
import ast
import asyncio
import logging
import unittest
from html import escape
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class TraineeRedoNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        cls.tree = tree
        wanted = {"_redo_module_lines", "build_trainee_redo_email", "_notify_trainee_redo_invite"}
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted]
        levels = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "TRAINING_LEVELS")
        cls.calls = {"notes": [], "mail": []}
        calls = cls.calls

        async def fake_note(uid, key, kind, title, body, **kw):
            calls["notes"].append((uid, kind, title, body, kw))
            return "id"

        async def fake_mail(address, subject, html, text, label="email"):
            calls["mail"].append((address, subject, html, text))
            return {"sent": True}

        import uuid
        cls.ns = {
            "escape": escape, "asyncio": asyncio, "uuid": uuid, "logger": logging.getLogger("t"),
            "APP_URL": "https://typemywordz.ai", "SUPPORT_EMAIL": "info@typemywordz.ai",
            "_create_user_notification": fake_note, "_send_resend_message": fake_mail,
            "firebase_auth": None,
        }
        exec(compile(ast.Module(body=[levels] + nodes, type_ignores=[]), "main", "exec"), cls.ns)

    def setUp(self):
        self.calls["notes"].clear()
        self.calls["mail"].clear()

    def test_notification_and_email_carry_modules_and_comments(self):
        asyncio.run(self.ns["_notify_trainee_redo_invite"]("u1", {"email": "t@example.test", "name": "Amina Otieno"}, [6], "Please fix the speaker labels"))
        self.assertEqual(len(self.calls["notes"]), 1)
        uid, kind, title, body, kw = self.calls["notes"][0]
        self.assertEqual((uid, kind, kw["route"], kw["requires_action"]), ("u1", "trainee_redo", "trainee", True))
        self.assertIn("module 6", title)
        self.assertIn("Please fix the speaker labels", body)
        address, subject, html, text = self.calls["mail"][0]
        self.assertEqual(address, "t@example.test")
        self.assertIn("module 6", subject)
        for blob in (html, text):
            self.assertIn("Final practical: Human Job audio", blob)
            self.assertIn("Please fix the speaker labels", blob)
        self.assertIn("Amina", html)

    def test_email_escapes_comments(self):
        _, html, _ = self.ns["build_trainee_redo_email"]("A", [1, 2], "<script>x</script>")
        self.assertNotIn("<script>", html)

    def test_failures_never_raise(self):
        async def boom(*a, **k):
            raise RuntimeError("down")
        original = self.ns["_create_user_notification"], self.ns["_send_resend_message"]
        self.ns["_create_user_notification"] = boom
        self.ns["_send_resend_message"] = boom
        try:
            asyncio.run(self.ns["_notify_trainee_redo_invite"]("u1", {"email": "t@example.test"}, [1], ""))
        finally:
            self.ns["_create_user_notification"], self.ns["_send_resend_message"] = original

    def test_decision_route_triggers_notice_and_admin_lists(self):
        src = MAIN_PATH.read_text(encoding="utf-8")
        decision = next(n for n in self.tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "admin_trainee_decision")
        self.assertIn("_notify_trainee_redo_invite(uid, profile, levels, message)", ast.unparse(decision))
        self.assertIn("ADMIN_EMAILS = ['typemywordz@gmail.com', 'info@typemywordz.ai']", src)
        self.assertIn("'donotgrowweary95@gmail.com'", src)


if __name__ == "__main__":
    unittest.main()
