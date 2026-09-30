"""Checks for the model catalogue, renamed-model aliases and audio byte ranges."""
import ast
import re
import unittest
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class FakeResponse:
    def __init__(self, content=b"", status_code=200, media_type=None, headers=None):
        self.content, self.status_code, self.media_type, self.headers = content, status_code, media_type, headers or {}


class FakeRequest:
    def __init__(self, range_header=""):
        self.headers = {"range": range_header} if range_header else {}


class ModelsAndAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted_funcs = {"_human_audio_response", "_ask_provider_ready", "ask_models_for", "resolve_ask_model"}
        wanted_assigns = {"ASK_MODEL_CATALOGUE", "ASK_MODEL_ALIASES", "ASK_DEFAULT_MODEL", "ASK_FALLBACK_MODEL", "PREMIUM_AI_PLANS"}
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
                body.append(node)
            elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") in wanted_assigns for t in node.targets):
                body.append(node)
        namespace = {
            "re": re, "os": __import__("os"), "Response": FakeResponse, "Request": FakeRequest,
            "is_ai_allowed": lambda plan, email, credits: True,
            "is_admin_user": lambda email: email == "admin@example.com",
            "is_comp_access_user": lambda email: False,
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        cls.ns = namespace

    def test_old_model_ids_move_to_their_successors(self):
        resolve = self.ns["resolve_ask_model"]
        self.assertEqual(resolve("claude-opus-4-6", "Monthly Plan", "a@b.c", True, True)[0], "claude-opus-5-5")
        self.assertEqual(resolve("claude-sonnet-5", "Monthly Plan", "a@b.c", True, True)[0], "claude-sonnet-5-5")
        self.assertEqual(resolve("gemini-3.6-flash", "Monthly Plan", "a@b.c", True, True)[0], "gemini-3.8-flash")

    def test_unknown_model_falls_back_to_default(self):
        self.assertEqual(self.ns["resolve_ask_model"]("nope", "Monthly Plan", "a@b.c", True, True)[0], "gpt-5.6-luna")

    def test_deepseek_is_hidden_until_its_key_is_set(self):
        import os
        os.environ.pop("DEEPSEEK_API_KEY", None)
        ids = [m["id"] for m in self.ns["ask_models_for"]("Monthly Plan", "a@b.c", True, True)]
        self.assertNotIn("deepseek-v4-flash", ids)
        os.environ["DEEPSEEK_API_KEY"] = "x"
        try:
            ids = [m["id"] for m in self.ns["ask_models_for"]("Monthly Plan", "a@b.c", True, True)]
            self.assertIn("deepseek-v4-flash", ids)
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)

    def test_standard_plan_does_not_get_premium_models(self):
        ids = [m["id"] for m in self.ns["ask_models_for"]("One-Week Plan", "a@b.c", True, True)]
        self.assertIn("gpt-5.6-luna", ids)
        self.assertNotIn("claude-opus-5-5", ids)

    def test_audio_full_and_ranges(self):
        respond = self.ns["_human_audio_response"]
        raw = bytes(range(100))
        full = respond(FakeRequest(), raw, "audio/mpeg", "part 1.mp3", False)
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full.headers["Content-Length"], "100")
        self.assertEqual(full.headers["Accept-Ranges"], "bytes")
        part = respond(FakeRequest("bytes=10-19"), raw, "audio/mpeg", "x.mp3", False)
        self.assertEqual((part.status_code, part.content), (206, raw[10:20]))
        self.assertEqual(part.headers["Content-Range"], "bytes 10-19/100")
        tail = respond(FakeRequest("bytes=90-"), raw, "audio/mpeg", "x.mp3", False)
        self.assertEqual(tail.content, raw[90:])
        suffix = respond(FakeRequest("bytes=-5"), raw, "audio/mpeg", "x.mp3", False)
        self.assertEqual(suffix.content, raw[95:])
        bad = respond(FakeRequest("bytes=500-600"), raw, "audio/mpeg", "x.mp3", False)
        self.assertEqual(bad.status_code, 416)

    def test_download_flag_and_safe_filename(self):
        respond = self.ns["_human_audio_response"]
        out = respond(FakeRequest(), b"abc", "audio/mpeg", 'we"ird name.mp3', True)
        self.assertTrue(out.headers["Content-Disposition"].startswith("attachment;"))
        self.assertNotIn('"ird', out.headers["Content-Disposition"])


if __name__ == "__main__":
    unittest.main()
