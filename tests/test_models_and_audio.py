"""Checks for the model catalogue, renamed-model aliases and audio byte ranges."""
import ast
import re
import unittest
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class FakeResponse:
    def __init__(self, content=b"", status_code=200, media_type=None, headers=None):
        self.content, self.status_code, self.media_type, self.headers = content, status_code, media_type, headers or {}


class FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self.lines, self.status_code, self.closed = lines, status_code, False
        self.text = ""

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def close(self):
        self.closed = True


class FakeRequests:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class FakeRequest:
    def __init__(self, range_header=""):
        self.headers = {"range": range_header} if range_header else {}


class ModelsAndAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted_funcs = {"_human_audio_response", "_normalize_human_job_instructions", "_ask_provider_ready", "ask_models_for", "resolve_ask_model", "_openai_format_content_text", "_openai_format_stream_text", "_ask_openai_format"}
        wanted_assigns = {"ASK_MODEL_CATALOGUE", "ASK_MODEL_ALIASES", "ASK_DEFAULT_MODEL", "ASK_LUNA_MODEL", "ASK_TRANSCRIPT_DEFAULT_MODEL", "ASK_FALLBACK_MODEL", "PREMIUM_AI_PLANS", "MISTRAL_API_MODEL_ALIASES", "MISTRAL_PLAIN_TEXT_MODELS"}
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
                body.append(node)
            elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") in wanted_assigns for t in node.targets):
                body.append(node)
        namespace = {
            "re": re, "os": __import__("os"), "json": __import__("json"),
            "logger": __import__("logging").getLogger("test.models"),
            "TYPEMYWORDZ_AI_NAME": "TypeMyworDz",
            "OPENAI_FORMAT_ENDPOINTS": {"mistral": ("https://api.mistral.ai/v1/chat/completions", lambda: "test-key")},
            "requests": None, "Response": FakeResponse, "Request": FakeRequest,
            "is_ai_allowed": lambda plan, email, credits: True,
            "is_admin_user": lambda email: email == "admin@example.com",
            "is_comp_access_user": lambda email: False,
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        cls.ns = namespace

    def test_old_model_ids_move_to_their_successors(self):
        resolve = self.ns["resolve_ask_model"]
        self.assertEqual(resolve("claude-opus-4-6", "Monthly Plan", "a@b.c", True, True)[0], "claude-opus-5-5")
        self.assertEqual(resolve("claude-sonnet-5", "Monthly Plan", "a@b.c", True, True)[0], "claude-haiku-5-5")
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

    def test_defaults_with_and_without_deepseek(self):
        import os
        resolve = self.ns["resolve_ask_model"]
        os.environ.pop("DEEPSEEK_API_KEY", None)
        self.assertEqual(resolve("", "Monthly Plan", "a@b.c", False, True)[0], "gpt-5.6-luna")
        os.environ["DEEPSEEK_API_KEY"] = "x"
        try:
            self.assertEqual(resolve("", "Monthly Plan", "a@b.c", False, True)[0], "deepseek-v4-flash")
            self.assertEqual(resolve("", "One-Week Plan", "a@b.c", True, True)[0], "deepseek-v4-flash")
            # Questions inside a transcript default to Gemini, but a chosen model wins.
            self.assertEqual(resolve("", "One-Week Plan", "a@b.c", True, True, True)[0], "gemini-3.1-flash-lite")
            self.assertEqual(resolve("gpt-5.6-luna", "One-Week Plan", "a@b.c", True, True, True)[0], "gpt-5.6-luna")
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)

    def test_gpt6_luna_and_mistral_small4_are_standard_and_key_gated(self):
        catalogue = {model["id"]: model for model in self.ns["ASK_MODEL_CATALOGUE"]}
        self.assertEqual((catalogue["gpt-6-luna"]["provider"], catalogue["gpt-6-luna"]["tier"], catalogue["gpt-6-luna"]["credits"]), ("openai", "standard", 1))
        self.assertEqual((catalogue["mistral-small-2603"]["provider"], catalogue["mistral-small-2603"]["tier"], catalogue["mistral-small-2603"]["credits"]), ("mistral", "standard", 1))
        import os
        previous = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            ids = [model["id"] for model in self.ns["ask_models_for"]("One-Week Plan", "a@b.c", True, True)]
            self.assertIn("gpt-6-luna", ids)
            self.assertNotIn("mistral-small-2603", ids)
            self.assertNotIn("mistral-large-4", ids)
            os.environ["MISTRAL_API_KEY"] = "test-key"
            ids = [model["id"] for model in self.ns["ask_models_for"]("One-Week Plan", "a@b.c", True, True)]
            self.assertIn("mistral-small-2603", ids)
        finally:
            if previous is None:
                os.environ.pop("MISTRAL_API_KEY", None)
            else:
                os.environ["MISTRAL_API_KEY"] = previous

    def test_mistral_large4_is_premium_and_uses_existing_plan_gates(self):
        import os
        catalogue = {model["id"]: model for model in self.ns["ASK_MODEL_CATALOGUE"]}
        large4 = catalogue["mistral-large-4"]
        self.assertEqual((large4["provider"], large4["tier"], large4["credits"]), ("mistral", "premium", 10))
        previous = os.environ.get("MISTRAL_API_KEY")
        os.environ["MISTRAL_API_KEY"] = "test-key"
        try:
            monthly_ids = {model["id"] for model in self.ns["ask_models_for"]("Monthly Plan", "a@b.c", True, True)}
            weekly_ids = {model["id"] for model in self.ns["ask_models_for"]("One-Week Plan", "a@b.c", True, True)}
            self.assertIn("mistral-large-4", monthly_ids)
            self.assertNotIn("mistral-large-4", weekly_ids)
        finally:
            if previous is None:
                os.environ.pop("MISTRAL_API_KEY", None)
            else:
                os.environ["MISTRAL_API_KEY"] = previous

    def test_openai_compatible_parser_handles_block_based_model_content(self):
        parse = self.ns["_openai_format_content_text"]
        self.assertEqual(parse("  A plain response.  "), "A plain response.")
        self.assertEqual(parse([
            {"type": "text", "text": "Mistral Large "},
            {"type": "text", "text": "4 is ready."},
        ]), "Mistral Large 4 is ready.")
        self.assertEqual(parse([
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "Visible answer."},
        ]), "Visible answer.")
        self.assertEqual(parse(None), "")

    def test_mistral_auto_continue_uses_streaming_chat_and_closes_response(self):
        response = FakeStreamResponse([
            b'data: {"choices":[{"delta":{"content":"A "},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"complete draft"},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ])
        fake_requests = FakeRequests(response)
        self.ns["requests"] = fake_requests
        try:
            answer = self.ns["_ask_openai_format"](
                "mistral", "mistral-large-4", "system", [], "format this transcript", [], 16000,
                auto_continue=True,
            )
        finally:
            self.ns["requests"] = None
        self.assertEqual(answer, "A complete draft")
        args, kwargs = fake_requests.calls[0]
        self.assertEqual(args[0], "https://api.mistral.ai/v1/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "mistral-large-4-0")
        self.assertEqual(kwargs["json"]["reasoning_effort"], "none")
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["timeout"], (15, 300))
        self.assertTrue(response.closed)

    def test_human_job_instructions_have_no_application_character_cap(self):
        normalize = self.ns["_normalize_human_job_instructions"]
        large_note = "x" * 100000
        self.assertEqual(normalize(large_note), large_note)
        self.assertEqual(normalize(f"  {large_note}  "), large_note)

    def test_mistral_large4_stream_reads_text_after_reasoning_chunks(self):
        response = FakeStreamResponse([
            b'data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"private reasoning"}]},{"type":"text","text":"Visible"}]},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":" answer"},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ])
        text, finish_reason = self.ns["_openai_format_stream_text"](response)
        self.assertEqual(text, "Visible answer")
        self.assertEqual(finish_reason, "stop")

    def test_completed_empty_mistral_stream_is_reported(self):
        response = FakeStreamResponse([
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ])
        with self.assertRaisesRegex(ValueError, "no text"):
            self.ns["_openai_format_stream_text"](response)

    def test_mistral_stream_requires_completion_marker(self):
        response = FakeStreamResponse([
            b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}'
        ])
        with self.assertRaisesRegex(RuntimeError, "completion marker"):
            self.ns["_openai_format_stream_text"](response)

    def test_standard_plan_does_not_get_premium_models(self):
        ids = [m["id"] for m in self.ns["ask_models_for"]("One-Week Plan", "a@b.c", True, True)]
        self.assertIn("gpt-5.6-luna", ids)
        self.assertIn("gemini-3.5-flash-lite", ids)
        self.assertNotIn("claude-opus-5-5", ids)
        research_ids = [m["id"] for m in self.ns["ask_models_for"]("One-Week Plan", "a@b.c", False, True)]
        self.assertNotIn("gemini-3.5-flash-lite", research_ids)

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
