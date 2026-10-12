"""Voxtral routing and parsing, rating code-fence fix, combined-parts Word download (no paid calls)."""
import ast
import re
import unittest
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.shared import Inches

MAIN = Path(__file__).resolve().parents[1] / "main.py"
SRC = MAIN.read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _load(names, extra=None):
    nodes = []
    for n in TREE.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names:
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            nodes.append(n)
    ns = dict(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main", "exec"), ns)
    return ns


def _routing(**overrides):
    base = {
        "MISTRAL_API_KEY": "k", "ASSEMBLYAI_API_KEY": "k", "OPENAI_WHISPER_SERVICE_RAILWAY_URL": "u", "DEEPGRAM_SERVICE_RAILWAY_URL": "u",
        "ADMIN_EMAILS": ["typemywordz@gmail.com"], "VOXTRAL_TESTER_EMAIL": "njokigituku@gmail.com",
        "DEEPGRAM_TESTER_EMAIL": "info@typemywordz.ai", "VOXTRAL_PRIMARY": True,
    }
    base.update(overrides)
    ns = _load({"is_admin_user", "get_transcription_services"}, base)
    return ns["get_transcription_services"]


class RoutingTests(unittest.TestCase):
    def test_everyone_gets_voxtral_then_assemblyai_deepgram_openai(self):
        f = _routing()
        for plan in ("free", "Monthly Plan", "Yearly Plan", "One-Week Plan"):
            r = f(plan, False, "client@example.com")
            self.assertEqual([r["tier_1"], r["tier_2"], r["tier_3"], r["tier_4"]], ["voxtral", "assemblyai", "deepgram", "openai_whisper"], plan)
        admin = f("free", True, "typemywordz@gmail.com")
        self.assertEqual(admin["tier_1"], "voxtral")

    def test_voxtral_tester_has_no_fallback(self):
        r = _routing()("Monthly Plan", True, "NjokiGituku@gmail.com")
        self.assertEqual((r["tier_1"], r["tier_2"], r["tier_3"], r["tier_4"]), ("voxtral", None, None, None))

    def test_deepgram_tester_unchanged(self):
        r = _routing()("free", False, "info@typemywordz.ai")
        self.assertEqual((r["tier_1"], r["tier_2"]), ("deepgram", "openai_whisper"))

    def test_missing_mistral_key_skips_voxtral_for_everyone_else(self):
        r = _routing(MISTRAL_API_KEY=None)("free", False, "client@example.com")
        self.assertEqual(r["tier_1"], "assemblyai")
        self.assertNotIn("voxtral", [r["tier_1"], r["tier_2"], r["tier_3"], r["tier_4"]])

    def test_switch_off_restores_plan_order(self):
        f = _routing(VOXTRAL_PRIMARY=False)
        self.assertEqual(f("Monthly Plan", False, "c@example.com")["tier_1"], "openai_whisper")
        self.assertEqual(f("free", False, "c@example.com")["tier_1"], "assemblyai")


class VoxtralParsingTests(unittest.TestCase):
    def setUp(self):
        self.f = _load({"_voxtral_result_from_payload", "VOXTRAL_TRANSCRIPTION_MODEL"}, {})["_voxtral_result_from_payload"]

    def test_plain_transcript_keeps_segments(self):
        out = self.f({"text": "Hello there. How are you?", "language": "en", "model": "voxtral-mini-2602",
                      "segments": [{"text": "Hello there.", "start": 0, "end": 1.2}, {"text": "How are you?", "start": 1.2, "end": 2.5}],
                      "usage": {"prompt_audio_seconds": 3}}, False)
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["transcription"], "Hello there. How are you?")
        self.assertEqual(len(out["segments"]), 2)
        self.assertEqual(out["duration"], 3)
        self.assertEqual(out["timings_source"], "Voxtral")

    def test_speaker_labels_merge_turns(self):
        out = self.f({"text": "x", "segments": [
            {"text": "Hi.", "start": 0, "end": 1, "speaker_id": "speaker_0"},
            {"text": "How are you?", "start": 1, "end": 2, "speaker_id": "speaker_0"},
            {"text": "Fine.", "start": 2, "end": 3, "speaker_id": "speaker_1"}]}, True)
        self.assertEqual(out["transcription"], "<strong>Speaker 1:</strong> Hi. How are you?\n<strong>Speaker 2:</strong> Fine.")
        self.assertEqual(len(out["segments"]), 2)
        self.assertTrue(out["has_speaker_labels"])

    def test_empty_transcript_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.f({"text": "", "segments": []}, False)

    def test_runner_knows_voxtral_in_every_tier(self):
        for tier in (1, 2, 3, 4):
            self.assertIn(f"attempt_voxtral({tier})", SRC)
        self.assertIn('"Authorization": f"Bearer {MISTRAL_API_KEY}"', SRC)
        self.assertIn("api.mistral.ai/v1/audio/transcriptions", SRC)


class RatingFenceTests(unittest.TestCase):
    def test_fenced_json_is_accepted_when_a_validator_exists(self):
        start = SRC.index("async def _human_call_model_chain")
        body = SRC[start:start + 1800]
        self.assertIn('stray == "the whole draft wrapped in a code block" and response_validator', body)
        ns = _load({"_parse_json_object", "_rating_agent_parse"}, {"json": __import__("json")})
        self.assertEqual(ns["_rating_agent_parse"]('```json\n{"rating": 4, "comments": "Good work."}\n```'), (4, "Good work."))


class CombinedPartsTests(unittest.TestCase):
    def test_docx_keeps_tabs_blank_lines_and_formatting(self):
        ns = _load({"_proofreading_marked_text_to_docx", "_PR_MARK_TOKEN_RE"}, {"Document": Document, "Inches": Inches, "BytesIO": BytesIO, "re": re})
        data = ns["_proofreading_marked_text_to_docx"]("\tFirst [[b]]bold[[/b]] line.\n\n\tSecond [[u]]under[[/u]] and [[i]]ital[[/i]].")
        doc = Document(BytesIO(data))
        paragraphs = doc.paragraphs
        self.assertEqual(len(paragraphs), 3)
        self.assertEqual(paragraphs[0].text, "\tFirst bold line.")
        self.assertEqual(paragraphs[1].text, "")
        self.assertTrue(any(run.bold for run in paragraphs[0].runs))
        self.assertTrue(any(run.underline for run in paragraphs[2].runs))
        self.assertTrue(any(run.italic for run in paragraphs[2].runs))

    def test_endpoint_is_admin_only_and_needs_every_part(self):
        start = SRC.index("async def human_combined_parts_docx")
        body = SRC[start:start + 1500]
        self.assertIn("_require_human_job_admin(request)", body)
        self.assertIn("Every part must be submitted", body)
        self.assertIn("/human-transcription/jobs/{job_id}/combined-parts.docx", SRC)

    def test_single_part_rule_is_hardcoded(self):
        self.assertIn("SINGLE-PART JOBS. If the job contains only one part", SRC)
        self.assertIn("skip the 'combining parts' step", SRC)


if __name__ == "__main__":
    unittest.main()
