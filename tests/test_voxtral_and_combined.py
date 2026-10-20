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
        "DEEPGRAM_TESTER_EMAIL": "info@typemywordz.ai", "OPENAI_TESTER_EMAIL": "openai-tester@example.com",
    }
    base.update(overrides)
    ns = _load({"is_admin_user", "get_transcription_services"}, base)
    return ns["get_transcription_services"]


class RoutingTests(unittest.TestCase):
    def test_clients_get_assemblyai_then_voxtral_openai_deepgram(self):
        f = _routing()
        for plan in ("free", "Monthly Plan", "Yearly Plan", "One-Week Plan"):
            for labels in (False, True):
                r = f(plan, labels, "client@example.com")
                self.assertEqual([r["tier_1"], r["tier_2"], r["tier_3"], r["tier_4"]], ["assemblyai", "voxtral", "openai_whisper", "deepgram"], plan)
        self.assertEqual(f("free", True, "typemywordz@gmail.com")["tier_1"], "assemblyai")

    def test_voxtral_tester_has_no_fallback(self):
        r = _routing()("Monthly Plan", True, "NjokiGituku@gmail.com")
        self.assertEqual((r["tier_1"], r["tier_2"], r["tier_3"], r["tier_4"]), ("voxtral", None, None, None))

    def test_deepgram_tester_unchanged(self):
        r = _routing()("free", False, "info@typemywordz.ai")
        self.assertEqual((r["tier_1"], r["tier_2"]), ("deepgram", "openai_whisper"))

    def test_missing_mistral_key_skips_voxtral(self):
        r = _routing(MISTRAL_API_KEY=None)("free", False, "client@example.com")
        self.assertEqual([r["tier_1"], r["tier_2"], r["tier_3"], r["tier_4"]], ["assemblyai", "openai_whisper", "deepgram", None])


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

    def test_form_data_is_a_dict_so_the_async_client_accepts_it(self):
        self.assertIn('form = {"model": VOXTRAL_TRANSCRIPTION_MODEL', SRC)
        self.assertNotIn('form = [("model"', SRC)

    def test_async_client_really_accepts_multipart_with_dict_data(self):
        import asyncio
        import httpx

        async def go():
            def handler(request):
                body = request.read()
                assert b'name="model"' in body and b'name="file"' in body and b'name="diarize"' in body
                return httpx.Response(200, json={"text": "ok"})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                r = await client.post("https://x.test/v1", files={"file": ("a.mp3", b"abc", "audio/mpeg")},
                                      data={"model": "m", "timestamp_granularities": "segment", "diarize": "true"})
                return r.status_code
        self.assertEqual(asyncio.run(go()), 200)


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

    def test_proofreader_specific_rules_are_no_longer_hardcoded(self):
        start = SRC.index("_PROOFREADING_AGENT_SYSTEM = (")
        end = SRC.index("_PROOFREADING_RESEARCH_RULES = (", start)
        prompt = SRC[start:end]
        self.assertIn("paragraph breaks", prompt)
        self.assertIn("minimal contextual corrections", prompt)
        self.assertIn("follow only the super-admin", prompt.lower())
        for removed_rule in ("SINGLE-PART JOBS", "STREETS, ADDRESSES AND LOCATIONS", "INDENTATION:", "A. You receive every part"):
            self.assertNotIn(removed_rule, prompt)


class HumanWorkSttAndCreditTests(unittest.TestCase):
    def test_orders_are_as_specified(self):
        ns = _load({"HUMAN_WORKER_STT_ORDER", "HUMAN_AGENT_STT_ORDER", "HUMAN_PROOFREADING_STT_ORDER",
                    "HUMAN_WORK_STT_PROVIDERS", "HUMAN_WORK_STT_DEFAULTS", "_human_work_stt_config", "_MODEL_ROUTE_CACHE"}, {})
        self.assertEqual(ns["HUMAN_WORKER_STT_ORDER"], ("deepgram", "voxtral", "openai"))
        self.assertEqual(ns["HUMAN_AGENT_STT_ORDER"], ("assemblyai",))
        self.assertEqual(ns["HUMAN_PROOFREADING_STT_ORDER"], ("deepgram",))
        self.assertNotIn("assemblyai", ns["HUMAN_WORKER_STT_ORDER"])
        self.assertEqual(ns["_human_work_stt_config"]("audio_agent"), {"order": ("assemblyai",), "comparison": ""})
        self.assertEqual(ns["_human_work_stt_config"]("proofreading_agent"), {"order": ("deepgram",), "comparison": ""})

    def test_each_audio_stage_reads_super_admin_stt_settings(self):
        worker = SRC[SRC.index("async def human_worker_ai_draft"):SRC.index("async def human_worker_ai_proofread_draft")]
        agent_start = SRC.index("async def _human_ai_transcribe_audio")
        agent = SRC[agent_start:SRC.index("def _human_term_snippets", agent_start)]
        proofreader = SRC[SRC.index("async def _proofreading_agent_full_audio_transcripts"):SRC.index("async def _proofreading_agent_audio")]
        self.assertIn('_human_work_stt_config("worker_draft")', worker)
        self.assertIn('_human_work_stt_config(stage)', agent)
        self.assertIn('_human_work_stt_config("proofreading_agent")', proofreader)
        self.assertIn('stage="audio_agent"', SRC)
        self.assertIn('stage="letter_review"', SRC)

    def test_stage_settings_allow_one_primary_and_optional_comparison(self):
        ns = _load({"HUMAN_WORKER_STT_ORDER", "HUMAN_AGENT_STT_ORDER", "HUMAN_PROOFREADING_STT_ORDER", "HUMAN_WORK_STT_PROVIDERS", "HUMAN_WORK_STT_DEFAULTS", "_human_work_stt_config", "_MODEL_ROUTE_CACHE"}, {})
        ns["_MODEL_ROUTE_CACHE"]["human_work_stt"] = {
            "audio_agent": {"order": ["assemblyai", "deepgram"], "comparison": "voxtral"},
            "proofreading_agent": {"order": ["deepgram"], "comparison": "assemblyai"},
        }
        self.assertEqual(ns["_human_work_stt_config"]("audio_agent"), {"order": ("assemblyai", "deepgram"), "comparison": "voxtral"})
        self.assertEqual(ns["_human_work_stt_config"]("proofreading_agent"), {"order": ("deepgram",), "comparison": "assemblyai"})

    def test_stage_routing_api_is_admin_controlled(self):
        self.assertIn("/api/admin/human-work-stt-routing", SRC)
        route = SRC[SRC.index("async def admin_put_human_work_stt_routing"):SRC.index('@app.put("/api/admin/model-routing")')]
        self.assertIn("_require_admin(request)", route)
        self.assertIn("primary provider and up to two fallbacks", route)
        self.assertIn("comparison provider must differ", route)

    def test_chain_falls_through_and_labels_provider(self):
        import asyncio
        calls = []

        async def voxtral(*a):
            calls.append("voxtral"); return {"status": "failed", "error": "boom"}

        async def assembly(*a):
            calls.append("assemblyai"); raise RuntimeError("down")

        async def deepgram(*a):
            calls.append("deepgram"); return {"status": "completed", "transcription": "hello there"}

        ns = _load({"_human_stt_chain"}, {
            "asyncio": asyncio, "logger": __import__("logging").getLogger("t"), "ASSEMBLYAI_API_KEY": "k", "DEEPGRAM_SERVICE_RAILWAY_URL": "u",
            "transcribe_with_voxtral": voxtral, "transcribe_with_assemblyai": assembly, "transcribe_with_deepgram": deepgram,
        })
        out = asyncio.run(ns["_human_stt_chain"]("x.mp3", ("voxtral", "assemblyai", "deepgram", "openai"), "id"))
        self.assertEqual(calls, ["voxtral", "assemblyai", "deepgram"])
        self.assertEqual((out["status"], out["provider"]), ("completed", "deepgram"))
        failed = asyncio.run(ns["_human_stt_chain"]("x.mp3", ("voxtral",), "id"))
        self.assertEqual(failed["status"], "failed")
        self.assertIn("voxtral: boom", failed["error"])

    def test_proofreading_agent_charges_three_credits_after_success_and_checks_first(self):
        self.assertIn("HUMAN_PROOFREADING_AGENT_CREDIT_COST = 3", SRC)
        self.assertIn("HUMAN_WORKER_DRAFT_FORMATTING_CREDITS = 3", SRC)
        start = SRC.index("async def human_proofreading_agent(")
        body = SRC[start:start + 4200]
        self.assertIn('_human_charge_ai_call(actor, job_id, "Proofreading Agent", credit_cost=HUMAN_PROOFREADING_AGENT_CREDIT_COST)', body)
        self.assertLess(body.index("_human_charge_ai_call"), body.index('"transcript": combined'))
        endpoint = SRC[SRC.index("async def human_proofreading_agent_start"):][:1500]
        self.assertIn("_human_require_ai_call_credits(actor, HUMAN_PROOFREADING_AGENT_CREDIT_COST)", endpoint)


class SpellingSectionAuthorityTests(unittest.TestCase):
    def setUp(self):
        import logging
        names = {"_human_ai_agent_research_footer", "_human_research_parse_findings", "_human_research_unverified_terms",
                 "_human_client_word_list_matches", "_human_client_word_list_entries"}
        self.footer = _load(names, {"re": re, "logger": logging.getLogger("t")})["_human_ai_agent_research_footer"]
        self.parts = [{"label": "Part 1", "text": "\tBody one.\n\nClient spellings: Redden; My spellings: Tucker, Ronan, Jason, Becker.\nI searched: Main Street."},
                      {"label": "Part 2", "text": "\tBody two.\n\nMy spellings: Emma, Redden."}]

    def test_agent_spelling_paragraph_is_used_exactly_as_written(self):
        model_text = "\tBody one.  Body two.\n\nClient spellings: None; My spellings: Tucker, Ronan, Jason, Emma."
        out = self.footer(model_text, "", source_parts=self.parts, thorough=True, model_spellings_authoritative=True)
        self.assertIn("Client spellings: None; My spellings: Tucker, Ronan, Jason, Emma", out)
        self.assertNotIn("Becker", out)
        self.assertNotIn("Redden", out)

    def test_old_behaviour_kept_for_other_callers(self):
        model_text = "\tBody.\n\nClient spellings: None; My spellings: Tucker."
        out = self.footer(model_text, "", source_parts=self.parts, thorough=True)
        self.assertIn("Becker", out)
        self.assertIn("Redden", out)

    def test_worker_entries_used_when_agent_wrote_no_spelling_lines(self):
        out = self.footer("\tBody one.  Body two.", "", source_parts=self.parts, thorough=True, model_spellings_authoritative=True)
        self.assertIn("Becker", out)
        self.assertIn("Emma", out)

    def test_client_spelling_moves_when_agent_says_so(self):
        model_text = "\tBody.\n\nClient spellings: Redden; My spellings: Tucker, Emma."
        out = self.footer(model_text, "", source_parts=self.parts, thorough=True, model_spellings_authoritative=True)
        self.assertIn("Client spellings: Redden; My spellings: Tucker, Emma", out)

    def test_worker_search_terms_and_standing_instructions_still_reach_the_agent(self):
        self.assertIn("I researched: Main Street", self.footer("\tBody.\n\nClient spellings: None; My spellings: Tucker.", "", source_parts=self.parts, thorough=True, model_spellings_authoritative=True))
        self.assertIn("only standing proofreading rules", SRC)
        self.assertIn("model_spellings_authoritative=True", SRC[SRC.index("async def _proofreading_agent_audio"):][:9000])
        self.assertIn("model_spellings_authoritative=True", SRC[SRC.index("async def _proofreading_agent_image"):][:7000])


class AgentDeepgramComparisonTests(unittest.TestCase):
    def test_prompt_helpers_label_primary_and_comparison(self):
        ns = _load({"_human_asr_prompt_text", "_human_asr_primary_text"}, {})
        data = {"Voxtral": "primary words", "Deepgram": "other words"}
        text = ns["_human_asr_prompt_text"](data)
        self.assertTrue(text.startswith("VOXTRAL TRANSCRIPT (PRIMARY):"))
        self.assertIn("DEEPGRAM TRANSCRIPT (COMPARISON ONLY", text)
        self.assertEqual(ns["_human_asr_primary_text"](data), "primary words")

    def test_label_follows_the_provider_that_produced_the_transcript(self):
        import asyncio
        ns = _load({"_human_ai_assembly_transcript"}, {"asyncio": asyncio})
        f = ns["_human_ai_assembly_transcript"]
        self.assertEqual(f({"status": "completed", "transcription": "x", "provider": "voxtral"}), {"Voxtral": "x"})
        self.assertEqual(f({"status": "completed", "transcription": "x"}), {"AssemblyAI": "x"})


if __name__ == "__main__":
    unittest.main()
