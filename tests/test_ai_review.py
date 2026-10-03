import ast
import asyncio
import math
import re
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt


def _load():
    src = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
    names = ["_SENTENCE_ABBREVIATIONS", "_review_two_space_style", "_review_normalise_sentence_spacing", "_review_text_to_html", "_review_enforce_indent", "_human_review_spelling_notes", "_human_worker_feedback", "_review_split_output"]
    chunks = []
    for name in names:
        start = src.index(name + " =") if name.startswith("_SENT") else src.index("def " + name)
        nxt = src.index("\n\n\n", start)
        chunks.append(src[start:nxt])
    ns = {"re": re, "math": math, "_parse_json_object": __import__("json").loads}
    exec("\n\n".join(chunks), ns)
    return ns


NS = _load()


class ReviewHelpers(unittest.TestCase):
    def test_spacing(self):
        out = NS["_review_normalise_sentence_spacing"]("He left. She stayed. Ms. Lee spoke. J. Smith agreed.")
        self.assertEqual(out, "He left.  She stayed.  Ms. Lee spoke.  J. Smith agreed.")

    def test_two_space_detect(self):
        self.assertTrue(NS["_review_two_space_style"](["One.  Two.  Three."]))
        self.assertFalse(NS["_review_two_space_style"](["One. Two. Three."]))

    def test_indent_enforced(self):
        src = ["\tFirst para.\n\n\tSecond para."]
        out = NS["_review_enforce_indent"]("First para.\n\nHEADING\n\nSecond para.", src)
        self.assertEqual(out, "\tFirst para.\n\n\nHEADING\n\n\tSecond para.".replace("\n\n\nHEADING", "\n\nHEADING"))

    def test_indent_applied_even_when_parts_lost_tabs(self):
        out = NS["_review_enforce_indent"]("First para.\n\nSecond para.\n\nClient spellings: Ann.", ["First para.\n\nSecond para."])
        self.assertEqual(out, "\tFirst para.\n\n\tSecond para.\n\nClient spellings: Ann.")

    def test_split(self):
        text, data = NS["_review_split_output"]('<<<TRANSCRIPT>>>\n\tHello.  World.\n<<<NOTES>>>\n{"summary": "ok"}')
        self.assertEqual(text, "\tHello.  World.")
        self.assertEqual(data["summary"], "ok")

    def test_cross_part_client_spellings_and_worker_research_are_collected(self):
        notes = NS["_human_review_spelling_notes"]([
            {"label": "Part 1", "text": "Transcript body.\nClient spellings: Jon, My spellings: Renee"},
            {"label": "Part 3", "text": "Transcript body.\nClient spellings: John; I searched: Summit Psych\nResearch Notes:\nSummit Psych is the provider named in the recording."},
        ])
        self.assertIn("Part 1: Jon", notes)
        self.assertIn("Part 3: John", notes)
        self.assertIn("searched terms: Summit Psych", notes)
        self.assertIn("provider named in the recording", notes)
        self.assertNotIn("Renee", notes)

    def test_no_worker_spelling_notes_returns_empty_context(self):
        self.assertEqual(NS["_human_review_spelling_notes"]([{"text": "Ordinary transcript text."}]), "")

    def test_worker_sees_only_own_feedback_as_admin(self):
        feedback = NS["_human_worker_feedback"]({
            "worker_uid": "w1",
            "admin_feedback": "Overall comment",
            "worker_rating": 3,
            "segments": [{"id": "part-a", "worker_uid": "w1"}, {"id": "part-b", "worker_uid": "w2"}],
            "part_ratings": {
                "part-a": {"worker_uid": "w1", "rating": 3, "note": "AI review: Verify the organization name.", "source": "ai"},
                "part-b": {"worker_uid": "w2", "rating": 2, "note": "Other worker note", "source": "admin"},
            },
        }, "w1")
        self.assertEqual(feedback, [{
            "label": "Part 1", "rating": 3.0,
            "note": "Verify the organization name.", "rater": "Admin",
        }])
        self.assertNotIn("source", feedback[0])
        self.assertNotIn("AI", feedback[0]["note"])

    def test_legacy_whole_job_feedback_is_included_for_assigned_worker(self):
        feedback = NS["_human_worker_feedback"]({
            "worker_uid": "w1", "worker_rating": 4,
            "admin_feedback": "Clear work; please verify names.",
        }, "w1")
        self.assertEqual(feedback[0]["label"], "Overall job review")
        self.assertEqual(feedback[0]["note"], "Clear work; please verify names.")
        self.assertEqual(feedback[0]["rater"], "Admin")

    def test_html(self):
        self.assertEqual(NS["_review_text_to_html"]("a<b\n\nc"), "<div>a&lt;b</div><div><br></div><div>c</div>")


class ProofreaderLabelSerialization(unittest.TestCase):
    def test_proofreader_parts_use_position_labels_without_ai_attribution(self):
        source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
        serializer = source[source.index("def _human_public_for("):source.index("def _human_available_public_for(")]
        self.assertIn('"author_label": f"Worker {index + 1}"', serializer)
        self.assertNotIn("AI draft:", serializer)


class WholeJobAiTakeover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = Path(__file__).resolve().parents[1].joinpath("main.py")
        cls.source = source_path.read_text()
        tree = ast.parse(cls.source)
        names = {"_human_is_split_job", "_human_whole_job_ai_takeover_allowed"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.helpers = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), cls.helpers)
        cls.routes = {node.name: node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}

    def test_takeover_is_allowed_only_before_any_part_or_proofreader_starts(self):
        allowed = self.helpers["_human_whole_job_ai_takeover_allowed"]
        base = {"split_mode": "dual", "status": "split_assigned", "segments": [
            {"id": "a", "status": "available"}, {"id": "b", "status": "approved"},
        ]}
        self.assertTrue(allowed(base))
        claimed = {**base, "segments": [dict(base["segments"][0]), {"id": "b", "status": "in_progress", "worker_uid": "worker-2"}]}
        self.assertFalse(allowed(claimed))
        submitted = {**base, "segments": [dict(base["segments"][0]), {"id": "b", "status": "submitted"}]}
        self.assertFalse(allowed(submitted))
        proofreader = {**base, "proofreader_status": "assigned"}
        self.assertFalse(allowed(proofreader))
        self.assertFalse(allowed({"split_mode": "single", "segments": base["segments"]}))

    def test_route_stores_paused_parts_and_restores_them_on_failure(self):
        route_source = ast.unparse(self.routes["human_admin_assign_ai_agent"])
        runner_source = ast.unparse(self.routes["_human_run_ai_agent"])
        self.assertIn("_require_admin", route_source)
        self.assertIn("_human_whole_job_ai_takeover_allowed", route_source)
        self.assertIn("ai_agent_paused_segments", route_source)
        self.assertIn("ai_agent_previous_split_mode", route_source)
        self.assertIn("ai_agent_paused_segments", runner_source)
        self.assertIn("restored_segments", runner_source)
        self.assertIn("firestore.DELETE_FIELD", runner_source)

    def test_worker_and_client_serialization_redacts_agent_identity(self):
        serializer = self.source[self.source.index("def _human_public_for("):self.source.index("def _human_available_public_for(")]
        self.assertIn('"ai_agent_name"', serializer)
        self.assertIn('"ai_agent_model_ids"', serializer)
        self.assertIn('"ai_agent_docx"', serializer)
        self.assertIn('"author_label": f"Worker {index + 1}"', serializer)


class TemplateDocxRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = Path(__file__).resolve().parents[1].joinpath("main.py")
        tree = ast.parse(source_path.read_text())
        names = {"_human_template_attachment", "_human_template_docx_profile", "_human_template_line_kind", "_human_template_source_kind", "_human_template_render_docx"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.helpers = {"Document": Document, "BytesIO": BytesIO, "deepcopy": deepcopy, "re": re}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), cls.helpers)

    def test_template_selection_never_guesses_between_word_files(self):
        choose = self.helpers["_human_template_attachment"]
        selected = choose({"instruction_attachments": [{"name": "notes.pdf"}, {"name": "letter.docx", "storage_path": "one"}]})
        self.assertEqual(selected[1]["storage_path"], "one")
        with self.assertRaisesRegex(ValueError, "More than one Word document"):
            choose({"instruction_attachments": [{"name": "one.docx"}, {"name": "two.docx"}]})
        selected = choose({"instruction_attachments": [
            {"name": "one.docx"}, {"name": "two.docx", "purpose": "template"},
        ]})
        self.assertEqual(selected[1]["name"], "two.docx")

    def test_render_keeps_the_attached_template_page_and_paragraph_formatting(self):
        template = Document()
        section = template.sections[0]
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
        template.styles["Normal"].font.name = "Times New Roman"
        template.styles["Normal"].font.size = Pt(12)
        heading = template.add_paragraph()
        heading.alignment = 1
        heading.add_run("EXAMPLE TITLE").bold = True
        template.add_paragraph()
        body = template.add_paragraph()
        body.paragraph_format.line_spacing = 1
        body.paragraph_format.space_before = Pt(0)
        body.paragraph_format.space_after = Pt(0)
        body.add_run("\tExample body placeholder.")
        template.add_paragraph("Client spellings: Example.")
        original = BytesIO()
        template.save(original)

        profile = self.helpers["_human_template_docx_profile"](original.getvalue())
        result_bytes = self.helpers["_human_template_render_docx"](
            original.getvalue(), "EXAMPLE TITLE\n\n\tBody paragraph.\n\nClient spellings: Jon."
        )
        result = Document(BytesIO(result_bytes))
        self.assertIn("Page size: 8.50 x 11.00 inches", profile)
        self.assertEqual([paragraph.text for paragraph in result.paragraphs], ["EXAMPLE TITLE", "", "\tBody paragraph.", "", "Client spellings: Jon."])
        self.assertEqual(result.sections[0].top_margin, section.top_margin)
        self.assertEqual(result.sections[0].page_width, section.page_width)
        self.assertEqual(result.styles["Normal"].font.name, "Times New Roman")
        self.assertEqual(result.styles["Normal"].font.size, Pt(12))
        self.assertEqual(result.paragraphs[0].alignment, 1)
        self.assertTrue(result.paragraphs[0].runs[0].bold)
        self.assertEqual(result.paragraphs[2].paragraph_format.line_spacing, 1)
        self.assertEqual(result.paragraphs[2].paragraph_format.space_after, Pt(0))

    def test_template_indent_is_not_doubled_with_default_transcript_tabs(self):
        template = Document()
        body = template.add_paragraph("Original body paragraph.")
        body.paragraph_format.first_line_indent = Inches(0.5)
        original = BytesIO()
        template.save(original)
        result_bytes = self.helpers["_human_template_render_docx"](original.getvalue(), "\tReplacement body paragraph.")
        result = Document(BytesIO(result_bytes))
        self.assertEqual(result.paragraphs[0].text, "Replacement body paragraph.")
        self.assertEqual(result.paragraphs[0].paragraph_format.first_line_indent, Inches(0.5))

    def test_table_templates_fail_closed_without_creating_a_misleading_docx(self):
        template = Document()
        template.add_table(rows=1, cols=1).cell(0, 0).text = "Name:"
        output = BytesIO()
        template.save(output)
        with self.assertRaisesRegex(ValueError, "uses tables"):
            self.helpers["_human_template_docx_profile"](output.getvalue())
        with self.assertRaisesRegex(ValueError, "uses tables"):
            self.helpers["_human_template_render_docx"](output.getvalue(), "Draft text")

    def test_assignment_uses_the_job_template_and_only_full_admin_can_download(self):
        source_path = Path(__file__).resolve().parents[1].joinpath("main.py")
        routes = {node.name: node for node in ast.parse(source_path.read_text()).body if isinstance(node, ast.AsyncFunctionDef)}
        assignment = ast.unparse(routes["human_admin_assign_ai_agent"])
        runner = ast.unparse(routes["_human_run_ai_agent"])
        download = ast.unparse(routes["human_admin_ai_agent_template_docx"])
        self.assertIn("_human_template_docx_bytes", assignment)
        self.assertIn("_human_template_docx_profile", assignment)
        self.assertIn("_human_template_render_docx", runner)
        self.assertIn("_human_store_raw_bytes", runner)
        self.assertIn("_require_admin", download)
        self.assertIn("human-workflow/{job_id}/ai-drafts/", download)


class AiAgentAuthorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
        cls.functions = {
            node.name: node for node in ast.parse(source).body
            if isinstance(node, ast.AsyncFunctionDef)
        }

    def test_catalog_and_assignment_require_full_admin(self):
        for name in ("human_admin_ai_agents", "human_admin_assign_ai_agent"):
            with self.subTest(route=name):
                calls = {
                    node.func.id for node in ast.walk(self.functions[name])
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                self.assertIn("_require_admin", calls)
                self.assertNotIn("_require_human_job_admin", calls)


if __name__ == "__main__":
    unittest.main()

class AiAgentCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
        tree = __import__("ast").parse(source)
        node = next(n for n in tree.body if isinstance(n, __import__("ast").Assign) and any(getattr(t, "id", None) == "HUMAN_AI_AGENTS" for t in n.targets))
        cls.agents = __import__("ast").literal_eval(node.value)

    def test_three_agents_have_requested_model_pairs(self):
        self.assertEqual(set(self.agents), {"general-gpt", "template-claude", "pdf-gemini"})
        expected_audio_models = ["claude-opus-5-5", "gpt-5.6-sol"]
        self.assertEqual(self.agents["general-gpt"]["models"], expected_audio_models)
        self.assertEqual(self.agents["template-claude"]["models"], expected_audio_models)
        self.assertEqual(self.agents["pdf-gemini"]["models"], ["gemini-3.8-flash"])

    def test_agents_are_internal_not_email_accounts(self):
        for agent in self.agents.values():
            self.assertNotIn("email", agent)
            self.assertNotIn("password", agent)


class HealthEndpointPrivacy(unittest.TestCase):
    def test_public_health_endpoint_returns_status_only(self):
        source_path = Path(__file__).resolve().parents[1].joinpath("main.py")
        tree = ast.parse(source_path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check")
        function.decorator_list = []
        isolated = ast.Module(body=[function], type_ignores=[])
        namespace = {}
        exec(compile(isolated, str(source_path), "exec"), namespace)
        self.assertEqual(asyncio.run(namespace["health_check"]()), {"status": "healthy"})


class AiModelRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = Path(__file__).resolve().parents[1].joinpath("main.py")
        cls.source = source_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        cls.assignments = {
            target.id: ast.literal_eval(node.value)
            for node in cls.tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
            and target.id in {"AI_REVIEW_MODEL_CHAIN", "HUMAN_AUDIO_AGENT_MODEL_CHAIN", "WORKER_DRAFT_FORMAT_MODEL_CHAIN"}
        }

    def test_requested_model_chains_are_primary_then_fallback(self):
        self.assertEqual(self.assignments["AI_REVIEW_MODEL_CHAIN"], (("claude-opus-5-5", "claude"), ("gpt-5.6-sol", "openai")))
        self.assertEqual(self.assignments["HUMAN_AUDIO_AGENT_MODEL_CHAIN"], (("claude-opus-5-5", "claude"), ("gpt-5.6-sol", "openai")))
        self.assertEqual(self.assignments["WORKER_DRAFT_FORMAT_MODEL_CHAIN"], (("gpt-5.6-sol", "openai"), ("gemini-3.8-flash", "gemini")))

    def test_audio_agents_compare_both_transcripts_and_fail_if_either_is_missing(self):
        transcriber = self.functions["_human_ai_transcribe_audio"]
        calls = {node.func.id for node in ast.walk(transcriber) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"transcribe_with_assemblyai", "transcribe_with_deepgram", "_human_ai_pair_asr_transcripts"}.issubset(calls))
        self.assertTrue(any(
            isinstance(node.func, ast.Attribute) and node.func.attr == "gather"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "asyncio"
            for node in ast.walk(transcriber) if isinstance(node, ast.Call)
        ))
        agent_source = "".join(node.value for node in ast.walk(self.functions["_human_ai_agent_generate"]) if isinstance(node, ast.Constant) and isinstance(node.value, str))
        self.assertIn("Compare the independent AssemblyAI and Deepgram transcripts", agent_source)
        self.assertIn("Compare both source transcripts", agent_source)

        helper = self.functions["_human_ai_pair_asr_transcripts"]
        namespace = {"asyncio": asyncio}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "main.py", "exec"), namespace)
        pair = namespace["_human_ai_pair_asr_transcripts"]
        result = pair({"status": "completed", "transcription": "Assembly words."}, {"status": "completed", "transcript": "Deepgram words."})
        self.assertEqual(result, {"AssemblyAI": "Assembly words.", "Deepgram": "Deepgram words."})
        with self.assertRaisesRegex(RuntimeError, "Both AssemblyAI and Deepgram"):
            pair({"status": "completed", "transcription": "Assembly words."}, {"status": "failed"})

    def test_ai_review_and_both_agent_passes_use_fallback_chain(self):
        review_calls = [node for node in ast.walk(self.functions["human_admin_ai_review"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_human_call_model_chain"]
        self.assertEqual(len(review_calls), 1)
        self.assertTrue(any(isinstance(arg, ast.Name) and arg.id == "AI_REVIEW_MODEL_CHAIN" for arg in review_calls[0].args))
        self.assertTrue(any(keyword.arg == "response_validator" and isinstance(keyword.value, ast.Name) and keyword.value.id == "_review_validate_output" for keyword in review_calls[0].keywords))
        self.assertNotIn("resolve_ask_model", {node.func.id for node in ast.walk(self.functions["human_admin_ai_review"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)})
        agent_calls = [node for node in ast.walk(self.functions["_human_ai_agent_generate"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_human_call_model_chain"]
        self.assertEqual(len(agent_calls), 2)
        self.assertTrue(all(any(isinstance(arg, ast.Name) and arg.id == "HUMAN_AUDIO_AGENT_MODEL_CHAIN" for arg in call.args) for call in agent_calls))

    def test_worker_draft_uses_guidelines_context_and_fallback_before_charging(self):
        formatter_calls = {node.func.id for node in ast.walk(self.functions["_human_worker_format_ai_draft"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"_admin_guidelines_text", "_human_review_context", "_human_ai_agent_research", "_human_call_model_chain", "_human_worker_ai_draft_system", "_review_normalise_sentence_spacing", "_review_enforce_indent"}.issubset(formatter_calls))
        call_lines = {node.func.id: node.lineno for node in ast.walk(self.functions["_human_worker_format_ai_draft"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertLess(call_lines["_human_ai_agent_research"], call_lines["_human_call_model_chain"])
        model_call = next(node for node in ast.walk(self.functions["_human_worker_format_ai_draft"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_human_call_model_chain")
        self.assertTrue(any(keyword.arg == "response_validator" for keyword in model_call.keywords))
        route = self.functions["human_worker_ai_draft"]
        call_lines = {node.func.id: node.lineno for node in ast.walk(route) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertLess(call_lines["_human_worker_format_ai_draft"], call_lines["charge_credits"])
        self.assertIn('"format_version": 4', self.source)

    def test_worker_draft_prompt_requires_research_notes_and_handles_clear_speaker_corrections(self):
        function = self.functions["_human_worker_ai_draft_system"]
        prompt_text = "".join(node.value for node in ast.walk(function) if isinstance(node, ast.Constant) and isinstance(node.value, str))
        self.assertIn("Research Notes:", prompt_text)
        self.assertIn("I searched:", prompt_text)
        self.assertIn("Dublin Granville-East Dublin Granville Children's Close To Home.", prompt_text)
        self.assertIn("East Dublin Granville Children's Close To Home.", prompt_text)
        self.assertIn("Do not remove ordinary repetition", prompt_text)

    def test_research_notes_are_suppressed_without_verified_google_search_metadata(self):
        class QuietLogger:
            def warning(self, *args, **kwargs):
                pass

        class Response:
            status_code = 200
            text = ""

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        class FakeRequests:
            response = Response({"candidates": [{"content": {"parts": [{"text": "A likely result."}]}}]})

            @classmethod
            def post(cls, *args, **kwargs):
                return cls.response

        function = self.functions["_gemini_research_blocking"]
        namespace = {"GEMINI_API_KEY": "test-key", "requests": FakeRequests, "logger": QuietLogger()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "main.py", "exec"), namespace)
        self.assertEqual(namespace["_gemini_research_blocking"]("prompt"), "NO_SEARCHED_TERMS")

        FakeRequests.response = Response({"candidates": [{
            "content": {"parts": [{"text": "Summit Psych | Summit Psych | provider | yes"}]},
            "groundingMetadata": {
                "webSearchQueries": ["Summit Psych Ohio provider"],
                "groundingChunks": [{"web": {"title": "Summit Psych", "uri": "https://example.test"}}],
            },
        }]})
        grounded = namespace["_gemini_research_blocking"]("prompt")
        self.assertIn("ACTUAL GOOGLE SEARCH QUERIES:\n- Summit Psych Ohio provider", grounded)
        self.assertIn("ACTUAL SEARCH SOURCES:\n- Summit Psych: https://example.test", grounded)

    def test_fallback_runs_only_after_failure_or_unusable_response(self):
        class QuietLogger:
            def warning(self, *args, **kwargs):
                pass

        function = self.functions["_human_call_model_chain"]
        namespace = {"asyncio": asyncio, "logger": QuietLogger()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "main.py", "exec"), namespace)
        calls = []

        def fake_run(model_id, provider, system_prompt, question, images, max_tokens):
            calls.append((model_id, provider))
            return "" if model_id == "primary" else "formatted transcript"

        namespace["_run_ask_model_with_images"] = fake_run
        answer, used = asyncio.run(namespace["_human_call_model_chain"](
            (("primary", "claude"), ("backup", "openai")), "system", "question"
        ))
        self.assertEqual(answer, "formatted transcript")
        self.assertEqual(used, "backup")
        self.assertEqual(calls, [("primary", "claude"), ("backup", "openai")])

        calls.clear()
        namespace["_run_ask_model_with_images"] = lambda model_id, *args: calls.append(model_id) or ("malformed" if model_id == "primary" else "valid transcript")

        def validate_transcript(answer):
            if answer != "valid transcript":
                raise ValueError("invalid transcript response")

        answer, used = asyncio.run(namespace["_human_call_model_chain"](
            (("primary", "claude"), ("backup", "openai")), "system", "question",
            response_validator=validate_transcript,
        ))
        self.assertEqual((answer, used), ("valid transcript", "backup"))
        self.assertEqual(calls, ["primary", "backup"])

        calls.clear()
        namespace["_run_ask_model_with_images"] = lambda *args: calls.append(args[0]) or "primary response"
        answer, used = asyncio.run(namespace["_human_call_model_chain"](
            (("primary", "claude"), ("backup", "openai")), "system", "question"
        ))
        self.assertEqual((answer, used), ("primary response", "primary"))
        self.assertEqual(calls, ["primary"])

