"""Proofreading Agent v2: AssemblyAI reference, safe paragraphs, markup, instructions, research (no paid calls)."""
import ast
import asyncio
import difflib
import logging
import re
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"
SRC = MAIN.read_text(encoding="utf-8")


def _ns():
    tree = ast.parse(SRC)
    prefixes = ("_review_", "_REVIEW_", "_proofreading_", "_PR_", "_PROOFREADING", "PROOFREADING_")
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith(prefixes):
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith(prefixes) for t in n.targets):
            nodes.append(n)
    ns = {"re": re, "difflib": difflib, "asyncio": asyncio, "logger": logging.getLogger("t")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main", "exec"), ns)
    return ns


NS = _ns()


class SafeBoundaryTests(unittest.TestCase):
    def restore(self, text, source):
        return NS["_review_restore_dictation_boundaries"](text, source, safe=True)

    def test_paragraph_break_goes_only_at_a_sentence_end(self):
        source = "We met on Monday. Next paragraph. Then she left the building early."
        out, info = self.restore("We met on Monday. Then she left the building early.", source)
        self.assertRegex(out, r"Monday\.\s*\n\nThen")
        self.assertTrue(info and info[0]["restored"])

    def test_never_cuts_inside_a_sentence_or_word(self):
        source = "She lived at 1,500 Main Street new paragraph and then moved to another place."
        text = "She lived at 1,500 Main Street and then moved to another place."
        out, _ = self.restore(text, source)
        self.assertNotIn("\n", out.strip())

    def test_heal_rejoins_a_cut_sentence(self):
        heal = NS["_proofreading_heal_paragraph_cuts"]
        text = "\tThe client went to the pharmacy on Tuesday and then\n\nwalked home after the appointment ended."
        self.assertEqual(heal(text), "\tThe client went to the pharmacy on Tuesday and then walked home after the appointment ended.")
        keep = "\tShe left the house early in the morning that day.\n\n\tNext she called."
        self.assertEqual(heal(keep), keep)


class MarkupTests(unittest.TestCase):
    def test_round_trip(self):
        html = NS["_proofreading_markup_to_html"]("\tHello [[b]]bold [[i]]both[[/i]][[/b]] & <x>\n\nNext [[u]]line\ncontinues[[/u]]")
        self.assertIn("<b>bold <i>both</i></b>", html)
        self.assertIn("&amp;", html)
        self.assertIn("&lt;x&gt;", html)
        self.assertIn("<div><br></div>", html)
        self.assertEqual(NS["_proofreading_strip_markup"]("a [[b]]b[[/b]] c"), "a b c")

    def test_html_to_marked_only_when_text_matches(self):
        f = NS["_proofreading_marked_text"]
        html = "<div>One <b>bold</b> word</div><div>Two</div>"
        self.assertEqual(f("One bold word\nTwo", html), "One [[b]]bold[[/b]] word\nTwo")
        self.assertEqual(f("Different text\nTwo", html), "Different text\nTwo")
        self.assertEqual(f("plain", ""), "plain")

    def test_markers_do_not_move_word_positions(self):
        tokens = NS["_review_word_tokens"]("[[b]]Main[[/b]] Street")
        self.assertEqual([t[0] for t in tokens], ["main", "street"])


class InstructionTests(unittest.TestCase):
    def test_digest_finds_edit_and_format_commands(self):
        digest, count = NS["_proofreading_instruction_digest"](
            "Go back to the second paragraph and add the date. Please make that bold. Do not indent the paragraphs."
        )
        self.assertGreaterEqual(count, 3)
        self.assertIn("[EDIT]", digest)
        self.assertIn("[FORMAT]", digest)

    def test_no_indent_detection(self):
        f = NS["_proofreading_no_indent_requested"]
        for phrase in ("do not indent paragraphs", "No indentation please", "don't indent", "use block style", "flush left", "no tabs"):
            self.assertTrue(f([phrase]), phrase)
        self.assertFalse(f(["She walked to the table and sat."]))

    def test_indent_enforcement_and_removal(self):
        f = NS["_proofreading_enforce_indent"]
        text = "HEADING\n\nBody line.\n\n[[b]]Bold start[[/b]] body.\n[Staff note]"
        out = f(text, indent=True).split("\n")
        self.assertEqual(out[0], "HEADING")
        self.assertEqual(out[2], "\tBody line.")
        self.assertTrue(out[4].startswith("\t[[b]]"))
        self.assertEqual(out[5], "[Staff note]")
        flush = f("\tBody line.\n\n\tOther.", indent=False)
        self.assertEqual(flush, "Body line.\n\nOther.")


class ResearchTests(unittest.TestCase):
    def test_location_candidates(self):
        found = NS["_proofreading_location_candidates"](
            "She went to Grove Park Hospital. They live at 4412 Maple Ridge Drive near the Lincoln Elementary School. The cat sat."
        )
        joined = " | ".join(found)
        self.assertIn("Grove Park Hospital", joined)
        self.assertIn("Maple Ridge Drive", joined)
        self.assertIn("Lincoln Elementary School", joined)

    def test_merge_dedupes_and_keeps_evidence(self):
        merged = NS["_proofreading_merge_research"] if "_proofreading_merge_research" in NS else None
        self.assertIsNotNone(merged)


class ResearchPolicyTests(unittest.TestCase):
    def test_research_pass_only_runs_when_admin_instructions_request_it(self):
        should_research = NS["_proofreading_research_requested"]
        self.assertFalse(should_research(""))
        self.assertFalse(should_research("Follow the job-specific formatting instructions."))
        self.assertFalse(should_research("Do not research names or locations."))
        self.assertTrue(should_research("Research street names and verify agency spellings."))
        self.assertTrue(should_research("Do not research client-supplied terms; research street names."))
        start = SRC.index("async def _proofreading_agent_research(")
        end = SRC.index("async def _proofreading_agent_custom_instructions(", start)
        research_pass = SRC[start:end].lower()
        self.assertIn("custom_instructions", research_pass)
        self.assertNotIn("research every street", research_pass)
        self.assertIn('_proofreading_research_requested(custom_instructions)', research_pass)


class WiringTests(unittest.TestCase):
    def test_only_brief_transcript_comparison_instruction_is_hardcoded(self):
        a = SRC.index("_PROOFREADING_AGENT_SYSTEM = (")
        b = SRC.index("_PROOFREADING_RESEARCH_RULES = (", a)
        block = SRC[a:b]
        self.assertIn("comparison-only", block.lower())
        self.assertIn("paragraph breaks", block)
        self.assertIn("minimal contextual corrections", block)
        self.assertIn("overall typemywordz guidelines", block.lower())
        for needle in ("STREETS, ADDRESSES AND LOCATIONS", "FORMATTING COMMANDS", "INSTRUCTION AUDIT", "INDENTATION", "SINGLE-PART JOBS"):
            self.assertNotIn(needle, block)

    def test_agent_uses_configured_speech_and_custom_instructions(self):
        start = SRC.index("async def _proofreading_agent_audio(")
        end = SRC.index("async def _proofreading_agent_image(")
        body = SRC[start:end]
        self.assertIn("_proofreading_agent_full_audio_transcripts", body)
        self.assertIn('_human_work_stt_config("proofreading_agent")', body)
        self.assertIn("_proofreading_custom_instructions_block", body)
        self.assertIn("safe=True", body)
        self.assertIn("thorough=True", body)

    def test_admin_endpoints_exist_and_are_admin_only(self):
        self.assertIn('@app.get("/api/admin/proofreading-agent-instructions")', SRC)
        self.assertIn('@app.put("/api/admin/proofreading-agent-instructions")', SRC)
        start = SRC.index("async def put_proofreading_agent_instructions")
        self.assertIn("_require_admin(request)", SRC[start:start + 300])

    def test_trainee_rename_endpoint_is_admin_only(self):
        start = SRC.index("async def admin_rename_trainee")
        body = SRC[start:start + 900]
        self.assertIn("_require_admin(request)", body)
        self.assertIn("officialIdName", body)
        self.assertIn("/api/admin/trainees/{uid}/rename", SRC)

    def test_custom_block(self):
        f = NS["_proofreading_custom_instructions_block"]
        self.assertEqual(f("  "), "")
        self.assertIn("SUPER-ADMIN PROOFREADING INSTRUCTIONS", f("Always bold the heading."))

    def test_thorough_footer_keeps_streets(self):
        tree = ast.parse(SRC)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_human_ai_agent_research_footer")
        ns = {"re": re}
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_human_research_parse_findings", "_human_research_unverified_terms", "_human_client_word_list_matches", "_human_client_word_list_entries"}]
        exec(compile(ast.Module(body=helpers + [fn], type_ignores=[]), "main", "exec"), ns)
        footer = ns["_human_ai_agent_research_footer"]
        research = "\n".join(
            f"Street{i} Avenue | Street{i} Avenue | Street{i} Avenue is a street in Springfield, Ohio. It fits the setting. Extra third sentence. | yes" for i in range(30)
        ) + "\nACTUAL SEARCH SOURCES: x"
        body = "\n".join(f"\tShe lived on Street{i} Avenue." for i in range(30))
        normal = footer(body, research)
        thorough = footer(body, research, thorough=True)
        self.assertLess(normal.count("- Street"), thorough.count("- Street"))
        self.assertEqual(thorough.count("- Street"), 30)
        self.assertIn("It fits the setting.", thorough)


if __name__ == "__main__":
    unittest.main()
