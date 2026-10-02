import ast
import math
import re
import unittest
from pathlib import Path


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
        self.assertEqual(self.agents["general-gpt"]["models"], ["gpt-5.6-terra", "gpt-5.6-sol"])
        self.assertEqual(self.agents["template-claude"]["models"], ["claude-opus-5-5", "claude-sonnet-5-5"])
        self.assertEqual(self.agents["pdf-gemini"]["models"], ["gemini-3.8-flash"])

    def test_agents_are_internal_not_email_accounts(self):
        for agent in self.agents.values():
            self.assertNotIn("email", agent)
            self.assertNotIn("password", agent)

