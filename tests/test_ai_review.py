import re
import unittest
from pathlib import Path


def _load():
    src = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
    names = ["_SENTENCE_ABBREVIATIONS", "_review_two_space_style", "_review_normalise_sentence_spacing", "_review_text_to_html", "_review_enforce_indent", "_review_split_output"]
    chunks = []
    for name in names:
        start = src.index(name + " =") if name.startswith("_SENT") else src.index("def " + name)
        nxt = src.index("\n\n\n", start)
        chunks.append(src[start:nxt])
    ns = {"re": re, "_parse_json_object": __import__("json").loads}
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

    def test_html(self):
        self.assertEqual(NS["_review_text_to_html"]("a<b\n\nc"), "<div>a&lt;b</div><div><br></div><div>c</div>")


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

