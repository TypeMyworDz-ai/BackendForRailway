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

    def test_split(self):
        text, data = NS["_review_split_output"]('<<<TRANSCRIPT>>>\n\tHello.  World.\n<<<NOTES>>>\n{"summary": "ok"}')
        self.assertEqual(text, "\tHello.  World.")
        self.assertEqual(data["summary"], "ok")

    def test_html(self):
        self.assertEqual(NS["_review_text_to_html"]("a<b\n\nc"), "<div>a&lt;b</div><div><br></div><div>c</div>")


if __name__ == "__main__":
    unittest.main()
