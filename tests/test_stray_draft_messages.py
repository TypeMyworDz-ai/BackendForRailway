import ast
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "main.py"


def _load():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    names = {"_STRAY_META_BRACKET", "_STRAY_PREFACE", "_STRAY_CLOSING", "_STRAY_REFUSAL", "_human_draft_stray_message"}
    body = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name in names)]
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), "main_subset", "exec"), ns)
    return ns["_human_draft_stray_message"]


class StrayDraftMessageTests(unittest.TestCase):
    def setUp(self):
        self.check = _load()

    def test_flags_machine_chatter(self):
        bad = [
            "Text.\n\n[The answer was cut short because it reached its length limit.  Ask me to continue.]",
            "Sure! Here is the formatted transcript:\n\nHello there.",
            "Here is the proofread draft:\nHello there.",
            "Hello there.\n\nLet me know if you would like me to adjust anything.",
            "I'm sorry, but I can't help with that.",
            "Hello.\n[Note: remaining text omitted]",
            "Hello.\n[Continued in next part]",
            "```\nHello there.\n```",
            "<thinking>hmm</thinking> Hello",
        ]
        for text in bad:
            self.assertTrue(self.check(text), text)

    def test_allows_normal_transcript_content(self):
        good = [
            "\tHello there.  The caller said [inaudible] twice.\n\n\tThank you.",
            "[Please send this to Christian for review.]\n\nDear Sir:",
            "\tShe said, \"Let me know if you need anything,\" and left.",
            "\tPlease let me know if you have any questions.",
            "Sure\tI will be there.",
            "\tHere is what happened at the home.  The worker arrived.",
            "I searched: Summit Psych.\n\nResearch Notes:\n- Summit: a clinic.",
        ]
        for text in good:
            self.assertEqual(self.check(text), "", text)


if __name__ == "__main__":
    unittest.main()
