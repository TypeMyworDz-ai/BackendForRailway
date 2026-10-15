import ast
import pathlib
import unittest

SRC = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text()
tree = ast.parse(SRC)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_human_conversation_closed")
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), ns)
closed = ns["_human_conversation_closed"]


class ConversationLockTests(unittest.TestCase):
    def test_open_while_work_is_live(self):
        for status in ("assigned", "in_progress", "split_in_progress", "approved"):
            self.assertFalse(closed({"status": status}))

    def test_closed_once_submitted(self):
        for status in ("submitted", "client_review", "client_approved", "released"):
            self.assertTrue(closed({"status": status}))

    def test_stays_open_if_a_part_or_proofreader_is_still_working(self):
        self.assertFalse(closed({"status": "submitted", "segments": [{"status": "in_progress"}]}))
        self.assertFalse(closed({"status": "submitted", "proofreader_status": "assigned"}))
        self.assertTrue(closed({"status": "submitted", "segments": [{"status": "submitted"}]}))


if __name__ == "__main__":
    unittest.main()
