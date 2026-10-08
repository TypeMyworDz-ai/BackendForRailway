import ast
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _ns():
    ns = {}
    for name in ("_human_is_split_job", "_human_worker_audio_gate", "_human_audio_lock_entries"):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    return ns


NS = _ns()
gate = NS["_human_worker_audio_gate"]
entries = NS["_human_audio_lock_entries"]


class WorkerAudioGate(unittest.TestCase):
    def whole(self, **extra):
        job = {"job_type": "general_job", "status": "assigned", "worker_uid": "w1"}
        job.update(extra)
        return job

    def test_whole_job_is_locked_until_the_draft_exists(self):
        self.assertEqual(gate(self.whole(), "w1"), (True, "main"))
        done = self.whole(ai_drafts={"main": {"worker_uid": "w1", "text": "Hello"}})
        self.assertEqual(gate(done, "w1"), (False, "main"))

    def test_empty_or_foreign_draft_does_not_unlock(self):
        self.assertTrue(gate(self.whole(ai_drafts={"main": {"worker_uid": "w1", "text": "  "}}), "w1")[0])
        self.assertTrue(gate(self.whole(ai_drafts={"main": {"worker_uid": "other", "text": "Hi"}}), "w1")[0])

    def test_admin_unlock_opens_the_recording(self):
        self.assertFalse(gate(self.whole(audio_unlocked={"main": True}), "w1")[0])

    def test_other_people_and_job_types_are_not_gated(self):
        self.assertFalse(gate(self.whole(), "proofreader")[0])
        self.assertFalse(gate(self.whole(job_type="letter_job"), "w1")[0])
        self.assertFalse(gate(self.whole(job_type="pdf_job"), "w1")[0])
        self.assertFalse(gate(self.whole(status="submitted"), "w1")[0])

    def split_job(self):
        return {
            "job_type": "general_job", "split_mode": "dual", "status": "split_in_progress",
            "segments": [
                {"id": "s1", "label": "Part 1", "worker_uid": "w1", "status": "assigned"},
                {"id": "s2", "label": "Part 2", "worker_uid": "w2", "status": "in_progress"},
                {"id": "s3", "label": "Part 3", "worker_uid": "", "status": "pending"},
            ],
            "ai_drafts": {"s1": {"worker_uid": "w1", "text": "Done"}},
        }

    def test_split_parts_are_gated_per_worker(self):
        job = self.split_job()
        self.assertEqual(gate(job, "w1"), (False, "s1"))
        self.assertEqual(gate(job, "w2"), (True, "s2"))
        self.assertEqual(gate(job, "w2", "s1"), (False, ""))
        self.assertEqual(gate(job, "nobody"), (False, ""))

    def test_admin_sees_who_is_still_waiting(self):
        self.assertEqual(entries(self.split_job()), [{"key": "s2", "label": "Part 2"}])
        self.assertEqual(entries(self.whole()), [{"key": "main", "label": "Whole job"}])
        self.assertEqual(entries(self.whole(audio_unlocked={"main": True})), [])


if __name__ == "__main__":
    unittest.main()
