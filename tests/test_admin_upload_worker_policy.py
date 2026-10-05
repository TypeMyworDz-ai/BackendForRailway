"""Synthetic-only regression checks for admin Human Work uploads and safeguards."""
import ast
import hashlib
import unittest
from datetime import datetime
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class _FakeFirestore:
    SERVER_TIMESTAMP = "SERVER_TIMESTAMP"


class AdminUploadWorkerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        cls.namespace = {
            "hashlib": hashlib,
            "datetime": datetime,
            "firestore": _FakeFirestore,
            "HUMAN_WORKER_DEADLINE_LOCKOUT_RETURNS": 11,
            "TRAINING_LEVELS": [{"level": level} for level in range(1, 7)],
        }
        wanted = [
            "_human_claim_item_key", "_human_claim_attempt_document_id",
            "_human_deadline_event_id", "_human_worker_retraining_updates",
        ]
        exec(compile(ast.Module(body=[cls.functions[name] for name in wanted], type_ignores=[]), str(MAIN_PATH), "exec"), cls.namespace)

    def test_item_claim_keys_are_scoped_to_worker_cycle_and_job_part(self):
        key = self.namespace["_human_claim_item_key"]
        doc_id = self.namespace["_human_claim_attempt_document_id"]
        item = key("job-1", "part_1")
        self.assertEqual(item, "job-1|part_1")
        self.assertNotEqual(item, key("job-1", "part_2"))
        self.assertNotEqual(doc_id("worker", "cycle-1", item), doc_id("worker", "cycle-2", item))
        self.assertEqual(doc_id("worker", "cycle-1", item), doc_id("worker", "cycle-1", item))

    def test_deadline_event_id_deduplicates_same_assignment_but_separates_parts(self):
        event_id = self.namespace["_human_deadline_event_id"]
        first = event_id("job-1", "transcriber", "part_1", "2026-10-05T12:00:00", "cycle-a")
        self.assertEqual(first, event_id("job-1", "transcriber", "part_1", "2026-10-05T12:00:00", "cycle-a"))
        self.assertNotEqual(first, event_id("job-1", "transcriber", "part_2", "2026-10-05T12:00:00", "cycle-a"))
        self.assertNotEqual(first, event_id("job-1", "transcriber", "part_1", "2026-10-05T12:00:00", "cycle-b"))

    def test_deadline_warning_threshold_does_not_lock_out_at_ten(self):
        lockout = self.namespace["_human_worker_retraining_updates"]
        self.assertEqual(lockout({"workerApproved": True}, 10, datetime(2026, 10, 5)), {})

    def test_eleventh_deadline_return_routes_worker_to_existing_paid_training(self):
        updates = self.namespace["_human_worker_retraining_updates"](
            {"workerApproved": True, "trainingPaymentStatus": "paid"}, 11, datetime(2026, 10, 5)
        )
        self.assertEqual(updates["role"], "trainee")
        self.assertIs(updates["workerApproved"], False)
        self.assertIs(updates["trainingRoomAccess"], True)
        self.assertEqual(updates["trainingLevel"], 1)
        self.assertEqual(updates["trainingRedoLevels"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(value == "redo_requested" for value in updates["trainingSubmissions"].values()))
        self.assertNotIn("trainingPaymentStatus", updates)
        self.assertTrue(updates["worker_retraining_required"])

    def test_admin_audio_upload_is_internal_and_directly_available(self):
        route = self.functions["human_admin_create_audio_job"]
        text = ast.unparse(route)
        for expected in (
            "admin_uploaded", "status", "approved", "quote_credits", "worker_amount_kes",
            "_human_build_available_segments", "_notify_available_workers",
            "GENERAL_JOB_DEFAULT_INSTRUCTION", "TEMPLATE_JOB_DEFAULT_INSTRUCTION",
            "A Template Job requires exactly one job-specific .docx template", "template_file",
            "is_template", "max_reference_count",
        ):
            self.assertIn(expected, text)

    def test_pdf_and_text_admin_uploads_are_claimable_but_old_pdf_and_letters_are_not(self):
        create_pdf = ast.unparse(self.functions["human_admin_create_pdf_jobs"])
        claim = ast.unparse(self.functions["_human_claim_assignment_transaction"])
        list_jobs = ast.unparse(self.functions["human_list_jobs"])
        self.assertIn("admin_uploaded", create_pdf)
        self.assertIn("job_type == 'letter_job'", claim)
        self.assertIn("job_type == 'pdf_job' and job.get('admin_uploaded') is not True", claim)
        self.assertIn("item_type == 'letter_job'", list_jobs)
        self.assertIn("item.get('admin_uploaded') is not True", list_jobs)

    def test_admin_uploads_bypass_client_approval_without_changing_client_queue(self):
        create_client = ast.unparse(self.functions["human_create_job"])
        create_audio = ast.unparse(self.functions["human_admin_create_audio_job"])
        self.assertIn("'pending_admin'", create_client)
        self.assertIn("'admin_uploaded': True", create_audio)
        self.assertIn("'quote_credits': 0", create_audio)

    def test_general_template_internal_review_requires_ai_review_to_be_applied(self):
        review = ast.unparse(self.functions["human_admin_review"])
        self.assertIn("'general_job', 'template_job'", review)
        self.assertIn("ai_review_applied", review)
        self.assertIn("Run the AI review", review)
        self.assertIn("Apply the AI-reviewed transcript", review)


if __name__ == "__main__":
    unittest.main()
