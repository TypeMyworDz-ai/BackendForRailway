"""Production-data-free checks for trainee enrollment and claim-board helpers."""
import ast
import math
import re
import unittest
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class FakeHTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class TraineeAndAvailableWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        wanted = {
            "_validated_trainee_typing_test",
            "_normalize_mpesa_details",
            "_human_build_available_segments",
            "_human_claim_is_active",
            "_human_available_public_for",
        }
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
        ]
        constants = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "TRAINEE_PRICE_USD", "FREE_TRIAL_CREDITS", "HUMAN_AVAILABLE_SLICE_MINUTES"
                }:
                    constants[target.id] = ast.literal_eval(node.value)
        cls.namespace = {
            **constants,
            "math": math,
            "re": re,
            "HTTPException": FakeHTTPException,
            "_human_public_for": lambda data, role, actor_uid: dict(data),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), cls.namespace)
        cls.tree = tree

    def test_prices_and_trial_allowance_match_the_product_copy(self):
        self.assertEqual(self.namespace["TRAINEE_PRICE_USD"], 1.0)
        self.assertEqual(self.namespace["FREE_TRIAL_CREDITS"], 5)
        self.assertEqual(self.namespace["HUMAN_AVAILABLE_SLICE_MINUTES"], 5)

    def test_typing_gate_requires_full_thirty_seconds_and_at_least_fifty_wpm(self):
        validate = self.namespace["_validated_trainee_typing_test"]
        self.assertEqual(validate({"correct_chars": 125, "elapsed_ms": 30000})["wpm"], 50.0)
        for value in (
            {"correct_chars": 124, "elapsed_ms": 30000},
            {"correct_chars": 125, "elapsed_ms": 29999},
            None,
        ):
            with self.subTest(value=value), self.assertRaises(FakeHTTPException):
                validate(value)

    def test_mpesa_number_is_normalized_without_claiming_external_verification(self):
        normalize = self.namespace["_normalize_mpesa_details"]
        self.assertEqual(normalize("  Amina   Wanjiku  ", "0712 345 678"), ("Amina Wanjiku", "254712345678"))
        self.assertEqual(normalize("Amina Wanjiku", "+254 (712) 345-678"), ("Amina Wanjiku", "254712345678"))
        with self.assertRaises(FakeHTTPException):
            normalize("Amina Wanjiku", "0712 345")

    def test_long_recording_is_split_into_open_balanced_slices(self):
        build = self.namespace["_human_build_available_segments"]
        segments = build({"seconds": 780, "minutes": 13}, None)
        self.assertEqual(len(segments), 3)
        self.assertEqual([item["minutes"] for item in segments], [5, 4, 4])
        self.assertTrue(all(item["status"] == "available" for item in segments))
        self.assertTrue(all(item["worker_uid"] is None and item["deadlineAt"] is None for item in segments))
        self.assertEqual(build({"seconds": 300, "minutes": 5}, None), [])

    def test_available_job_serializer_hides_files_and_drafts_until_claimed(self):
        serialize = self.namespace["_human_available_public_for"]
        result = serialize({
            "audio": {"name": "interview.mp3", "content_type": "audio/mpeg", "size": 800, "storage_path": "private/audio"},
            "instruction_attachments": [{"name": "names.pdf", "storage_path": "private/reference"}],
            "transcript": "private draft", "worker_notes": "private notes",
            "final_attachment": {"storage_path": "private/final"},
            "proofreader_parts": [{"transcript": "private part"}],
            "worker_assignment": {"transcript": "earlier part"},
            "segments": [{"worker_email": "private@example.com", "transcript": "another part"}],
            "assigned_worker_uids": ["worker-2"], "worker_uid": "worker-2",
            "worker_email": "private@example.com", "worker_name": "Private Worker",
            "last_auto_reassigned_worker_name": "Former Worker",
        }, "worker-1", [{"id": "part_1", "label": "Part 1"}], False, False, "Finish your current assignment first.")
        self.assertEqual(result["audio"], {"name": "interview.mp3", "content_type": "audio/mpeg", "size": 800})
        self.assertEqual(result["instruction_attachments"], [])
        self.assertEqual(result["transcript"], "")
        self.assertEqual(result["worker_notes"], "")
        self.assertIsNone(result["final_attachment"])
        self.assertEqual(result["proofreader_parts"], [])
        for private_field in ("worker_assignment", "segments", "assigned_worker_uids", "worker_uid", "worker_email", "worker_name", "last_auto_reassigned_worker_name"):
            self.assertNotIn(private_field, result)
        self.assertEqual(result["claimable_parts"], [{"id": "part_1", "label": "Part 1"}])
        self.assertFalse(result["can_claim"])
        self.assertEqual(result["claim_block_reason"], "Finish your current assignment first.")

    def test_claim_is_only_active_for_the_matching_live_assignment(self):
        is_active = self.namespace["_human_claim_is_active"]
        job = {"segments": [{"id": "part_1", "worker_uid": "worker-1", "status": "assigned"}]}
        claim = {"role": "transcriber", "segment_id": "part_1"}
        self.assertTrue(is_active(job, "worker-1", claim))
        self.assertFalse(is_active(job, "worker-2", claim))
        job["segments"][0]["status"] = "submitted"
        self.assertFalse(is_active(job, "worker-1", claim))

    def test_claim_transaction_reads_and_writes_the_job_and_worker_lock(self):
        node = next(item for item in self.tree.body if isinstance(item, ast.FunctionDef) and item.name == "_human_claim_assignment_transaction")
        source = ast.unparse(node)
        self.assertIn("firestore.transactional", source)
        self.assertIn("claim_ref.get(transaction=tx)", source)
        self.assertIn("worker_ref.get(transaction=tx)", source)
        self.assertIn("workerApproved", source)
        self.assertIn("is_available", source)
        self.assertIn("tx.update(job_ref", source)
        self.assertIn("tx.set(claim_ref", source)
        self.assertIn("already claimed that part", source)

    def test_admin_legacy_route_blocks_new_manual_splits_and_uses_claim_lock(self):
        route = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "human_admin_assign")
        source = ast.unparse(route)
        self.assertIn("New jobs are split automatically", source)
        self.assertIn("_human_claim_assignment_transaction", source)
        self.assertIn("workerApproved", source)

    def test_claim_locks_are_released_after_submit_takeback_and_expiry(self):
        names = {"human_worker_submit", "human_admin_take_back", "_human_check_expiry"}
        nodes = [item for item in self.tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in names]
        self.assertEqual({node.name for node in nodes}, names)
        for node in nodes:
            calls = {
                item.func.id for item in ast.walk(node)
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
            }
            self.assertIn("_human_release_worker_claim", calls, node.name)


if __name__ == "__main__":
    unittest.main()
