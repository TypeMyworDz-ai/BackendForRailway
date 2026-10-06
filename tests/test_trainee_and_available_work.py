"""Production-data-free checks for trainee enrollment and claim-board helpers."""
import ast
import logging
import math
import os
import re
import unittest
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps
from pypdf import PdfWriter


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
            "_human_claim_item_key",
            "_human_job_worker_uids",
            "_human_worker_rating_for_job",
            "_human_worker_rating_summary_from_jobs",
            "_human_worker_earning_items",
            "_pdf_job_jpeg_bytes",
            "_pdf_job_images_from_upload",
            "_as_dt",
            "_int",
            "grant_free_trial",
            "free_trial_correction",
            "read_balance",
            "backfill_credits",
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
                    "TRAINEE_PRICE_USD", "TRAINEE_MIN_WPM", "FREE_TRIAL_CREDITS", "REFILL_DAYS",
                    "HUMAN_AVAILABLE_SLICE_MINUTES", "MIN_HUMAN_WORKER_RATING",
                    "HUMAN_WORKER_MAX_CLAIMS_PER_ITEM",
                    "HUMAN_LEGACY_STANDARD_PAYOUT_KES", "HUMAN_PROOFREADING_PAYOUT_KES",
                    "PDF_JOB_WORKER_PAY_KES", "PDF_JOB_MAX_PAGES_PER_FILE", "PDF_JOB_WORD_EXTENSIONS",
                    "PLAN_CREDITS"
                }:
                    constants[target.id] = ast.literal_eval(node.value)
        cls.namespace = {
            **constants,
            "math": math,
            "hashlib": __import__("hashlib"),
            "os": os,
            "re": re,
            "datetime": datetime,
            "timedelta": timedelta,
            "BytesIO": BytesIO,
            "Image": Image,
            "ImageOps": ImageOps,
            "pdfium": pdfium,
            "doc_tools": __import__("doc_tools"),
            "logger": logging.getLogger(__name__),
            "HTTPException": FakeHTTPException,
            "_human_public_for": lambda data, role, actor_uid: dict(data),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), cls.namespace)
        cls.tree = tree

    def test_prices_trial_allowance_and_worker_threshold_match_product_policy(self):
        self.assertEqual(self.namespace["TRAINEE_PRICE_USD"], 1.0)
        self.assertEqual(self.namespace["FREE_TRIAL_CREDITS"], 30)
        self.assertEqual(self.namespace["HUMAN_AVAILABLE_SLICE_MINUTES"], 5)
        self.assertEqual(self.namespace["MIN_HUMAN_WORKER_RATING"], 3.5)

    def test_every_new_account_gets_exactly_30_credits_and_old_5_grants_are_corrected(self):
        now = datetime(2026, 9, 30, 9, 0, 0)
        grant_trial = self.namespace["grant_free_trial"]
        new_account = grant_trial({}, now)
        self.assertEqual(new_account["planCredits"], 30)
        self.assertTrue(new_account["hasReceivedInitialFreeMinutes"])
        self.assertNotIn("LEGACY_FREE_TRIAL_CREDITS", self.namespace)

        backfill = self.namespace["backfill_credits"]
        for profile in ({}, {"freeTrialVersion": 2}, {"role": "trainee"}):
            updates, detail = backfill(profile, now)
            self.assertEqual(updates["planCredits"], 30, profile)
            self.assertEqual(detail["granted"], 30, profile)

        existing = {"planCredits": 17, "topUpCredits": 23}
        self.assertEqual(grant_trial(existing, now), {})
        updates, detail = backfill(existing, now)
        self.assertEqual(updates, {"creditsBackfilledAt": now})
        self.assertEqual(existing["planCredits"], 17)

        fix = self.namespace["free_trial_correction"]
        reduced = {"planCredits": 5, "planCreditsExpireAt": now + timedelta(days=20), "plan": "free"}
        corrected = fix(reduced, True, now)
        self.assertEqual(corrected["planCredits"], 30)
        self.assertEqual(fix({**reduced, **corrected}, True, now), {})
        self.assertEqual(fix(reduced, False, now), {})
        used = fix({"planCredits": 2, "planCreditsExpireAt": now + timedelta(days=20)}, True, now)
        self.assertEqual(used["planCredits"], 27)

    def test_typing_gate_requires_full_thirty_seconds_and_at_least_forty_wpm(self):
        validate = self.namespace["_validated_trainee_typing_test"]
        self.assertEqual(validate({"correct_chars": 100, "elapsed_ms": 30000})["wpm"], 40.0)
        for value in (
            {"correct_chars": 99, "elapsed_ms": 30000},
            {"correct_chars": 100, "elapsed_ms": 29999},
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
        self.assertEqual(result["claimable_parts"][0]["id"], "part_1")
        self.assertEqual(result["claimable_parts"][0]["claim_attempt_count"], 0)
        self.assertFalse(result["claimable_parts"][0]["can_claim"])
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

    def test_split_worker_ratings_use_job_history_and_ignore_unrelated_workers(self):
        summarize = self.namespace["_human_worker_rating_summary_from_jobs"]
        jobs = [
            {"worker_rating": 5, "segments": [{"worker_uid": "worker-1"}]},
            {"worker_rating": 3, "assigned_worker_uids": ["worker-1", "worker-2"]},
            {"worker_rating": 1, "worker_uid": "worker-3"},
            {"worker_rating": 7, "worker_uid": "worker-1"},
        ]
        self.assertEqual(summarize(jobs, "worker-1"), {"average": 4.0, "count": 2})
        self.assertEqual(summarize(jobs, "worker-2"), {"average": 3.0, "count": 1})
        self.assertEqual(summarize(jobs, "new-worker"), {"average": None, "count": 0})

    def test_per_worker_split_rating_overrides_parent_rating(self):
        summarize = self.namespace["_human_worker_rating_summary_from_jobs"]
        jobs = [{
            "worker_rating": 5,
            "worker_ratings": {"worker-1": 3.5, "worker-2": 4.5},
            "segments": [{"worker_uid": "worker-1"}, {"worker_uid": "worker-2"}],
        }]
        self.assertEqual(summarize(jobs, "worker-1"), {"average": 3.5, "count": 1})
        self.assertEqual(summarize(jobs, "worker-2"), {"average": 4.5, "count": 1})

    def test_claim_transaction_reads_and_writes_the_job_and_worker_lock(self):
        node = next(item for item in self.tree.body if isinstance(item, ast.FunctionDef) and item.name == "_human_claim_assignment_transaction")
        source = ast.unparse(node)
        self.assertIn("firestore.transactional", source)
        self.assertIn("claim_ref.get(transaction=tx)", source)
        self.assertIn("worker_ref.get(transaction=tx)", source)
        self.assertIn("workerApproved", source)
        self.assertIn("HUMAN_SHIFT_COLLECTION", source)
        self.assertIn("_human_shift_status_payload", source)
        self.assertIn("claim_status.get('can_claim')", source)
        self.assertNotIn("is_available", source)
        self.assertIn("tx.update(job_ref", source)
        self.assertIn("tx.set(claim_ref", source)
        self.assertIn("already claimed that part", source)

    def test_worker_claims_require_a_qualifying_rating_and_admin_scope_is_private(self):
        claim_route = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "human_worker_claim")
        claim_source = ast.unparse(claim_route)
        self.assertIn("_human_worker_rating_summary", claim_source)
        self.assertIn("MIN_HUMAN_WORKER_RATING", claim_source)
        list_route = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "human_list_jobs")
        list_source = ast.unparse(list_route)
        self.assertIn('scope == \'admin\' and actor[\'role\'] != \'admin\'', list_source)
        self.assertIn("worker_meets_rating", list_source)

    def test_admin_legacy_route_blocks_new_manual_splits_and_uses_claim_lock(self):
        route = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "human_admin_assign")
        source = ast.unparse(route)
        self.assertIn("supervised_starter", source)
        self.assertIn("_human_worker_rating_summary", source)
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
            if node.name == "_human_check_expiry":
                self.assertIn("_human_reclaim_expired_job", calls, node.name)
            else:
                self.assertIn("_human_release_worker_claim", calls, node.name)
        reclaim = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "_human_reclaim_expired_job")
        reclaim_calls = {item.func.id for item in ast.walk(reclaim) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
        self.assertIn("_human_release_worker_claim", reclaim_calls)
        self.assertIn("_human_deadline_event_id", reclaim_calls)

    def test_pdf_pages_and_images_become_single_normalized_image_jobs(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        source = BytesIO()
        writer.write(source)

        convert = self.namespace["_pdf_job_images_from_upload"]
        pages = convert("client form.pdf", source.getvalue())
        self.assertEqual([page["page_number"] for page in pages], [1, 2])
        self.assertEqual([page["page_count"] for page in pages], [2, 2])
        self.assertEqual([page["name"] for page in pages], ["client_form-page-001.jpg", "client_form-page-002.jpg"])
        for page in pages:
            with Image.open(BytesIO(page["raw"])) as rendered:
                self.assertEqual(rendered.format, "JPEG")
                self.assertEqual(rendered.mode, "RGB")

        image_source = BytesIO()
        Image.new("RGBA", (16, 12), (20, 90, 160, 255)).save(image_source, format="PNG")
        images = convert("photo.png", image_source.getvalue())
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source_filename"], "photo.png")
        self.assertEqual(images[0]["page_number"], 1)
        with Image.open(BytesIO(images[0]["raw"])) as normalized:
            self.assertEqual(normalized.format, "JPEG")
            self.assertEqual(normalized.mode, "RGB")

    def test_pdf_job_earning_is_exactly_100_kes_and_separately_categorized(self):
        earn = self.namespace["_human_worker_earning_items"]
        job = {
            "job_type": "pdf_job",
            "worker_uid": "worker-1",
            "worker_email": "worker@example.test",
            "worker_minutes": 1,
            "workerCompletedAt": datetime(2026, 9, 27, 8, 0),
        }
        items = list(earn("pdf-1", job))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "pdf")
        self.assertEqual(items[0]["gross_amount_kes"], 100)
        self.assertEqual(items[0]["amount_kes"], 100)
        self.assertEqual(self.namespace["PDF_JOB_WORKER_PAY_KES"], 100)

    def test_training_progress_is_automatic_and_admin_promotion_waits_for_all_six(self):
        submit = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "trainee_submit_training")
        submit_source = ast.unparse(submit)
        self.assertIn("expected_checklist", submit_source)
        self.assertIn("TRAINING_QUIZ_ANSWERS[level]", submit_source)
        self.assertIn("level >= 4", submit_source)
        self.assertIn("not transcript", submit_source)
        self.assertIn("pending_final_review", submit_source)
        self.assertIn("next_level", submit_source)

        decision = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "admin_trainee_decision")
        decision_source = ast.unparse(decision)
        self.assertIn("approve_level", decision_source)
        self.assertIn("len(TRAINING_LEVELS) + 1", decision_source)
        self.assertIn("final_transcript", decision_source)
        self.assertIn("if not complete or not final_transcript", decision_source)

    def test_private_materials_and_pdf_images_require_authorized_access(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        image_route = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "human_pdf_job_image")
        image_source = ast.unparse(image_route)
        self.assertIn("_human_assert_access", image_source)
        self.assertIn("Cache-Control", image_source)
        self.assertIn("private, no-store", image_source)
        self.assertIn("not in", image_source)
        self.assertIn("worker", image_source)

        material_route = next(item for item in self.tree.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "trainee_training_material")
        material_source = ast.unparse(material_route)
        self.assertIn("paid_trainee", material_source)
        self.assertIn("TRAINING_ASSET_STORAGE_PREFIX", material_source)
        self.assertIn("_training_assets_bucket()", material_source)
        self.assertNotIn("_human_bucket()", material_source)
        self.assertIn("bucket.blob", material_source)
        self.assertIn("download_as_bytes", material_source)
        self.assertNotIn("open(", material_source)
        self.assertIn("Cache-Control", material_source)
        self.assertIn("private, no-store", material_source)
        training_bucket = next(item for item in self.tree.body if isinstance(item, ast.FunctionDef) and item.name == "_training_assets_bucket")
        training_bucket_source = ast.unparse(training_bucket)
        self.assertIn("TRAINING_ASSET_STORAGE_BUCKET", training_bucket_source)
        self.assertIn("firebase_storage.bucket(configured)", training_bucket_source)
        for filename in (
            "human-job-practical.mp3",
            "transcription-guidelines.docx",
            "formatting-default.docx",
        ):
            self.assertIn(filename, source)

        serializer = next(item for item in self.tree.body if isinstance(item, ast.FunctionDef) and item.name == "_human_public_for")
        serializer_source = ast.unparse(serializer)
        self.assertIn("page_count", serializer_source)
        self.assertNotIn("storage_path", serializer_source)


if __name__ == "__main__":
    unittest.main()
