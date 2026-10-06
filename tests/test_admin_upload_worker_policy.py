"""Synthetic-only regression checks for admin Human Work uploads and safeguards."""
import ast
import asyncio
import hashlib
import re
import unittest
from io import BytesIO
from datetime import datetime, time as datetime_time, timedelta, timezone
from PIL import Image
import pypdfium2 as pdfium
from pathlib import Path
from zoneinfo import ZoneInfo


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class _FakeFirestore:
    SERVER_TIMESTAMP = "SERVER_TIMESTAMP"


class _FakeLogger:
    def info(self, *args, **kwargs):
        pass


class AdminUploadWorkerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        timezone_nairobi = ZoneInfo("Africa/Nairobi")
        cls.namespace = {
            "hashlib": hashlib,
            "re": re,
            "logger": _FakeLogger(),
            "datetime": datetime,
            "datetime_time": datetime_time,
            "timedelta": timedelta,
            "timezone": timezone,
            "ZoneInfo": ZoneInfo,
            "HUMAN_SHIFT_TIMEZONE": timezone_nairobi,
            "HUMAN_SHIFT_START": datetime_time(15, 0),
            "HUMAN_SHIFT_END": datetime_time(20, 0),
            "HUMAN_SHIFT_WARNING_MISSES": 5,
            "HUMAN_SHIFT_ONLINE_TTL_SECONDS": 150,
            "HUMAN_SHIFT_CALL_TTL_HOURS": 4,
            "firestore": _FakeFirestore,
            "HUMAN_WORKER_DEADLINE_LOCKOUT_RETURNS": 11,
            "TRAINING_LEVELS": [{"level": level} for level in range(1, 7)],
        }
        wanted = [
            "_human_claim_item_key", "_human_claim_attempt_document_id",
            "_human_deadline_event_id", "_human_worker_retraining_updates",
            "_human_shift_parse_datetime", "_human_shift_is_scheduled",
            "_human_shift_call_in_active", "_human_shift_status_payload",
            "_human_ai_draft_finish_eligible", "_human_collapse_duplicate_image_page_blocks",
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

    def test_shift_window_uses_nairobi_time_and_excludes_8pm(self):
        scheduled = self.namespace["_human_shift_is_scheduled"]
        zone = self.namespace["HUMAN_SHIFT_TIMEZONE"]
        self.assertFalse(scheduled(datetime(2026, 10, 5, 14, 59, tzinfo=zone)))
        self.assertTrue(scheduled(datetime(2026, 10, 5, 15, 0, tzinfo=zone)))
        self.assertTrue(scheduled(datetime(2026, 10, 5, 19, 59, tzinfo=zone)))
        self.assertFalse(scheduled(datetime(2026, 10, 5, 20, 0, tzinfo=zone)))
        self.assertFalse(scheduled(datetime(2026, 10, 10, 16, 0, tzinfo=zone)))
        self.assertFalse(scheduled(datetime(2026, 10, 11, 16, 0, tzinfo=zone)))

    def test_off_shift_self_claim_requires_explicit_opt_in_and_fresh_workroom_presence(self):
        status_payload = self.namespace["_human_shift_status_payload"]
        zone = self.namespace["HUMAN_SHIFT_TIMEZONE"]
        now = datetime(2026, 10, 5, 20, 15, tzinfo=zone)
        recent = {"lastPresenceAt": "2026-10-05T20:14:00+03:00"}
        opted_in = status_payload("worker-1", {"workerApproved": True, "is_available": True}, recent, now)
        self.assertTrue(opted_in["can_claim"])
        self.assertTrue(opted_in["off_shift_self_claim"])
        self.assertTrue(opted_in["workroom_online"])
        self.assertFalse(opted_in["clocked_in"])
        self.assertFalse(opted_in["online"], "off-shift presence must not count as attendance or admin overtime eligibility")

        opted_out = status_payload("worker-1", {"workerApproved": True, "is_available": False}, recent, now)
        self.assertFalse(opted_out["can_claim"])
        self.assertFalse(opted_out["off_shift_self_claim"])
        missing_preference = status_payload("worker-1", {"workerApproved": True}, recent, now)
        self.assertFalse(missing_preference["can_claim"], "unset availability is not opt-in")
        stale = status_payload("worker-1", {"workerApproved": True, "is_available": True}, {"lastPresenceAt": "2026-10-05T20:10:00+03:00"}, now)
        self.assertFalse(stale["can_claim"])
        called_in = status_payload(
            "worker-1", {"workerApproved": True, "is_available": True, "humanShiftCallInExpiresAt": "2026-10-06T00:00:00+03:00"}, recent, now,
        )
        self.assertFalse(called_in["off_shift_self_claim"], "admin call-ins use clock-in, not the self-claim path")

    def test_scheduled_shift_claim_does_not_require_availability_toggle(self):
        status_payload = self.namespace["_human_shift_status_payload"]
        zone = self.namespace["HUMAN_SHIFT_TIMEZONE"]
        now = datetime(2026, 10, 5, 18, 0, tzinfo=zone)
        result = status_payload("worker-1", {"workerApproved": True, "is_available": False}, {"clockedInAt": "2026-10-05T15:00:00+03:00"}, now)
        self.assertTrue(result["scheduled_now"])
        self.assertFalse(result["availability_required"])
        self.assertTrue(result["can_claim"])
        self.assertFalse(result["off_shift_self_claim"])

    def test_off_shift_presence_refresh_does_not_record_shift_attendance(self):
        presence = ast.unparse(self.functions["human_worker_shift_presence"])
        self.assertIn("off_shift_available", presence)
        self.assertIn("lastPresenceAt", presence)
        self.assertNotIn('updates["clockedInAt"]', presence)
        self.assertNotIn('updates["clockedOutAt"]', presence)

    def test_admin_call_in_allows_clocked_in_worker_to_claim_after_shift(self):
        status_payload = self.namespace["_human_shift_status_payload"]
        zone = self.namespace["HUMAN_SHIFT_TIMEZONE"]
        now = datetime(2026, 10, 5, 20, 15, tzinfo=zone)
        profile = {"workerApproved": True, "humanShiftCallInExpiresAt": "2026-10-06T00:00:00+03:00"}
        shift = {"clockedInAt": "2026-10-05T20:05:00+03:00", "lastPresenceAt": "2026-10-05T20:14:00+03:00"}
        called_in = status_payload("worker-1", profile, shift, now)
        self.assertTrue(called_in["can_claim"])
        self.assertTrue(called_in["call_in_active"])
        shift["clockedOutAt"] = "2026-10-05T20:10:00+03:00"
        clocked_out = status_payload("worker-1", profile, shift, now)
        self.assertFalse(clocked_out["can_claim"])

    def test_assigned_work_can_start_and_submit_after_shift_while_clocked_in(self):
        start = ast.unparse(self.functions["human_worker_start"])
        submit = ast.unparse(self.functions["human_worker_submit"])
        self.assertGreaterEqual(start.count("_human_shift_assert_can_start_assigned"), 3)
        self.assertGreaterEqual(submit.count("_human_shift_assert_can_start_assigned"), 4)
        self.assertNotIn("_human_shift_assert_can_claim", start)
        self.assertNotIn("_human_shift_assert_can_claim", submit)
        self.assertIn("Finish your active job before clocking out.", ast.unparse(self.functions["human_worker_shift_clock_out"]))

    def test_overtime_admin_assignment_routes_check_live_worker_presence(self):
        for name in ("human_admin_assign", "human_admin_assign_whole", "human_admin_assign_proofreader"):
            with self.subTest(route=name):
                source = ast.unparse(self.functions[name])
                self.assertIn("_human_require_worker_online_for_overtime", source)

    def test_main_admin_only_shift_attendance_and_call_in_routes(self):
        for name in ("human_admin_shift_attendance", "human_admin_shift_call_in"):
            with self.subTest(route=name):
                self.assertIn("_require_human_shift_admin", ast.unparse(self.functions[name]))
        self.assertIn('HUMAN_SHIFT_ADMIN_EMAIL = "typemywordz@gmail.com"', self.source)

    def test_whole_file_review_collapses_exact_duplicate_pages_only_once(self):
        collapse = self.namespace["_human_collapse_duplicate_image_page_blocks"]
        pages = ["Hello there.\nHow can I help?", "Hello there.\nHow can I help?"]
        answer = "Hello there.\nHow can I help?\n\nHello there.\nHow can I help?"
        self.assertEqual(collapse(answer, ["same-image", "same-image"], pages), "Hello there.\nHow can I help?")
        prompt = ast.unparse(self.functions["_human_image_review_draft"])
        self.assertIn("adjacent overlapping screenshots must appear exactly once", prompt)
        self.assertIn("do not remove a repeated message that is visibly repeated within the original conversation itself", prompt)

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

    def test_internal_ai_finish_requires_all_parts_and_no_proofreader(self):
        eligible = self.namespace["_human_ai_draft_finish_eligible"]
        base = {
            "admin_uploaded": True, "ai_agent_status": "submitted", "status": "proofreading_available",
            "segments": [
                {"id": "part-1", "status": "submitted", "transcript": "First completed section."},
                {"id": "part-2", "status": "submitted", "transcript": "Second completed section."},
            ],
        }
        self.assertTrue(eligible(base))
        self.assertFalse(eligible({**base, "segments": [*base["segments"], {"id": "part-3", "status": "in_progress", "transcript": "unfinished"}]}))
        self.assertFalse(eligible({**base, "proofreader_status": "assigned"}))
        self.assertFalse(eligible({**base, "proofreader_status": "in_progress"}))
        self.assertFalse(eligible({**base, "proofreader_status": "submitted"}))
        self.assertFalse(eligible({**base, "reviewer_choice": "human"}))
        self.assertFalse(eligible({**base, "admin_uploaded": False}))
        self.assertFalse(eligible({**base, "status": "in_progress"}))
        self.assertFalse(eligible({**base, "segments": [{"id": "part-1", "status": "submitted", "transcript": "  "}]}))
        self.assertTrue(eligible({
            "admin_uploaded": True, "ai_agent_status": "submitted", "status": "submitted",
            "job_type": "pdf_job", "pdf_review": {"page_count": 2}, "transcript": "Complete PDF draft.",
        }))

    def test_internal_ai_finish_only_marks_finished_without_client_side_effects(self):
        route = ast.unparse(self.functions["human_admin_finish_ai_agent_draft"])
        self.assertIn("_human_ai_draft_finish_eligible", route)
        self.assertIn("'status': 'released'", route)
        self.assertIn("'client_charged': False", route)
        self.assertIn("'client_notified': False", route)
        self.assertNotIn("charge_credits", route)
        self.assertNotIn("_create_user_notification", route)

    def test_audio_jobs_require_a_completed_ai_or_human_review_before_admin_approval(self):
        review = ast.unparse(self.functions["human_admin_review"])
        self.assertIn("not is_pdf_job", review)
        self.assertIn("not is_letter_job", review)
        self.assertIn("reviewer_choice == 'human'", review)
        self.assertIn("proofreader_status", review)
        self.assertIn("reviewer_choice == 'ai'", review)
        self.assertIn("ai_review_applied", review)
        self.assertIn("Run the AI reviewer", review)
        self.assertIn("Apply the AI-reviewed transcript", review)
        self.assertIn("Choose an AI reviewer or assign a human reviewer", review)


class ShiftAuthorizationRules(unittest.TestCase):
    @staticmethod
    def _functions():
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def test_main_admin_guard_and_after_shift_assigned_start_gate(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        shift = {"clockedInAt": "2026-10-05T15:00:00+03:00"}
        async def reconcile(_uid, profile):
            return profile
        async def current_record(_uid):
            return None, shift
        def status_payload(*_args):
            return {
                "online": not bool(shift.get("clockedOutAt")), "scheduled_now": False,
                "clocked_in": bool(shift.get("clockedInAt") and not shift.get("clockedOutAt")),
            }

        namespace = {
            "HTTPException": RequestError,
            "Request": object,
            "HUMAN_SHIFT_ADMIN_EMAIL": "typemywordz@gmail.com",
            "_verified_user": lambda request: request,
            "_human_shift_reconcile": reconcile,
            "_human_shift_current_record": current_record,
            "_human_shift_status_payload": status_payload,
            "_human_shift_local_now": lambda: datetime(2026, 10, 5, 21, 0, tzinfo=ZoneInfo("Africa/Nairobi")),
            "_human_shift_is_scheduled": lambda _now=None: False,
            "_load_profile": lambda _uid: None,
        }
        exec(compile(ast.Module(body=[functions["_require_human_shift_admin"], functions["_human_shift_assert_can_start_assigned"], functions["_human_require_worker_online_for_overtime"]], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        self.assertEqual(namespace["_require_human_shift_admin"]({"email": "typemywordz@gmail.com"}), {"email": "typemywordz@gmail.com"})
        with self.assertRaises(RequestError):
            namespace["_require_human_shift_admin"]({"email": "info@typemywordz.ai"})

        actor = {"uid": "worker-1", "profile": {"workerApproved": True}}
        result = asyncio.run(namespace["_human_shift_assert_can_start_assigned"](actor))
        self.assertFalse(result["scheduled_now"])
        shift["clockedOutAt"] = "2026-10-05T20:30:00+03:00"
        with self.assertRaisesRegex(RequestError, "Clock in before starting"):
            asyncio.run(namespace["_human_shift_assert_can_start_assigned"](actor))
        permitted_claim = asyncio.run(namespace["_human_shift_assert_can_start_assigned"](
            actor, allow_after_hours_self_claim=True,
        ))
        self.assertFalse(permitted_claim["clocked_in"])

    def test_accepted_after_hours_claim_marker_is_used_for_start_and_submit(self):
        start = ast.unparse(self._functions()["human_worker_start"])
        submit = ast.unparse(self._functions()["human_worker_submit"])
        claim = ast.unparse(self._functions()["human_worker_claim"])
        self.assertIn("allow_after_hours_self_claim=target.get('after_hours_self_claim') is True", start)
        self.assertIn("allow_after_hours_self_claim=job.get('after_hours_self_claim') is True", start)
        self.assertIn("allow_after_hours_self_claim=target.get('after_hours_self_claim') is True", submit)
        self.assertIn("allow_after_hours_self_claim=job.get('after_hours_self_claim') is True", submit)
        self.assertIn("require_availability=claim_window['availability_required']", claim)
        self.assertIn("after_hours_self_claim=claim_window['off_shift_self_claim']", claim)

    def test_direct_overtime_assignment_does_not_depend_on_self_claim_toggle(self):
        for name in ("human_admin_assign", "human_admin_assign_whole", "human_admin_assign_proofreader"):
            with self.subTest(route=name):
                source = ast.unparse(self._functions()[name])
                self.assertIn("_human_require_worker_online_for_overtime", source)
                self.assertNotIn("is_available", source)

    def test_overtime_assignment_requires_online_presence(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_human_require_worker_online_for_overtime")

        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code

        async def current_record(_uid, _now=None):
            return None, {}
        def payload(*_args):
            return {"online": online}
        async def load_profile(_uid):
            return {"workerApproved": True}
        async def run_case(online_now):
            nonlocal online
            online = online_now
            return await namespace["_human_require_worker_online_for_overtime"]("worker-1", {"workerApproved": True})

        online = False
        namespace = {
            "datetime": datetime, "ZoneInfo": ZoneInfo, "HUMAN_SHIFT_TIMEZONE": ZoneInfo("Africa/Nairobi"),
            "_human_shift_local_now": lambda: datetime(2026, 10, 5, 21, 0, tzinfo=ZoneInfo("Africa/Nairobi")),
            "_human_shift_is_scheduled": lambda _now=None: False,
            "_human_shift_current_record": current_record, "_human_shift_status_payload": payload,
            "_load_profile": load_profile, "HTTPException": RequestError,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        with self.assertRaisesRegex(RequestError, "only to a worker who is clocked in and online"):
            asyncio.run(run_case(False))
        self.assertTrue(asyncio.run(run_case(True)))


class WeekendShiftReconciliation(unittest.TestCase):
    def test_weekends_are_skipped_without_adding_missed_shifts(self):
        class Snapshot:
            def __init__(self, value):
                self.value = value
                self.exists = value is not None

            def to_dict(self):
                return dict(self.value or {})

        class Document:
            def __init__(self, records, key):
                self.records = records
                self.key = key

            def get(self):
                return Snapshot(self.records.get(self.key))

            def set(self, data, merge=False):
                current = dict(self.records.get(self.key) or {}) if merge else {}
                current.update(data)
                self.records[self.key] = current

        class Collection:
            def __init__(self):
                self.records = {}

            def document(self, key):
                return Document(self.records, key)

        class Database:
            def __init__(self):
                self.collections = {}

            def collection(self, name):
                return self.collections.setdefault(name, Collection())

        class Firestore:
            SERVER_TIMESTAMP = "SERVER_TIMESTAMP"

        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_human_shift_doc_id")
        db = Database()

        async def unused_archive(_uid):
            return None

        namespace = {
            "asyncio": asyncio, "datetime": datetime, "timedelta": timedelta,
            "HUMAN_SHIFT_TIMEZONE": ZoneInfo("Africa/Nairobi"),
            "HUMAN_SHIFT_END": datetime_time(20, 0),
            "HUMAN_SHIFT_WARNING_MISSES": 5, "HUMAN_SHIFT_LOCKOUT_MISSES": 6,
            "HUMAN_WORKER_DEADLINE_LOCKOUT_RETURNS": 11,
            "HUMAN_SHIFT_COLLECTION": "human_shifts", "firestore": Firestore,
            "_human_worker_retraining_updates": lambda *_args: {},
            "_human_archive_retraining_attempts": unused_archive,
            "db": db, "logger": _FakeLogger(),
        }
        exec(compile(ast.Module(body=[helper, functions["_human_shift_reconcile"]], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        profile = {
            "workerApproved": True, "humanShiftLastEvaluatedDate": "2026-10-02",
            "humanShiftMissesConsecutive": 4,
        }
        now = datetime(2026, 10, 5, 20, 0, tzinfo=ZoneInfo("Africa/Nairobi"))
        updated = asyncio.run(namespace["_human_shift_reconcile"]("worker-1", profile, now))
        self.assertEqual(updated["humanShiftMissesConsecutive"], 5)
        self.assertEqual(updated["humanShiftLastEvaluatedDate"], "2026-10-05")
        shift_records = db.collections["human_shifts"].records
        self.assertEqual(set(shift_records), {"worker-1_2026-10-05"})


class PdfBatchDownloadTests(unittest.TestCase):
    def test_private_page_images_are_combined_into_pdf_in_page_order(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_admin_download_pdf_batch")
        route.decorator_list = []

        class Snapshot:
            def __init__(self, job_id, data):
                self.id = job_id
                self.data = data

            def to_dict(self):
                return dict(self.data)

        class Query:
            def __init__(self, snapshots):
                self.snapshots = snapshots

            def where(self, **_kwargs):
                return self

            def stream(self):
                return iter(self.snapshots)

        class Database:
            def __init__(self, snapshots):
                self.snapshots = snapshots

            def collection(self, _name):
                return Query(self.snapshots)

        class Blob:
            def __init__(self, raw):
                self.raw = raw

            def download_as_bytes(self):
                return self.raw

        class Bucket:
            def __init__(self, files):
                self.files = files

            def blob(self, path):
                return Blob(self.files[path])

        class HttpError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code

        class TestResponse:
            def __init__(self, content, media_type, headers):
                self.body = content
                self.media_type = media_type
                self.headers = headers

        def png(color):
            handle = BytesIO()
            Image.new("RGB", (16, 16), color).save(handle, format="PNG")
            return handle.getvalue()

        red, blue = png((240, 10, 10)), png((10, 10, 240))
        job_two = {"job_type": "pdf_job", "pdf_batch_id": "batch-a", "pdf_image": {
            "page_number": 2, "source_filename": "source.docx", "storage_path": "human-workflow/job-2/pdf/page-2.jpg",
        }}
        job_one = {"job_type": "pdf_job", "pdf_batch_id": "batch-a", "pdf_image": {
            "page_number": 1, "source_filename": "source.docx", "storage_path": "human-workflow/job-1/pdf/page-1.jpg",
        }}
        snapshots = [Snapshot("job-2", job_two), Snapshot("job-1", job_one)]
        files = {job_two["pdf_image"]["storage_path"]: blue, job_one["pdf_image"]["storage_path"]: red}
        email = "typemywordz@gmail.com"

        async def actor(_request):
            return {"email": email}

        namespace = {
            "asyncio": asyncio, "BytesIO": BytesIO, "Image": Image, "Response": TestResponse,
            "HTTPException": HttpError, "Request": object, "FieldFilter": lambda *args: args,
            "HUMAN_JOB_COLLECTION": "human_jobs", "PDF_JOB_ADMIN_EMAILS": {"typemywordz@gmail.com", "info@typemywordz.ai"},
            "db": Database(snapshots), "_human_actor": actor, "_human_bucket": lambda: Bucket(files),
            "os": __import__("os"), "re": re, "logger": _FakeLogger(),
        }
        exec(compile(ast.Module(body=[route], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        response = asyncio.run(namespace["human_admin_download_pdf_batch"]("batch-a", object()))
        self.assertEqual(response.media_type, "application/pdf")
        document = pdfium.PdfDocument(response.body)
        self.assertEqual(len(document), 2)
        first = document[0].render(scale=0.5).to_pil().convert("RGB")
        second = document[1].render(scale=0.5).to_pil().convert("RGB")
        self.assertGreater(first.getpixel((first.width // 2, first.height // 2))[0], first.getpixel((first.width // 2, first.height // 2))[2])
        self.assertGreater(second.getpixel((second.width // 2, second.height // 2))[2], second.getpixel((second.width // 2, second.height // 2))[0])
        document.close()
        email = "subadmin@example.com"
        with self.assertRaises(HttpError):
            asyncio.run(namespace["human_admin_download_pdf_batch"]("batch-a", object()))


if __name__ == "__main__":
    unittest.main()
