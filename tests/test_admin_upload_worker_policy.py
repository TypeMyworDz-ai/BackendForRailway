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
            "HUMAN_JOB_DASHBOARD_RETENTION_DAYS": 3,
            "_as_dt": lambda value: value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None,
            "firestore": _FakeFirestore,
            "HUMAN_WORKER_DEADLINE_LOCKOUT_RETURNS": 11,
            "TRAINING_LEVELS": [{"level": level} for level in range(1, 7)],
        }
        wanted = [
            "_human_claim_item_key", "_human_claim_attempt_document_id",
            "_human_deadline_event_id", "_human_worker_retraining_updates",
            "_human_shift_parse_datetime", "_human_shift_is_scheduled",
            "_human_shift_call_in_active", "_human_shift_status_payload",
            "_human_job_dashboard_archived",
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

    def test_off_shift_availability_is_presence_only_and_never_enables_self_claims(self):
        status_payload = self.namespace["_human_shift_status_payload"]
        zone = self.namespace["HUMAN_SHIFT_TIMEZONE"]
        now = datetime(2026, 10, 5, 20, 15, tzinfo=zone)
        recent = {"lastPresenceAt": "2026-10-05T20:14:00+03:00"}
        online = status_payload("worker-1", {"workerApproved": True, "is_available": True}, recent, now)
        self.assertFalse(online["can_claim"], "online presence is not attendance or permission to claim")
        self.assertFalse(online["off_shift_self_claim"])
        self.assertTrue(online["workroom_online"])
        self.assertTrue(online["online"], "the toggle and fresh heartbeat together make the worker visible online")
        self.assertFalse(online["clocked_in"])

        offline = status_payload("worker-1", {"workerApproved": True, "is_available": False}, recent, now)
        self.assertFalse(offline["can_claim"])
        self.assertFalse(offline["online"], "turning the toggle off hides online presence even with a fresh heartbeat")
        stale = status_payload("worker-1", {"workerApproved": True, "is_available": True}, {"lastPresenceAt": "2026-10-05T20:10:00+03:00"}, now)
        self.assertFalse(stale["online"], "a stale heartbeat is not online presence")
        called_in = status_payload(
            "worker-1", {"workerApproved": True, "is_available": True, "humanShiftCallInExpiresAt": "2026-10-06T00:00:00+03:00"}, recent, now,
        )
        self.assertFalse(called_in["can_claim"], "an admin call-in still requires clock-in before claiming")
        self.assertFalse(called_in["off_shift_self_claim"])
        called_back = status_payload(
            "worker-1",
            {"workerApproved": True, "is_available": True, "humanShiftCallInExpiresAt": "2026-10-06T00:00:00+03:00"},
            {"clockedInAt": "2026-10-05T15:00:00+03:00", "clockedOutAt": "2026-10-05T20:00:00+03:00", **recent},
            now,
        )
        self.assertEqual(called_back["status"], "called_in_not_clocked_in")
        self.assertTrue(called_back["can_clock_in"], "an active admin call-in permits a new overtime clock-in after clocking out")
        self.assertFalse(called_back["can_claim"])

    def test_scheduled_shift_claim_does_not_require_availability_toggle(self):
        status_payload = self.namespace["_human_shift_status_payload"]
        zone = self.namespace["HUMAN_SHIFT_TIMEZONE"]
        now = datetime(2026, 10, 5, 18, 0, tzinfo=zone)
        result = status_payload("worker-1", {"workerApproved": True, "is_available": False}, {"clockedInAt": "2026-10-05T15:00:00+03:00"}, now)
        self.assertTrue(result["scheduled_now"])
        self.assertFalse(result["availability_required"])
        self.assertTrue(result["can_claim"])
        self.assertFalse(result["off_shift_self_claim"])

    def test_presence_refresh_never_records_shift_attendance_and_requires_toggle_or_clock_in(self):
        presence = ast.unparse(self.functions["human_worker_shift_presence"])
        self.assertIn("online_toggle", presence)
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
        self.assertIn("Run AI proofreading", review)
        self.assertIn("Apply the AI-proofread transcript", review)
        self.assertIn("Choose an AI proofreader or assign a human proofreader", review)


    def test_three_day_archive_boundary_is_inclusive_and_non_destructive(self):
        archived = self.namespace["_human_job_dashboard_archived"]
        now = datetime(2026, 10, 7, 12, 0)
        self.assertFalse(archived({"createdAt": datetime(2026, 10, 4, 12, 0)}, now=now - timedelta(seconds=1)))
        self.assertTrue(archived({"createdAt": datetime(2026, 10, 4, 12, 0)}, now=now))
        self.assertTrue(archived({"createdAt": datetime(2026, 10, 4, 12, 0), "status": "in_progress"}, now=now))
        self.assertFalse(archived({"status": "released"}, now=now))
        source = ast.unparse(self.functions["human_list_jobs"])
        self.assertIn("scope == 'archived'", source)
        self.assertIn("_human_job_dashboard_archived", source)
        self.assertNotIn("_human_delete_job_safely", source)

    def test_scheduled_worker_can_clock_in_again_without_an_after_hours_call_in(self):
        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FakeFirestore:
            DELETE_FIELD = object()

        class ShiftRef:
            def __init__(self):
                self.saved = None
            def set(self, updates, merge=False):
                self.saved = (updates, merge)

        class FakeApp:
            def post(self, *_args, **_kwargs):
                return lambda function: function

        shift_ref = ShiftRef()
        shift = {"clockedInAt": "2026-10-05T15:00:00+03:00", "clockedOutAt": "2026-10-05T15:30:00+03:00", "sessions": []}
        async def actor(_request):
            return {"uid": "worker-1", "email": "worker@example.com", "role": "worker", "profile": {"workerApproved": True}}
        async def reconcile(_uid, profile):
            return profile
        async def current_record(_uid, _now):
            return shift_ref, dict(shift)
        def payload(_uid, _profile, current, _now):
            return {"clocked_in": bool(current.get("clockedInAt") and not current.get("clockedOutAt")), "online": False}

        namespace = {
            "app": FakeApp(), "asyncio": asyncio, "Request": object, "HTTPException": RequestError,
            "firestore": FakeFirestore, "_human_actor": actor, "_human_shift_reconcile": reconcile,
            "_human_shift_call_in_active": lambda *_args: False,
            "_human_shift_is_scheduled": lambda *_args: True,
            "_human_shift_local_now": lambda: datetime(2026, 10, 5, 16, 0, tzinfo=ZoneInfo("Africa/Nairobi")),
            "_human_shift_current_record": current_record, "_human_shift_status_payload": payload,
        }
        route = self.functions["human_worker_shift_clock_in"]
        exec(compile(ast.Module(body=[route], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        result = asyncio.run(namespace["human_worker_shift_clock_in"](object()))
        saved, merge = shift_ref.saved
        self.assertTrue(merge)
        self.assertIs(saved["clockedOutAt"], FakeFirestore.DELETE_FIELD)
        self.assertEqual(saved["sessions"][0]["clockedOutAt"], shift["clockedOutAt"])
        self.assertTrue(result["clocked_in"])

    def test_finished_job_conversation_remains_open_to_assigned_worker(self):
        assert_access = self.functions["_human_assert_access"]
        assert_thread_access = self.functions["_human_assert_job_conversation_access"]
        thread_for = self.functions["_human_thread_for"]
        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail
        namespace = {
            "HTTPException": RequestError,
            "_human_job_worker_uids": lambda job: {job.get("worker_uid"), *(item.get("worker_uid") for item in job.get("segments", []))} - {None},
        }
        exec(compile(ast.Module(body=[assert_access, assert_thread_access, thread_for], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        job = {"status": "released", "worker_uid": "worker-1", "segments": []}
        actor = {"uid": "worker-1", "role": "worker"}
        self.assertIsNone(asyncio.run(namespace["_human_assert_access"](job, actor)))
        self.assertIsNone(namespace["_human_assert_job_conversation_access"](job, actor))
        self.assertEqual(namespace["_human_thread_for"](actor), "worker")
        for name in ("human_messages", "human_send_message"):
            source = ast.unparse(self.functions[name])
            self.assertIn("_human_assert_access", source)
            self.assertIn("_human_assert_job_conversation_access", source)
            self.assertNotRegex(source, r"\bstatus\b")

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

    def test_worker_can_reclock_after_admin_calls_them_in_again_after_clocking_out(self):
        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FakeFirestore:
            DELETE_FIELD = object()

        class ShiftRef:
            def __init__(self):
                self.saved = None
            def set(self, updates, merge=False):
                self.saved = (updates, merge)

        class FakeApp:
            def post(self, *_args, **_kwargs):
                return lambda function: function

        shift_ref = ShiftRef()
        shift = {"clockedInAt": "2026-10-05T15:00:00+03:00", "clockedOutAt": "2026-10-05T20:00:00+03:00", "sessions": []}
        async def actor(_request):
            return {"uid": "worker-1", "email": "worker@example.com", "role": "worker", "profile": {"workerApproved": True}}
        async def reconcile(_uid, profile):
            return profile
        async def current_record(_uid, _now):
            return shift_ref, dict(shift)
        def payload(_uid, _profile, current, _now):
            return {"clocked_in": bool(current.get("clockedInAt") and not current.get("clockedOutAt")), "online": False}

        namespace = {
            "app": FakeApp(), "asyncio": asyncio, "Request": object, "HTTPException": RequestError,
            "firestore": FakeFirestore, "_human_actor": actor,
            "_human_shift_reconcile": reconcile, "_human_shift_call_in_active": lambda *_args: True,
            "_human_shift_is_scheduled": lambda *_args: False,
            "_human_shift_local_now": lambda: datetime(2026, 10, 5, 21, 0, tzinfo=ZoneInfo("Africa/Nairobi")),
            "_human_shift_current_record": current_record, "_human_shift_status_payload": payload,
        }
        route = self._functions()["human_worker_shift_clock_in"]
        exec(compile(ast.Module(body=[route], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        result = asyncio.run(namespace["human_worker_shift_clock_in"](object()))
        saved, merge = shift_ref.saved
        self.assertTrue(merge)
        self.assertIs(saved["clockedOutAt"], FakeFirestore.DELETE_FIELD)
        self.assertTrue(result["clocked_in"])

    def test_accepted_after_hours_claim_marker_is_used_for_start_and_submit(self):
        start = ast.unparse(self._functions()["human_worker_start"])
        submit = ast.unparse(self._functions()["human_worker_submit"])
        claim = ast.unparse(self._functions()["human_worker_claim"])
        self.assertIn("allow_after_hours_self_claim=target.get('after_hours_self_claim') is True", start)
        self.assertIn("allow_after_hours_self_claim=job.get('after_hours_self_claim') is True", start)
        self.assertIn("allow_after_hours_self_claim=target.get('after_hours_self_claim') is True", submit)
        self.assertIn("allow_after_hours_self_claim=job.get('after_hours_self_claim') is True", submit)
        self.assertIn("require_availability=False", claim)
        self.assertIn("after_hours_self_claim=False", claim)
        self.assertIn("_human_shift_assert_can_claim(actor)", claim)

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
            return {"online": online, "clocked_in": clocked_in, "call_in_active": call_in_active}
        async def load_profile(_uid):
            return {"workerApproved": True}
        async def run_case(online_now, clocked_in_now, call_in_now):
            nonlocal online, clocked_in, call_in_active
            online = online_now
            clocked_in = clocked_in_now
            call_in_active = call_in_now
            return await namespace["_human_require_worker_online_for_overtime"]("worker-1", {"workerApproved": True})

        online = False
        clocked_in = False
        call_in_active = False
        namespace = {
            "datetime": datetime, "ZoneInfo": ZoneInfo, "HUMAN_SHIFT_TIMEZONE": ZoneInfo("Africa/Nairobi"),
            "_human_shift_local_now": lambda: datetime(2026, 10, 5, 21, 0, tzinfo=ZoneInfo("Africa/Nairobi")),
            "_human_shift_is_scheduled": lambda _now=None: False,
            "_human_shift_current_record": current_record, "_human_shift_status_payload": payload,
            "_load_profile": load_profile, "HTTPException": RequestError,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        with self.assertRaisesRegex(RequestError, "call the worker in"):
            asyncio.run(run_case(False, True, True))
        with self.assertRaisesRegex(RequestError, "call the worker in"):
            asyncio.run(run_case(True, False, True))
        with self.assertRaisesRegex(RequestError, "call the worker in"):
            asyncio.run(run_case(True, True, False))
        self.assertTrue(asyncio.run(run_case(True, True, True)))


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
    def test_admin_uploads_share_one_download_batch_across_selected_images(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_admin_create_pdf_jobs")
        source = ast.unparse(route)
        self.assertLess(source.index("upload_batch_id = uuid.uuid4().hex"), source.index("for upload in files:"))
        self.assertIn("pdf_upload_batch_id", source)
        self.assertIn("pdf_upload_page_number", source)
        self.assertIn("upload_batch_id", source)
        rendered_loop = source.index("for rendered in rendered_images:")
        self.assertLess(source.index("for upload in files:"), rendered_loop)
        self.assertLess(source.index("upload_page_number += 1", rendered_loop), source.index("pdf_upload_page_number", rendered_loop))

    def test_private_page_images_are_combined_into_pdf_in_page_order(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_admin_download_pdf_batch")
        order_helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_human_pdf_upload_order_key")
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

            def where(self, **kwargs):
                field, operator, value = kwargs.get("filter")
                if operator == "==":
                    return Query([snap for snap in self.snapshots if (snap.to_dict() or {}).get(field) == value])
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

        red, blue, green = png((240, 10, 10)), png((10, 10, 240)), png((10, 220, 10))
        job_two = {
            "job_type": "pdf_job", "status": "approved", "worker_uid": None,
            "pdf_batch_id": "source-file-b", "pdf_upload_batch_id": "upload-batch-a",
            "pdf_upload_batch_name": "conversation.png + 1 more", "pdf_upload_download_name": "conversation-combined",
            "pdf_upload_page_number": 2, "pdf_image": {
                "page_number": 1, "source_filename": "follow-up.png", "storage_path": "human-workflow/job-2/pdf/page-1.jpg",
            },
        }
        job_one = {
            "job_type": "pdf_job", "status": "approved", "worker_uid": None,
            "pdf_batch_id": "source-file-a", "pdf_upload_batch_id": "upload-batch-a",
            "pdf_upload_batch_name": "conversation.png + 1 more", "pdf_upload_download_name": "conversation-combined",
            "pdf_upload_page_number": 1, "pdf_image": {
                "page_number": 1, "source_filename": "conversation.png", "storage_path": "human-workflow/job-1/pdf/page-1.jpg",
            },
        }
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
        exec(compile(ast.Module(body=[order_helper, route], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        response = asyncio.run(namespace["human_admin_download_pdf_batch"]("upload-batch-a", object()))
        self.assertEqual(response.media_type, "application/pdf")
        self.assertIn("conversation-combined.pdf", response.headers["Content-Disposition"])
        document = pdfium.PdfDocument(response.body)
        self.assertEqual(len(document), 2)
        first = document[0].render(scale=0.5).to_pil().convert("RGB")
        second = document[1].render(scale=0.5).to_pil().convert("RGB")
        self.assertGreater(first.getpixel((first.width // 2, first.height // 2))[0], first.getpixel((first.width // 2, first.height // 2))[2])
        self.assertGreater(second.getpixel((second.width // 2, second.height // 2))[2], second.getpixel((second.width // 2, second.height // 2))[0])
        document.close()
        legacy_path = "human-workflow/legacy-job/pdf/legacy.jpg"
        files[legacy_path] = green
        legacy_job = {"job_type": "pdf_job", "status": "approved", "pdf_batch_id": "legacy-batch", "pdf_image": {
            "page_number": 1, "source_filename": "legacy.png", "storage_path": legacy_path,
        }}
        namespace["db"] = Database([Snapshot("legacy-job", legacy_job)])
        legacy_response = asyncio.run(namespace["human_admin_download_pdf_batch"]("legacy-batch", object()))
        legacy_pdf = pdfium.PdfDocument(legacy_response.body)
        self.assertEqual(len(legacy_pdf), 1)
        legacy_pdf.close()
        email = "subadmin@example.com"
        with self.assertRaises(HttpError):
            asyncio.run(namespace["human_admin_download_pdf_batch"]("upload-batch-a", object()))

    def test_order_and_group_helpers_preserve_upload_sequence_across_source_files(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        wanted = {"_human_pdf_upload_order_key", "_human_pdf_same_upload_batch", "_human_pdf_same_source_file"}
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        first = {"pdf_upload_batch_id": "upload-1", "pdf_upload_page_number": 1, "pdf_batch_id": "file-a", "pdf_image": {"page_number": 1, "source_filename": "first.png"}}
        second = {"pdf_upload_batch_id": "upload-1", "pdf_upload_page_number": 2, "pdf_batch_id": "file-b", "pdf_image": {"page_number": 1, "source_filename": "second.png"}}
        other_upload = {"pdf_upload_batch_id": "upload-2", "pdf_upload_page_number": 1, "pdf_batch_id": "file-c", "pdf_image": {"page_number": 1, "source_filename": "first.png"}}
        unordered = [("second", second), ("first", first)]
        ordered = sorted(unordered, key=lambda entry: namespace["_human_pdf_upload_order_key"](entry[1], entry[0]))
        self.assertEqual([job_id for job_id, _ in ordered], ["first", "second"])
        self.assertTrue(namespace["_human_pdf_same_upload_batch"](first, second))
        self.assertFalse(namespace["_human_pdf_same_upload_batch"](first, other_upload))
        self.assertFalse(namespace["_human_pdf_same_source_file"](first, second))

    def test_whole_batch_ai_and_review_use_upload_identity_and_original_order(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
        assignment = ast.unparse(functions["human_admin_assign_ai_agent"])
        review = ast.unparse(functions["human_admin_create_file_review"])
        reviewer = ast.unparse(functions["_human_image_review_draft"])
        for source in (assignment, review):
            self.assertIn("requested_upload_batch_id", source)
            self.assertIn("_human_pdf_same_upload_batch", source)
            self.assertIn("_human_pdf_upload_order_key", source)
        self.assertIn("'pdf_images'", reviewer)
        self.assertIn("metas = sorted(metas", reviewer)

    def test_whole_upload_review_stores_images_and_drafts_in_upload_order(self):
        import uuid

        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_human_pdf_upload_order_key", "_human_pdf_same_upload_batch", "_human_pdf_same_source_file"}]
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_admin_create_file_review")
        route.decorator_list = []
        jobs = {
            "image-second": {
                "job_type": "pdf_job", "job_category": "pdf", "pdf_batch_id": "file-b", "pdf_upload_batch_id": "upload-1",
                "pdf_upload_batch_name": "first.png + 1 more", "pdf_upload_page_number": 2, "transcript": "Second image text",
                "pdf_image": {"name": "second.png", "source_filename": "second.png", "page_number": 1, "storage_path": "source/second"},
            },
            "image-first": {
                "job_type": "pdf_job", "job_category": "pdf", "pdf_batch_id": "file-a", "pdf_upload_batch_id": "upload-1",
                "pdf_upload_batch_name": "first.png + 1 more", "pdf_upload_page_number": 1, "transcript": "First image text",
                "pdf_image": {"name": "first.png", "source_filename": "first.png", "page_number": 1, "storage_path": "source/first"},
            },
        }
        stored_jobs = {}
        storage = {"source/first": b"first-image", "source/second": b"second-image"}
        stored_names = []

        class FakeHttpError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code

        class FakeQuery:
            def where(self, **_kwargs):
                return self

            def limit(self, _limit):
                return self

            def stream(self):
                return iter(())

        class FakeDocument:
            def __init__(self, doc_id):
                self.doc_id = doc_id

            def set(self, value):
                stored_jobs[self.doc_id] = value

        class FakeCollection:
            def where(self, **_kwargs):
                return FakeQuery()

            def document(self, doc_id):
                return FakeDocument(doc_id)

        class FakeDatabase:
            def collection(self, _name):
                return FakeCollection()

        class FakeBlob:
            def __init__(self, path):
                self.path = path

            def download_as_bytes(self):
                return storage[self.path]

        class FakeBucket:
            def blob(self, path):
                return FakeBlob(path)

        class FakeRequest:
            async def json(self):
                return {"job_ids": ["image-second", "image-first"], "upload_batch_id": "upload-1"}

        async def actor(_request):
            return {"email": "typemywordz@gmail.com"}

        async def get_job(job_id):
            return jobs[job_id]

        async def page_text(item):
            return item.get("transcript", "")

        def store_bytes(job_id, name, raw, content_type, folder):
            stored_names.append(name)
            path = f"human-workflow/{job_id}/{folder}/{name}"
            storage[path] = raw
            return {"name": name, "storage_path": path, "content_type": content_type}

        fake_firestore = type("FakeFirestore", (), {"SERVER_TIMESTAMP": "SERVER_TIMESTAMP"})
        namespace = {
            "asyncio": asyncio, "uuid": uuid, "hashlib": hashlib, "logger": _FakeLogger(),
            "HTTPException": FakeHttpError, "Request": object,
            "PDF_JOB_ADMIN_EMAILS": {"typemywordz@gmail.com", "info@typemywordz.ai"},
            "HUMAN_JOB_COLLECTION": "human_jobs", "db": FakeDatabase(), "firestore": fake_firestore,
            "FieldFilter": lambda *args: args, "PDF_JOB_REVIEW_PAY_KES_PER_PAGE": 100,
            "_human_actor": actor, "_human_job": get_job, "_pdf_page_draft_text": page_text,
            "_human_bucket": lambda: FakeBucket(), "_human_store_raw_bytes": store_bytes,
            "_human_pdf_upload_order_key": None, "_human_pdf_same_upload_batch": None,
            "_human_pdf_same_source_file": None, "_human_collapse_duplicate_image_page_blocks": lambda text, *_args: text,
            "human_image_tat_seconds": lambda count: count * 60,
            "_review_text_to_html": lambda text: text, "_sanitize_editor_html": lambda text: text,
        }
        namespace["_human_pdf_upload_order_key"] = next(node for node in helpers if node.name == "_human_pdf_upload_order_key")
        namespace["_human_pdf_same_upload_batch"] = next(node for node in helpers if node.name == "_human_pdf_same_upload_batch")
        namespace["_human_pdf_same_source_file"] = next(node for node in helpers if node.name == "_human_pdf_same_source_file")
        exec(compile(ast.Module(body=helpers + [route], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        result = asyncio.run(namespace["human_admin_create_file_review"](FakeRequest()))
        self.assertEqual(result["pages"], 2)
        self.assertEqual(stored_names, ["first.png", "second.png"])
        review_job = stored_jobs[result["job_id"]]
        self.assertEqual(review_job["pdf_review"]["source_job_ids"], ["image-first", "image-second"])
        self.assertEqual(review_job["pdf_review"]["page_texts"], ["First image text", "Second image text"])
        self.assertEqual(review_job["pdf_review"]["source_upload_batch_id"], "upload-1")
        self.assertEqual([image["page_number"] for image in review_job["pdf_images"]], [1, 2])
        self.assertIn("first.png + 1 more", review_job["pdf_review"]["source_filename"])


class HumanWorkAiBillingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        cls.functions = {
            node.name: node for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_each_human_work_ai_agent_or_reviewer_route_charges_before_queueing(self):
        routes = (
            "human_admin_assign_ai_agent", "human_admin_assign_letter_agent",
            "human_admin_retry_letter_ai_review", "human_admin_ai_review",
        )
        for name in routes:
            with self.subTest(route=name):
                function = self.functions[name]
                calls = [node for node in ast.walk(function)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "_human_charge_ai_call"]
                self.assertEqual(len(calls), 1, f"{name} must charge exactly once")
                if name != "human_admin_ai_review":
                    source = ast.unparse(function)
                    self.assertLess(source.index("_human_charge_ai_call"), source.index("background_tasks.add_task"))

    def test_failed_billing_restores_queued_job_state_before_provider_start(self):
        generic = ast.unparse(self.functions["human_admin_assign_ai_agent"])
        letter = ast.unparse(self.functions["human_admin_assign_letter_agent"])
        letter_review = ast.unparse(self.functions["human_admin_retry_letter_ai_review"])
        self.assertIn("if job_updated", generic)
        self.assertIn("restore_updates", generic)
        self.assertLess(generic.index("_human_charge_ai_call"), generic.index("background_tasks.add_task"))
        self.assertIn("if queued", letter)
        self.assertIn("restore_keys", letter)
        self.assertLess(letter.index("_human_charge_ai_call"), letter.index("background_tasks.add_task"))
        self.assertIn("review_ref.update", letter_review)
        self.assertIn("_human_charge_ai_call", letter_review)
        self.assertLess(letter_review.index("_human_charge_ai_call"), letter_review.index("background_tasks.add_task"))

    def test_internal_finish_and_apply_existing_ai_review_are_not_new_ai_calls(self):
        for name in ("human_admin_finish_ai_agent_draft", "human_admin_apply_ai_review"):
            with self.subTest(route=name):
                source = ast.unparse(self.functions[name])
                self.assertNotIn("_human_charge_ai_call", source)
                self.assertNotIn("charge_credits", source)

    def test_ai_call_charge_is_two_credits_and_force_bills_admin_and_subadmin(self):
        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        calls = []
        async def charge(*args, **kwargs):
            calls.append((args, kwargs))
            return {"charged": 2, "remaining": 8}

        namespace = {
            "HUMAN_WORK_AI_CREDIT_COST": 2,
            "Optional": __import__("typing").Optional,
            "HTTPException": RequestError,
            "charge_credits": charge,
        }
        exec(compile(ast.Module(body=[self.functions["_human_charge_ai_call"]], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        helper = namespace["_human_charge_ai_call"]
        for email in ("typemywordz@gmail.com", "info@typemywordz.ai"):
            result = asyncio.run(helper({"uid": "operator-1", "email": email}, "job-1", "AI reviewer"))
            self.assertEqual(result, {"credits_deducted": 2, "credits_remaining": 8})
        self.assertEqual(len(calls), 2)
        for index, (args, kwargs) in enumerate(calls):
            self.assertEqual(args[:3], ("operator-1", ("typemywordz@gmail.com", "info@typemywordz.ai")[index], 2))
            self.assertTrue(str(args[3]).startswith("Human Work AI reviewer"))
            self.assertEqual(kwargs["usage_category"], "human_work_ai")
            self.assertTrue(kwargs["force_charge"])
            self.assertTrue(kwargs["require_saved"])

    def test_proofreading_can_use_the_scoped_five_credit_rate(self):
        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        calls = []
        async def charge(*args, **kwargs):
            calls.append((args, kwargs))
            return {"charged": 5, "remaining": 3}

        namespace = {
            "HUMAN_WORK_AI_CREDIT_COST": 2,
            "Optional": __import__("typing").Optional,
            "HTTPException": RequestError,
            "charge_credits": charge,
        }
        exec(compile(ast.Module(body=[self.functions["_human_charge_ai_call"]], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        result = asyncio.run(namespace["_human_charge_ai_call"](
            {"uid": "worker-1", "email": "worker@example.com"},
            "job-1", "AI proofreader", credit_cost=5,
        ))
        self.assertEqual(result, {"credits_deducted": 5, "credits_remaining": 3})
        self.assertEqual(calls[0][0][2], 5)
        self.assertTrue(str(calls[0][0][3]).startswith("Human Work AI proofreader"))

    def test_insufficient_balance_or_failed_save_prevents_ai_call(self):
        class RequestError(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        outcomes = iter([
            {"charged": 0, "available": 1, "remaining": 1},
            {"charged": 0, "error": "credit update could not be saved"},
        ])
        async def charge(*_args, **_kwargs):
            return next(outcomes)

        namespace = {
            "HUMAN_WORK_AI_CREDIT_COST": 2,
            "Optional": __import__("typing").Optional,
            "HTTPException": RequestError,
            "charge_credits": charge,
        }
        exec(compile(ast.Module(body=[self.functions["_human_charge_ai_call"]], type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
        helper = namespace["_human_charge_ai_call"]
        with self.assertRaises(RequestError) as insufficient:
            asyncio.run(helper({"uid": "operator-1", "email": "info@typemywordz.ai"}, "job-1", "AI agent"))
        self.assertEqual(insufficient.exception.status_code, 402)
        with self.assertRaises(RequestError) as save_failure:
            asyncio.run(helper({"uid": "operator-1", "email": "info@typemywordz.ai"}, "job-1", "AI agent"))
        self.assertEqual(save_failure.exception.status_code, 503)

    def test_billable_main_admin_balance_is_verified_and_scoped_to_same_account(self):
        route = ast.unparse(self.functions["credits_balance"])
        self.assertIn("if billable_ai_balance", route)
        self.assertIn("_verified_user(request)", route)
        self.assertIn("decoded.get('sub')", route)
        self.assertIn("verified_email != str(user_email", route)


if __name__ == "__main__":
    unittest.main()
