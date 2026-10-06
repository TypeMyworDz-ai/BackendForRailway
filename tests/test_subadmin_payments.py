"""Pure, synthetic-only regression tests for sub-admin Human Work earnings."""
import ast
import hashlib
import math
import re
import unittest
from datetime import datetime, time as datetime_time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class _FakeFirestore:
    SERVER_TIMESTAMP = "SERVER_TIMESTAMP"


def _is_admin_user(email):
    return str(email or "").strip().lower() == "typemywordz@gmail.com"


class SubadminPaymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        wanted = {
            "_human_subadmin_rate_defaults", "_human_subadmin_rate_values",
            "_human_subadmin_rate_public", "_human_subadmin_parse_rate_update",
            "_human_subadmin_word_count", "_human_subadmin_audio_minutes",
            "_human_subadmin_ai_used", "_human_subadmin_assignment_metadata",
            "_human_subadmin_submission_actor",
            "_human_subadmin_earning_specs", "_human_admin_finish_job_eligible", "_human_shift_local_now",
            "_human_shift_parse_datetime", "_human_shift_is_scheduled",
            "is_human_subadmin",
        }
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted]
        cls.namespace = {
            "hashlib": hashlib,
            "math": math,
            "re": re,
            "unescape": unescape,
            "Decimal": Decimal,
            "InvalidOperation": InvalidOperation,
            "datetime": datetime,
            "datetime_time": datetime_time,
            "timedelta": timedelta,
            "timezone": timezone,
            "ZoneInfo": ZoneInfo,
            "firestore": _FakeFirestore,
            "HUMAN_JOB_ADMIN_EMAILS": ["info@typemywordz.ai"],
            "HUMAN_SHIFT_TIMEZONE": ZoneInfo("Africa/Nairobi"),
            "HUMAN_SHIFT_START": datetime_time(15, 0),
            "HUMAN_SHIFT_END": datetime_time(20, 0),
            "is_admin_user": _is_admin_user,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(MAIN_PATH), "exec"), cls.namespace)
        cls.zone = cls.namespace["HUMAN_SHIFT_TIMEZONE"]
        cls.shift_time = datetime(2026, 10, 5, 15, 30, tzinfo=cls.zone)
        cls.subadmin = {"uid": "subadmin-1", "email": "info@typemywordz.ai"}

    def test_defaults_preserve_exact_audio_and_image_units(self):
        defaults = self.namespace["_human_subadmin_rate_defaults"]()
        public = self.namespace["_human_subadmin_rate_public"](defaults)
        self.assertEqual(defaults["audio_human_kes_per_minute"], 10)
        self.assertEqual(defaults["audio_ai_kes_per_minute"], 20)
        self.assertEqual(defaults["image_human_rate_milli_kes_per_word"], 1)
        self.assertEqual(defaults["image_ai_rate_milli_kes_per_word"], 2)
        self.assertEqual(public["image_human_cents_per_word"], 0.1)
        self.assertEqual(public["image_ai_cents_per_word"], 0.2)

    def test_admin_rate_updates_validate_and_store_tenths_of_a_cent_exactly(self):
        parse = self.namespace["_human_subadmin_parse_rate_update"]
        values = parse({
            "audio_human_kes_per_minute": 10,
            "audio_ai_kes_per_minute": 20,
            "image_human_cents_per_word": 0.1,
            "image_ai_cents_per_word": 0.2,
        })
        self.assertEqual(values["image_human_rate_milli_kes_per_word"], 1)
        self.assertEqual(values["image_ai_rate_milli_kes_per_word"], 2)
        for invalid in (0.15, -0.1, 1001, "not-a-rate"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse({
                    "audio_human_kes_per_minute": 10,
                    "audio_ai_kes_per_minute": 20,
                    "image_human_cents_per_word": invalid,
                    "image_ai_cents_per_word": 0.2,
                })

    def test_word_count_handles_html_and_apostrophes(self):
        count = self.namespace["_human_subadmin_word_count"]
        self.assertEqual(count("I can't <b>proof-read</b> Children's."), 5)

    def test_audio_duration_rounds_up_to_a_whole_minute(self):
        minutes = self.namespace["_human_subadmin_audio_minutes"]
        self.assertEqual(minutes({"seconds": 61}), 2)
        self.assertEqual(minutes({"seconds": 10}, {"minutes": 3}), 3)

    def test_parent_ai_status_does_not_relabel_a_human_submitted_slice(self):
        used = self.namespace["_human_subadmin_ai_used"]
        parent = {"ai_agent_status": "submitted"}
        self.assertFalse(used(parent, {"status": "submitted", "worker_uid": "worker-1"}))
        self.assertTrue(used(parent, {"status": "submitted", "ai_agent_status": "submitted"}))
        self.assertTrue(used({"ai_review_applied": True}, {"status": "submitted"}))

    def test_submission_attribution_uses_the_assigned_subadmin_or_sole_configured_admin(self):
        choose_actor = self.namespace["_human_subadmin_submission_actor"]
        assigned = {
            "human_work_assigned_by_uid": "subadmin-2",
            "human_work_assigned_by_email": " INFO@TYPEMYWORDZ.AI ",
            "human_work_assignedAt": self.shift_time,
        }
        self.assertEqual(choose_actor(assigned), {"uid": "subadmin-2", "email": "info@typemywordz.ai"})
        self.assertEqual(choose_actor({}), {"uid": "", "email": "info@typemywordz.ai"})

    def test_every_human_submit_branch_persists_before_triggering_immediate_accrual(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_worker_submit")
        source = ast.unparse(route)
        persistence = "await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update"
        persist_positions = [index for index in range(len(source)) if source.startswith(persistence, index)]
        accrual = "await _human_subadmin_accrue_job_earnings"
        accrual_positions = [index for index in range(len(source)) if source.startswith(accrual, index)]
        self.assertEqual(len(persist_positions), 4)
        self.assertEqual(len(accrual_positions), 4)
        self.assertTrue(all(saved < earned for saved, earned in zip(persist_positions, accrual_positions)))
        self.assertIn("_human_subadmin_submission_actor", source)
        self.assertIn("segment_ids", source)

    def test_audio_earnings_apply_the_human_and_ai_rates_per_minute(self):
        build = self.namespace["_human_subadmin_earning_specs"]
        job = {
            "job_type": "general_job", "seconds": 65, "worker_uid": "worker-1",
            "workerCompletedAt": "submitted", "status": "submitted",
        }
        human = build("human-job", job, self.subadmin, now=self.shift_time)[0]
        self.assertEqual(human["category"], "audio_human")
        self.assertEqual(human["quantity"], 2)
        self.assertEqual(human["rate_milli_kes_per_unit"], 10000)
        self.assertEqual(human["amount_kes_milli"], 20000)

        ai_job = {**job, "ai_agent_status": "submitted"}
        ai = build("ai-job", ai_job, self.subadmin, now=self.shift_time)[0]
        self.assertEqual(ai["category"], "audio_ai")
        self.assertEqual(ai["rate_milli_kes_per_unit"], 20000)
        self.assertEqual(ai["amount_kes_milli"], 40000)

    def test_mixed_submitted_slices_are_each_classified_and_idempotently_identified(self):
        build = self.namespace["_human_subadmin_earning_specs"]
        job = {
            "job_type": "general_job", "ai_agent_status": "submitted",
            "segments": [
                {"id": "human-part", "status": "submitted", "worker_uid": "worker-1", "minutes": 1, "transcript": "human text"},
                {"id": "ai-part", "status": "submitted", "ai_agent_status": "submitted", "minutes": 2, "transcript": "AI text"},
            ],
        }
        rows = build("mixed-job", job, self.subadmin, now=self.shift_time)
        by_segment = {row["segment_id"]: row for row in rows}
        self.assertEqual(by_segment["human-part"]["category"], "audio_human")
        self.assertEqual(by_segment["human-part"]["amount_kes_milli"], 10000)
        self.assertEqual(by_segment["ai-part"]["category"], "audio_ai")
        self.assertEqual(by_segment["ai-part"]["amount_kes_milli"], 40000)
        again = build("mixed-job", job, self.subadmin, now=self.shift_time)
        self.assertEqual({row["earning_id"] for row in rows}, {row["earning_id"] for row in again})

    def test_immediate_ai_accrual_can_be_limited_to_the_successful_ai_slice(self):
        build = self.namespace["_human_subadmin_earning_specs"]
        job = {
            "job_type": "general_job", "ai_agent_status": "submitted",
            "segments": [
                {"id": "human-part", "status": "submitted", "worker_uid": "worker-1", "minutes": 1, "transcript": "human text"},
                {"id": "ai-part", "status": "submitted", "ai_agent_status": "submitted", "ai_agent_assigned_by_uid": "subadmin-1", "ai_agent_assigned_by_email": "info@typemywordz.ai", "ai_agent_assignedAt": self.shift_time, "minutes": 2, "transcript": "AI text"},
            ],
        }
        rows = build("mixed-job", job, {}, now=self.shift_time, segment_ids=["ai-part"], ai_only=True)
        self.assertEqual([row["segment_id"] for row in rows], ["ai-part"])
        self.assertEqual(rows[0]["category"], "audio_ai")
        self.assertEqual(build("mixed-job", job, {}, now=self.shift_time, segment_ids=["human-part"], ai_only=True), [])

    def test_admin_finish_requires_all_submitted_parts_and_no_active_proofreader(self):
        eligible = self.namespace["_human_admin_finish_job_eligible"]
        job = {
            "status": "proofreading_available",
            "segments": [
                {"id": "part-1", "status": "submitted", "transcript": "First part."},
                {"id": "part-2", "status": "submitted", "final_attachment": {"storage_path": "private/final.docx"}},
            ],
        }
        self.assertTrue(eligible(job))
        self.assertFalse(eligible({**job, "proofreader_status": "assigned"}))
        self.assertFalse(eligible({**job, "proofreader_status": "in_progress"}))
        self.assertFalse(eligible({**job, "reviewer_status": "processing"}))
        self.assertFalse(eligible({**job, "segments": [*job["segments"], {"id": "part-3", "status": "in_progress", "transcript": "Not done."}]}))
        self.assertFalse(eligible({"status": "submitted", "segments": [{"id": "empty", "status": "submitted"}]}))
        self.assertTrue(eligible({"status": "submitted", "transcript": "Complete transcript."}))

    def test_image_earnings_keep_fractional_kes_without_rounding(self):
        build = self.namespace["_human_subadmin_earning_specs"]
        human_job = {
            "job_type": "pdf_job", "pdf_review": {"page_count": 1}, "transcript": "one two three four",
            "worker_uid": "worker-1", "workerCompletedAt": "submitted",
        }
        human = build("image-human", human_job, self.subadmin, now=self.shift_time)[0]
        self.assertEqual(human["category"], "image_human")
        self.assertEqual(human["quantity"], 4)
        self.assertEqual(human["amount_kes_milli"], 4)
        ai_job = {**human_job, "ai_agent_status": "submitted"}
        ai = build("image-ai", ai_job, self.subadmin, now=self.shift_time)[0]
        self.assertEqual(ai["category"], "image_ai")
        self.assertEqual(ai["amount_kes_milli"], 8)

    def test_main_admin_review_uses_subadmin_ai_assignment_and_shift_timestamp(self):
        build = self.namespace["_human_subadmin_earning_specs"]
        assigned_at = datetime(2026, 10, 5, 15, 5, tzinfo=self.zone)
        job = {
            "job_type": "general_job", "seconds": 60, "ai_agent_status": "submitted",
            "ai_agent_assigned_by_uid": "subadmin-1",
            "ai_agent_assigned_by_email": "info@typemywordz.ai",
            "ai_agent_assignedAt": assigned_at,
        }
        main_admin = {"uid": "main-admin", "email": "typemywordz@gmail.com"}
        row = build("assigned-ai-job", job, main_admin, now=datetime(2026, 10, 5, 21, 0, tzinfo=self.zone))[0]
        self.assertEqual(row["subadmin_uid"], "subadmin-1")
        self.assertEqual(row["category"], "audio_ai")
        self.assertEqual(row["shift_date"], "2026-10-05")

    def test_earnings_are_excluded_before_shift_and_on_weekends(self):
        build = self.namespace["_human_subadmin_earning_specs"]
        job = {"job_type": "general_job", "seconds": 60, "worker_uid": "worker-1", "workerCompletedAt": "submitted"}
        before = datetime(2026, 10, 5, 14, 59, tzinfo=self.zone)
        saturday = datetime(2026, 10, 10, 16, 0, tzinfo=self.zone)
        self.assertEqual(build("before-shift", job, self.subadmin, now=before), [])
        self.assertEqual(build("weekend", job, self.subadmin, now=saturday), [])

    def test_payroll_and_rate_routes_use_the_correct_role_guards(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        routes = {
            node.name: node for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {
                "human_admin_get_subadmin_rates", "human_admin_update_subadmin_rates",
                "human_admin_subadmin_options", "human_admin_subadmin_earnings",
                "human_admin_subadmin_payouts", "human_admin_mark_subadmin_payout_paid",
                "human_subadmin_payment_history",
            }
        }
        expected = {
            "human_admin_get_subadmin_rates", "human_admin_update_subadmin_rates",
            "human_admin_subadmin_options", "human_admin_subadmin_earnings",
            "human_admin_subadmin_payouts", "human_admin_mark_subadmin_payout_paid",
            "human_subadmin_payment_history",
        }
        self.assertEqual(set(routes), expected)
        for name, node in routes.items():
            calls = {item.func.id for item in ast.walk(node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
            self.assertIn("_require_human_subadmin" if name == "human_subadmin_payment_history" else "_require_admin", calls, name)

    def test_successful_agent_review_and_finish_paths_call_idempotent_accrual(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
        for name in (
            "human_admin_review", "human_admin_finish_ai_agent_draft", "human_admin_finish_job",
            "human_admin_ai_review", "_human_run_ai_agent", "_human_run_letter_agent", "_human_run_letter_ai_review",
        ):
            calls = {item.func.id for item in ast.walk(functions[name]) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
            self.assertIn("_human_subadmin_accrue_job_earnings", calls, name)

    def test_generic_finish_route_is_admin_only_and_preserves_client_approval_flow(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_admin_finish_job")
        route_text = ast.unparse(route)
        calls = {item.func.id for item in ast.walk(route) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
        self.assertIn("_require_human_job_admin", calls)
        self.assertIn("_human_admin_finish_job_eligible", calls)
        self.assertIn("_human_subadmin_accrue_job_earnings", calls)
        self.assertIn("client_review", route_text)
        self.assertIn("released", route_text)
        self.assertNotIn("charge_credits", route_text)


if __name__ == "__main__":
    unittest.main()
