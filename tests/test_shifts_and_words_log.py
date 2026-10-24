import ast
import os
import unittest
from datetime import datetime, time as datetime_time, timedelta, date as datetime_date
from zoneinfo import ZoneInfo

SRC = open(os.path.join(os.path.dirname(__file__), "..", "main.py"), encoding="utf-8").read()
TREE = ast.parse(SRC)
TZ = ZoneInfo("Africa/Nairobi")


def load(funcs, consts):
    ns = {"datetime": datetime, "timedelta": timedelta, "datetime_time": datetime_time, "datetime_date": datetime_date,
          "ZoneInfo": ZoneInfo, "HUMAN_SHIFT_TIMEZONE": TZ}
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in consts for t in node.targets):
            exec(compile(ast.Module([node], []), "m", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            exec(compile(ast.Module([node], []), "m", "exec"), ns)
    return ns


def at(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=TZ)


class WorkerShiftWindowTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({"_human_shift_window", "_human_shift_is_scheduled", "_human_shift_effective_date", "_human_shift_local_now"},
                       {"HUMAN_SHIFT_START", "HUMAN_SHIFT_END", "HUMAN_SHIFT_SECOND_START", "HUMAN_SHIFT_SECOND_END"})
        self.sched = self.ns["_human_shift_is_scheduled"]

    def test_first_shift_is_4_to_8(self):
        # 2026-10-26 is a Monday
        self.assertFalse(self.sched(at(2026, 10, 26, 15, 30)))
        self.assertTrue(self.sched(at(2026, 10, 26, 16, 0)))
        self.assertTrue(self.sched(at(2026, 10, 26, 19, 59)))
        self.assertFalse(self.sched(at(2026, 10, 26, 20, 0)))
        self.assertFalse(self.sched(at(2026, 10, 26, 21, 59)))

    def test_second_shift_is_10pm_to_1am_and_belongs_to_start_day(self):
        self.assertTrue(self.sched(at(2026, 10, 26, 22, 0)))
        self.assertTrue(self.sched(at(2026, 10, 27, 0, 59)))
        self.assertFalse(self.sched(at(2026, 10, 27, 1, 0)))
        self.assertEqual(self.ns["_human_shift_window"](at(2026, 10, 27, 0, 30)), ("second", datetime_date(2026, 10, 26)))
        self.assertEqual(self.ns["_human_shift_effective_date"](at(2026, 10, 27, 0, 30)), datetime_date(2026, 10, 26))
        self.assertEqual(self.ns["_human_shift_effective_date"](at(2026, 10, 27, 9, 0)), datetime_date(2026, 10, 27))

    def test_weekend_rules(self):
        # Friday late shift runs into Saturday morning; Saturday night has none.
        self.assertTrue(self.sched(at(2026, 10, 30, 23, 0)))
        self.assertTrue(self.sched(at(2026, 10, 31, 0, 30)))
        self.assertFalse(self.sched(at(2026, 10, 31, 17, 0)))
        self.assertFalse(self.sched(at(2026, 10, 31, 23, 0)))
        self.assertFalse(self.sched(at(2026, 11, 1, 0, 30)))
        self.assertFalse(self.sched(at(2026, 11, 1, 17, 0)))

    def test_four_shifts_per_week_and_reconcile_waits_for_1am(self):
        self.assertIn("HUMAN_SHIFT_MIN_WEEKLY = 4", SRC)
        self.assertIn("through = (local_now - timedelta(hours=1)).date() - timedelta(days=1)", SRC)
        self.assertIn('"windows_clocked"', SRC)


class SubadminShiftTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({"_human_subadmin_shift_active", "_human_shift_local_now"}, {"HUMAN_SUBADMIN_PAYMENT_SHIFTS"})
        self.f = self.ns["_human_subadmin_shift_active"]

    def test_donotgrowweary_and_priscilla_shifts(self):
        for email in ("donotgrowweary95@gmail.com", "priscillamuthoni010@gmail.com"):
            self.assertTrue(self.f(email, at(2026, 10, 26, 15, 0)))
            self.assertFalse(self.f(email, at(2026, 10, 26, 20, 0)))
            self.assertFalse(self.f(email, at(2026, 10, 26, 21, 30)))
            self.assertTrue(self.f(email, at(2026, 10, 26, 22, 0)))
            self.assertTrue(self.f(email, at(2026, 10, 27, 0, 59)))
            self.assertFalse(self.f(email, at(2026, 10, 27, 1, 0)))
            self.assertFalse(self.f(email, at(2026, 10, 31, 16, 0)))


class WordsLogTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({"_human_subadmin_word_rate_milli", "_human_subadmin_word_entry_counts", "_human_subadmin_shift_active", "_human_shift_local_now"},
                       {"HUMAN_SUBADMIN_PAYMENT_SHIFTS", "HUMAN_SUBADMIN_WORD_RATES_MILLI_KES", "HUMAN_SUBADMIN_WORD_SHIFT_GATED"})

    def test_rates(self):
        rate = self.ns["_human_subadmin_word_rate_milli"]
        self.assertEqual(rate("donotgrowweary95@gmail.com"), 350)
        self.assertEqual(rate("priscillamuthoni010@gmail.com"), 300)
        self.assertEqual(rate("gracenyaitara@gmail.com"), 400)
        self.assertEqual(rate("someone@example.com"), 0)
        self.assertEqual(1000 * rate("donotgrowweary95@gmail.com"), 350000)  # 1,000 words = KES 350

    def test_only_two_are_shift_gated(self):
        counts = self.ns["_human_subadmin_word_entry_counts"]
        self.assertTrue(counts("donotgrowweary95@gmail.com", at(2026, 10, 26, 17, 0)))
        self.assertFalse(counts("donotgrowweary95@gmail.com", at(2026, 10, 26, 21, 0)))
        self.assertFalse(counts("priscillamuthoni010@gmail.com", at(2026, 10, 26, 8, 0)))
        self.assertTrue(counts("gracenyaitara@gmail.com", at(2026, 10, 26, 8, 0)))
        self.assertFalse(counts("nobody@example.com", at(2026, 10, 26, 17, 0)))

    def test_wiring(self):
        self.assertIn('@app.post("/human-transcription/subadmin/words-log")', SRC)
        self.assertIn('@app.get("/human-transcription/subadmin/words-log")', SRC)
        self.assertIn('"category": "words_entry"', SRC)
        self.assertIn('"payout_status": "accruing"', SRC)


class GeneralProofreadingPausedTests(unittest.TestCase):
    def test_blocker_pauses_general_jobs_only(self):
        ns = load({"_proofreading_agent_blocker"}, {"HUMAN_GENERAL_PROOFREADING_ENABLED"})
        ns["_proofreading_agent_scope"] = lambda job: "audio"
        ns["_human_is_split_job"] = lambda job: False
        f = ns["_proofreading_agent_blocker"]
        self.assertIn("paused", f({"job_type": "general_job"}))
        self.assertNotIn("paused", f({"job_type": "template_job", "status": "submitted", "transcript": "x"}))
        ns["HUMAN_GENERAL_PROOFREADING_ENABLED"] = True
        self.assertNotIn("paused", f({"job_type": "general_job", "status": "submitted", "transcript": "x"}))


if __name__ == "__main__":
    unittest.main()
