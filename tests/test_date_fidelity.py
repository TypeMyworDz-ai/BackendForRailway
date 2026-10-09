import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()


def _load():
    start = SRC.index("_DATE_MONTHS = {")
    end = SRC.index("\n\n\nHUMAN_GENERAL_SELF_CORRECTION_GUIDANCE", start)
    ns = {"re": re}
    exec(SRC[start:end], ns)
    return ns


NS = _load()
restore = NS["_human_restore_dictated_dates"]


class DateRestoreTests(unittest.TestCase):
    def test_numeric_date_restored_to_dictated_month_name(self):
        out = restore("The visit was on 05/01/2026 at noon.", ["The visit was on May 1, 2026 at noon."])
        self.assertEqual(out, "The visit was on May 1, 2026 at noon.")

    def test_iso_and_short_year_restored(self):
        refs = ["Hearing on September 3, 2025."]
        self.assertEqual(restore("Hearing 2025-09-03.", refs), "Hearing September 3, 2025.")
        self.assertEqual(restore("Hearing 9/3/25.", refs), "Hearing September 3, 2025.")
        self.assertEqual(restore("Hearing 9/3/25.", ["Hearing Sept 3, 2025."]), "Hearing Sept 3, 2025.")

    def test_dictated_numeric_date_is_untouched(self):
        self.assertEqual(restore("Dated 05/01/2026.", ["Dated 05/01/2026 and May 1, 2026."]), "Dated 05/01/2026.")

    def test_unmatched_date_left_alone(self):
        self.assertEqual(restore("Seen 06/02/2026.", ["Seen May 1, 2026."]), "Seen 06/02/2026.")

    def test_empty_inputs(self):
        self.assertEqual(restore("Text 05/01/2026", []), "Text 05/01/2026")
        self.assertEqual(restore("", ["May 1, 2026"]), "")


class PromptTests(unittest.TestCase):
    def test_rules_present_in_prompts(self):
        self.assertIn("HUMAN_DATE_FIDELITY_RULES = (", SRC)
        self.assertGreaterEqual(SRC.count("HUMAN_DATE_FIDELITY_RULES"), 3)
        self.assertIn("5b. DATES STAY AS DICTATED", SRC)
        self.assertIn("STRUCTURE CHECK only", SRC)
        self.assertNotIn("_human_review_full_audio_assemblyai", SRC)


if __name__ == "__main__":
    unittest.main()

