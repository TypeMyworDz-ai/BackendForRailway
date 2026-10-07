"""Pure tests for two-image Text Messages jobs (no production data)."""
import ast
import re
import unittest
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class ImageJobPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        wanted = {"_human_pdf_job_image_metas", "human_image_tat_seconds"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        scope = {"PDF_JOB_TAT_SECONDS": 1200}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "main_subset", "exec"), scope)
        cls.scope = scope

    def test_constants(self):
        self.assertIn("TEXT_MESSAGES_IMAGES_PER_JOB = 1", self.source)
        self.assertIn("TEXT_MESSAGES_WORKER_PAY_KES = 50", self.source)
        self.assertEqual(re.search(r"^PDF_JOB_WORKER_PAY_KES = (\d+)", self.source, re.M).group(1), "100")

    def test_two_image_job_exposes_both_images(self):
        job = {"pdf_image": {"storage_path": "a"}, "pdf_images": [{"storage_path": "a"}, {"storage_path": "b"}]}
        metas = self.scope["_human_pdf_job_image_metas"](job)
        self.assertEqual([m["storage_path"] for m in metas], ["a", "b"])

    def test_single_image_job_exposes_one_image(self):
        f = self.scope["_human_pdf_job_image_metas"]
        self.assertEqual(len(f({"pdf_image": {"storage_path": "a"}})), 1)
        self.assertEqual(f({}), [])

    def test_tat_scales_with_images(self):
        self.assertEqual(self.scope["human_image_tat_seconds"](2), 2400)


if __name__ == "__main__":
    unittest.main()
