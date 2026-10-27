import ast, re, unittest
from datetime import datetime
from io import BytesIO

SRC = open("main.py", encoding="utf-8").read()
TREE = ast.parse(SRC)
NAMES = {"_invoice_text_width", "_invoice_pdf_escape", "_invoice_pdf_bytes", "_invoice_docx_bytes", "_gt_parse_length", "_gt_length_label", "_gt_amount_kes", "_invoice_day_label", "_invoice_in_range", "_invoice_period_text"}
CONSTS = {"HUMAN_GT_WORK_KES_PER_MINUTE"}
body = []
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in NAMES:
        body.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CONSTS for t in node.targets):
        body.append(node)


class _HTTPException(Exception):
    def __init__(self, status_code=0, detail=""):
        self.status_code, self.detail = status_code, detail


from docx import Document
ns = {"re": re, "datetime": datetime, "HTTPException": _HTTPException, "Document": Document, "BytesIO": BytesIO}
exec(compile(ast.Module(body=body, type_ignores=[]), "main.py", "exec"), ns)


class InvoiceAndGtTests(unittest.TestCase):
    def test_gt_length_parsing(self):
        p = ns["_gt_parse_length"]
        self.assertEqual(p("6:30"), 390)
        self.assertEqual(p("12"), 720)
        self.assertEqual(p("6.5"), 390)
        self.assertEqual(p("1:02:03"), 3723)
        for bad in ("", "abc", "0", "1:2:3:4"):
            with self.assertRaises(_HTTPException):
                p(bad)

    def test_gt_amount_and_label(self):
        self.assertEqual(ns["_gt_amount_kes"](600), 150)
        self.assertEqual(ns["_gt_amount_kes"](390), 98)
        self.assertEqual(ns["_gt_length_label"](390), "6:30")
        self.assertEqual(ns["_gt_length_label"](3723), "1:02:03")

    def test_pdf_and_docx(self):
        rows = [["Mon, 26 Oct 2026", "JOB-%d" % i, "1,000", "300.00"] for i in range(80)]
        pdf = ns["_invoice_pdf_bytes"]("T", ["a (b)"], ["A", "B", "C", "D"], rows, ["TOTAL", "", "80,000", "24,000.00"], [150, 150, 90, 120], ["left", "left", "right", "right"])
        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
        self.assertGreaterEqual(pdf.count(b"/Type /Page "), 2)
        docx = ns["_invoice_docx_bytes"]("T", ["x"], ["A", "B"], [["1", "2"]], ["T", "3"], ["left", "right"])
        self.assertTrue(docx.startswith(b"PK"))

    def test_range_and_routes(self):
        self.assertTrue(ns["_invoice_in_range"]("2026-10-05", "2026-10-01", "2026-10-31"))
        self.assertFalse(ns["_invoice_in_range"]("2026-11-01", "2026-10-01", "2026-10-31"))
        for route in ("/api/admin/gt-work-invoice", "/api/admin/subadmin-words-invoice", "/human-transcription/worker/gt-work", "/api/public/home-tips", "/human-transcription/subadmin/words-invoice"):
            self.assertIn(route, SRC)
        self.assertIn('BROADCAST_AUDIENCES = ("worker", "subadmin", "home")', SRC)


class ZaiThinkingTests(unittest.TestCase):
    def test_zai_never_sends_thinking_disabled(self):
        self.assertIn('if provider == "zai":\n            # GLM 5.3 Flash cannot run with thinking off', SRC)
        self.assertNotIn('if provider == "zai" or model_id == "mimo-v2.6-flash"', SRC)


class WordTimingAndRatingRouteTests(unittest.TestCase):
    def test_routes_and_word_times(self):
        self.assertIn('"rating_agent": ("Rating Agent"', SRC)
        self.assertIn('_route_chain("rating_agent", RATING_AGENT_MODEL_CHAIN)', SRC)
        self.assertGreaterEqual(SRC.count('"wt"'), 4)
        fn = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_word_start_times"][0]
        local = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), local)
        self.assertEqual(local["_word_start_times"]([{"start": 1500}, {"start": 2250}]), [1.5, 2.25])


if __name__ == "__main__":
    unittest.main()
