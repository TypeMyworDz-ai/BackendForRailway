import io
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doc_tools


def _rate_limit_fn():
    src = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
    start = src.index("def _tools_rate_limited")
    end = src.index("\n\n\n", start)
    ns = {"time": time, "_tools_hits": {}, "TOOLS_RATE_LIMIT": 3, "TOOLS_RATE_WINDOW_SECONDS": 100}
    exec(src[start:end], ns)
    return ns["_tools_rate_limited"]


class ToolsTests(unittest.TestCase):
    def test_rate_limit(self):
        fn = _rate_limit_fn()
        hits = {}
        self.assertFalse(fn("a", 0, hits, 3, 100))
        self.assertFalse(fn("a", 1, hits, 3, 100))
        self.assertFalse(fn("a", 2, hits, 3, 100))
        self.assertTrue(fn("a", 3, hits, 3, 100))
        self.assertFalse(fn("b", 3, hits, 3, 100))
        self.assertFalse(fn("a", 500, hits, 3, 100))

    def test_unsupported(self):
        with self.assertRaises(doc_tools.ConversionError):
            doc_tools.convert("x.xyz", b"abc", "pdf")
        with self.assertRaises(doc_tools.ConversionError):
            doc_tools.convert("x.png", b"abc", "gif")

    def test_image_to_pdf_and_back(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
        name, mime, data = doc_tools.convert("pic.png", buf.getvalue(), "pdf")
        self.assertEqual((name, mime), ("pic.pdf", "application/pdf"))
        self.assertTrue(data.startswith(b"%PDF"))
        name, mime, data = doc_tools.convert("pic.pdf", data, "jpg")
        self.assertEqual(mime, "image/jpeg")

    def test_safe_stem(self):
        self.assertEqual(doc_tools.safe_stem("../My Report (final).docx"), "My Report final")


if __name__ == "__main__":
    unittest.main()
