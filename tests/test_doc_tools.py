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


class PdfAndAudioToolsTests(unittest.TestCase):
    @staticmethod
    def _pdf(pages=3):
        from PIL import Image

        imgs = [Image.new("RGB", (400, 500), (200 + i * 10, 100, 100)) for i in range(pages)]
        out = io.BytesIO()
        imgs[0].save(out, format="PDF", save_all=True, append_images=imgs[1:])
        return out.getvalue()

    def test_merge_and_split(self):
        from pypdf import PdfReader

        _, _, merged = doc_tools.merge_pdfs([("a.pdf", self._pdf(2)), ("b.pdf", self._pdf(3))])
        self.assertEqual(len(PdfReader(io.BytesIO(merged)).pages), 5)
        with self.assertRaises(doc_tools.ConversionError):
            doc_tools.merge_pdfs([("a.pdf", self._pdf(1))])
        name, mime, data = doc_tools.split_pdf("m.pdf", merged, "extract", "1-2, 5")
        self.assertEqual(len(PdfReader(io.BytesIO(data)).pages), 3)
        name, mime, data = doc_tools.split_pdf("m.pdf", merged, "each")
        self.assertTrue(name.endswith(".zip"))
        name, mime, data = doc_tools.split_pdf("m.pdf", merged, "ranges", "1-2, 3-")
        self.assertTrue(name.endswith(".zip"))
        with self.assertRaises(doc_tools.ConversionError):
            doc_tools.split_pdf("m.pdf", merged, "extract", "9")

    def test_ranges(self):
        self.assertEqual(doc_tools.parse_page_ranges("1-3, 5, 8-", 10), [(1, 3), (5, 5), (8, 10)])
        with self.assertRaises(doc_tools.ConversionError):
            doc_tools.parse_page_ranges("x", 10)

    def test_compress_never_grows(self):
        raw = self._pdf(2)
        name, mime, data, original = doc_tools.compress_pdf("a.pdf", raw, "strong")
        self.assertLessEqual(len(data), original)
        self.assertTrue(data[:5] == b"%PDF-")

    def test_audio_convert(self):
        import shutil, wave, struct

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed here")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"".join(struct.pack("<h", int(8000 * ((i // 40) % 2 - 0.5))) for i in range(16000)))
        name, mime, data = doc_tools.audio_convert("beep.wav", buf.getvalue(), "mp3", "64")
        self.assertEqual(name, "beep.mp3")
        self.assertGreater(len(data), 500)
        with self.assertRaises(doc_tools.ConversionError):
            doc_tools.audio_convert("x.txt", b"not audio", "mp3")
