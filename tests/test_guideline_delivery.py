import asyncio
import ast
import time
import unittest
from html import escape
from io import BytesIO
from pathlib import Path
from docx import Document

SOURCE = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

def selected(*names):
    return [node for node in TREE.body if getattr(node, "name", None) in set(names)]

class SavedGuidelineDeliveryTests(unittest.TestCase):
    def test_html_renderer_uses_exact_text_and_escapes_markup(self):
        namespace = {"escape": escape}
        functions = selected("_guidelines_text_to_html_and_text")
        exec(compile(ast.Module(body=functions, type_ignores=[]), "main.py", "exec"), namespace)
        original = "General Guidelines\nUse A & B.\n\nKeep <spoken> wording."
        html, text = namespace["_guidelines_text_to_html_and_text"](original)
        self.assertEqual(text, original)
        self.assertIn("Use A &amp; B.", html)
        self.assertIn("Keep &lt;spoken&gt; wording.", html)
        self.assertIn('class="guideline-spacer"', html)

    def test_download_docx_is_built_from_the_saved_text(self):
        namespace = {"Document": Document, "BytesIO": BytesIO}
        exec(compile(ast.Module(body=selected("_guidelines_text_to_docx"), type_ignores=[]), "main.py", "exec"), namespace)
        raw = namespace["_guidelines_text_to_docx"]("First rule.\n\nSecond rule.")
        document = Document(BytesIO(raw))
        self.assertEqual([paragraph.text for paragraph in document.paragraphs], ["First rule.", "", "Second rule."])

    def test_worker_guideline_page_prefers_admin_saved_firestore_text(self):
        class Snapshot:
            exists = True
            def to_dict(self):
                return {"text": "General Guidelines\nSaved admin rule."}
        class DocumentRef:
            def get(self):
                return Snapshot()
        class Collection:
            def document(self, name):
                self.document_name = name
                return DocumentRef()
        class Database:
            def collection(self, name):
                self.collection_name = name
                return Collection()
        class QuietLogger:
            def warning(self, *args, **kwargs):
                pass
            def exception(self, *args, **kwargs):
                pass

        namespace = {
            "asyncio": asyncio, "time": time, "escape": escape, "logger": QuietLogger(),
            "db": Database(), "_GUIDELINES_CACHE": {"at": 0.0, "html": "", "text": ""},
            "_training_assets_bucket": lambda: None, "TRAINING_ASSET_STORAGE_PREFIX": "training-materials/",
            "_docx_to_html_and_text": lambda raw: ("<p>fallback</p>", "fallback"), "HTTPException": Exception,
        }
        helpers = selected("_guidelines_text_to_html_and_text", "_guidelines_content")
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "main.py", "exec"), namespace)
        html, text = asyncio.run(namespace["_guidelines_content"]())
        self.assertEqual(text, "General Guidelines\nSaved admin rule.")
        self.assertIn("Saved admin rule.", html)
        self.assertEqual(namespace["_GUIDELINES_CACHE"]["text"], text)

    def test_admin_save_invalidates_page_cache_and_download_uses_firestore_text(self):
        save_source = next(node for node in TREE.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "admin_put_ai_guidelines")
        download_source = next(node for node in TREE.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "trainee_training_material")
        save_text = ast.unparse(save_source)
        download_text = ast.unparse(download_source)
        self.assertIn("_GUIDELINES_CACHE.update", save_text)
        self.assertIn("admin_settings", download_text)
        self.assertIn("_guidelines_text_to_docx", download_text)

if __name__ == "__main__":
    unittest.main()
