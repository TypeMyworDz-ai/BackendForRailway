"""Focused, production-data-free tests for individual Human Work rates."""
import ast
import asyncio
import re
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"
SOURCE = MAIN.read_text(encoding="utf-8")


class FakeHTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeSnapshot:
    def __init__(self, data=None):
        self.data = data
        self.exists = data is not None

    def to_dict(self):
        return self.data


class FakeDoc:
    def __init__(self, data):
        self.data = data

    def get(self):
        return FakeSnapshot(self.data)


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def document(self, uid):
        return FakeDoc(self.records.get(uid))


class FakeDB:
    def __init__(self, overrides):
        self.overrides = overrides

    def collection(self, _name):
        return FakeCollection(self.overrides)


class FakeLogger:
    def exception(self, *_args, **_kwargs):
        pass


def _rate_namespace(overrides):
    tree = ast.parse(SOURCE)
    wanted = {"_validate_human_transcriber_rate", "_human_worker_transcriber_rate_kes"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted or isinstance(n, ast.AsyncFunctionDef) and n.name in wanted]
    ns = {
        "re": re, "asyncio": asyncio, "db": FakeDB(overrides), "HTTPException": FakeHTTPException,
        "logger": FakeLogger(), "HUMAN_WORKER_RATE_OVERRIDES_COLLECTION": "human_worker_rate_overrides",
        "HUMAN_LEGACY_STANDARD_PAYOUT_KES": 25, "HUMAN_MIN_TRANSCRIBER_RATE_KES": 1,
        "HUMAN_MAX_TRANSCRIBER_RATE_KES": 500,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAIN), "exec"), ns)
    return ns


class IndividualWorkerRateTests(unittest.TestCase):
    def test_individual_rate_overrides_quote_and_missing_override_keeps_quote(self):
        ns = _rate_namespace({"worker-a": {"transcriber_rate_kes_per_minute": 40}})
        rate = ns["_human_worker_transcriber_rate_kes"]
        self.assertEqual(asyncio.run(rate("worker-a", 30)), 40)
        self.assertEqual(asyncio.run(rate("worker-b", 30)), 30)
        self.assertEqual(asyncio.run(rate("worker-b", None)), 25)

    def test_invalid_saved_override_fails_closed(self):
        ns = _rate_namespace({"worker-a": {"transcriber_rate_kes_per_minute": 900}})
        with self.assertRaises(FakeHTTPException) as error:
            asyncio.run(ns["_human_worker_transcriber_rate_kes"]("worker-a", 30))
        self.assertEqual(error.exception.status_code, 503)

    def test_override_routes_are_main_admin_only_and_validate_worker(self):
        tree = ast.parse(SOURCE)
        routes = {
            node.name: node for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"human_admin_get_worker_rate_overrides", "human_admin_update_worker_rate_override"}
        }
        self.assertEqual(set(routes), {"human_admin_get_worker_rate_overrides", "human_admin_update_worker_rate_override"})
        for name, node in routes.items():
            calls = {item.func.id for item in ast.walk(node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
            self.assertIn("_require_admin", calls, name)
        update = ast.unparse(routes["human_admin_update_worker_rate_override"])
        self.assertIn('workerApproved', update)
        self.assertIn('status_code=404', update)
        self.assertIn("_validate_human_transcriber_rate", update)
        self.assertIn("ref.delete", update)

    def test_audio_worker_submission_uses_override_for_split_and_whole_jobs(self):
        tree = ast.parse(SOURCE)
        submit = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "human_worker_submit")
        calls = [node for node in ast.walk(submit) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_human_worker_transcriber_rate_kes"]
        self.assertEqual(len(calls), 2)
        source = ast.unparse(submit)
        self.assertIn("if is_pdf_job", source)


if __name__ == "__main__":
    unittest.main()
