import asyncio
import re
import unittest
import uuid
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


def load():
    start = SRC.index('BROADCAST_DOC = (')
    end = SRC.index('@app.get("/human-transcription/broadcast")')
    ns = {"re": re, "uuid": uuid, "HTTPException": HTTPException, "time": __import__("time")}
    exec(SRC[start:end], ns)
    return ns


class Broadcasts(unittest.TestCase):
    def test_workers_keep_the_monday_message_by_default_and_sub_admins_do_not_get_it(self):
        ns = load()
        self.assertEqual(ns["BROADCAST_DEFAULTS"]["worker"][0]["title"], "Monday shifts")
        self.assertIn("Mondays are our slow shift days", ns["BROADCAST_DEFAULTS"]["worker"][0]["text"])
        self.assertEqual(ns["BROADCAST_DEFAULTS"]["subadmin"], [])

    def test_clean_items_trims_and_requires_https_links(self):
        ns = load()
        items = ns["_broadcast_clean_items"]([
            {"title": "  Update ", "text": " Hello ", "link_url": "https://example.com/a", "link_label": ""},
            {"title": "", "text": ""},
            {"title": "Hidden", "text": "x", "active": False},
        ])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["title"], "Update")
        self.assertEqual(items[0]["link_label"], "Open link")
        self.assertFalse(items[1]["active"])
        for bad in ("javascript:alert(1)", "http://example.com", "https://a b.com"):
            with self.assertRaises(HTTPException):
                ns["_broadcast_clean_items"]([{"title": "t", "text": "x", "link_url": bad}])

    def test_item_limit(self):
        ns = load()
        many = [{"title": str(i), "text": "x"} for i in range(30)]
        self.assertEqual(len(ns["_broadcast_clean_items"](many)), ns["BROADCAST_MAX_ITEMS"])

    def test_endpoints_are_scoped(self):
        self.assertIn('@app.put("/api/admin/broadcasts")\nasync def admin_put_broadcasts(request: Request):\n    admin = _require_admin(request)', SRC)
        self.assertIn('elif actor["role"] == "admin" and is_human_subadmin(actor.get("email")):', SRC)


if __name__ == "__main__":
    unittest.main()
