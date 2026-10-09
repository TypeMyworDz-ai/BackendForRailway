import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()


def load():
    start = SRC.index("MODEL_ROUTES = {")
    end = SRC.index('@app.get("/api/admin/model-routing")')
    ns = {
        "time": __import__("time"),
        "ASK_MODEL_CATALOGUE": [{"id": "gemini-3.8-flash", "provider": "gemini", "label": "G"}, {"id": "gpt-5.6-terra", "provider": "openai", "label": "T"}],
        "HUMAN_GENERAL_AGENT_MODEL_CHAIN": (("gemini-3.8-flash", "gemini"), ("gpt-5.6-terra", "openai")),
        "HUMAN_AUDIO_AGENT_MODEL_CHAIN": (("a", "x"),), "HUMAN_PDF_AGENT_MODEL_CHAIN": (("a", "x"),),
        "HUMAN_TEXT_MESSAGES_MODEL_CHAIN": (("a", "x"),), "WORKER_DRAFT_FORMAT_MODEL_CHAIN": (("a", "x"),),
        "WORKER_DRAFT_PROOFREAD_MODEL_CHAIN": (("a", "x"),), "AI_REVIEW_MODEL_CHAIN": (("a", "x"),),
        "logger": __import__("logging").getLogger("t"),
    }
    exec(SRC[start:end], ns)
    return ns


class ModelRouting(unittest.TestCase):
    def test_general_default_is_gemini_then_terra(self):
        self.assertIn('HUMAN_GENERAL_AGENT_MODEL_CHAIN = (("gemini-3.8-flash", "gemini"), ("gpt-5.6-terra", "openai"))', SRC)

    def test_override_used_and_unknown_ignored(self):
        ns = load()
        ns["_MODEL_ROUTE_CACHE"].update({"at": __import__("time").time(), "routes": {"general": ["gpt-5.6-terra", "bogus", "claude-sonnet-5-5"]}})
        chain = ns["_route_chain"]("general", ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"])
        self.assertEqual(chain, (("gpt-5.6-terra", "openai"), ("claude-sonnet-5-5", "claude")))

    def test_default_when_no_override(self):
        ns = load()
        ns["_MODEL_ROUTE_CACHE"].update({"at": __import__("time").time(), "routes": {}})
        self.assertEqual(ns["_route_chain"]("general", ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"]), ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"])

    def test_endpoints_require_super_admin(self):
        for name in ("admin_get_model_routing", "admin_put_model_routing"):
            body = SRC.split("async def " + name)[1][:200]
            self.assertIn("_require_admin(request)", body)


if __name__ == "__main__":
    unittest.main()
