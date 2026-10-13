import ast
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()


def load():
    start = SRC.index("MODEL_ROUTES = {")
    end = SRC.index('@app.get("/api/admin/model-routing")')
    ns = {
        "time": __import__("time"),
        "ASK_MODEL_CATALOGUE": [
            {"id": "gemini-3.8-flash", "provider": "gemini", "label": "G"},
            {"id": "gpt-5.6-terra", "provider": "openai", "label": "T"},
            {"id": "mistral-large-4", "provider": "mistral", "label": "Mistral Large 4 (Preview)"},
        ],
        "HUMAN_GENERAL_AGENT_MODEL_CHAIN": (("gemini-3.8-flash", "gemini"), ("gpt-5.6-terra", "openai")),
        "HUMAN_GENERAL_SUBADMIN_AGENT_MODEL_CHAIN": (("gpt-5.6-terra", "openai"), ("gemini-3.8-flash", "gemini")),
        "is_human_subadmin": lambda email: str(email or "").strip().lower().startswith("subadmin@"),
        "HUMAN_AUDIO_AGENT_MODEL_CHAIN": (("a", "x"),), "HUMAN_PDF_AGENT_MODEL_CHAIN": (("a", "x"),),
        "HUMAN_TEXT_MESSAGES_MODEL_CHAIN": (("a", "x"),), "WORKER_DRAFT_FORMAT_MODEL_CHAIN": (("a", "x"),),
        "WORKER_DRAFT_PROOFREAD_MODEL_CHAIN": (("a", "x"),), "AI_REVIEW_MODEL_CHAIN": (("a", "x"),),
        "logger": __import__("logging").getLogger("t"),
    }
    exec(SRC[start:end], ns)
    return ns


class ModelRouting(unittest.TestCase):
    def test_general_defaults_are_separate_for_super_admin_and_sub_admins(self):
        self.assertIn('HUMAN_GENERAL_AGENT_MODEL_CHAIN = (("gemini-3.8-flash", "gemini"), ("gpt-5.6-terra", "openai"))', SRC)
        self.assertIn('HUMAN_GENERAL_SUBADMIN_AGENT_MODEL_CHAIN = (("gpt-5.6-terra", "openai"), ("gemini-3.8-flash", "gemini"))', SRC)
        self.assertIn('"general_subadmin": ("General Jobs agent (Sub-admins)"', SRC)

    def test_general_route_overrides_are_independent(self):
        ns = load()
        ns["_MODEL_ROUTE_CACHE"].update({"at": __import__("time").time(), "routes": {
            "general": ["gemini-3.8-flash"], "general_subadmin": ["gpt-5.6-terra"],
        }})
        self.assertEqual(ns["_route_chain"]("general", ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"]), (("gemini-3.8-flash", "gemini"),))
        self.assertEqual(ns["_route_chain"]("general_subadmin", ns["HUMAN_GENERAL_SUBADMIN_AGENT_MODEL_CHAIN"]), (("gpt-5.6-terra", "openai"),))

    def test_mistral_large_4_is_scoped_to_super_admin_general_jobs(self):
        ns = load()
        general = ns["_model_route_options"]("general")
        self.assertEqual(general["mistral-large-4"], ("mistral", "Mistral Large 4 (Preview)"))
        for route_key in ("general_subadmin", "template", "letter", "pdf", "text_messages", "worker_format", "worker_proofread", "proofreading_agent", "letter_review"):
            self.assertNotIn("mistral-large-4", ns["_model_route_options"](route_key))
        ns["_MODEL_ROUTE_CACHE"].update({"at": __import__("time").time(), "routes": {
            "general": ["mistral-large-4", "gemini-3.8-flash"],
            "general_subadmin": ["mistral-large-4"],
        }})
        self.assertEqual(ns["_route_chain"]("general", ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"]), (("mistral-large-4", "mistral"), ("gemini-3.8-flash", "gemini")))
        self.assertEqual(ns["_route_chain"]("general_subadmin", ns["HUMAN_GENERAL_SUBADMIN_AGENT_MODEL_CHAIN"]), ns["HUMAN_GENERAL_SUBADMIN_AGENT_MODEL_CHAIN"])
        self.assertNotIn("mistral-large-4", ns["_model_route_options"]())
        tree = ast.parse(SRC)
        catalogue_node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ASK_MODEL_CATALOGUE" for target in node.targets))
        catalogue = {model["id"]: model for model in ast.literal_eval(catalogue_node.value)}
        self.assertEqual((catalogue["mistral-large-4"]["provider"], catalogue["mistral-large-4"]["tier"]), ("mistral", "premium"))

    def test_only_verified_general_job_assigner_role_selects_subadmin_route(self):
        ns = load()
        route_key = ns["_human_ai_agent_model_route_key"]
        self.assertEqual(route_key("general-gpt", "subadmin@example.com"), "general_subadmin")
        self.assertEqual(route_key("general-gpt", "admin@example.com"), "general")
        self.assertEqual(route_key("template-claude", "subadmin@example.com"), "")

    def test_override_used_and_unknown_ignored(self):
        ns = load()
        ns["_MODEL_ROUTE_CACHE"].update({"at": __import__("time").time(), "routes": {"general": ["gpt-5.6-terra", "bogus", "claude-sonnet-5-5"]}})
        chain = ns["_route_chain"]("general", ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"])
        self.assertEqual(chain, (("gpt-5.6-terra", "openai"), ("claude-sonnet-5-5", "claude")))

    def test_default_when_no_override(self):
        ns = load()
        ns["_MODEL_ROUTE_CACHE"].update({"at": __import__("time").time(), "routes": {}})
        self.assertEqual(ns["_route_chain"]("general", ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"]), ns["HUMAN_GENERAL_AGENT_MODEL_CHAIN"])

    def test_endpoints_require_super_admin_and_validate_route_scoped_models(self):
        for name in ("admin_get_model_routing", "admin_put_model_routing"):
            body = SRC.split("async def " + name)[1][:300]
            self.assertIn("_require_admin(request)", body)
        self.assertIn('"available_models": [{"id": model_id', SRC)
        self.assertIn("options = _model_route_options(key)", SRC.split("async def admin_put_model_routing")[1][:400])


if __name__ == "__main__":
    unittest.main()
