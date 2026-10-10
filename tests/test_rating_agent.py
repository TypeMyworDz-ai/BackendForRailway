import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
TREE = ast.parse(SRC)
FUNCS = {n.name: n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _routes():
    found = {}
    for name, node in FUNCS.items():
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") in {"get", "post", "put", "delete"} and dec.args:
                found[(dec.func.attr, ast.literal_eval(dec.args[0]))] = name
    return found


class RatingAgentTests(unittest.TestCase):
    def test_parse_validates_rating_and_comments(self):
        import json
        ns = {"json": json}
        exec(compile(ast.Module(body=[FUNCS["_parse_json_object"], FUNCS["_rating_agent_parse"]], type_ignores=[]), "main.py", "exec"), ns)
        parse = ns["_rating_agent_parse"]
        self.assertEqual(parse('{"rating": 4, "comments": "Good work."}'), (4, "Good work."))
        self.assertEqual(parse('```json\n{"rating": 4.4, "comments": "ok"}\n```')[0], 4)
        for bad in ('{"rating": 7, "comments": "x"}', '{"rating": 3, "comments": ""}', "no json"):
            with self.assertRaises(Exception):
                parse(bad)

    def test_only_super_admin_routes_and_gemini_38(self):
        routes = _routes()
        for key in (("get", "/api/admin/worker-ratings"), ("put", "/api/admin/worker-ratings/{rating_id}"),
                    ("post", "/api/admin/worker-ratings/{rating_id}/apply"), ("post", "/api/admin/worker-ratings/{rating_id}/dismiss"),
                    ("get", "/api/admin/general-job-guidelines"), ("put", "/api/admin/general-job-guidelines")):
            self.assertIn(key, routes)
            self.assertIn("_require_admin", ast.unparse(FUNCS[routes[key]]))
        self.assertIn('RATING_AGENT_MODEL_CHAIN = (("gemini-3.8-flash", "gemini"),)', SRC)
        self.assertIn("HUMAN_RATING_EVERY_N = 3", SRC)

    def test_old_rating_routes_are_gone(self):
        routes = set(_routes())
        self.assertNotIn(("post", "/human-transcription/jobs/{job_id}/rate-part"), routes)
        self.assertNotIn(("post", "/human-transcription/jobs/{job_id}/proofreader-ratings"), routes)
        review = ast.unparse(FUNCS["human_admin_review"])
        self.assertNotIn("worker_rating", review)
        self.assertNotIn("rating", ast.unparse(FUNCS["human_admin_ai_review"]).replace("_rating_agent", ""))  # AI review no longer rates

    def test_every_submission_path_counts_for_rating(self):
        self.assertEqual(ast.unparse(FUNCS["human_worker_submit"]).count("_rating_agent_on_submission("), 4)

    def test_general_jobs_use_their_own_guidelines(self):
        self.assertIn("'general_job'", ast.unparse(FUNCS["_guidelines_for_job"]))
        for fn in ("_human_ai_agent_generate", "_human_worker_format_ai_draft", "human_worker_ai_proofread_draft", "human_admin_ai_review"):
            self.assertIn("_guidelines_for_job", ast.unparse(FUNCS[fn]))


if __name__ == "__main__":
    unittest.main()
