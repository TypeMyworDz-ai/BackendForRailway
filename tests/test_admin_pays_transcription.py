import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("main.py").read_text()
TREE = ast.parse(SRC)


def _fn(name):
    return next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _ns(comp=()):
    ns = {"COMP_ACCESS_EMAILS": list(comp), "ADMIN_EMAILS": ["typemywordz@gmail.com"]}
    ns["is_admin_user"] = lambda e: str(e or "").strip().lower() in ns["ADMIN_EMAILS"]
    for name in ("is_comp_access_user", "credits_exempt", "transcription_credits_exempt"):
        exec(compile(ast.Module(body=[_fn(name)], type_ignores=[]), "main.py", "exec"), ns)
    return ns


class AdminPaysForTranscription(unittest.TestCase):
    def test_super_admin_is_not_exempt_from_transcription_charges(self):
        ns = _ns()
        self.assertFalse(ns["transcription_credits_exempt"]("typemywordz@gmail.com"))
        self.assertFalse(ns["transcription_credits_exempt"]("TypeMyworDz@Gmail.com "))

    def test_ordinary_users_not_exempt(self):
        self.assertFalse(_ns()["transcription_credits_exempt"]("client@example.com"))

    def test_only_an_explicit_complimentary_account_is_exempt(self):
        ns = _ns(comp=["free@example.com"])
        self.assertTrue(ns["transcription_credits_exempt"]("free@example.com"))

    def test_transcription_charge_forces_charge_for_admin(self):
        self.assertIn("force_charge=not transcription_credits_exempt(user_email)", SRC)


if __name__ == "__main__":
    unittest.main()
