import ast, pathlib, re, unittest

SRC = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text()


def catalogue():
    start = SRC.index("ASK_MODEL_CATALOGUE = [")
    end = SRC.index("# Model ids that were renamed or replaced.")
    return ast.literal_eval(SRC[start:end].split("=", 1)[1].strip())


class StreamlinedModelTests(unittest.TestCase):
    def setUp(self):
        self.cat = {m["id"]: m for m in catalogue()}

    def test_removed_models_are_gone_from_the_catalogue(self):
        for gone in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-5-5", "claude-haiku-4-5-20251001",
                     "gemini-3.1-flash-lite", "deepseek-v4-flash", "deepseek-v4-pro"):
            self.assertNotIn(gone, self.cat)

    def test_kept_and_new_models_are_present(self):
        for keep in ("gpt-6.1-sol", "gpt-6-luna", "claude-sonnet-5-5", "claude-haiku-5-5", "gemini-3.8-flash",
                     "gemini-3.5-flash-lite", "deepseek-v4-1-flash", "glm-5.3-flash", "mimo-v2.6-pro",
                     "mimo-v2.6-flash", "grok-4.3", "grok-4.7"):
            self.assertIn(keep, self.cat)

    def test_argon_is_admin_routing_only(self):
        self.assertNotIn("gemini-4-argon", self.cat)
        self.assertIn('"gemini-4-argon": ("gemini"', SRC)

    def test_expensive_new_models_are_in_the_advanced_list(self):
        for premium in ("mimo-v2.6-pro", "grok-4.3", "grok-4.7", "gpt-6.1-sol", "claude-sonnet-5-5"):
            self.assertEqual(self.cat[premium]["tier"], "premium")
        for standard in ("glm-5.3-flash", "mimo-v2.6-flash", "deepseek-v4-1-flash", "gpt-6-luna"):
            self.assertEqual(self.cat[standard]["tier"], "standard")

    def test_every_catalogue_provider_can_be_called_and_is_key_gated(self):
        for m in self.cat.values():
            if m["provider"] in ("zai", "xiaomi", "xai"):
                self.assertIn('"%s": ("https://' % m["provider"], SRC)
                self.assertIn('provider == "%s"' % m["provider"], SRC)

    def test_keys_are_read_from_environment_never_written_in_code(self):
        for env in ("ZAI_API_KEY", "XIAOMI_API_KEY", "XAI_API_KEY"):
            self.assertIn('os.environ.get("%s")' % env, SRC)
        self.assertNotRegex(SRC, r"grok_[0-9a-f]{20}")
        self.assertNotRegex(SRC, r"sk-sph[0-9a-z]{10}")
        self.assertNotRegex(SRC, r"[0-9a-f]{32}\.[A-Za-z0-9]{16}")

    def test_old_ids_map_to_successors(self):
        for old in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-5-5", "gemini-3.1-flash-lite",
                    "deepseek-v4-pro", "deepseek-v4-flash", "claude-haiku-4-5-20251001"):
            self.assertRegex(SRC, r'"%s": "' % re.escape(old))

    def test_graces_fallback_is_gpt_61_sol(self):
        self.assertIn('HUMAN_GENERAL_GRACE_AGENT_MODEL_CHAIN = (("gemini-3.8-flash", "gemini"), ("gpt-6.1-sol", "openai"))', SRC)

    def test_stt_testers_are_wired(self):
        self.assertIn("XAI_STT_TESTER_EMAIL = 'njokigituku@gmail.com'", SRC)
        self.assertIn("XIAOMI_STT_TESTER_EMAIL = 'jaymoest2@gmail.com'", SRC)
        self.assertIn("attempt_tester_stt(tier_1_service, 1)", SRC)


class XaiPayloadTests(unittest.TestCase):
    def test_words_become_timed_segments(self):
        start = SRC.index("def _xai_result_from_payload")
        end = SRC.index("async def transcribe_with_xai")
        ns = {"XAI_STT_MODEL": "grok-voice-transcribe-2.0"}
        exec(SRC[start:end], ns)
        out = ns["_xai_result_from_payload"]({"text": "Hello there. How are you?", "language": "en", "duration": 3.4,
            "words": [{"text": "Hello", "start": 0.1, "end": 0.4}, {"text": "there.", "start": 0.4, "end": 0.9},
                      {"text": "How", "start": 1.0, "end": 1.2}, {"text": "are", "start": 1.2, "end": 1.4}, {"text": "you?", "start": 1.4, "end": 1.8}]}, False)
        self.assertEqual([s["text"] for s in out["segments"]], ["Hello there.", "How are you?"])
        self.assertEqual(out["segments"][1]["start"], 1.0)
        self.assertEqual(out["timings_source"], "xAI")


if __name__ == "__main__":
    unittest.main()
