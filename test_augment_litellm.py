import unittest
from augment_litellm import generation_payload, reasoning_status, accepted_text, resolve_models


class AugmentationTests(unittest.TestCase):
    def test_explicit_aliases(self):
        self.assertEqual(resolve_models(
            ["claude-opus-4-8", "claude-opus-4-6", "claude-fable-5"],
            {"claude-opus-4.8", "claude-opus-4.6", "claude-fable-5.1"}),
            {"claude-opus-4-8": "claude-opus-4.8", "claude-opus-4-6": "claude-opus-4.6"})

    def test_no_reasoning_request(self):
        p = generation_payload("model", "prompt", 768)
        self.assertEqual(p["reasoning_effort"], "none")
        self.assertEqual(p["max_tokens"], 768)
        self.assertEqual(generation_payload("m", "p", 768, "disable")["reasoning_effort"], "disable")

    def test_usage(self):
        self.assertEqual(reasoning_status({}), (None, "unknown"))
        for n, status in [(0, "reported_zero"), (100, "nonzero")]:
            self.assertEqual(reasoning_status({"usage": {"completion_tokens_details":
                                                         {"reasoning_tokens": n}}}), (n, status))

    def test_final_text_only(self):
        response = {"model": "m", "choices": [{"finish_reason": "stop", "message": {
            "content": "x" * 100, "reasoning_content": "not for training"}}]}
        self.assertEqual(accepted_text(response, "m"), ("x" * 100, None))
        self.assertIsNone(accepted_text(response, "other")[0])


if __name__ == "__main__":
    unittest.main()
