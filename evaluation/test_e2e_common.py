import unittest

from e2e_common import aggregate_numeric, estimate_tokens, lexical_scores
from run_e2e_answer_eval import RAG_INSTRUCTIONS, build_messages, reusable_result


class E2ECommonTest(unittest.TestCase):
    def test_token_estimate_is_explicit_and_stable(self):
        self.assertEqual(estimate_tokens("通信原理"), 2)

    def test_lexical_scores(self):
        scores = lexical_scores("QPSK主瓣宽度是一半", "QPSK的主瓣宽度为一半")
        self.assertGreater(scores["reference_char_f1"], 0.8)

    def test_concise_prompt_is_used(self):
        messages = build_messages("什么是QPSK？", "教材证据", RAG_INSTRUCTIONS["concise"])
        self.assertIn(RAG_INSTRUCTIONS["concise"], messages[-1]["content"])
        self.assertNotIn("思考分析", messages[-1]["content"])

    def test_resume_rejects_a_different_rag_prompt(self):
        legacy = {"status": "success", "variant": "e02", "prompt": {"rag_mode": "legacy"}}
        no_rag = {"status": "success", "variant": "no_rag", "prompt": {"rag_mode": "not_applicable"}}
        self.assertFalse(reusable_result(legacy, "concise"))
        self.assertTrue(reusable_result(no_rag, "concise"))

    def test_aggregate_includes_tail_percentiles(self):
        aggregate = aggregate_numeric([{"x": value} for value in range(1, 101)], ("x",))
        self.assertEqual(aggregate["p50"], 50.5)
        self.assertAlmostEqual(aggregate["p99"], 99.01)


if __name__ == "__main__":
    unittest.main()
