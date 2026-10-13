import unittest

from run_ragflow_baseline import retrieval_metrics


class RetrievalMetricsTest(unittest.TestCase):
    def test_single_gold_at_rank_two(self):
        result = retrieval_metrics(["x", "gold", "z"], ["gold"], [1, 2, 3])
        self.assertEqual(result["hit_at_1"], 0.0)
        self.assertEqual(result["recall_at_2"], 1.0)
        self.assertEqual(result["reciprocal_rank"], 0.5)

    def test_multiple_gold_recall_and_ndcg(self):
        result = retrieval_metrics(["g1", "x", "g2"], ["g1", "g2"], [1, 3])
        self.assertEqual(result["recall_at_1"], 0.5)
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertLess(result["ndcg_at_3"], 1.0)

    def test_no_gold_has_no_formal_metrics(self):
        self.assertEqual(retrieval_metrics(["x"], [], [1]), {})


if __name__ == "__main__":
    unittest.main()
