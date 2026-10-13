import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("local_evaluate", MODULE_PATH)
evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"doc_id": "gold", "content": "QPSK 四相 相移 键控 两比特", "source_type": "textbook"},
            {"doc_id": "leak", "content": "QPSK 为什么有四个相位 完整教师答案", "source_type": "teacher_qa"},
            {"doc_id": "other", "content": "模拟调幅载波振幅", "source_type": "textbook"},
        ]

    def test_holdout_is_filtered_without_mutating_corpus(self):
        retriever = evaluate.BM25Retriever(self.corpus)
        before = list(self.corpus)
        ranked = retriever.search("QPSK 为什么四个相位", 4, ["leak"])
        self.assertNotIn("leak", [row.doc_id for row in ranked])
        self.assertEqual(before, self.corpus)

    def test_retrieval_metrics(self):
        case = {
            "case_id": "1", "question": "QPSK 两比特 四相", "reference_answer": "答案",
            "gold_doc_ids": ["gold"], "holdout_doc_ids": ["leak"], "split": "test",
        }
        result = evaluate.evaluate_case(case, evaluate.BM25Retriever(self.corpus), 2)
        self.assertEqual(result["retrieval"]["recall_at_k"], 1.0)
        self.assertEqual(result["retrieval"]["mrr"], 1.0)

    def test_output_is_written_only_to_requested_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            evaluate.write_json(path, {"ok": True})
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
