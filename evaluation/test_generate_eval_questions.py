import unittest

from generate_eval_questions import build_passages, endpoint, stratified_sample, validate_item


class GenerationHelpersTest(unittest.TestCase):
    def test_endpoint(self):
        self.assertEqual(endpoint("https://example.test/v1"), "https://example.test/v1/chat/completions")
        self.assertEqual(endpoint("https://example.test/v1/chat/completions"), "https://example.test/v1/chat/completions")

    def test_build_and_sample_passages(self):
        chunks = [
            {"chunk_id": str(i), "document_name": "book.pdf", "content": "第1章 通信系统的基本组成包括发送端信道和接收端。" * 3}
            for i in range(6)
        ]
        passages = build_passages(chunks, target_chars=120, max_chars=300)
        self.assertGreaterEqual(len(passages), 2)
        first = stratified_sample(passages, 2, seed=7)
        second = stratified_sample(passages, 2, seed=7)
        self.assertEqual([x["passage_id"] for x in first], [x["passage_id"] for x in second])

    def test_validation_accepts_grounded_item(self):
        passage = {
            "chunk_ids": ["c1"],
            "chunks": [{"chunk_id": "c1", "content": "数字通信系统通常包括信源编码、信道编码和调制等环节。"}],
        }
        item = {
            "source_question": "数字通信系统包括哪些环节？",
            "paraphrased_question": "一个数字通信系统通常由哪些主要处理步骤构成？",
            "reference_answer": "通常包括信源编码、信道编码和调制等环节。",
            "evidence_quote": "数字通信系统通常包括信源编码、信道编码和调制等环节。",
            "supporting_chunk_ids": ["c1"],
            "question_type": "definition",
            "difficulty": "easy",
        }
        self.assertEqual(validate_item(item, passage), [])


if __name__ == "__main__":
    unittest.main()
