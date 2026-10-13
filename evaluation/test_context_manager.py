import sys
import unittest
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parents[1] / "tasks_distribute"
sys.path.insert(0, str(ROUTER_DIR))

from context_manager import select_recent_context


class ContextManagerTest(unittest.TestCase):
    def test_keeps_recent_complete_turns(self):
        history = []
        for index in range(6):
            history.extend([
                {"sender_role": "user", "content": f"问题{index}"},
                {"sender_role": "assistant", "content": f"回答{index}"},
            ])
        selected, stats = select_recent_context(history, token_budget=100, max_messages=4)
        self.assertEqual([item["content"] for item in selected], ["问题4", "回答4", "问题5", "回答5"])
        self.assertEqual(stats["kept_messages"], 4)

    def test_budget_drops_old_turns(self):
        history = [
            {"role": "user", "content": "旧问题" * 20},
            {"role": "assistant", "content": "旧回答" * 20},
            {"role": "user", "content": "最近问题" * 10},
            {"role": "assistant", "content": "最近回答" * 10},
        ]
        selected, stats = select_recent_context(history, token_budget=45, max_messages=10)
        self.assertTrue(any("最近问题" in item["content"] for item in selected))
        self.assertGreater(stats["dropped_messages"], 0)
        self.assertLessEqual(stats["estimated_tokens"], 45)

    def test_truncates_oversized_latest_message(self):
        selected, stats = select_recent_context(
            [{"role": "user", "content": "A" * 1000}],
            token_budget=50,
            max_messages=4,
            max_message_chars=2000,
        )
        self.assertEqual(len(selected), 1)
        self.assertIn("历史消息已截断", selected[0]["content"])
        self.assertLessEqual(stats["estimated_tokens"], 50)


if __name__ == "__main__":
    unittest.main()

