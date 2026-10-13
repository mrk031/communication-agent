import sys
import unittest
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parents[1] / "tasks_distribute"
sys.path.insert(0, str(ROUTER_DIR))

from intent_router import classify_business_intents


class IntentRouterTest(unittest.TestCase):
    def route(self, question):
        return classify_business_intents(question)

    def test_ordinary_concept_question_only_uses_main_answer(self):
        result = self.route("OFDM为什么能够抗频率选择性衰落？")
        self.assertEqual(result["大模型概念回答"], 1)
        self.assertEqual(sum(v for k, v in result.items() if k != "问题内容"), 1)

    def test_explicit_question_bank_and_teacher_history(self):
        self.assertEqual(self.route("给我找几道QPSK练习题")["通信原理题目库"], 1)
        self.assertEqual(self.route("查一下老师以前的答疑记录")["通信原理课堂教师答疑记录"], 1)

    def test_hardware_word_in_explanation_does_not_open_lab(self):
        question = "扩频系统做粗同步时，串行相关检测硬件上省不省？"
        self.assertEqual(self.route(question)["通信原理课堂在线硬件实验"], 0)

    def test_explicit_hardware_navigation(self):
        self.assertEqual(self.route("打开QPSK硬件实验")["通信原理课堂在线硬件实验"], 1)
        self.assertEqual(self.route("星座图实验")["通信原理课堂在线硬件实验"], 1)

    def test_interactive_other_course_and_ideology(self):
        self.assertEqual(self.route("给我一个OFDM软件仿真案例")["通信原理课堂交互式案例"], 1)
        other = self.route("找一个数字电路实验")
        self.assertEqual(other["其他课程实验案例"], 1)
        self.assertEqual(other["通信原理课堂在线硬件实验"], 0)
        self.assertEqual(self.route("介绍香农的生平故事")["思政教育案例"], 1)


if __name__ == "__main__":
    unittest.main()
