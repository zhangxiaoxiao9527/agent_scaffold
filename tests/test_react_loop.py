import os
import unittest

from agent_scaffold.llm_call import LLMClient, Message, MiniMaxLLMProvider
from agent_scaffold.memory import InMemoryStore
from agent_scaffold.process import ReActAgent, ReActConfig
from agent_scaffold.session import SessionManager, SessionStatus
from agent_scaffold.tool_register import ToolRegistry


class ReActLoopTests(unittest.TestCase):
    def test_react_loop_real_call_returns_final_answer(self):

        llm_client = LLMClient(
            {"minimax": MiniMaxLLMProvider()},
            default_provider="minimax",
        )
        session_manager = SessionManager()
        memory = InMemoryStore()
        memory.save("user-1", "The user prefers short answers.")

        result = ReActAgent(
            llm_client=llm_client,
            tool_registry=ToolRegistry(),
            memory_store=memory,
            session_manager=session_manager,
            config=ReActConfig(
                max_steps=2,
                system_prompt=(
                    "You are a ReAct agent. Return JSON only. "
                    "Do not use tools. The JSON must have keys thought and final_answer."
                ),
            ),
        ).run(user_id="user-1", query="Say hello in five words or fewer.")

        session = session_manager.get(result.session_id)

        self.assertTrue(result.completed)
        self.assertTrue(result.answer.strip())
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertIsNotNone(result.steps[0].final_answer)

    def test_parse_response_supports_json_code_block(self):
        parsed = ReActAgent._parse_response(
            '```json\n{"thought": "done", "final_answer": "ok"}\n```'
        )

        self.assertEqual(parsed["thought"], "done")
        self.assertEqual(parsed["final_answer"], "ok")


if __name__ == "__main__":
    unittest.main()
