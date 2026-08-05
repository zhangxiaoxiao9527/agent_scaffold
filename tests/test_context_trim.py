import unittest

from agent_scaffold.context_trim import ContextTrimRequest, PassthroughContextTrimmer
from agent_scaffold.llm_call import Message


class ContextTrimTests(unittest.TestCase):
    def test_passthrough_context_trimmer_keeps_messages(self):
        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
        ]

        result = PassthroughContextTrimmer().trim(
            ContextTrimRequest(
                messages=messages,
                tools=[{"type": "function"}],
                session_id="session-1",
                user_id="user-1",
                query="hello",
                step_index=1,
            )
        )

        self.assertEqual(result.messages, messages)
        self.assertEqual(result.metadata["strategy"], "passthrough")
        self.assertEqual(result.metadata["original_message_count"], 2)


if __name__ == "__main__":
    unittest.main()
