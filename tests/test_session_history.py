import tempfile
import unittest
from pathlib import Path

from agent_scaffold.session import MarkdownSessionHistoryStore, SessionMessage


class SessionHistoryTests(unittest.TestCase):
    def test_markdown_session_history_store_appends_and_reads_messages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MarkdownSessionHistoryStore(temp_dir)
            session_id = "session-1"

            store.append(
                session_id,
                SessionMessage(
                    role="user",
                    content="remember this",
                    metadata={"type": "user_input"},
                ),
            )
            store.append(
                session_id,
                SessionMessage(
                    role="assistant",
                    content="remembered",
                    metadata={"type": "final_answer", "completed": True},
                ),
            )

            messages = store.list(session_id)
            path = Path(temp_dir) / "session-1.md"

            self.assertTrue(path.exists())
            self.assertEqual([message.role for message in messages], ["user", "assistant"])
            self.assertEqual(messages[0].content, "remember this")
            self.assertTrue(messages[1].metadata["completed"])


if __name__ == "__main__":
    unittest.main()
