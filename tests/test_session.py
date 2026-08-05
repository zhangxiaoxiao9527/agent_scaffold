import unittest

import tempfile

from agent_scaffold.session import MarkdownSessionHistoryStore, SessionManager, SessionStatus


class SessionTests(unittest.TestCase):
    def test_create_session_activates_by_default(self):
        manager = SessionManager()

        session = manager.create(context={"user_id": "u1"})

        self.assertEqual(session.status, SessionStatus.ACTIVE)
        self.assertEqual(manager.get_context(session.id, "user_id"), "u1")

    def test_context_and_data_can_be_updated(self):
        manager = SessionManager()
        session = manager.create()

        manager.set_context(session.id, "topic", "agent")
        manager.set_data(session.id, "turn_count", 1)

        self.assertEqual(manager.get_context(session.id, "topic"), "agent")
        self.assertEqual(manager.get_data(session.id, "turn_count"), 1)

    def test_valid_and_invalid_state_transitions(self):
        manager = SessionManager()
        session = manager.create()

        paused = manager.transition(session.id, SessionStatus.PAUSED)
        self.assertEqual(paused.status, SessionStatus.PAUSED)

        manager.transition(session.id, SessionStatus.CLOSED)
        with self.assertRaises(ValueError):
            manager.transition(session.id, SessionStatus.ACTIVE)

    def test_list_and_delete_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(
                history_store=MarkdownSessionHistoryStore(temp_dir)
            )
            session = manager.create()

            self.assertEqual(len(manager.list()), 1)
            manager.delete(session.id)

            self.assertEqual(manager.list(), [])

    def test_session_manager_appends_and_reads_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(
                history_store=MarkdownSessionHistoryStore(temp_dir)
            )
            session = manager.create()

            manager.append_history(session.id, "user", "hello")
            manager.append_history(session.id, "assistant", "hi")

            history = manager.get_history(session.id)

            self.assertEqual([message.content for message in history], ["hello", "hi"])


if __name__ == "__main__":
    unittest.main()
