import json
import tempfile
import unittest
from pathlib import Path

from agent_scaffold.trace import log_event


class TraceTests(unittest.TestCase):
    def test_log_event_writes_newest_event_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"

            log_event(str(path), "first", data={"order": 1})
            log_event(str(path), "second", data={"order": 2})

            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual([event["type"] for event in events], ["second", "first"])
            self.assertEqual(events[0]["data"], {"order": 2})


if __name__ == "__main__":
    unittest.main()
