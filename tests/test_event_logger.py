import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import event_logger  # noqa: E402


class EventLoggerResolutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "events.db")
        self.old_dir = event_logger.DB_DIR
        self.old_path = event_logger.DB_PATH
        event_logger.DB_DIR = self.tmp
        event_logger.DB_PATH = self.db
        self.logger = event_logger.EventLogger()

    def tearDown(self):
        self.logger.conn.close()
        try:
            os.remove(self.db)
            os.remove(self.db + "-wal")
            os.remove(self.db + "-shm")
            os.rmdir(self.tmp)
        except OSError:
            pass
        event_logger.DB_DIR = self.old_dir
        event_logger.DB_PATH = self.old_path

    def test_disambiguation_request_is_persisted_as_json(self):
        payload = {
            "scan_id": "scan-1", "resolution_id": "resolution-1",
            "probe_id": "probe-1", "candidate_scores": [
                {"location_id": 2, "similarity": 0.81}],
        }
        self.logger.on_disambiguation_needed(payload)
        row = self.logger.conn.execute(
            "SELECT topic, payload_json FROM events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(row[0], "picarx/exploration/disambiguation_needed")
        self.assertEqual(json.loads(row[1]), payload)

    def test_run_subscribes_to_disambiguation_requests(self):
        self.logger.run = lambda: None  # avoid the module's infinite loop
        self.logger.bus.subscribe(
            "picarx/exploration/disambiguation_needed",
            self.logger.on_disambiguation_needed)
        self.logger.bus.deliver("picarx/exploration/disambiguation_needed", {
            "probe_id": "probe-2"})
        self.assertEqual(self.logger.conn.execute(
            "SELECT COUNT(*) FROM events WHERE topic = ?",
            ("picarx/exploration/disambiguation_needed",)).fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
