import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from remote_assist import RemoteAssist, RemoteSession, valid_host, valid_user  # noqa: E402


def load_helper():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "picarx_host_helper.py")
    spec = importlib.util.spec_from_file_location("picarx_host_helper_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HostHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "main.py"), "w") as f:
            f.write("print('hello')\n")
        with open(os.path.join(self.root, "notes.txt"), "w") as f:
            f.write("needle here\nsecond line\n")
        self.mod = load_helper()
        self.helper = self.mod.HostHelper(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scoped_list_read_search_and_path_escape_rejection(self):
        self.assertEqual(self.helper.read({"path": "main.py"})["text"], "print('hello')")
        self.assertEqual(len(self.helper.search({"pattern": "needle"})["results"]), 1)
        self.assertTrue(any(e["name"] == "main.py"
                            for e in self.helper.list({"path": "."})["entries"]))
        with self.assertRaises(self.mod.HelperError):
            self.helper.read({"path": "../outside"})

    def test_symlink_escape_is_rejected(self):
        outside = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            outside.write("secret\n")
            outside.close()
            os.symlink(outside.name, os.path.join(self.root, "outside-link"))
            with self.assertRaises(self.mod.HelperError):
                self.helper.read({"path": "outside-link"})
        finally:
            try:
                os.unlink(outside.name)
            except OSError:
                pass

    def test_binary_and_oversized_files_are_not_read(self):
        with open(os.path.join(self.root, "binary.bin"), "wb") as f:
            f.write(b"x\x00y")
        with self.assertRaises(self.mod.HelperError):
            self.helper.read({"path": "binary.bin"})

    def test_commands_are_allowlisted_and_outputs_bounded(self):
        result = self.helper.run({"argv": ["python3", "-m", "unittest", "--help"],
                                  "timeout_sec": 5})
        self.assertEqual(result["returncode"], 0)
        with self.assertRaises(self.mod.HelperError):
            self.helper.run({"command": "sh -c 'echo unsafe'"})

    def test_writes_require_explicit_host_write_enablement(self):
        with self.assertRaises(self.mod.HelperError):
            self.helper.apply_patch({"patch": "diff --git a/main.py b/main.py\n"})

    def test_standalone_jsonl_process_needs_no_host_install(self):
        helper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "tools", "picarx_host_helper.py")
        requests = [
            {"op": "status", "request_id": "status-1"},
            {"op": "read", "path": "main.py", "request_id": "read-1"},
        ]
        proc = subprocess.run(
            [sys.executable, helper_path, "--root", self.root],
            input="\n".join(json.dumps(r) for r in requests) + "\n",
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        responses = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([r["request_id"] for r in responses], ["status-1", "read-1"])
        self.assertEqual(responses[1]["result"]["text"], "print('hello')")


class FakeSession:
    def __init__(self):
        self.connected = False
        self.requests = []

    def connect(self, host, user=None, port=None, project_root="."):
        self.connected = True
        return {"host": host, "user": user, "port": port,
                "project_root": project_root, "bootstrapped": False}

    def request(self, payload):
        self.requests.append(payload)
        return {"ok": True, "result": {"path": payload.get("path", ".")}}

    def close(self):
        self.connected = False


class RemoteAssistTests(unittest.TestCase):
    def test_host_and_user_validation(self):
        self.assertEqual(valid_host("192.168.1.20"), "192.168.1.20")
        self.assertEqual(valid_host("dev-box.local"), "dev-box.local")
        self.assertIsNone(valid_host("192.168.1.20; rm -rf /"))
        self.assertIsNone(valid_host("../etc"))
        self.assertEqual(valid_user("lucas"), "lucas")
        self.assertIsNone(valid_user("lucas;rm"))

    def test_remote_writes_and_commands_need_confirmation(self):
        session = FakeSession()
        remote = RemoteAssist(session=session)
        remote._handle({"command": "connect", "host": "192.168.1.20"})
        self.assertTrue(remote.connected)
        remote._handle({"command": "read", "path": "main.py"})
        self.assertEqual(session.requests[-1]["op"], "read")
        before = len(session.requests)
        remote._handle({"command": "run", "argv": ["python3", "-m", "pytest"]})
        self.assertEqual(len(session.requests), before)
        remote._handle({"command": "run", "argv": ["python3", "-m", "pytest"],
                        "confirmed": True})
        self.assertEqual(session.requests[-1]["op"], "run")
        remote._handle({"command": "disconnect"})
        self.assertFalse(remote.connected)

    def test_session_bootstraps_robot_owned_helper_before_starting_it(self):
        class Proc:
            def __init__(self, bootstrap):
                self.bootstrap = bootstrap
                self.returncode = 0
                self.stdin = type("In", (), {"write": lambda *_: None,
                                               "flush": lambda *_: None,
                                               "close": lambda *_: None})()
                self.stdout = type("Out", (), {"readline": lambda *_: ""})()
                self.stderr = type("Err", (), {})()

            def communicate(self, *_args, **_kwargs):
                return "", ""

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, **_kwargs):
                return 0

            def kill(self):
                pass

        calls = []
        def fake_popen(argv, **kwargs):
            calls.append(argv)
            return Proc(len(calls) == 1)

        session = RemoteSession(popen=fake_popen)
        target = session.connect("192.168.1.20", "lucas", project_root="~/project")
        self.assertTrue(target["bootstrapped"])
        self.assertEqual(len(calls), 2)
        self.assertIn("python3", calls[0])
        self.assertIn("-c", calls[0])
        self.assertIn("--root", calls[1])
        self.assertIn("~/project", calls[1])
        self.assertIn("--allow-write", calls[1])


if __name__ == "__main__":
    unittest.main()
