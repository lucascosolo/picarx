import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
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

    def test_active_command_can_be_canceled_without_ending_helper(self):
        helper = self.mod.HostHelper(
            self.root, command_prefixes=((sys.executable, "-c"),))
        result = {}

        def run_command():
            result["run"] = helper.handle({
                "op": "run", "request_id": "run-1", "confirmed": True,
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"]})

        worker = threading.Thread(target=run_command)
        worker.start()
        deadline = time.time() + 3
        while time.time() < deadline:
            with helper._active_lock:
                if helper._active_command is not None:
                    break
            time.sleep(0.01)
        canceled = helper.handle({"op": "cancel", "request_id": "cancel-1",
                                  "target_request_id": "run-1"})
        worker.join(timeout=3)
        self.assertTrue(canceled["canceled"])
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["run"]["canceled"])
        self.assertEqual(helper.handle({"op": "status"})["requests"], 2)

    def test_jsonl_mutating_operations_require_confirmation_and_logs_are_bounded(self):
        with self.assertRaises(self.mod.HelperError):
            self.helper.handle({"op": "run", "argv": ["git", "status"]})
        self.helper.handle({"op": "status", "request_id": "s1"})
        logs = self.helper.handle({"op": "logs", "limit": 5})
        self.assertTrue(logs["entries"])
        self.assertEqual(logs["entries"][0]["op"], "run")
        self.assertFalse(any("argv" in entry for entry in logs["entries"]))

    def test_writes_require_explicit_host_write_enablement(self):
        with self.assertRaises(self.mod.HelperError):
            self.helper.apply_patch({"patch": "diff --git a/main.py b/main.py\n"})

    def test_coding_session_writes_are_scoped_atomic_and_hash_checked(self):
        writable = self.mod.HostHelper(self.root, allow_write=True)
        created = writable.handle({"op": "write_file", "path": "src/new.py",
                                    "content": "print('new')\n", "confirmed": True})
        self.assertTrue(created["created"])
        self.assertEqual(writable.read({"path": "src/new.py"})["text"],
                         "print('new')")
        with self.assertRaises(self.mod.HelperError):
            writable.handle({"op": "write_file", "path": "main.py",
                             "content": "changed\n", "expected_sha256": "bad",
                             "confirmed": True})
        updated = writable.handle({
            "op": "write_file", "path": "main.py", "content": "changed\n",
            "expected_sha256": hashlib.sha256(
                b"print('hello')\n").hexdigest(), "confirmed": True})
        self.assertFalse(updated["created"])
        with self.assertRaises(self.mod.HelperError):
            writable.handle({"op": "delete_path", "path": "src/new.py"})
        deleted = writable.handle({"op": "delete_path", "path": "src/new.py",
                                    "confirmed": True})
        self.assertTrue(deleted["deleted"])
        logs = writable.handle({"op": "logs", "limit": 20})["entries"]
        self.assertFalse(any("print('new')" in repr(entry) or "changed\\n" in repr(entry)
                             for entry in logs))

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
        self.cancel_calls = 0
        self.password = None
        self.close_calls = 0
        self.request_error = None

    def connect(self, host, user=None, port=None, project_root=".", password=None):
        self.connected = True
        self.password = password
        return {"host": host, "user": user, "port": port,
                "project_root": project_root, "bootstrapped": False}

    def request(self, payload):
        self.requests.append(payload)
        if self.request_error is not None:
            raise self.request_error
        return {"ok": True, "result": {"path": payload.get("path", ".")}}

    def close(self):
        self.connected = False
        self.close_calls += 1

    def cancel(self, target_request_id=None):
        self.cancel_calls += 1
        return True


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

    def test_remote_cancel_uses_transport_without_disconnecting(self):
        session = FakeSession()
        remote = RemoteAssist(session=session)
        remote._handle({"command": "connect", "host": "192.168.1.20"})
        remote._handle({"command": "cancel", "silent": True})
        self.assertEqual(session.cancel_calls, 1)
        self.assertTrue(remote.connected)

    def test_transport_failure_clears_claim_and_session_authority(self):
        session = FakeSession()
        session.request_error = RuntimeError("SSH helper closed the session")
        bus = harness.FakeBus()
        remote = RemoteAssist(session=session, bus=bus)
        remote._handle({"command": "connect", "host": "192.168.1.20"})
        remote._handle({"command": "begin_coding", "confirmed": True})
        coding_id = remote.coding_session_id
        remote._handle({"command": "authorize_write", "confirmed": True})
        self.assertTrue(remote.write_authorized)

        remote._handle({"command": "read", "path": "main.py"})

        self.assertFalse(remote.connected)
        self.assertIsNone(remote.target)
        self.assertIsNone(remote.coding_session_id)
        self.assertFalse(remote.write_authorized)
        self.assertGreaterEqual(session.close_calls, 1)
        result = bus.last("picarx/tools/remote_assist/result")
        self.assertTrue(result["disconnected"])
        self.assertEqual(bus.last("picarx/state/release")["owner"],
                         "remote_assist")
        self.assertNotEqual(coding_id, remote.coding_session_id)

    def test_thinking_destructive_work_requires_active_coding_session(self):
        session = FakeSession()
        bus = harness.FakeBus()
        remote = RemoteAssist(session=session, bus=bus)
        remote._handle({"command": "connect", "host": "192.168.1.20"})
        remote._handle({"command": "begin_coding", "confirmed": True})
        coding_id = remote.coding_session_id
        remote._handle({"command": "authorize_write", "confirmed": True,
                        "source": "thinking"})
        self.assertFalse(remote.write_authorized)
        remote._handle({"command": "authorize_write", "confirmed": True,
                        "source": "thinking", "coding_session_id": coding_id})
        self.assertTrue(remote.write_authorized)
        remote._handle({"command": "run", "argv": ["python3", "-m", "pytest"],
                        "confirmed": True, "source": "thinking",
                        "coding_session_id": coding_id})
        self.assertEqual(session.requests[-1]["op"], "run")
        remote._handle({"command": "end_coding", "source": "thinking",
                        "coding_session_id": coding_id})
        self.assertIsNone(remote.coding_session_id)
        self.assertIsNone(bus.last("picarx/tools/remote_assist/result")
                          ["result"]["coding_session_id"])

    def test_remote_connect_forwards_password_only_to_session(self):
        session = FakeSession()
        bus = harness.FakeBus()
        remote = RemoteAssist(session=session, bus=bus)
        remote._handle({"command": "connect", "host": "192.168.1.20",
                        "password": "temporary secret"})
        self.assertEqual(session.password, "temporary secret")
        published = bus.last("picarx/tools/remote_assist/result")
        self.assertNotIn("password", published)
        self.assertNotIn("temporary secret", repr(published))

    def test_remote_coding_file_edit_requires_session_authorization(self):
        session = FakeSession()
        remote = RemoteAssist(session=session)
        remote._handle({"command": "connect", "host": "192.168.1.20"})
        before = len(session.requests)
        remote._handle({"command": "write_file", "path": "main.py",
                        "content": "print('edited')\n"})
        self.assertEqual(len(session.requests), before)
        remote._handle({"command": "authorize_write", "confirmed": True})
        remote._handle({"command": "write_file", "path": "main.py",
                        "content": "print('edited')\n", "expected_sha256": "abc"})
        self.assertEqual(session.requests[-1]["op"], "write_file")
        self.assertTrue(session.requests[-1]["confirmed"])

    def test_write_authorization_persists_until_revoke_or_disconnect(self):
        session = FakeSession()
        remote = RemoteAssist(session=session)
        remote._handle({"command": "connect", "host": "192.168.1.20"})
        before = len(session.requests)
        remote._handle({"command": "apply_patch", "patch": "diff"})
        self.assertEqual(len(session.requests), before)
        self.assertFalse(remote.write_authorized)
        remote._handle({"command": "authorize_write", "confirmed": True})
        self.assertTrue(remote.write_authorized)
        remote._handle({"command": "apply_patch", "patch": "diff"})
        self.assertEqual(session.requests[-1]["op"], "apply_patch")
        self.assertTrue(session.requests[-1]["confirmed"])
        remote._handle({"command": "apply_patch", "patch": "diff"})
        self.assertTrue(session.requests[-1]["confirmed"])
        remote._handle({"command": "revoke_write"})
        self.assertFalse(remote.write_authorized)
        count = len(session.requests)
        remote._handle({"command": "rollback"})
        self.assertEqual(len(session.requests), count)
        remote._handle({"command": "disconnect"})
        self.assertFalse(remote.write_authorized)

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

    def test_password_is_pipe_only_and_not_returned_or_put_in_argv(self):
        class Proc:
            def __init__(self):
                self.returncode = 0
                self.stdin = type("In", (), {"close": lambda *_: None})()
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
        session = RemoteSession(popen=lambda argv, **kwargs: (
            calls.append((argv, kwargs)) or Proc()))
        session._password = "temporary secret"
        # Replace executable discovery at the narrow seam; this test must not
        # need the optional host utility installed in the test environment.
        import remote_assist
        saved = remote_assist.shutil.which
        remote_assist.shutil.which = lambda name: "/usr/bin/sshpass"
        try:
            result = session._popen_ssh(session._base_ssh("host"),
                                       stdin=subprocess.PIPE)
        finally:
            remote_assist.shutil.which = saved
        argv, kwargs = calls[0]
        self.assertEqual(argv[:2], ["/usr/bin/sshpass", "-d"])
        self.assertNotIn("temporary secret", argv)
        self.assertIn("BatchMode=no", argv)
        self.assertEqual(len(kwargs["pass_fds"]), 1)
        self.assertEqual(result.returncode, 0)

    def test_password_is_not_in_connection_metadata(self):
        session = RemoteSession(bootstrap=False,
                                helper_command="python3 helper.py")
        calls = []
        session._popen = lambda argv, **kwargs: (
            calls.append((argv, kwargs)) or type("P", (), {
                "returncode": 0, "stdin": None, "stdout": None,
                "stderr": None, "poll": lambda self: None})())
        import remote_assist
        saved = remote_assist.shutil.which
        remote_assist.shutil.which = lambda name: "/usr/bin/sshpass"
        try:
            target = session.connect("192.168.1.20", password="secret")
        finally:
            remote_assist.shutil.which = saved
        self.assertNotIn("password", target)
        self.assertNotIn("secret", repr(calls))
        session.close()


if __name__ == "__main__":
    unittest.main()
