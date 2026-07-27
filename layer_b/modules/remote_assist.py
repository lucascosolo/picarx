#!/usr/bin/env python3
"""SSH client for the host-side ``picarx_host_helper.py`` protocol."""
import ipaddress
import base64
import json
import os
import re
import select
import shlex
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus

try:
    import robot_config
except Exception:  # pragma: no cover
    robot_config = None

REQUEST_TOPIC = "picarx/tools/remote_assist"
RESULT_TOPIC = "picarx/tools/remote_assist/result"
STATE_CLAIM_TOPIC = "picarx/state/claim"
STATE_RELEASE_TOPIC = "picarx/state/release"
OWNER = "remote_assist"
LOCAL_HELPER_SOURCE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "picarx_host_helper.py"))
BOOTSTRAP_CODE = (
    "import base64,os,sys; p=sys.argv[1]; os.makedirs(os.path.dirname(p),exist_ok=True); "
    "open(p,'wb').write(base64.b64decode(sys.stdin.buffer.read())); os.chmod(p,0o700)"
)


def valid_host(value):
    """Accept an IP or conservative DNS hostname, never shell syntax."""
    value = str(value or "").strip()
    if len(value) > 253 or any(ord(c) < 32 or c in " /\\\t\r\n" for c in value):
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", value):
            return value
    return None


def valid_user(value):
    value = str(value or "").strip()
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}", value) else None


class RemoteSession:
    def __init__(self, ssh_bin="ssh", helper_command="picarx-host-helper",
                 connect_timeout=8.0, request_timeout=30.0, popen=None,
                 helper_source=LOCAL_HELPER_SOURCE, bootstrap=True):
        self.ssh_bin = ssh_bin
        self.helper_command = helper_command
        self.connect_timeout = float(connect_timeout)
        self.request_timeout = float(request_timeout)
        self._popen = popen or subprocess.Popen
        self.helper_source = helper_source
        self.bootstrap = bool(bootstrap)
        self.proc = None
        self.host = None
        self.remote_script = None
        self._destination = None
        self._port = None

    def _base_ssh(self, destination, port=None):
        argv = [self.ssh_bin, "-T", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes",
                "-o", f"ConnectTimeout={int(self.connect_timeout)}"]
        if port is not None:
            argv += ["-p", str(port)]
        return argv + [destination]

    def _bootstrap_helper(self, destination, port=None):
        """Copy the helper from the robot tree to a private remote temp file.

        This is a one-shot SSH command using only Python's base64 decoder on
        the host; no package install, shell profile edit, or manual host setup
        is required. A subsequent SSH channel runs the copied helper.
        """
        if not self.helper_source or not os.path.isfile(self.helper_source):
            raise RuntimeError("robot-side host helper source is missing")
        with open(self.helper_source, "rb") as stream:
            source = stream.read()
        token = uuid.uuid4().hex
        self.remote_script = f"/tmp/picarx_host_helper_{token}.py"
        argv = self._base_ssh(destination, port) + ["python3", "-c",
                                                     BOOTSTRAP_CODE, self.remote_script]
        proc = self._popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            encoded = base64.b64encode(source).decode("ascii")
            out, err = proc.communicate(encoded, timeout=self.connect_timeout + 5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            raise
        if proc.returncode != 0:
            raise RuntimeError((err or out or "could not bootstrap host helper")[-500:])
        return self.remote_script

    def connect(self, host, user=None, port=None, project_root="."):
        host = valid_host(host)
        if not host:
            raise ValueError("invalid host or IP address")
        if user is not None and not valid_user(user):
            raise ValueError("invalid SSH user")
        project_root = str(project_root or ".").strip()
        if (not project_root or len(project_root) > 500 or
                any(ord(c) < 32 for c in project_root)):
            raise ValueError("invalid remote project root")
        destination = f"{user}@{host}" if user else host
        if port is not None:
            try:
                port = int(port)
            except (TypeError, ValueError):
                raise ValueError("invalid SSH port")
            if not 1 <= port <= 65535:
                raise ValueError("invalid SSH port")
        if self.bootstrap:
            remote_script = self._bootstrap_helper(destination, port)
            # The helper is a robot-owned source file copied into a private
            # per-session path. The host need only have python3, not a package
            # installation or a pre-existing script.
            helper_argv = ["python3", "-u", remote_script,
                           "--root", project_root, "--allow-write"]
        else:
            helper_argv = shlex.split(self.helper_command)
        argv = self._base_ssh(destination, port) + helper_argv
        self.proc = self._popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        self.host = host
        self._destination = destination
        self._port = port
        return {"host": host, "user": user, "port": port,
                "project_root": project_root, "bootstrapped": self.bootstrap}

    def request(self, payload, timeout=None):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("SSH session is not connected")
        request_id = payload.get("request_id") or uuid.uuid4().hex
        payload = dict(payload, request_id=request_id)
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + (self.request_timeout if timeout is None else float(timeout))
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            ready, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if not ready:
                continue
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("SSH helper closed the session")
            response = json.loads(line)
            if response.get("request_id") in (None, request_id):
                return response
        raise TimeoutError("remote helper response timed out")

    def close(self):
        proc, self.proc = self.proc, None
        old_destination, old_port = self._destination, self._port
        old_script = self.remote_script
        self.host = None
        self._destination = None
        self._port = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Best-effort cleanup of the robot-uploaded helper. The path is a
        # random /tmp token created by this session, never caller input.
        if old_script and old_destination:
            cleanup_code = "import os,sys; os.unlink(sys.argv[1]) if os.path.exists(sys.argv[1]) else None"
            try:
                cleanup = self._popen(
                    self._base_ssh(old_destination, old_port) +
                    ["python3", "-c", cleanup_code, old_script],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1)
                cleanup.communicate(timeout=3)
            except Exception:
                pass
        self.remote_script = None


class RemoteAssist:
    def __init__(self, session=None, bus=None):
        self.bus = bus or Bus()
        self.session = session or RemoteSession()
        self.lock = threading.RLock()
        self.connected = False
        self.target = None

    def _publish(self, payload):
        message = dict(payload, ts=time.time())
        self.bus.publish(RESULT_TOPIC, message)
        # Voice-only sessions still get a useful bounded summary; full source
        # and command output remain available on the result topic/web console.
        if not message.get("ok"):
            text = "Remote assist failed: " + str(message.get("error") or "unknown error")
        elif message.get("command") == "connect":
            text = "Connected. The remote project helper is ready."
        elif message.get("command") == "disconnect":
            text = "Remote project session closed."
        elif message.get("command") == "run":
            result = message.get("result") or {}
            text = f"Remote command finished with status {result.get('returncode')}."
            output = (result.get("stderr") or result.get("stdout") or "").strip()
            if output:
                text += " " + " ".join(output.split())[:220]
        else:
            text = "Remote operation complete. I sent the details to the tools console."
        self.bus.publish("picarx/audio/speak", {"text": text[:400], "ts": time.time()})

    def _claim(self):
        self.bus.publish(STATE_CLAIM_TOPIC, {
            "owner": OWNER, "state": "REMOTE_ASSIST", "ttl": 6.0,
            "reason": "remote project session active", "ts": time.time()})

    def _release(self):
        self.bus.publish(STATE_RELEASE_TOPIC, {"owner": OWNER, "ts": time.time()})

    def on_request(self, payload):
        threading.Thread(target=self._handle, args=(dict(payload),),
                         daemon=True, name="remote-assist-request").start()

    def _handle(self, payload):
        command = str(payload.get("command") or payload.get("op") or "").lower()
        try:
            with self.lock:
                if command == "connect":
                    self.session.close()
                    target = self.session.connect(payload.get("host"), payload.get("user"),
                                                  payload.get("port"),
                                                  payload.get("project_root", "."))
                    self.target, self.connected = target, True
                    self._claim()
                    response = {"ok": True, "command": command, "target": target}
                elif command in {"disconnect", "stop"}:
                    self.session.close()
                    self.connected = False
                    self.target = None
                    self._release()
                    response = {"ok": True, "command": "disconnect"}
                elif command == "status":
                    response = {"ok": True, "command": command,
                                "connected": self.connected, "target": self.target}
                elif not self.connected:
                    raise RuntimeError("not connected; connect to a host first")
                elif command in {"list", "read", "search", "stat", "preview_patch", "apply_patch", "run"}:
                    request = dict(payload)
                    request["op"] = command
                    if command in {"apply_patch", "run"} and not payload.get("confirmed"):
                        raise PermissionError("explicit confirmation is required for remote writes/commands")
                    self._claim()
                    result = self.session.request(request)
                    response = {"ok": bool(result.get("ok")), "command": command,
                                "result": result.get("result"), "error": result.get("error")}
                else:
                    raise ValueError(f"unsupported remote command: {command}")
            self._publish(response)
        except Exception as e:
            if command == "connect":
                # A failed bootstrap or helper launch may have opened a
                # partial SSH process; do not leave it attached or report a
                # stale session after a failed connection attempt.
                try:
                    self.session.close()
                except Exception:
                    pass
                self.connected = False
                self.target = None
            self._publish({"ok": False, "command": command, "error": str(e)[:500]})

    def run(self):
        self.bus.subscribe(REQUEST_TOPIC, self.on_request)
        print(f"Remote assist active on {REQUEST_TOPIC}")
        while True:
            if self.connected:
                # Keep REMOTE_ASSIST active between file operations so a
                # speaking/RC/gesture transition has an observable owner and
                # the lease cannot expire during a long debugging pause.
                self._claim()
            time.sleep(2)


if __name__ == "__main__":
    RemoteAssist().run()
