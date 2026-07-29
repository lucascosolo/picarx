#!/usr/bin/env python3
"""Safety-gated repository update orchestration.

This module owns the *decision* to update the robot checkout; the Layer B
orchestrator owns process quiescing and re-executes itself after a successful
fast-forward.  Keeping git and rollback logic here makes the dangerous part
testable without a broker, systemd, camera, or robot hardware.

The updater never accepts a branch or ref from an incoming request.  It uses
the configured remote/branch, refuses tracked local changes, requires a safe
RobotState, and leaves a small ignored marker so the newly started process can
verify its first boot and roll back automatically if startup health fails.
"""
import json
import os
import subprocess
import sys
import threading
import time


CONTROL_TOPIC = "picarx/system/update/control"
STATUS_TOPIC = "picarx/system/update/status"
STATE_TOPIC = "picarx/state/current"
HEALTH_TOPIC = "picarx/health/state"

SAFE_STATES = {"IDLE", "OBJECT_DETECTION"}
DEFAULT_POLL_INTERVAL_SEC = 3600.0
DEFAULT_COMMAND_TIMEOUT_SEC = 30.0
DEFAULT_HEALTH_TIMEOUT_SEC = 30.0


class UpdateError(RuntimeError):
    """A recoverable update/preflight failure."""


def safe_to_update(state_payload, health_payload=None):
    """Return whether the observed robot state permits maintenance.

    ``OBJECT_DETECTION`` is accepted as a quiescible state: the orchestrator
    stops the detector and camera consumers before touching the checkout.  Any
    state that can drive, speak, own a remote session, or indicate a safety
    stop is rejected instead of being interrupted underneath the user.
    """
    if not isinstance(state_payload, dict):
        return False
    state = str(state_payload.get("state") or "").upper()
    if state not in SAFE_STATES:
        return False
    claims = state_payload.get("claims") or []
    if not isinstance(claims, list):
        return False
    for claim in claims:
        if not isinstance(claim, dict):
            return False
        if str(claim.get("state") or "").upper() not in SAFE_STATES:
            return False
    if isinstance(health_payload, dict) and health_payload.get("low_power"):
        return False
    return True


class RepositoryUpdater:
    """Run one-at-a-time, configured-branch updates with rollback markers."""

    def __init__(self, repo_path, remote="origin", branch="master",
                 enabled=False, poll_interval=DEFAULT_POLL_INTERVAL_SEC,
                 command_timeout=DEFAULT_COMMAND_TIMEOUT_SEC,
                 health_timeout=DEFAULT_HEALTH_TIMEOUT_SEC,
                 marker_path=None, publish=None, quiesce=None, resume=None,
                 restart=None, health_check=None, runtime_health_check=None,
                 run_command=None,
                 clock=None):
        self.repo_path = os.path.abspath(os.path.expanduser(str(repo_path)))
        self.remote = str(remote or "origin").strip() or "origin"
        self.branch = str(branch or "master").strip() or "master"
        self.enabled = bool(enabled)
        self.poll_interval = max(60.0, float(poll_interval))
        self.command_timeout = max(1.0, float(command_timeout))
        self.health_timeout = max(1.0, float(health_timeout))
        self.marker_path = marker_path or os.path.join(
            self.repo_path, "layer_b", "data", "repository_update.json")
        self.publish = publish or (lambda topic, payload: None)
        self.quiesce = quiesce or (lambda: True)
        self.resume = resume or (lambda: None)
        self.restart = restart or (lambda: None)
        self.health_check = health_check or self._default_health_check
        self.runtime_health_check = runtime_health_check
        self.run_command = run_command or subprocess.run
        self.clock = clock or time.time
        self.state_payload = {}
        self.health_payload = {}
        self.last_status = {"state": "disabled" if not self.enabled else "idle"}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._in_progress = False

    def _emit(self, state, **fields):
        payload = {
            "state": state,
            "enabled": self.enabled,
            "remote": self.remote,
            "branch": self.branch,
            "repo": self.repo_path,
            "ts": self.clock(),
        }
        payload.update(fields)
        self.last_status = dict(payload)
        self.publish(STATUS_TOPIC, payload)

    def on_state(self, payload):
        if isinstance(payload, dict):
            self.state_payload = dict(payload)

    def on_health(self, payload):
        if isinstance(payload, dict):
            self.health_payload = dict(payload)

    def on_control(self, payload):
        """Handle status/update requests; updates run outside MQTT's callback."""
        payload = payload if isinstance(payload, dict) else {}
        operation = str(payload.get("operation") or "status").strip().lower()
        request_id = payload.get("request_id")
        if operation == "status":
            self._emit(self.last_status.get("state", "idle"),
                       request_id=request_id)
            return
        if operation != "update":
            self._emit("error", request_id=request_id,
                       error="operation must be update or status")
            return
        if payload.get("confirmed") is not True:
            self._emit("rejected", request_id=request_id,
                       error="confirmed=true is required")
            return
        self.request_update(trigger="approved", request_id=request_id)

    def request_update(self, trigger="periodic", request_id=None):
        """Queue an update, returning False when one is already running."""
        with self._lock:
            if self._in_progress:
                self._emit("busy", request_id=request_id)
                return False
            self._in_progress = True
        thread = threading.Thread(
            target=self._run_update,
            args=(str(trigger), request_id),
            name="repository-update", daemon=True)
        thread.start()
        return True

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop,
                                        name="repository-update-poll",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _poll_loop(self):
        while not self._stop.wait(self.poll_interval):
            self.request_update(trigger="periodic")

    def _run(self, *args, timeout=None):
        command = ["git", "-C", self.repo_path, *args]
        try:
            result = self.run_command(
                command, capture_output=True, text=True,
                timeout=timeout or self.command_timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise UpdateError(str(exc)) from exc
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        if getattr(result, "returncode", 1) != 0:
            detail = stderr or stdout or f"git exited {result.returncode}"
            raise UpdateError(detail[:500])
        return stdout

    def _git(self, *args):
        return self._run(*args)

    def _default_health_check(self):
        """Compile the tracked Python tree and validate the module catalog."""
        files = self._git("ls-files", "*.py").splitlines()
        if files:
            try:
                result = self.run_command(
                    [sys.executable, "-m", "py_compile", *files],
                    cwd=self.repo_path, capture_output=True, text=True,
                    timeout=self.health_timeout, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                return False, str(exc)
            if result.returncode != 0:
                return False, ((result.stderr or result.stdout or "compile failed")
                               .strip()[-500:])
        registry = os.path.join(self.repo_path, "layer_b", "module_registry.json")
        try:
            with open(registry, encoding="utf-8") as stream:
                value = json.load(stream)
            if not isinstance(value, list):
                return False, "module_registry.json is not a list"
        except (OSError, ValueError) as exc:
            return False, str(exc)
        return True, "python compile and module registry checks passed"

    def _write_marker(self, payload):
        directory = os.path.dirname(self.marker_path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary = self.marker_path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.marker_path)
            # The rename is not durable across a sudden power loss until the
            # containing directory is synced. Refuse to merge if that cannot
            # be established; the old checkout is safer than an unmarked new
            # checkout that cannot self-rollback on the next boot.
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise UpdateError(f"could not durably write update marker: {exc}") from exc

    def _read_marker(self):
        try:
            with open(self.marker_path, encoding="utf-8") as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else None
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _remove_marker(self):
        try:
            os.unlink(self.marker_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._emit("warning", error=f"could not remove update marker: {exc}")

    def startup_recover(self):
        """Validate the commit that just restarted the orchestrator.

        Returns True when startup may continue.  On a failed check, the old
        revision is restored and ``restart`` is invoked to exec it.  The
        callback normally does not return; returning False keeps this helper
        safe in tests and in a manually invoked process.
        """
        marker = self._read_marker()
        if not marker:
            return True
        old = str(marker.get("previous_commit") or "").strip()
        new = str(marker.get("new_commit") or "").strip()
        self._emit("health_check", previous_commit=old, new_commit=new,
                   trigger=marker.get("trigger"))
        try:
            healthy, detail = self.health_check()
        except Exception as exc:  # health checks must fail closed
            healthy, detail = False, f"health check raised: {exc}"
        if healthy and self.runtime_health_check is not None:
            try:
                healthy, detail = self.runtime_health_check()
            except Exception as exc:
                healthy, detail = False, f"runtime health check raised: {exc}"
        if healthy:
            self._remove_marker()
            self._emit("success", previous_commit=old, new_commit=new,
                       detail=detail, trigger=marker.get("trigger"))
            return True
        try:
            self._git("reset", "--hard", old)
        except UpdateError as exc:
            self._emit("rollback_error", previous_commit=old, new_commit=new,
                       error=str(exc), detail=detail)
            return False
        self._remove_marker()
        self._emit("rollback", previous_commit=old, new_commit=new,
                   error=detail, trigger=marker.get("trigger"))
        self.restart()
        return False

    def _run_update(self, trigger, request_id):
        quiesced = False
        previous = None
        try:
            if not safe_to_update(self.state_payload, self.health_payload):
                self._emit("waiting", trigger=trigger, request_id=request_id,
                           error="robot is not safely idle")
                return
            self._emit("checking", trigger=trigger, request_id=request_id)
            dirty = self._git("status", "--porcelain", "--untracked-files=no")
            if dirty:
                raise UpdateError("tracked local changes present; refusing to overwrite them")
            previous = self._git("rev-parse", "HEAD")
            self._emit("fetching", trigger=trigger, request_id=request_id,
                       previous_commit=previous)
            self._git("fetch", "--prune", self.remote, self.branch)
            remote_ref = f"{self.remote}/{self.branch}"
            target = self._git("rev-parse", remote_ref)
            if target == previous:
                self._emit("no_change", trigger=trigger, request_id=request_id,
                           commit=previous)
                return
            try:
                self._run("merge-base", "--is-ancestor", previous, remote_ref)
            except UpdateError as exc:
                raise UpdateError(f"remote is not a fast-forward: {exc}") from exc

            self._emit("quiescing", trigger=trigger, request_id=request_id,
                       previous_commit=previous, new_commit=target)
            if not self.quiesce():
                raise UpdateError("could not quiesce Layer B safely")
            quiesced = True
            # Write and fsync the rollback marker *before* changing HEAD. If
            # power is lost during the merge or before re-exec, startup still
            # knows which known-good commit to restore.
            self._write_marker({
                "previous_commit": previous,
                "new_commit": target,
                "trigger": trigger,
                "request_id": request_id,
                "started_at": self.clock(),
            })
            self._git("merge", "--ff-only", remote_ref)
            healthy, detail = self.health_check()
            if not healthy:
                raise UpdateError(f"pre-restart health check failed: {detail}")
            self._emit("restarting", trigger=trigger, request_id=request_id,
                       previous_commit=previous, new_commit=target)
            self.restart()
            raise UpdateError("restart callback returned without replacing the process")
        except Exception as exc:
            if previous and quiesced:
                try:
                    self._git("reset", "--hard", previous)
                    self._remove_marker()
                    self._emit("rollback", trigger=trigger,
                               request_id=request_id,
                               previous_commit=previous, error=str(exc))
                except UpdateError as rollback_error:
                    self._emit("rollback_error", trigger=trigger,
                               request_id=request_id,
                               previous_commit=previous,
                               error=f"{exc}; rollback failed: {rollback_error}")
            else:
                self._emit("error", trigger=trigger, request_id=request_id,
                           error=str(exc))
        finally:
            if quiesced:
                self.resume()
            with self._lock:
                self._in_progress = False
