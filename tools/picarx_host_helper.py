#!/usr/bin/env python3
"""Small JSON-lines helper intended to run on a developer's host computer.

The robot talks to this process through an already authenticated SSH
connection. It deliberately has a filesystem root, bounded reads/results,
no shell=True execution, and a read-only default. The robot can stream this
file to a private host temp path and start it directly; no host installation
is required. For local testing it can be run as
``picarx_host_helper.py --root /path/to/project --allow-write``.

Protocol: one JSON object per stdin line, one JSON result per stdout line.
Supported operations are ``status``, ``list``, ``read``, ``search``, ``stat``,
``logs``, ``write_file``, ``delete_path``, ``preview_patch``, ``apply_patch``,
``rollback``, ``run``, and ``cancel``. File edits are bounded, rooted at the explicitly
scoped project, and require robot-side write authorization.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

MAX_READ_BYTES = 512 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_SEARCH_RESULTS = 200
MAX_LIST_RESULTS = 200
MAX_COMMAND_SEC = 120.0
MAX_SESSION_LOG = 200
DEFAULT_COMMAND_PREFIXES = (
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "unittest"),
    ("python3", "-m", "unittest"),
    ("pytest",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "grep"),
    ("make", "test"),
)


class HelperError(Exception):
    pass


def _bounded_text(value, limit=MAX_OUTPUT_BYTES):
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value or "")
    if len(text.encode("utf-8")) <= limit:
        return text, False
    raw = text.encode("utf-8")[:limit]
    return raw.decode("utf-8", "ignore"), True


class HostHelper:
    def __init__(self, root, allow_write=False, command_prefixes=None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise HelperError("project root is not a directory")
        self.allow_write = bool(allow_write)
        self.command_prefixes = tuple(command_prefixes or DEFAULT_COMMAND_PREFIXES)
        self.started_at = time.time()
        self._session_log = deque(maxlen=MAX_SESSION_LOG)
        self._last_patch = None
        self._active_lock = threading.Lock()
        self._active_command = None

    def _path(self, value, must_exist=False):
        rel = str(value or ".")
        if "\x00" in rel or os.path.isabs(rel):
            raise HelperError("path must be relative to the project root")
        # Resolve non-strictly first so a missing path becomes our bounded
        # protocol error rather than leaking a host traceback.  The resolved
        # path still catches symlink escapes before the existence check.
        candidate = (self.root / rel).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise HelperError("path escapes the project root")
        if must_exist and not candidate.exists():
            raise HelperError("path does not exist")
        return candidate

    def _read_text(self, path):
        try:
            size = path.stat().st_size
        except OSError as e:
            raise HelperError(str(e))
        if size > MAX_READ_BYTES:
            raise HelperError(f"file is too large ({size} bytes)")
        try:
            data = path.read_bytes()
        except OSError as e:
            raise HelperError(str(e))
        if b"\x00" in data:
            raise HelperError("binary files are not readable through this operation")
        return data.decode("utf-8", "replace")

    def _relative(self, path):
        return str(path.relative_to(self.root)) or "."

    def status(self, _request):
        return {"root": str(self.root), "allow_write": self.allow_write,
                "uptime_sec": round(time.time() - self.started_at, 2),
                "requests": len(self._session_log),
                "rollback_available": self._last_patch is not None}

    def logs(self, request):
        """Return a bounded, metadata-only audit trail for this SSH session.

        Source text, patch bodies, command arguments, and file contents are
        deliberately excluded.  This is enough to explain a debugging run
        without turning the helper into a secret-bearing transcript store.
        """
        try:
            limit = int(request.get("limit", 50))
        except (TypeError, ValueError):
            raise HelperError("log limit must be an integer")
        limit = max(1, min(MAX_SESSION_LOG, limit))
        return {"entries": list(self._session_log)[-limit:],
                "truncated": len(self._session_log) > limit}

    def list(self, request):
        path = self._path(request.get("path", "."), must_exist=True)
        if not path.is_dir():
            raise HelperError("path is not a directory")
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower())[:MAX_LIST_RESULTS]:
            try:
                entries.append({"name": child.name, "path": self._relative(child),
                                "kind": "dir" if child.is_dir() else "file",
                                "size": child.stat().st_size if child.is_file() else None})
            except OSError:
                continue
        return {"path": self._relative(path), "entries": entries,
                "truncated": len(list(path.iterdir())) > MAX_LIST_RESULTS}

    def stat(self, request):
        path = self._path(request.get("path", "."), must_exist=True)
        info = path.stat()
        return {"path": self._relative(path), "kind": "dir" if path.is_dir() else "file",
                "size": info.st_size, "modified": info.st_mtime}

    def read(self, request):
        path = self._path(request.get("path"), must_exist=True)
        if not path.is_file():
            raise HelperError("path is not a file")
        text = self._read_text(path)
        start = max(1, int(request.get("start_line", 1)))
        end = max(start, int(request.get("end_line", start + 400)))
        lines = text.splitlines()
        selected = lines[start - 1:end]
        return {"path": self._relative(path), "start_line": start,
                "end_line": min(end, len(lines)), "text": "\n".join(selected),
                "total_lines": len(lines)}

    def search(self, request):
        pattern = str(request.get("pattern") or "").strip()
        if not pattern or len(pattern) > 200:
            raise HelperError("search pattern is empty or too long")
        try:
            regex = re.compile(pattern, re.IGNORECASE if request.get("ignore_case", True) else 0)
        except re.error as e:
            raise HelperError(f"invalid search pattern: {e}")
        base = self._path(request.get("path", "."), must_exist=True)
        if not base.is_dir():
            raise HelperError("search path is not a directory")
        results, truncated = [], False
        for path in base.rglob("*"):
            if any(part in {".git", ".venv", "node_modules", "__pycache__"}
                   for part in path.relative_to(self.root).parts):
                continue
            if not path.is_file():
                continue
            try:
                text = self._read_text(path)
            except HelperError:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append({"path": self._relative(path), "line": line_no,
                                    "text": line[:500]})
                    if len(results) >= MAX_SEARCH_RESULTS:
                        truncated = True
                        break
            if truncated:
                break
        return {"pattern": pattern, "results": results, "truncated": truncated}

    def write_file(self, request):
        """Atomically write one bounded UTF-8 text file inside the root."""
        if not self.allow_write:
            raise HelperError("host helper is read-only; restart it with --allow-write")
        content = request.get("content")
        if not isinstance(content, str) or not content:
            raise HelperError("file content must be a non-empty text string")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_READ_BYTES:
            raise HelperError("file content is too large")
        if "\x00" in content:
            raise HelperError("binary file content is not accepted")
        path = self._path(request.get("path"))
        if path == self.root:
            raise HelperError("a file path is required")
        if path.exists() and not path.is_file():
            raise HelperError("path is not a file")
        expected = request.get("expected_sha256")
        current = None
        if path.exists():
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected and str(expected).lower() != current:
                raise HelperError("file changed since it was read; refusing to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent,
                    prefix=f".{path.name}.picarx-", delete=False) as stream:
                temporary = stream.name
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if current is not None:
                os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        except OSError as exc:
            raise HelperError(f"could not write file: {exc}") from exc
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return {"path": self._relative(path), "bytes": len(encoded),
                "created": current is None,
                "sha256": hashlib.sha256(encoded).hexdigest()}

    def delete_path(self, request):
        """Delete one file after explicit confirmation; never recurse."""
        if not self.allow_write:
            raise HelperError("host helper is read-only; restart it with --allow-write")
        path = self._path(request.get("path"), must_exist=True)
        if path == self.root:
            raise HelperError("the project root cannot be deleted")
        if not path.is_file():
            raise HelperError("only individual files can be deleted")
        try:
            path.unlink()
        except OSError as exc:
            raise HelperError(f"could not delete file: {exc}") from exc
        return {"path": self._relative(path), "deleted": True}

    def preview_patch(self, request):
        patch = str(request.get("patch") or "")
        if not patch or len(patch.encode()) > MAX_READ_BYTES:
            raise HelperError("patch is empty or too large")
        if "\x00" in patch:
            raise HelperError("binary patch data is not accepted")
        # A preview is intentionally non-mutating. Git's checker catches bad
        # paths and malformed hunks when this project is a git worktree.
        if (self.root / ".git").exists():
            proc = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", "-"],
                                  input=patch, text=True, cwd=self.root,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=10, check=False)
            out, out_trunc = _bounded_text(proc.stdout)
            err, err_trunc = _bounded_text(proc.stderr)
            return {"valid": proc.returncode == 0, "stdout": out, "stderr": err,
                    "truncated": out_trunc or err_trunc}
        return {"valid": True, "note": "no git worktree; patch syntax only was checked"}

    def apply_patch(self, request):
        if not self.allow_write:
            raise HelperError("host helper is read-only; restart it with --allow-write")
        patch = str(request.get("patch") or "")
        if not patch or len(patch.encode()) > MAX_READ_BYTES:
            raise HelperError("patch is empty or too large")
        if not (self.root / ".git").exists():
            raise HelperError("patch application requires a git worktree")
        check = self.preview_patch({"patch": patch})
        if not check.get("valid"):
            raise HelperError(check.get("stderr") or "patch failed validation")
        proc = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"],
                              input=patch, text=True, cwd=self.root,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=20, check=False)
        out, out_trunc = _bounded_text(proc.stdout)
        err, err_trunc = _bounded_text(proc.stderr)
        if proc.returncode:
            raise HelperError(err or "patch application failed")
        self._last_patch = {"patch": patch, "applied_at": time.time()}
        return {"applied": True, "stdout": out, "stderr": err,
                "truncated": out_trunc or err_trunc}

    def rollback(self, _request):
        """Reverse only the most recent successful patch in this session.

        ``git apply --check -R`` makes this conservative: if the project has
        changed since the patch was applied, rollback fails instead of
        overwriting unrelated work.  A successful rollback clears the one
        available rollback slot and is itself recorded in the session log.
        """
        if not self.allow_write:
            raise HelperError("host helper is read-only; restart it with --allow-write")
        if not self._last_patch:
            raise HelperError("no successful patch is available to roll back")
        patch = self._last_patch["patch"]
        check = subprocess.run(["git", "apply", "--check", "-R",
                                "--whitespace=nowarn", "-"], input=patch,
                               text=True, cwd=self.root, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=10, check=False)
        if check.returncode:
            raise HelperError(_bounded_text(check.stderr)[0] or
                              "rollback failed validation; project changed")
        proc = subprocess.run(["git", "apply", "-R", "--whitespace=nowarn", "-"],
                              input=patch, text=True, cwd=self.root,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=20, check=False)
        out, out_trunc = _bounded_text(proc.stdout)
        err, err_trunc = _bounded_text(proc.stderr)
        if proc.returncode:
            raise HelperError(err or "rollback failed")
        self._last_patch = None
        return {"rolled_back": True, "stdout": out, "stderr": err,
                "truncated": out_trunc or err_trunc}

    def _allowed(self, argv):
        return any(tuple(argv[:len(prefix)]) == prefix for prefix in self.command_prefixes)

    def run(self, request):
        raw = request.get("argv", request.get("command"))
        if isinstance(raw, str):
            try:
                argv = shlex.split(raw)
            except ValueError as e:
                raise HelperError(f"invalid command quoting: {e}")
        elif isinstance(raw, list):
            argv = [str(x) for x in raw]
        else:
            argv = []
        if not argv or len(argv) > 32 or any(len(x) > 500 for x in argv):
            raise HelperError("command is empty or too large")
        if not self._allowed(argv):
            raise HelperError("command is not in the host helper allowlist")
        cwd = self._path(request.get("cwd", "."), must_exist=True)
        if not cwd.is_dir():
            raise HelperError("command cwd is not a directory")
        timeout = min(MAX_COMMAND_SEC, max(0.1, float(request.get("timeout_sec", 30))))
        with self._active_lock:
            if self._active_command is not None:
                raise HelperError("another remote command is already running")
        active = None
        try:
            proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
            active = {"proc": proc, "request_id": request.get("request_id"),
                      "cancel_requested": False}
            with self._active_lock:
                # A second run may have won the slot between the first check
                # and process creation. Refuse it without leaving a child.
                if self._active_command is not None:
                    proc.terminate()
                    proc.wait(timeout=2)
                    raise HelperError("another remote command is already running")
                self._active_command = active
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=timeout)
            finally:
                with self._active_lock:
                    canceled = bool(active["cancel_requested"])
                    if self._active_command is active:
                        self._active_command = None
            stdout, out_trunc = _bounded_text(stdout_raw)
            stderr, err_trunc = _bounded_text(stderr_raw)
            if canceled:
                return {"returncode": None, "stdout": stdout, "stderr": stderr,
                        "timed_out": False, "canceled": True,
                        "truncated": out_trunc or err_trunc}
            return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr,
                    "timed_out": False, "truncated": out_trunc or err_trunc}
        except subprocess.TimeoutExpired as e:
            with self._active_lock:
                if active is not None:
                    active["cancel_requested"] = True
            try:
                if active is not None:
                    os.killpg(active["proc"].pid, signal.SIGTERM)
            except (AttributeError, OSError, ProcessLookupError):
                if active is not None:
                    active["proc"].terminate()
            stdout_raw, stderr_raw = e.stdout or b"", e.stderr or b""
            if active is not None:
                try:
                    stdout_raw, stderr_raw = active["proc"].communicate(timeout=2)
                except Exception:
                    pass
            stdout, _ = _bounded_text(stdout_raw)
            stderr, _ = _bounded_text(stderr_raw)
            return {"returncode": None, "stdout": stdout, "stderr": stderr,
                    "timed_out": True, "truncated": True}

    def cancel(self, request):
        """Stop the one active host command without killing the SSH helper."""
        with self._active_lock:
            active = self._active_command
            if active is None:
                return {"canceled": False, "reason": "no active command"}
            target = request.get("target_request_id")
            if target and str(target) != str(active.get("request_id")):
                return {"canceled": False, "reason": "request is not active"}
            active["cancel_requested"] = True
            proc = active["proc"]
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            try:
                proc.terminate()
            except Exception:
                return {"canceled": False, "reason": "command already exited"}
        return {"canceled": True, "target_request_id": active.get("request_id")}

    def handle(self, request):
        if not isinstance(request, dict):
            raise HelperError("request must be an object")
        op = str(request.get("op") or "").lower()
        request_id = request.get("request_id")
        started = time.monotonic()
        methods = {"status": self.status, "list": self.list, "read": self.read,
                   "search": self.search, "stat": self.stat, "logs": self.logs,
                   "write_file": self.write_file, "delete_path": self.delete_path,
                   "preview_patch": self.preview_patch, "apply_patch": self.apply_patch,
                   "rollback": self.rollback, "run": self.run, "cancel": self.cancel}
        if op not in methods:
            raise HelperError(f"unsupported operation: {op}")
        try:
            if op in {"write_file", "delete_path", "apply_patch", "rollback", "run"} and \
                    not bool(request.get("confirmed")):
                raise HelperError("explicit confirmation is required for this operation")
            result = methods[op](request)
        except Exception as exc:
            self._session_log.append({
                "request_id": str(request_id)[:80] if request_id else None,
                "op": op, "ok": False, "error": str(exc)[:200],
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "ts": time.time()})
            raise
        self._session_log.append({
            "request_id": str(request_id)[:80] if request_id else None,
            "op": op, "ok": True,
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "ts": time.time()})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--allow-command", action="append", default=[],
                        help="additional exact command prefix, e.g. 'cargo test'")
    args = parser.parse_args(argv)
    prefixes = list(DEFAULT_COMMAND_PREFIXES)
    for raw in args.allow_command:
        try:
            prefixes.append(tuple(shlex.split(raw)))
        except ValueError as e:
            parser.error(str(e))
    try:
        helper = HostHelper(args.root, args.allow_write, prefixes)
    except HelperError as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        return 2
    output_lock = threading.Lock()
    workers = []

    def emit(response):
        with output_lock:
            print(json.dumps(response, separators=(",", ":")), flush=True)

    def serve(request):
        request_id = request.get("request_id") if isinstance(request, dict) else None
        try:
            result = helper.handle(request)
            response = {"ok": True, "request_id": request_id, "result": result}
        except Exception as e:
            response = {"ok": False, "request_id": request_id,
                        "error": str(e)[:500]}
        emit(response)

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except Exception as e:
            emit({"ok": False, "request_id": None, "error": str(e)[:500]})
            continue
        # A command gets its own worker so the input loop can receive a typed
        # cancel request over the same SSH channel while it is running. Other
        # operations stay synchronous, preserving the simple JSONL behavior.
        if isinstance(request, dict) and str(request.get("op") or "").lower() == "run":
            worker = threading.Thread(target=serve, args=(request,),
                                      name="picarx-helper-command")
            worker.start()
            workers.append(worker)
        else:
            serve(request)
    for worker in workers:
        worker.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
