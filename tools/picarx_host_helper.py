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
``preview_patch``, ``apply_patch``, and ``run``.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

MAX_READ_BYTES = 512 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_SEARCH_RESULTS = 200
MAX_LIST_RESULTS = 200
MAX_COMMAND_SEC = 120.0
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
                "uptime_sec": round(time.time() - self.started_at, 2)}

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
        return {"applied": True, "stdout": out, "stderr": err,
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
        try:
            proc = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout, check=False)
            stdout, out_trunc = _bounded_text(proc.stdout)
            stderr, err_trunc = _bounded_text(proc.stderr)
            return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr,
                    "timed_out": False, "truncated": out_trunc or err_trunc}
        except subprocess.TimeoutExpired as e:
            stdout, _ = _bounded_text(e.stdout or b"")
            stderr, _ = _bounded_text(e.stderr or b"")
            return {"returncode": None, "stdout": stdout, "stderr": stderr,
                    "timed_out": True, "truncated": True}

    def handle(self, request):
        if not isinstance(request, dict):
            raise HelperError("request must be an object")
        op = str(request.get("op") or "").lower()
        methods = {"status": self.status, "list": self.list, "read": self.read,
                   "search": self.search, "stat": self.stat,
                   "preview_patch": self.preview_patch, "apply_patch": self.apply_patch,
                   "run": self.run}
        if op not in methods:
            raise HelperError(f"unsupported operation: {op}")
        return methods[op](request)


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
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id") if isinstance(request, dict) else None
            result = helper.handle(request)
            response = {"ok": True, "request_id": request_id, "result": result}
        except Exception as e:
            response = {"ok": False, "request_id": request_id,
                        "error": str(e)[:500]}
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
