"""Process-wide lock for the Pi camera pipeline.

libcamera reports a rather unhelpful "pipeline handler in use" error when two
Picamera2 instances race to open the same sensor.  RobotState coordinates the
*intended* owner, but its MQTT transition is asynchronous, so camera users
also need a kernel-enforced handoff while a Picamera2 instance is alive.
"""
import fcntl
import os


LOCK_PATH = os.environ.get("PICARX_CAMERA_LOCK", "/tmp/picarx-camera.lock")


class CameraBusy(RuntimeError):
    """Another process currently owns the camera pipeline."""


class CameraLease:
    def __init__(self, path=LOCK_PATH):
        self.path = path
        self._fd = None

    def acquire(self):
        if self._fd is not None:
            return self
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in (11, 13):
                raise CameraBusy("camera pipeline is owned by another process")
            raise
        self._fd = fd
        return self

    def release(self):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()

