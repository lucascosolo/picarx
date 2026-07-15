"""Pytest entrypoint: importing harness installs the dep stubs and sys.path
so `pytest tests/` works the same as `python3 -m unittest discover tests`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: F401,E402
