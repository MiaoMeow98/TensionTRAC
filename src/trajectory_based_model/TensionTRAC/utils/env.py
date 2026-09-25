"""Environment setup for the inference-only TensionTRAC subset."""

from __future__ import annotations

from pathlib import Path

_ENV_SETUP_DONE = False


class LocalPathManager:
    def open(self, path, mode="r", buffering=-1):
        return open(path, mode, buffering=buffering)

    def exists(self, path):
        return Path(path).exists()

    def mkdirs(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


pathmgr = LocalPathManager()
checkpoint_pathmgr = LocalPathManager()


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
