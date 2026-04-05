"""Global test fixtures — keep subprocess-heavy probes fast and block real LLM calls."""

from __future__ import annotations

import subprocess

import pytest

from tests.support import patch_loopforge_backend_constructors


class _FakeCompletedProcess:
    """Mimics a successful subprocess.run result for the execution probe."""

    def __init__(self, stdout="loopforge_execution_probe_ok\n/tmp\npython\n"):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


_REAL_RUN = subprocess.run


def _fast_subprocess_run(cmd, **kwargs):
    """Intercept the loopforge execution probe; pass everything else through quickly."""
    if isinstance(cmd, (list, tuple)):
        cmd_str = " ".join(str(c) for c in cmd)
        if "loopforge_execution_probe_ok" in cmd_str:
            return _FakeCompletedProcess()
    return _REAL_RUN(cmd, **kwargs)


@pytest.fixture(autouse=True)
def _no_slow_subprocesses(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fast_subprocess_run)


@pytest.fixture(autouse=True)
def _block_real_llm_calls(monkeypatch):
    """Prevent any test from accidentally making real LLM API calls via litellm."""
    try:
        import litellm

        def _blocked_completion(**kwargs):
            raise RuntimeError(
                "Real LLM call blocked by conftest — use a fake/stub backend in your test."
            )

        monkeypatch.setattr(litellm, "completion", _blocked_completion)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _stub_loopforge_backend_constructors(monkeypatch):
    """Prevent Loopforge tests from constructing concrete LiteLLM backend classes."""
    patch_loopforge_backend_constructors(monkeypatch)
