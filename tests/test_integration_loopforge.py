from __future__ import annotations

import json
import sys

import pytest

from loopforge import (
    CapabilityContext,
    ExecutionStep,
    PrimaryMetric,
    ReflectionSummary,
    ReviewDecision,
)
from loopforge.cli import main
from tests.support import (
    build_adapter_setup,
    build_candidate,
    build_spec,
    passed_check,
    patch_litellm_test_backends,
)

pytestmark = pytest.mark.integration


def test_cli_main_run_succeeds_on_first_iteration_with_generic_autonomous_executor(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    spec = build_spec(
        allowed_actions=["run_experiment", "fix_failure"],
        stop_conditions={"max_iterations": 1},
        metadata={"execution_mode": "autonomous_after_bootstrap"},
        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    patch_litellm_test_backends(
        monkeypatch,
        candidate=build_candidate(
            hypothesis="Run the first experiment end-to-end.",
            action_type="run_experiment",
            instructions="Write and run a metric-producing script.",
            execution_steps=[
                ExecutionStep(
                    kind="write_file",
                    path="emit_metric.py",
                    content="print('ordinal_loss=0.18')\n",
                ),
                ExecutionStep(
                    kind="shell",
                    command=f'"{sys.executable}" emit_metric.py',
                    cwd=str(repo_root),
                ),
            ],
        ),
        reflection=ReflectionSummary(assessment="First iteration succeeded."),
        review=ReviewDecision(status="accepted", reason="ok"),
    )

    def factory(spec, memory_root):
        return build_adapter_setup(
            handlers={},
            capability_context=CapabilityContext(
                environment_facts={
                    "execution_backend_kind": "generic_agentic",
                    "autonomous_execution_supported": True,
                    "execution_shell": "cmd.exe"
                    if sys.platform.startswith("win")
                    else "/bin/sh",
                    "shell_family": "windows_cmd"
                    if sys.platform.startswith("win")
                    else "posix_sh",
                    "repo_root": str(repo_root),
                    "python_executable": sys.executable,
                }
            ),
            preflight_checks=passed_check(
                detail="Generic autonomous executor is configured."
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: factory)

    exit_code = main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--repo-root",
            str(repo_root),
            "--memory-root",
            str(tmp_path / "memory"),
            "--executor-factory",
            "fake.module:factory",
            "--worker-model",
            "openai/gpt-5.4",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload[0]["record"]["outcome"]["status"] == "success"
    assert payload[0]["record"]["outcome"]["primary_metric_value"] == 0.18
