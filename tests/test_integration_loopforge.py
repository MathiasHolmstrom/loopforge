from __future__ import annotations

import json
import sys

import pytest

from loopforge import (
    BootstrapTurn,
    CapabilityContext,
    ExecutionStep,
    ExperimentSpec,
    ExperimentSpecProposal,
    Loopforge,
    PrimaryMetric,
    ReflectionSummary,
    ReviewDecision,
)
from loopforge.cli import main
from tests.support import (
    FakeAccessAdvisorBackend,
    FakeConsultationBackend,
    FakeNarrationBackend,
    FakeReflectionBackend,
    FakeReviewBackend,
    FakeWorkerBackend,
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


def test_loopforge_generic_executor_recovers_from_socketpair_permission_error_with_real_subprocess(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Generic executor is ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["run_experiment", "fix_failure"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class StaticFixBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str]] = []
            self.model = "anthropic/claude-opus-4-6-v1"

        def fix_execution_plan(
            self, candidate, failed_step_index, failure_summary, step_results
        ):
            self.calls.append((failed_step_index, failure_summary))
            return [
                ExecutionStep(
                    kind="write_file",
                    path="repro_socketpair_permission.py",
                    content="print('ordinal_loss=0.31')\n",
                ),
                ExecutionStep(
                    kind="shell",
                    command=f'"{sys.executable}" repro_socketpair_permission.py',
                    cwd=str(repo_root),
                ),
            ]

    fix_backend = StaticFixBackend()
    progress_log: list[tuple[str, str]] = []
    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=FakeWorkerBackend(
            [
                build_candidate(
                    hypothesis="Run a real repro for the socketpair permission error.",
                    action_type="run_experiment",
                    instructions="Reproduce the multiprocessing/socketpair permission error and recover from it.",
                    execution_steps=[
                        ExecutionStep(
                            kind="write_file",
                            path="repro_socketpair_permission.py",
                            content=(
                                "import asyncio\n"
                                "import socket\n"
                                "\n"
                                "def _fail(*args, **kwargs):\n"
                                "    raise PermissionError('[WinError 5] Access is denied')\n"
                                "\n"
                                "socket.socketpair = _fail\n"
                                "asyncio.new_event_loop()\n"
                            ),
                        ),
                        ExecutionStep(
                            kind="shell",
                            command=f'"{sys.executable}" repro_socketpair_permission.py',
                            cwd=str(repo_root),
                        ),
                    ],
                )
            ]
        ),
        consultation_backend=FakeConsultationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Recovered.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
        progress_fn=lambda stage, msg: progress_log.append((stage, msg)),
    )
    app.execution_fix_backend = fix_backend

    result = app.start(user_goal="Build a LoL kills model.")

    assert result["status"] == "started"
    record = result["results"][0]["record"]
    outcome = record["outcome"]
    assert outcome["status"] == "success"
    assert fix_backend.calls
    attempts = outcome["execution_details"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["success"] is False
    assert (
        "self._ssock, self._csock = socket.socketpair()"
        in attempts[0]["step_results"][1]["stderr"]
    )
    assert "asyncio\\proactor_events.py" in attempts[0]["step_results"][1]["stderr"]
    assert attempts[1]["success"] is True
    assert any(
        "[anthropic/claude-opus-4-6-v1] Step 2 failed" in message
        for _, message in progress_log
    )
    assert any("Revised plan with 2 step(s)" in message for _, message in progress_log)
