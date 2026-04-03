from __future__ import annotations

import sys
import time

import pytest

import loopforge.bootstrap as bootstrap_module
from loopforge import (
    BootstrapTurn,
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentSpec,
    ExperimentSpecProposal,
    Loopforge,
    MemorySnapshot,
    MetricSpec,
    PrimaryMetric,
    ReflectionSummary,
    ReviewDecision,
    RoutingExperimentExecutor,
)
from loopforge.core.agentic_execution import GenericExecutionPlanExecutor
from tests.support import (
    FakeAccessAdvisorBackend,
    FakeConsultationBackend,
    FakeNarrationBackend,
    FakeReflectionBackend,
    FakeReviewBackend,
    FakeWorkerBackend,
    build_candidate,
    build_spec,
)

pytestmark = pytest.mark.slow


def _snapshot(
    spec: ExperimentSpec,
    capability_context: CapabilityContext | None = None,
) -> MemorySnapshot:
    return MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=capability_context or CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )


def test_probe_data_asset_timeout_returns_without_waiting_for_probe_completion(
    tmp_path, monkeypatch
) -> None:
    asset_path = tmp_path / "matches.csv"
    asset_path.write_text("kills\n1\n", encoding="utf-8")

    class SlowPandasModule:
        @staticmethod
        def read_csv(path, nrows=None):
            time.sleep(0.2)
            return []

    monkeypatch.setitem(sys.modules, "pandas", SlowPandasModule)
    monkeypatch.setattr(bootstrap_module, "DATA_PROBE_TIMEOUT_SECONDS", 0.01)

    started_at = time.monotonic()
    schema = bootstrap_module._probe_data_asset(str(asset_path), tmp_path)
    elapsed = time.monotonic() - started_at

    assert schema.load_error == "Probe timed out after 0.01s"
    assert elapsed < 0.15


def test_probe_data_asset_skips_new_probe_while_timed_out_probe_is_still_running(
    tmp_path, monkeypatch
) -> None:
    first_asset = tmp_path / "first.csv"
    second_asset = tmp_path / "second.csv"
    first_asset.write_text("kills\n1\n", encoding="utf-8")
    second_asset.write_text("kills\n2\n", encoding="utf-8")

    class SlowPandasModule:
        @staticmethod
        def read_csv(path, nrows=None):
            time.sleep(0.2)
            return []

    monkeypatch.setitem(sys.modules, "pandas", SlowPandasModule)
    monkeypatch.setattr(bootstrap_module, "DATA_PROBE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(bootstrap_module, "_ACTIVE_DATA_PROBES", {})

    first_schema = bootstrap_module._probe_data_asset(str(first_asset), tmp_path)
    second_schema = bootstrap_module._probe_data_asset(str(second_asset), tmp_path)

    assert first_schema.load_error == "Probe timed out after 0.01s"
    assert (
        second_schema.load_error is None
        or "timed out" in second_schema.load_error
        or "Probe skipped" in second_schema.load_error
    )
    time.sleep(0.25)


def test_probe_data_asset_skips_reprobe_for_same_asset_while_previous_timeout_is_still_running(
    tmp_path, monkeypatch
) -> None:
    asset_path = tmp_path / "matches.csv"
    asset_path.write_text("kills\n1\n", encoding="utf-8")

    class SlowPandasModule:
        @staticmethod
        def read_csv(path, nrows=None):
            time.sleep(0.2)
            return []

    monkeypatch.setitem(sys.modules, "pandas", SlowPandasModule)
    monkeypatch.setattr(bootstrap_module, "DATA_PROBE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(bootstrap_module, "_ACTIVE_DATA_PROBES", {})

    first_schema = bootstrap_module._probe_data_asset(str(asset_path), tmp_path)
    second_schema = bootstrap_module._probe_data_asset(str(asset_path), tmp_path)

    assert first_schema.load_error == "Probe timed out after 0.01s"
    assert (
        second_schema.load_error
        == "Probe skipped because a previous timed-out probe is still running for this asset."
    )
    time.sleep(0.25)


def test_generic_execution_fix_uses_latest_revised_plan_on_subsequent_repairs(
    tmp_path,
) -> None:
    spec = build_spec(allowed_actions=["run_experiment", "fix_failure"])
    snapshot = _snapshot(
        spec,
        CapabilityContext(
            environment_facts={
                "execution_shell": "cmd.exe",
                "shell_family": "windows_cmd",
            }
        ),
    )
    candidate = ExperimentCandidate(
        hypothesis="Repair the failing command incrementally.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run the experiment.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command="python missing_script.py",
                cwd=str(tmp_path),
            )
        ],
    )

    class RecordingFixBackend:
        def __init__(self) -> None:
            self.model = "anthropic/claude-opus-4-6-v1"
            self.seen_commands: list[str] = []

        def fix_execution_plan(
            self, candidate, failed_step_index, failure_summary, step_results
        ):
            self.seen_commands.append(candidate.execution_steps[0].command)
            if len(self.seen_commands) == 1:
                return [
                    ExecutionStep(
                        kind="shell",
                        command="head missing_script.py",
                        cwd=str(tmp_path),
                    )
                ]
            return [
                ExecutionStep(
                    kind="shell",
                    command=f'"{sys.executable}" -c "print(123)"',
                    cwd=str(tmp_path),
                )
            ]

    fix_backend = RecordingFixBackend()

    outcome = GenericExecutionPlanExecutor(
        repo_root=tmp_path,
        fix_backend=fix_backend,
        max_retries=2,
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert fix_backend.seen_commands == [
        "python missing_script.py",
        "head missing_script.py",
    ]


def test_loopforge_generic_executor_preserves_stdout_for_multiline_python_c_commands(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n", encoding="utf-8"
    )

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
                        allowed_actions=["inspect_repo", "fix_failure"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=FakeWorkerBackend(
            [
                build_candidate(
                    hypothesis="Inspect a repo file with inline Python.",
                    action_type="inspect_repo",
                    change_type="inspect",
                    instructions="Use a multiline python -c command to print the repo file.",
                    execution_steps=[
                        ExecutionStep(
                            kind="shell",
                            command=(
                                f'"{sys.executable}" -c "from pathlib import Path; files=[\'pyproject.toml\'];\n'
                                "for fp in files:\n"
                                "    print(fp)\n"
                                "print('ordinal_loss=0.27')"
                                '"'
                            ),
                            cwd=str(repo_root),
                        )
                    ],
                )
            ]
        ),
        consultation_backend=FakeConsultationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Inspection succeeded.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    result = app.start(user_goal="Inspect the repo.")

    assert result["status"] == "started"
    step_result = result["results"][0]["record"]["outcome"]["execution_details"][
        "step_results"
    ][0]
    assert step_result["returncode"] == 0
    assert "pyproject.toml" in step_result["stdout"]


def test_generic_execution_plan_executor_captures_metrics_from_json_stdout(
    tmp_path,
) -> None:
    spec = build_spec(
        allowed_actions=["run_experiment"],
        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
        secondary_metrics=[MetricSpec(name="accuracy", goal="maximize")],
        guardrail_metrics=[MetricSpec(name="latency_ms", goal="minimize")],
    )
    snapshot = _snapshot(spec)
    candidate = ExperimentCandidate(
        hypothesis="Emit metrics in machine-readable form.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run the experiment and print metrics as JSON.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="emit_metrics.py",
                content=(
                    "import json\n"
                    "print(json.dumps({\n"
                    '    "metric_results": {"ordinal_loss": 0.19, "accuracy": 0.81},\n'
                    '    "guardrail_metrics": {"latency_ms": 123.0},\n'
                    "}))\n"
                ),
            ),
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" emit_metrics.py',
                cwd=str(tmp_path),
            ),
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "success"
    assert outcome.primary_metric_value == 0.19
    assert outcome.metric_results["ordinal_loss"].value == 0.19
    assert outcome.secondary_metrics["accuracy"] == 0.81
    assert outcome.guardrail_metrics["latency_ms"] == 123.0
    assert "Captured metric output" in outcome.notes[-1]


def test_generic_execution_plan_executor_treats_permission_denied_as_recoverable_failure(
    tmp_path,
) -> None:
    spec = build_spec(allowed_actions=["run_experiment"])
    snapshot = _snapshot(spec)
    candidate = ExperimentCandidate(
        hypothesis="Surface a permission problem without killing the whole run.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run a command that exits with a permission error.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=(
                    f'"{sys.executable}" -c '
                    "\"import sys; sys.stderr.write('PermissionError: [WinError 5] Access is denied'); sys.exit(1)\""
                ),
                cwd=str(tmp_path),
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "ShellPermissionDenied"
    assert "writable repo path" in outcome.recovery_actions[0]


def test_generic_execution_plan_executor_runs_shell_steps(tmp_path) -> None:
    spec = build_spec(allowed_actions=["baseline"])
    snapshot = _snapshot(spec)
    candidate = ExperimentCandidate(
        hypothesis="Run a direct command.",
        action_type="baseline",
        change_type="baseline",
        instructions="Run a bounded shell command.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" -c "print(123)"',
                cwd=str(tmp_path),
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "success"
    assert "Executed" in outcome.notes[0] and "step" in outcome.notes[0]
    assert outcome.execution_details["step_results"][0]["returncode"] == 0
    assert outcome.execution_details["latest_stdout_preview"] == "123"


@pytest.mark.parametrize(
    ("allowed_actions", "capability_context", "action_type", "printed_value"),
    [
        (["baseline"], CapabilityContext(), "baseline", "456"),
        (
            ["run_experiment"],
            CapabilityContext(
                environment_facts={
                    "execution_backend_kind": "generic_agentic",
                    "autonomous_execution_supported": True,
                }
            ),
            "baseline",
            "789",
        ),
    ],
)
def test_routing_executor_uses_plan_executor_for_execution_steps(
    tmp_path,
    allowed_actions: list[str],
    capability_context: CapabilityContext,
    action_type: str,
    printed_value: str,
) -> None:
    spec = build_spec(allowed_actions=allowed_actions)
    snapshot = _snapshot(spec, capability_context)
    candidate = ExperimentCandidate(
        hypothesis="Use generic execution.",
        action_type=action_type,
        change_type="baseline",
        instructions="Run bounded shell steps.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" -c "print({printed_value})"',
                cwd=str(tmp_path),
            )
        ],
    )

    executor = RoutingExperimentExecutor(
        handlers={},
        plan_executor=GenericExecutionPlanExecutor(repo_root=tmp_path),
    )

    outcome = executor.execute(candidate, snapshot)

    assert outcome.status == "success"
    assert outcome.execution_details["step_results"][0]["returncode"] == 0
