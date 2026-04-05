from __future__ import annotations

import json

import loopforge.core.backends as backend_module
import loopforge.bootstrap as bootstrap_module
from loopforge import (
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentOutcome,
    IterationRecord,
    MemorySnapshot,
    MarkdownMemoryNote,
    ReflectionSummary,
    ReviewDecision,
)
from tests.support import build_spec  # noqa: F401 (used in later tests)


def test_compact_recent_records_omits_full_attempt_payloads() -> None:
    spec = build_spec(allowed_actions=["run_experiment"])
    record = IterationRecord(
        iteration_id=3,
        parent_iteration_id=2,
        candidate=ExperimentCandidate(
            hypothesis="Retry the baseline with repair context.",
            action_type="run_experiment",
            change_type="repair",
            instructions="Retry after inspecting the last failure.",
            execution_steps=[
                ExecutionStep(kind="shell", command="python src/train.py")
            ],
        ),
        outcome=ExperimentOutcome(
            status="recoverable_failure",
            failure_type="ShellCommandFailed",
            failure_summary="Traceback...\n" + ("x" * 1200),
            recoverable=True,
            notes=["Step failed."],
            execution_details={
                "step_results": [
                    {
                        "index": 1,
                        "kind": "shell",
                        "command": "python src/train.py",
                        "stderr": "Traceback...\n" + ("y" * 1200),
                        "stdout": "",
                        "returncode": 1,
                    }
                ],
                "attempts": [
                    {
                        "attempt": 1,
                        "steps": [{"command": "python src/train.py"}],
                        "step_results": [{"stderr": "big"}],
                    }
                ],
                "intra_iteration_attempts": [{"attempt_number": 1}],
            },
        ),
        reflection=ReflectionSummary(assessment="Recoverable failure."),
        review=ReviewDecision(status="accepted", reason="Useful failure."),
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[record],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=4,
    )

    compact = backend_module._compact_recent_records(snapshot)

    assert len(compact) == 1
    execution_details = compact[0]["outcome"]["execution_details"]
    assert "attempts" not in execution_details
    assert "intra_iteration_attempts" not in execution_details
    assert execution_details["attempt_count"] == 1
    assert execution_details["latest_step_result"]["stderr_preview"].endswith("...")


def test_worker_markdown_handoff_truncates_and_prefers_bootstrap_notes() -> None:
    spec = build_spec()
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[
            MarkdownMemoryNote(
                path="agent_markdown/bootstrap_handoff.md",
                content="A" * 1400,
            ),
            MarkdownMemoryNote(
                path="agent_markdown/execution_runbook.md",
                content="B" * 100,
            ),
            MarkdownMemoryNote(
                path="agent_markdown/extra.md",
                content="C" * 100,
            ),
        ],
        next_iteration_id=1,
    )

    handoff = backend_module._worker_markdown_handoff(
        snapshot,
        iteration_policy={"generic_autonomous": True, "first_iteration": True},
    )

    assert len(handoff) == 2
    assert handoff[0]["path"].endswith("bootstrap_handoff.md")
    assert handoff[0]["content"].endswith("...")
    assert len(handoff[0]["content"]) == backend_module.MARKDOWN_NOTE_CHAR_LIMIT
