from __future__ import annotations

from loopforge import (
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    MemorySnapshot,
)
from tests.support import build_spec


def test_generic_execution_plan_executor_rejects_plan_that_ignores_baseline_contract(
    tmp_path,
) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
                "enforcement_scope": "until_first_successful_iteration",
            }
        },
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
        ),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )
    candidate = ExperimentCandidate(
        hypothesis="Write a new scratch pipeline.",
        action_type="run_experiment",
        change_type="run",
        instructions="Create a new standalone script.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="scratch/new_model.py",
                content="print('new pipeline')\n",
            ),
            ExecutionStep(
                kind="shell",
                command="python scratch/new_model.py",
                cwd=str(tmp_path),
            ),
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "InvalidExecutionPlan"
    assert "src/train.py" in outcome.failure_summary
    assert "existing framework" in outcome.failure_summary.lower()


def test_generic_execution_plan_executor_can_write_files(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["baseline"])
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
        markdown_memory=[],
        next_iteration_id=1,
    )
    candidate = ExperimentCandidate(
        hypothesis="Write a file directly.",
        action_type="baseline",
        change_type="baseline",
        instructions="Write a repo file.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="notes/generated.txt",
                content="hello\n",
            ),
            ExecutionStep(
                kind="append_file",
                path="notes/generated.txt",
                content="world\n",
            ),
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "success"
    assert (tmp_path / "notes" / "generated.txt").read_text(
        encoding="utf-8"
    ) == "hello\nworld\n"


def test_generic_execution_plan_executor_blocks_writes_outside_repo(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["baseline"])
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
        markdown_memory=[],
        next_iteration_id=1,
    )
    candidate = ExperimentCandidate(
        hypothesis="Try an unsafe write.",
        action_type="baseline",
        change_type="baseline",
        instructions="Attempt to write outside the repo root.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="../outside.txt",
                content="nope",
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "blocked"
    assert outcome.failure_type == "UnsafePath"
