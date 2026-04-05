from __future__ import annotations

from loopforge import (
    CapabilityContext,
    ExperimentCandidate,
    ExperimentOutcome,
    FileMemoryStore,
    HumanIntervention,
    IterationRecord,
    IterationSummary,
    ReflectionSummary,
    ReviewDecision,
)
from tests.support import build_spec


def test_memory_store_load_snapshot_applies_human_interventions_and_markdown_memory(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec(allowed_actions=["baseline", "train"])
    store.initialize(spec)
    store.append_human_intervention(
        HumanIntervention(
            author="user",
            type="feedback",
            message="Disable baseline and suggest a bounded follow-up.",
            timestamp="2026-04-04T10:00:00Z",
            effects={
                "disable_actions": ["baseline"],
                "metadata_updates": {"force_next_action": "train"},
            },
        )
    )
    guide_path = tmp_path / "memory" / "agent_markdown" / "execution_runbook.md"
    guide_path.write_text("# Runbook\n\nUse the verified path.\n", encoding="utf-8")

    snapshot = store.load_snapshot(capability_context=CapabilityContext())

    assert snapshot.effective_spec.allowed_actions == ["train"]
    assert snapshot.effective_spec.metadata["force_next_action"] == "train"
    assert any(
        note.path == "agent_markdown/execution_runbook.md"
        for note in snapshot.markdown_memory
    )


def test_memory_store_initialize_reset_state_clears_records_and_best_summary(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec()
    store.initialize(spec)
    record = IterationRecord(
        iteration_id=1,
        parent_iteration_id=None,
        candidate=ExperimentCandidate(
            hypothesis="Baseline",
            action_type="baseline",
            change_type="baseline",
            instructions="Run baseline.",
        ),
        outcome=ExperimentOutcome(primary_metric_value=0.5),
        reflection=ReflectionSummary(assessment="Fine."),
        review=ReviewDecision(status="accepted", reason="ok"),
    )
    summary = IterationSummary(
        iteration_id=1,
        parent_iteration_id=None,
        hypothesis="Baseline",
        action_type="baseline",
        change_type="baseline",
        instructions="Run baseline.",
        config_patch={},
        primary_metric_name="log_loss",
        primary_metric_value=0.5,
        secondary_metrics={},
        result="improved",
        artifacts=[],
        lessons=[],
        next_ideas=[],
        do_not_repeat=[],
        reflection_assessment="Fine.",
        review_reason="ok",
    )
    store.append_iteration_record(record)
    store.append_accepted_summary(summary)
    store.write_best_summary(summary)

    store.initialize(spec, reset_state=True)
    snapshot = store.load_snapshot(capability_context=CapabilityContext())

    assert snapshot.recent_records == []
    assert snapshot.recent_summaries == []
    assert snapshot.best_summary is None


def test_memory_store_initialize_reset_state_clears_stale_agent_markdown(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec()
    store.initialize(spec)
    stale_path = tmp_path / "memory" / "agent_markdown" / "bootstrap_handoff.md"
    stale_path.write_text("# Old handoff\n", encoding="utf-8")

    store.initialize(spec, reset_state=True)
    snapshot = store.load_snapshot(capability_context=CapabilityContext())

    assert not stale_path.exists()
    assert all(
        note.path != "agent_markdown/bootstrap_handoff.md"
        for note in snapshot.markdown_memory
    )
