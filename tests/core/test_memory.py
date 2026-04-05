from __future__ import annotations

from loopforge import (
    AgentUpdate,
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


def test_memory_store_reopen_last_iteration_rewinds_accepted_state(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec()
    store.initialize(spec)

    first_record = IterationRecord(
        iteration_id=1,
        parent_iteration_id=None,
        candidate=ExperimentCandidate(
            hypothesis="Baseline",
            action_type="baseline",
            change_type="baseline",
            instructions="Run baseline.",
        ),
        outcome=ExperimentOutcome(primary_metric_value=0.5, artifacts=["baseline.pkl"]),
        reflection=ReflectionSummary(assessment="Baseline set."),
        review=ReviewDecision(status="accepted", reason="ok"),
    )
    second_record = IterationRecord(
        iteration_id=2,
        parent_iteration_id=1,
        candidate=ExperimentCandidate(
            hypothesis="Leaky feature tweak",
            action_type="train",
            change_type="train",
            instructions="Train with the champion column.",
        ),
        outcome=ExperimentOutcome(primary_metric_value=0.3, artifacts=["leaky.pkl"]),
        reflection=ReflectionSummary(assessment="Looks better."),
        review=ReviewDecision(status="accepted", reason="ok"),
    )
    first_summary = IterationSummary(
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
        result="unchanged",
        artifacts=["baseline.pkl"],
        lessons=["Keep the split fixed."],
        next_ideas=[],
        do_not_repeat=[],
        reflection_assessment="Baseline set.",
        review_reason="ok",
    )
    second_summary = IterationSummary(
        iteration_id=2,
        parent_iteration_id=1,
        hypothesis="Leaky feature tweak",
        action_type="train",
        change_type="train",
        instructions="Train with the champion column.",
        config_patch={},
        primary_metric_name="log_loss",
        primary_metric_value=0.3,
        secondary_metrics={},
        result="improved",
        artifacts=["leaky.pkl"],
        lessons=["Do not use post-outcome features."],
        next_ideas=[],
        do_not_repeat=[],
        reflection_assessment="Looks better.",
        review_reason="ok",
    )

    store.append_iteration_record(first_record)
    store.append_accepted_summary(first_summary)
    store.write_best_summary(first_summary)
    store.append_agent_update(
        AgentUpdate(stage="iteration", iteration_id=1, message="Iteration 1 accepted.")
    )
    store.append_iteration_record(second_record)
    store.append_accepted_summary(second_summary)
    store.write_best_summary(second_summary)
    store.append_agent_update(
        AgentUpdate(stage="iteration", iteration_id=2, message="Iteration 2 accepted.")
    )

    reopened = store.reopen_last_iteration()
    snapshot = store.load_snapshot(capability_context=CapabilityContext())
    updates = store.read_agent_updates()
    journal = (
        tmp_path / "memory" / "agent_markdown" / "experiment_journal.md"
    ).read_text(encoding="utf-8")

    assert reopened is not None
    assert reopened.iteration_id == 2
    assert snapshot.next_iteration_id == 2
    assert [record.iteration_id for record in snapshot.recent_records] == [1]
    assert [summary.iteration_id for summary in snapshot.recent_summaries] == [1]
    assert snapshot.best_summary is not None
    assert snapshot.best_summary.iteration_id == 1
    assert [update.iteration_id for update in updates] == [1]
    assert "Iteration 2" not in journal
    assert "Do not use post-outcome features." not in (
        tmp_path / "memory" / "agent_markdown" / "lessons_learned.md"
    ).read_text(encoding="utf-8")
