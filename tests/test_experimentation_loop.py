from __future__ import annotations

import json
from pathlib import Path

from loopforge import (
    AdapterSetup,
    CapabilityContext,
    ExperimentCandidate,
    ExperimentOrchestrator,
    ExperimentOutcome,
    ExperimentSpec,
    FileMemoryStore,
    MetricResult,
    MetricSpec,
    PrimaryMetric,
    ReflectionSummary,
    ReviewDecision,
    RoutingExperimentExecutor,
)
from loopforge.cli import append_human_intervention, run_from_spec


class FakeWorkerBackend:
    def __init__(self, candidates: list[ExperimentCandidate]) -> None:
        self._candidates = list(candidates)
        self.snapshots = []

    def propose_next_experiment(self, snapshot):
        self.snapshots.append(snapshot)
        return self._candidates.pop(0)


class StaticActionExecutor:
    def __init__(self, outcomes: list[ExperimentOutcome]) -> None:
        self._outcomes = list(outcomes)

    def execute(self, candidate: ExperimentCandidate, snapshot):
        return self._outcomes.pop(0)


class FakeReflectionBackend:
    def __init__(self, reflections: list[ReflectionSummary]) -> None:
        self._reflections = list(reflections)

    def reflect(self, snapshot, candidate, outcome):
        return self._reflections.pop(0)


class FakeReviewBackend:
    def __init__(self, decisions: list[ReviewDecision]) -> None:
        self._decisions = list(decisions)

    def review(self, snapshot, candidate, outcome, reflection):
        return self._decisions.pop(0)


def build_spec(**overrides) -> ExperimentSpec:
    payload = {
        "objective": "Improve pass outcome validation loss.",
        "primary_metric": PrimaryMetric(name="log_loss", goal="minimize"),
        "allowed_actions": ["baseline", "eda", "train", "tune", "evaluate"],
        "constraints": {"max_runtime_minutes": 30},
        "search_space": {"learning_rate": [0.01, 0.03, 0.05]},
        "stop_conditions": {"max_iterations": 4, "patience": 2},
        "metadata": {"model_key": "pass_outcome"},
    }
    payload.update(overrides)
    return ExperimentSpec(**payload)


def test_only_accepted_memory_reaches_the_next_worker_and_human_notes_modify_effective_spec(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            ExperimentCandidate(
                hypothesis="Do EDA next",
                action_type="eda",
                change_type="slice",
                instructions="Slice by rest days.",
            ),
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={
                "baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["Baseline done."])]),
                "eda": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["Rest-day slice found drift."])]),
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(assessment="Useful baseline.", lessons=["Baseline established."]),
                ReflectionSummary(assessment="Rest-day slice is worth following up.", lessons=["Rest days matter."]),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="Use this as starting memory."),
                ReviewDecision(status="accepted", reason="EDA finding is actionable."),
            ]
        ),
    )
    orchestrator.initialize(spec=build_spec())

    first_cycle = orchestrator.run_iteration()
    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Focus on EDA before more tuning.",
        effects={"disable_actions": ["tune"], "force_next_action": "eda", "suggested_hypothesis": "Check rest-day drift"},
    )
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary is not None
    assert worker.snapshots[0].recent_summaries == []
    assert [summary.hypothesis for summary in worker.snapshots[1].recent_summaries] == ["Baseline"]
    assert worker.snapshots[1].effective_spec.allowed_actions == ["baseline", "eda", "train", "evaluate"]
    assert worker.snapshots[1].effective_spec.metadata["force_next_action"] == "eda"
    assert worker.snapshots[1].effective_spec.metadata["suggested_hypotheses"] == ["Check rest-day drift"]


def test_orchestrator_injects_capability_context_into_worker_snapshot(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41)])}
        ),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
        capability_provider=lambda effective_spec: CapabilityContext(
            available_actions={"baseline": "Run baseline"},
            available_entities={"model_keys": ["pass_outcome", "fumble_expanded"]},
            available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            available_metrics={
                "log_loss": {"scorer_ref": "metrics.log_loss_cv"},
                "recall_at_p90": {"scorer_ref": "metrics.recall_at_p90"},
            },
            environment_facts={"adapter": "test"},
        ),
    )
    orchestrator.initialize(spec=build_spec())

    orchestrator.run_iteration()

    assert worker.snapshots[0].capability_context.available_entities["model_keys"] == [
        "pass_outcome",
        "fumble_expanded",
    ]
    assert worker.snapshots[0].capability_context.available_data_assets == ["gold_cross_validated_pass_outcome_ncaaf"]
    assert worker.snapshots[0].capability_context.available_metrics["log_loss"]["scorer_ref"] == "metrics.log_loss_cv"


def test_rejected_review_stays_in_raw_records_but_not_in_accepted_memory(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Suspicious train run",
                    action_type="train",
                    change_type="feature_set",
                    instructions="Try a risky feature.",
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            handlers={"train": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.3)])}
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Looks leaky.", lessons=["Need leakage check."])]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="rejected", reason="Do not trust this run.")]
        ),
    )
    orchestrator.initialize(spec=build_spec(allowed_actions=["train"]))

    cycle = orchestrator.run_iteration()
    snapshot = store.load_snapshot()
    records_text = (tmp_path / "loop" / "iteration_records.jsonl").read_text(encoding="utf-8")

    assert cycle.accepted_summary is None
    assert snapshot.latest_summary is None
    assert '"status": "rejected"' in records_text


def test_cli_interjection_appends_human_note_and_changes_future_effective_spec(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())

    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Only do EDA next.",
        effects={"disable_actions": ["train", "tune"], "force_next_action": "eda"},
        author="mhol",
        type_="override",
    )
    snapshot = store.load_snapshot()

    assert snapshot.recent_human_interventions[0].author == "mhol"
    assert snapshot.effective_spec.allowed_actions == ["baseline", "eda", "evaluate"]
    assert snapshot.effective_spec.metadata["force_next_action"] == "eda"


def test_run_from_spec_supports_factory_and_returns_reviewed_cycle_results(tmp_path, monkeypatch) -> None:
    spec = build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 1})
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    class StubWorker:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )

    class StubReflection:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def reflect(self, snapshot, candidate, outcome):
            return ReflectionSummary(assessment="Baseline is fine.", lessons=["Keep this baseline."])

    class StubReview:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def review(self, snapshot, candidate, outcome, reflection):
            return ReviewDecision(status="accepted", reason="Approved into memory.")

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.39)])},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_entities={"model_keys": ["pass_outcome"]},
                available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            ),
        )

    monkeypatch.setattr("loopforge.cli.LiteLLMWorkerBackend", StubWorker)
    monkeypatch.setattr("loopforge.cli.LiteLLMReflectionBackend", StubReflection)
    monkeypatch.setattr("loopforge.cli.LiteLLMReviewBackend", StubReview)
    monkeypatch.setattr("loopforge.cli.load_factory", lambda _: fake_factory)

    results = run_from_spec(
        spec_path=spec_path,
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        worker_model="openai/gpt-5.4",
    )

    assert results[0]["accepted_summary"]["hypothesis"] == "Baseline"
    assert results[0]["record"]["review"]["status"] == "accepted"


def test_metric_results_and_guardrails_drive_summary_classification(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Push recall while holding precision guardrail.",
                action_type="train",
                change_type="threshold",
                instructions="Tune the scorer threshold.",
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={
                "train": StaticActionExecutor(
                    [
                        ExperimentOutcome(
                            metric_results={
                                "recall_at_p90": MetricResult(
                                    name="recall_at_p90",
                                    value=0.74,
                                    scorer_ref="metrics.recall_at_p90",
                                ),
                                "precision_floor": MetricResult(
                                    name="precision_floor",
                                    value=0.88,
                                    scorer_ref="metrics.precision_floor",
                                ),
                            },
                            notes=["Recall improved but precision dipped."],
                        )
                    ]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Primary metric improved but the guardrail failed.")]
        ),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="Track this in memory.")]),
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["train"],
            primary_metric=PrimaryMetric(
                name="recall_at_p90",
                goal="maximize",
                scorer_ref="metrics.recall_at_p90",
                display_name="Recall @ Precision>=0.90",
            ),
            secondary_metrics=[
                MetricSpec(
                    name="log_loss",
                    goal="minimize",
                    scorer_ref="metrics.log_loss_cv",
                )
            ],
            guardrail_metrics=[
                MetricSpec(
                    name="precision_floor",
                    goal="maximize",
                    scorer_ref="metrics.precision_floor",
                    constraints={"min_value": 0.9},
                )
            ],
        )
    )

    cycle = orchestrator.run_iteration()

    assert cycle.accepted_summary is not None
    assert cycle.accepted_summary.primary_metric_value == 0.74
    assert cycle.accepted_summary.metric_results["recall_at_p90"].scorer_ref == "metrics.recall_at_p90"
    assert cycle.accepted_summary.guardrail_failures == ["precision_floor"]
    assert cycle.accepted_summary.result == "regressed"

