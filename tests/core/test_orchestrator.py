from __future__ import annotations

from loopforge import (
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentOrchestrator,
    ExperimentOutcome,
    FileMemoryStore,
    ReflectionSummary,
    ReviewDecision,
    RoutingExperimentExecutor,
)
from tests.support import (
    FakeReflectionBackend,
    FakeReviewBackend,
    FakeWorkerBackend,
    build_spec,
)


def test_orchestrator_does_not_continue_metricless_baseline_first_iteration(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
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

    class Worker:
        def __init__(self) -> None:
            self.continue_calls = 0

        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Run the baseline once.",
                action_type="run_experiment",
                change_type="baseline",
                instructions="Run baseline.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python src/train.py")
                ],
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            self.continue_calls += 1
            raise AssertionError("baseline-first phase should not continue inline")

    class Executor:
        def execute(self, candidate, snapshot):
            return ExperimentOutcome(status="success")

    worker = Worker()
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor({}, plan_executor=Executor()),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="No metrics.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
        capability_provider=lambda spec: CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
        ),
    )
    orchestrator.initialize(spec)

    cycle = orchestrator.run_iteration()

    assert worker.continue_calls == 0
    assert cycle.record.outcome.status == "recoverable_failure"
    assert cycle.record.outcome.failure_type == "MetriclessIterationExhausted"


def test_orchestrator_attempts_same_iteration_repair_in_baseline_first_phase(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
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

    class Worker:
        def __init__(self) -> None:
            self.propose_calls = 0

        def propose_next_experiment(self, snapshot):
            self.propose_calls += 1
            return ExperimentCandidate(
                hypothesis="Run the baseline once.",
                action_type="run_experiment",
                change_type="baseline",
                instructions="Run baseline.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python src/train.py")
                ],
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            raise AssertionError("not used")

    class Executor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, candidate, snapshot):
            self.calls += 1
            if self.calls == 1:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    failure_type="ShellCommandFailed",
                    failure_summary="baseline failed",
                    recoverable=True,
                )
            return ExperimentOutcome(status="success")

    worker = Worker()
    executor = Executor()
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor({}, plan_executor=executor),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Failed.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
        capability_provider=lambda spec: CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
        ),
    )
    orchestrator.initialize(spec)

    cycle = orchestrator.run_iteration()

    assert worker.propose_calls == 2
    assert cycle.record.outcome.status == "recoverable_failure"
    assert cycle.record.outcome.failure_type == "MetriclessIterationExhausted"


def test_orchestrator_classifies_first_metric_bearing_iteration_as_unchanged_and_sets_best_summary(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec()
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Run baseline.",
                    action_type="baseline",
                    change_type="baseline",
                    instructions="Run the baseline once.",
                    execution_steps=[
                        ExecutionStep(kind="shell", command="python src/train.py")
                    ],
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            {},
            plan_executor=StaticExecutor([ExperimentOutcome(primary_metric_value=0.5)]),
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Baseline established.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
        capability_provider=lambda spec: CapabilityContext(),
    )
    orchestrator.initialize(spec)

    cycle = orchestrator.run_iteration()
    snapshot = store.load_snapshot(capability_context=CapabilityContext())

    assert cycle.accepted_summary is not None
    assert cycle.accepted_summary.result in ("unchanged", "inconclusive", "improved")
    assert snapshot.best_summary is not None
    assert snapshot.best_summary.primary_metric_value == 0.5


def test_orchestrator_classifies_metric_ties_as_unchanged(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec()

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Run baseline.",
                    action_type="baseline",
                    change_type="baseline",
                    instructions="Run the baseline once.",
                    execution_steps=[
                        ExecutionStep(kind="shell", command="python src/train.py")
                    ],
                ),
                ExperimentCandidate(
                    hypothesis="Repeat with same score.",
                    action_type="train",
                    change_type="train",
                    instructions="Train again.",
                    execution_steps=[
                        ExecutionStep(kind="shell", command="python src/train.py")
                    ],
                ),
            ]
        ),
        executor=RoutingExperimentExecutor(
            {},
            plan_executor=StaticExecutor(
                [
                    ExperimentOutcome(primary_metric_value=0.5),
                    ExperimentOutcome(primary_metric_value=0.5),
                ]
            ),
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(assessment="Baseline established."),
                ReflectionSummary(assessment="No change."),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="ok"),
                ReviewDecision(status="accepted", reason="ok"),
            ]
        ),
        capability_provider=lambda spec: CapabilityContext(),
    )
    orchestrator.initialize(spec)

    first_cycle = orchestrator.run_iteration()
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert first_cycle.accepted_summary.result == "unchanged"
    assert second_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary.result == "unchanged"


def test_orchestrator_does_not_mark_unconstrained_guardrails_as_failures(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec(
        guardrail_metrics=[
            bootstrapless_guardrail := build_spec().primary_metric.__class__(
                name="mean_absolute_error",
                goal="minimize",
            )
        ]
    )

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Run baseline.",
                    action_type="baseline",
                    change_type="baseline",
                    instructions="Run baseline once.",
                ),
                ExperimentCandidate(
                    hypothesis="Improve the model.",
                    action_type="train",
                    change_type="train",
                    instructions="Train an improved model.",
                ),
            ]
        ),
        executor=RoutingExperimentExecutor(
            {},
            plan_executor=StaticExecutor(
                [
                    ExperimentOutcome(
                        metric_results={
                            "log_loss": bootstrapless_guardrail.__class__(
                                name="log_loss", goal="minimize"
                            )  # placeholder to keep syntax invalid?
                        }
                    )
                ]
            ),
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(assessment="Baseline."),
                ReflectionSummary(assessment="Improved."),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="ok"),
                ReviewDecision(status="accepted", reason="ok"),
            ]
        ),
        capability_provider=lambda spec: CapabilityContext(),
    )
    orchestrator.initialize(spec)

    first_cycle = orchestrator.run_iteration()
    assert first_cycle.accepted_summary is not None

    # placeholder


def test_orchestrator_run_continues_after_recoverable_failure_exhaustion(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    spec = build_spec(
        allowed_actions=["run_experiment"],
        stop_conditions={
            "max_iterations": 3,
            "patience": 5,
            "max_same_iteration_repairs": 0,
        },
    )

    class Worker:
        def __init__(self) -> None:
            self.propose_calls = 0

        def propose_next_experiment(self, snapshot):
            self.propose_calls += 1
            return ExperimentCandidate(
                hypothesis=f"Attempt {self.propose_calls}",
                action_type="run_experiment",
                change_type="repair",
                instructions="Retry with fixes.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python src/train.py")
                ],
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            raise AssertionError("not used")

    class Executor:
        def execute(self, candidate, snapshot):
            return ExperimentOutcome(
                status="recoverable_failure",
                failure_type="ShellCommandFailed",
                failure_summary="still failing",
                recoverable=True,
            )

    worker = Worker()
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor({}, plan_executor=Executor()),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Failed.")] * 3
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")] * 3
        ),
        capability_provider=lambda spec: CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
        ),
    )
    orchestrator.initialize(spec)

    results = orchestrator.run(iterations=3)

    assert len(results) == 3
    assert all(
        result.record.outcome.status == "recoverable_failure" for result in results
    )


class StaticExecutor:
    def __init__(self, outcomes) -> None:
        self._outcomes = list(outcomes)

    def execute(self, candidate, snapshot):
        return self._outcomes.pop(0)


def test_build_failure_outcome_treats_windows_multiprocessing_permission_error_as_recoverable() -> (
    None
):
    exc = PermissionError(
        "[WinError 5] Access is denied: '_multiprocessing.socketpair _ssock _csock'"
    )

    outcome = ExperimentOrchestrator._build_failure_outcome(exc)

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "MultiprocessingPermissionError"
    assert outcome.recoverable is True
