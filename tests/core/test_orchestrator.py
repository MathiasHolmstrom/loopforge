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
from tests.support import FakeReflectionBackend, FakeReviewBackend, build_spec


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
                execution_steps=[ExecutionStep(kind="shell", command="python src/train.py")],
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
    assert all(result.record.outcome.status == "recoverable_failure" for result in results)
