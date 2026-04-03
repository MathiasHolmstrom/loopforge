from __future__ import annotations

from typing import Protocol

from loopforge.core.backends import ReflectionBackend, ReviewBackend, WorkerBackend
from loopforge.core.memory import FileMemoryStore
from loopforge.core.types import (
    CapabilityContext,
    ExperimentCandidate,
    ExperimentOutcome,
    ExperimentSpec,
    IterationCycleResult,
    IterationRecord,
    IterationResult,
    IterationSummary,
    MemorySnapshot,
    PrimaryMetric,
)


class ActionExecutor(Protocol):
    def execute(self, candidate: ExperimentCandidate, snapshot: MemorySnapshot) -> ExperimentOutcome: ...


class RoutingExperimentExecutor:
    def __init__(self, handlers: dict[str, ActionExecutor]) -> None:
        self.handlers = handlers

    def execute(self, candidate: ExperimentCandidate, snapshot: MemorySnapshot) -> ExperimentOutcome:
        if candidate.action_type not in snapshot.effective_spec.allowed_actions:
            raise ValueError(
                f"Worker proposed action_type={candidate.action_type!r}, "
                f"which is not allowed by the effective spec."
            )
        try:
            handler = self.handlers[candidate.action_type]
        except KeyError as exc:
            raise ValueError(f"No executor registered for action_type={candidate.action_type!r}.") from exc
        return handler.execute(candidate, snapshot)


class ExperimentOrchestrator:
    def __init__(
        self,
        memory_store: FileMemoryStore,
        worker_backend: WorkerBackend,
        executor: RoutingExperimentExecutor,
        reflection_backend: ReflectionBackend,
        review_backend: ReviewBackend,
        capability_provider=None,
        summary_window: int = 5,
        human_window: int = 10,
    ) -> None:
        self.memory_store = memory_store
        self.worker_backend = worker_backend
        self.executor = executor
        self.reflection_backend = reflection_backend
        self.review_backend = review_backend
        self.capability_provider = capability_provider
        self.summary_window = summary_window
        self.human_window = human_window

    def initialize(self, spec: ExperimentSpec) -> None:
        self.memory_store.initialize(spec=spec)

    def run_iteration(self) -> IterationCycleResult:
        snapshot = self._load_snapshot()
        candidate = self.worker_backend.propose_next_experiment(snapshot)
        outcome = self.executor.execute(candidate, snapshot)
        reflection = self.reflection_backend.reflect(snapshot, candidate, outcome)
        review = self.review_backend.review(snapshot, candidate, outcome, reflection)
        record = IterationRecord(
            iteration_id=snapshot.next_iteration_id,
            parent_iteration_id=snapshot.latest_summary.iteration_id if snapshot.latest_summary is not None else None,
            candidate=candidate,
            outcome=outcome,
            reflection=reflection,
            review=review,
        )
        self.memory_store.append_iteration_record(record)

        accepted_summary: IterationSummary | None = None
        if review.status == "accepted" and review.should_update_memory:
            accepted_summary = self._build_summary(snapshot, record)
            self.memory_store.append_accepted_summary(accepted_summary)
            if accepted_summary.result == "improved":
                self.memory_store.write_best_summary(accepted_summary)
        return IterationCycleResult(record=record, accepted_summary=accepted_summary)

    def run(self, iterations: int | None = None) -> list[IterationCycleResult]:
        snapshot = self._load_snapshot()
        max_iterations = iterations
        if max_iterations is None:
            max_iterations = int(snapshot.effective_spec.stop_conditions.get("max_iterations", 1))
        patience = snapshot.effective_spec.stop_conditions.get("patience")
        no_improvement_streak = 0
        results: list[IterationCycleResult] = []
        for _ in range(max_iterations):
            cycle_result = self.run_iteration()
            results.append(cycle_result)
            accepted_summary = cycle_result.accepted_summary
            if accepted_summary is not None and accepted_summary.result == "improved":
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
            if patience is not None and no_improvement_streak >= int(patience):
                break
        return results

    def _load_snapshot(self) -> MemorySnapshot:
        preview_snapshot = self.memory_store.load_snapshot(
            summary_window=self.summary_window,
            human_window=self.human_window,
            capability_context=CapabilityContext(),
        )
        capability_context = (
            self.capability_provider(preview_snapshot.effective_spec)
            if self.capability_provider is not None
            else CapabilityContext()
        )
        return self.memory_store.load_snapshot(
            summary_window=self.summary_window,
            human_window=self.human_window,
            capability_context=capability_context,
        )

    def _build_summary(self, snapshot: MemorySnapshot, record: IterationRecord) -> IterationSummary:
        metric_results = record.outcome.resolved_metric_results(snapshot.effective_spec)
        primary_result = metric_results.get(snapshot.effective_spec.primary_metric.name)
        if primary_result is None or primary_result.value is None:
            raise ValueError(
                f"Outcome did not include a value for the primary metric "
                f"{snapshot.effective_spec.primary_metric.name!r}."
            )
        guardrail_failures = self._guardrail_failures(snapshot.effective_spec, metric_results)
        result = self._classify_result(
            primary_metric=snapshot.effective_spec.primary_metric,
            candidate_value=primary_result.value,
            best_value=snapshot.best_summary.primary_metric_value if snapshot.best_summary is not None else None,
            guardrail_failures=guardrail_failures,
        )
        secondary_metric_names = {metric.name for metric in snapshot.effective_spec.secondary_metrics}
        secondary_metrics = {
            name: result.value
            for name, result in metric_results.items()
            if name in secondary_metric_names and result.value is not None
        }
        return IterationSummary(
            iteration_id=record.iteration_id,
            parent_iteration_id=record.parent_iteration_id,
            hypothesis=record.candidate.hypothesis,
            action_type=record.candidate.action_type,
            change_type=record.candidate.change_type,
            instructions=record.candidate.instructions,
            config_patch=record.candidate.config_patch,
            primary_metric_name=snapshot.effective_spec.primary_metric.name,
            primary_metric_value=primary_result.value,
            secondary_metrics=secondary_metrics,
            result=result,
            artifacts=record.outcome.artifacts,
            lessons=[*record.outcome.notes, *record.reflection.lessons],
            next_ideas=record.outcome.next_ideas,
            do_not_repeat=record.outcome.do_not_repeat,
            reflection_assessment=record.reflection.assessment,
            review_reason=record.review.reason,
            metric_results=metric_results,
            guardrail_failures=guardrail_failures,
            dataset_version=record.outcome.dataset_version,
            code_or_config_changes=record.outcome.code_or_config_changes,
            candidate_fingerprint=record.outcome.candidate_fingerprint,
        )

    @staticmethod
    def _classify_result(
        primary_metric: PrimaryMetric,
        candidate_value: float,
        best_value: float | None,
        guardrail_failures: list[str],
    ) -> IterationResult:
        if guardrail_failures:
            return "regressed"
        if best_value is None:
            return "improved"
        if primary_metric.is_improvement(candidate=candidate_value, incumbent=best_value):
            return "improved"
        return "regressed"

    @staticmethod
    def _guardrail_failures(
        spec: ExperimentSpec,
        metric_results,
    ) -> list[str]:
        failures = []
        for metric in spec.guardrail_metrics:
            result = metric_results.get(metric.name)
            if metric.resolve_passed(result) is not True:
                failures.append(metric.name)
        return failures

