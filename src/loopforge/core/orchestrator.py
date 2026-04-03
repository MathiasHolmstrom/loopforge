from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from loopforge.core.backends import (
    NarrationBackend,
    ReflectionBackend,
    ReviewBackend,
    WorkerBackend,
)
from loopforge.core.memory import FileMemoryStore
from loopforge.core.runtime import is_generic_autonomous
from loopforge.core.types import (
    AgentUpdate,
    CapabilityContext,
    ExperimentCandidate,
    ExperimentInterrupted,
    ExperimentOutcome,
    ExperimentSpec,
    IterationCycleResult,
    IterationRecord,
    IterationResult,
    IterationSummary,
    MemorySnapshot,
    PrimaryMetric,
    ProgressFn,
    ReflectionSummary,
    ReviewDecision,
    RoleModelConfig,
    _noop_progress,
)

IterationCallback = Callable[[IterationCycleResult], None]


class ActionExecutor(Protocol):
    def execute(
        self, candidate: ExperimentCandidate, snapshot: MemorySnapshot
    ) -> ExperimentOutcome: ...


ExecutionRecoveryHandler = Callable[
    [ActionExecutor, ExperimentCandidate, MemorySnapshot, Exception],
    ExperimentOutcome | None,
]


class RoutingExperimentExecutor:
    def __init__(
        self,
        handlers: dict[str, ActionExecutor],
        plan_executor: ActionExecutor | None = None,
        recovery_handler: ExecutionRecoveryHandler | None = None,
    ) -> None:
        self.handlers = handlers
        self.plan_executor = plan_executor
        self.recovery_handler = recovery_handler

    def execute(
        self, candidate: ExperimentCandidate, snapshot: MemorySnapshot
    ) -> ExperimentOutcome:
        generic_autonomous = is_generic_autonomous(snapshot=snapshot)
        if candidate.execution_steps and self.plan_executor is not None:
            return self.plan_executor.execute(candidate, snapshot)
        if candidate.action_type not in snapshot.effective_spec.allowed_actions:
            if generic_autonomous:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[
                        f"Worker proposed a generic autonomous step label ({candidate.action_type}) without a concrete execution plan."
                    ],
                    failure_type="UnboundActionProposal",
                    failure_summary=(
                        f"The worker proposed action_type={candidate.action_type!r}, which is not part of the current "
                        "generic autonomous action set and did not include execution_steps."
                    ),
                    recoverable=True,
                    recovery_actions=[
                        "Propose bounded execution_steps for the next iteration.",
                        "Or relabel the step using the generic autonomous actions and include a concrete command or file edit.",
                    ],
                    execution_details={"candidate": candidate.to_dict()},
                )
            raise ValueError(
                f"Worker proposed action_type={candidate.action_type!r}, "
                f"which is not allowed by the effective spec."
            )
        try:
            handler = self.handlers[candidate.action_type]
        except KeyError as exc:
            if generic_autonomous:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[
                        f"No bound executor exists for generic autonomous action {candidate.action_type!r}."
                    ],
                    failure_type="MissingExecutionSteps",
                    failure_summary=(
                        f"No executor registered for action_type={candidate.action_type!r}. "
                        "In generic autonomous mode, the worker must provide concrete execution_steps."
                    ),
                    recoverable=True,
                    recovery_actions=[
                        "Propose bounded shell or file execution_steps for the next iteration.",
                        "If the step is exploratory, inspect the repo or run a minimal diagnostic command first.",
                    ],
                    execution_details={"candidate": candidate.to_dict()},
                )
            raise ValueError(
                f"No executor registered for action_type={candidate.action_type!r}."
            ) from exc
        try:
            return handler.execute(candidate, snapshot)
        except Exception as exc:
            if self.recovery_handler is not None:
                recovered = self.recovery_handler(handler, candidate, snapshot, exc)
                if recovered is not None:
                    return recovered
            raise


class ExperimentOrchestrator:
    def __init__(
        self,
        memory_store: FileMemoryStore,
        worker_backend: WorkerBackend,
        executor: RoutingExperimentExecutor,
        reflection_backend: ReflectionBackend,
        review_backend: ReviewBackend,
        narrator_backend: NarrationBackend | None = None,
        capability_provider=None,
        summary_window: int = 5,
        human_window: int = 10,
        monotonic_fn=time.monotonic,
        progress_fn: ProgressFn | None = None,
        role_models: RoleModelConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.worker_backend = worker_backend
        self.executor = executor
        self.reflection_backend = reflection_backend
        self.review_backend = review_backend
        self.narrator_backend = narrator_backend
        self.capability_provider = capability_provider
        self.summary_window = summary_window
        self.human_window = human_window
        self.monotonic_fn = monotonic_fn
        self.progress_fn: ProgressFn = progress_fn or _noop_progress
        self.role_models = role_models

    def initialize(self, spec: ExperimentSpec, *, reset_state: bool = False) -> None:
        self.memory_store.initialize(spec=spec, reset_state=reset_state)

    def run_iteration(self) -> IterationCycleResult:
        snapshot = self._load_snapshot()
        generic_autonomous = self._is_generic_autonomous(snapshot)
        max_continuations = int(
            snapshot.effective_spec.stop_conditions.get(
                "max_metricless_continuations",
                12 if generic_autonomous else 5,
            )
        )
        max_same_iteration_repairs = int(
            snapshot.effective_spec.stop_conditions.get(
                "max_same_iteration_repairs",
                8 if generic_autonomous else 0,
            )
        )
        intra_iteration_attempts: list[dict[str, object]] = []
        pending_success_outcome: ExperimentOutcome | None = None

        # ── Worker proposes ──
        worker_tag = f"[{self.role_models.worker}] " if self.role_models else ""
        self.progress_fn(
            "worker_propose", f"{worker_tag}Worker proposing next experiment..."
        )
        candidate = self.worker_backend.propose_next_experiment(snapshot)
        self.progress_fn("worker_detail", self._format_candidate(candidate))

        outcome: ExperimentOutcome | None = None
        metricless_continuations = 0
        repair_round = 0
        while True:
            if candidate.execution_steps:
                self.progress_fn("executor_run", "Executing autonomous agent plan...")
            else:
                self.progress_fn(
                    "executor_run", f"Executing {candidate.action_type}..."
                )
            try:
                outcome = self.executor.execute(candidate, snapshot)
            except Exception as exc:
                outcome = self._build_failure_outcome(exc)
            if pending_success_outcome is not None and outcome.status == "success":
                outcome = self._merge_outcomes(pending_success_outcome, outcome)
                pending_success_outcome = None
            intra_iteration_attempts.append(
                {
                    "attempt_number": len(intra_iteration_attempts) + 1,
                    "candidate": candidate.to_dict(),
                    "outcome": outcome.to_dict(),
                }
            )
            outcome = self._attach_intra_iteration_attempts(
                outcome, intra_iteration_attempts
            )
            self.progress_fn(
                "executor_detail",
                self._format_outcome(outcome, snapshot.effective_spec),
            )

            if outcome.status == "success" and self._outcome_has_metrics(
                outcome, snapshot.effective_spec
            ):
                break

            if outcome.status == "success":
                if metricless_continuations >= max_continuations:
                    if generic_autonomous:
                        outcome = self._block_metricless_iteration(
                            outcome,
                            intra_iteration_attempts,
                            max_continuations,
                        )
                    break
                self.progress_fn(
                    f"continue_{metricless_continuations}",
                    "No metrics produced — asking agent to continue the same iteration "
                    f"({metricless_continuations + 1}/{max_continuations})...",
                )
                try:
                    continuation_candidate = self.worker_backend.continue_experiment(
                        self._make_inflight_snapshot(snapshot, candidate, outcome),
                        candidate,
                        outcome,
                    )
                except Exception as exc:
                    self.progress_fn(
                        f"continue_{metricless_continuations}_error",
                        f"Continuation planning failed internally: {exc}. Proceeding with the current outcome.",
                    )
                    if generic_autonomous:
                        outcome = self._block_metricless_iteration(
                            outcome,
                            intra_iteration_attempts,
                            metricless_continuations,
                        )
                    break
                self.progress_fn(
                    f"continue_{metricless_continuations}_detail",
                    self._format_candidate(continuation_candidate),
                )
                pending_success_outcome = outcome
                candidate = continuation_candidate
                metricless_continuations += 1
                continue

            if (
                generic_autonomous
                and outcome.status == "recoverable_failure"
                and repair_round < max_same_iteration_repairs
            ):
                self.progress_fn(
                    f"repair_{repair_round}",
                    "Recoverable failure did not produce a successful experiment — asking worker to repair "
                    f"within the same iteration ({repair_round + 1}/{max_same_iteration_repairs})...",
                )
                try:
                    repaired_candidate = self.worker_backend.propose_next_experiment(
                        self._make_inflight_snapshot(snapshot, candidate, outcome),
                    )
                except Exception as exc:
                    self.progress_fn(
                        f"repair_{repair_round}_error",
                        f"Same-iteration repair planning failed internally: {exc}.",
                    )
                    outcome = self._block_repair_exhaustion(
                        outcome,
                        intra_iteration_attempts,
                        repair_round,
                        max_same_iteration_repairs,
                    )
                    break
                self.progress_fn(
                    f"repair_{repair_round}_detail",
                    self._format_candidate(repaired_candidate),
                )
                candidate = repaired_candidate
                repair_round += 1
                continue

            if generic_autonomous and outcome.status == "recoverable_failure":
                outcome = self._block_repair_exhaustion(
                    outcome,
                    intra_iteration_attempts,
                    repair_round,
                    max_same_iteration_repairs,
                )
            break

        if outcome is None:
            outcome = ExperimentOutcome(
                status="blocked",
                notes=["Iteration ended without executing a candidate."],
                failure_type="MissingIterationOutcome",
                failure_summary="The orchestrator finished the iteration without an execution outcome.",
            )

        # ── Reflect ──
        reflect_tag = f"[{self.role_models.reflection}] " if self.role_models else ""
        self.progress_fn("reflect", f"{reflect_tag}Reflecting on results...")
        try:
            reflection = self.reflection_backend.reflect(snapshot, candidate, outcome)
        except Exception as exc:
            reflection = self._fallback_reflection(outcome, exc)
        self.progress_fn("reflect_detail", self._format_reflection(reflection))

        # ── Review ──
        review_tag = f"[{self.role_models.review}] " if self.role_models else ""
        self.progress_fn("review", f"{review_tag}Reviewing iteration...")
        try:
            review = self.review_backend.review(
                snapshot, candidate, outcome, reflection
            )
        except Exception as exc:
            review = self._fallback_review(outcome, exc)
        self.progress_fn("review_detail", self._format_review(review))
        record = IterationRecord(
            iteration_id=snapshot.next_iteration_id,
            parent_iteration_id=snapshot.latest_summary.iteration_id
            if snapshot.latest_summary is not None
            else None,
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
        human_update = None
        if self.narrator_backend is not None:
            narrate_tag = f"[{self.role_models.narrator}] " if self.role_models else ""
            self.progress_fn("narrate", f"{narrate_tag}Writing iteration summary...")
            try:
                human_update = self.narrator_backend.summarize_iteration(
                    snapshot=snapshot,
                    candidate=candidate,
                    outcome=outcome,
                    reflection=reflection,
                    review=review,
                    accepted_summary=accepted_summary,
                )
            except Exception as exc:
                human_update = (
                    f"Iteration {record.iteration_id} completed with review status {review.status}. "
                    f"Narration backend failed: {exc}"
                )
            self.memory_store.append_agent_update(
                AgentUpdate(
                    stage="iteration",
                    iteration_id=record.iteration_id,
                    message=human_update,
                )
            )
        return IterationCycleResult(
            record=record, accepted_summary=accepted_summary, human_update=human_update
        )

    def run(
        self,
        iterations: int | None = None,
        iteration_callback: IterationCallback | None = None,
    ) -> list[IterationCycleResult]:
        snapshot = self._load_snapshot()
        max_iterations = iterations
        if max_iterations is None:
            max_iterations = int(
                snapshot.effective_spec.stop_conditions.get("max_iterations", 30)
            )
        max_autonomous_hours = snapshot.effective_spec.stop_conditions.get(
            "max_autonomous_hours", 6
        )
        max_runtime_seconds = None
        if max_autonomous_hours is not None:
            max_runtime_seconds = float(max_autonomous_hours) * 60 * 60
        patience = snapshot.effective_spec.stop_conditions.get("patience")
        no_improvement_streak = 0
        results: list[IterationCycleResult] = []
        started_at = self.monotonic_fn()
        for i in range(max_iterations):
            if (
                max_runtime_seconds is not None
                and (self.monotonic_fn() - started_at) >= max_runtime_seconds
            ):
                break
            self.progress_fn(
                "iteration_start", f"Starting iteration {i + 1}/{max_iterations}..."
            )
            try:
                cycle_result = self.run_iteration()
            except KeyboardInterrupt:
                raise ExperimentInterrupted(
                    results_so_far=results,
                    current_stage="iteration",
                    snapshot=self._load_snapshot(),
                )
            if iteration_callback is not None:
                iteration_callback(cycle_result)
            results.append(cycle_result)
            if cycle_result.record.outcome.status == "blocked":
                break
            accepted_summary = cycle_result.accepted_summary
            if accepted_summary is not None and accepted_summary.result == "improved":
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
            if patience is not None and no_improvement_streak >= int(patience):
                break
        return results

    @staticmethod
    def _format_candidate(candidate: ExperimentCandidate) -> str:
        lines = [
            f"  Hypothesis : {candidate.hypothesis}",
        ]
        if candidate.instructions:
            instr = candidate.instructions
            if len(instr) > 400:
                instr = instr[:400] + "..."
            lines.append(f"  Plan       : {instr}")
        if candidate.execution_steps:
            # Summarize steps at a high level — don't dump raw commands
            step_summaries = []
            for step in candidate.execution_steps:
                if step.rationale:
                    step_summaries.append(step.rationale)
                elif step.kind in ("write_file", "append_file") and step.path:
                    step_summaries.append(f"create {step.path}")
                elif step.kind == "shell":
                    # Show a brief description, not the full command
                    cmd = step.command
                    if "python -c" in cmd:
                        step_summaries.append("run inline Python inspection")
                    elif "python " in cmd and ".py" in cmd:
                        import re

                        script_match = re.search(r"python\s+(\S+\.py)", cmd)
                        step_summaries.append(
                            f"run {script_match.group(1)}"
                            if script_match
                            else "run Python script"
                        )
                    elif cmd:
                        short = cmd[:80] + "..." if len(cmd) > 80 else cmd
                        step_summaries.append(short)
            if step_summaries:
                lines.append(f"  Steps      : {' → '.join(step_summaries[:5])}")
        return "\n".join(lines)

    @staticmethod
    def _format_outcome(outcome: ExperimentOutcome, spec: ExperimentSpec) -> str:
        lines = []
        # Metrics first — this is what the user cares about most
        metrics_parts = []
        resolved = outcome.resolved_metric_results(spec)
        for name, result in resolved.items():
            if result.value is not None:
                metrics_parts.append(f"{name} = {result.value:.4g}")
        if not metrics_parts and outcome.primary_metric_value is not None:
            metrics_parts.append(f"primary = {outcome.primary_metric_value:.4g}")
        if metrics_parts:
            lines.append(f"  Metrics    : {' | '.join(metrics_parts)}")
        elif outcome.status == "success":
            lines.append("  Result     : completed successfully (no metrics reported)")
        # Failure info
        if outcome.status != "success":
            summary = outcome.failure_summary or "execution failed"
            if len(summary) > 200:
                summary = summary[:200] + "..."
            lines.append(f"  Issue      : {summary}")
            if outcome.recovery_actions:
                lines.append(f"  Next       : {outcome.recovery_actions[0]}")
        # Key notes (skip noise)
        for note in outcome.notes[:2]:
            if len(note) > 150:
                note = note[:150] + "..."
            lines.append(f"  Note       : {note}")
        return "\n".join(lines)

    @staticmethod
    def _format_reflection(reflection: ReflectionSummary) -> str:
        assessment = reflection.assessment
        if len(assessment) > 400:
            assessment = assessment[:400] + "..."
        lines = [f"  Assessment : {assessment}"]
        if reflection.lessons:
            for lesson in reflection.lessons[:5]:
                lines.append(f"    Lesson   : {lesson}")
        if reflection.risks:
            for risk in reflection.risks[:3]:
                lines.append(f"    Risk     : {risk}")
        if reflection.recommended_next_action:
            next_action = reflection.recommended_next_action
            if isinstance(next_action, dict):
                # Extract human-readable parts from the dict
                action_type = next_action.get("action_type", "")
                reason = next_action.get("reason", next_action.get("hypothesis", ""))
                if reason:
                    lines.append(
                        f"  Next       : {action_type} — {reason}"
                        if action_type
                        else f"  Next       : {reason}"
                    )
                elif action_type:
                    lines.append(f"  Next       : {action_type}")
            elif isinstance(next_action, str):
                lines.append(f"  Next       : {next_action}")
        return "\n".join(lines)

    @staticmethod
    def _format_review(review: ReviewDecision) -> str:
        return f"  Decision   : {review.status} — {review.reason}"

    @staticmethod
    def _is_generic_autonomous(snapshot: MemorySnapshot) -> bool:
        return is_generic_autonomous(snapshot=snapshot)

    @staticmethod
    def _attach_intra_iteration_attempts(
        outcome: ExperimentOutcome,
        attempts: list[dict[str, object]],
    ) -> ExperimentOutcome:
        return replace(
            outcome,
            execution_details={
                **outcome.execution_details,
                "intra_iteration_attempts": list(attempts),
            },
        )

    def _make_inflight_snapshot(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
    ) -> MemorySnapshot:
        synthetic_record = IterationRecord(
            iteration_id=snapshot.next_iteration_id,
            parent_iteration_id=snapshot.latest_summary.iteration_id
            if snapshot.latest_summary is not None
            else None,
            candidate=candidate,
            outcome=outcome,
            reflection=ReflectionSummary(
                assessment="In-flight same-iteration state. Use the concrete failure or missing-metrics outcome to repair the current iteration.",
            ),
            review=ReviewDecision(
                status="accepted",
                reason="Synthetic same-iteration record for worker repair context.",
                should_update_memory=False,
            ),
        )
        return replace(
            snapshot, recent_records=[*snapshot.recent_records, synthetic_record]
        )

    def _block_metricless_iteration(
        self,
        outcome: ExperimentOutcome,
        attempts: list[dict[str, object]],
        continuation_count: int,
    ) -> ExperimentOutcome:
        blocked = replace(
            outcome,
            status="blocked",
            recoverable=False,
            failure_type="MetriclessIterationExhausted",
            failure_summary=(
                "Generic autonomous mode exhausted same-iteration continuations without producing the configured "
                "primary metric."
            ),
            recovery_actions=[
                "Inspect the recorded same-iteration attempts and fix the metric reporting or evaluation path before retrying.",
            ],
            notes=[
                *outcome.notes,
                "Loopforge refused to advance to the next iteration because no successful metric-producing experiment was completed.",
                f"Same-iteration continuations attempted: {continuation_count}.",
            ],
        )
        return self._attach_intra_iteration_attempts(blocked, attempts)

    def _block_repair_exhaustion(
        self,
        outcome: ExperimentOutcome,
        attempts: list[dict[str, object]],
        repair_round: int,
        max_same_iteration_repairs: int,
    ) -> ExperimentOutcome:
        blocked = replace(
            outcome,
            status="blocked",
            recoverable=False,
            failure_type="IterationRepairBudgetExceeded",
            failure_summary=(
                "Generic autonomous mode exhausted same-iteration repair attempts before producing a successful "
                "experiment."
            ),
            recovery_actions=[
                "Inspect the recorded same-iteration attempts and the latest execution failure before retrying.",
            ],
            notes=[
                *outcome.notes,
                "Loopforge refused to advance to the next iteration because the current iteration never reached a successful experiment.",
                f"Same-iteration worker repairs attempted: {repair_round}/{max_same_iteration_repairs}.",
            ],
        )
        return self._attach_intra_iteration_attempts(blocked, attempts)

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

    def _build_summary(
        self, snapshot: MemorySnapshot, record: IterationRecord
    ) -> IterationSummary:
        metric_results = record.outcome.resolved_metric_results(snapshot.effective_spec)
        primary_result = metric_results.get(snapshot.effective_spec.primary_metric.name)
        guardrail_failures = self._guardrail_failures(
            snapshot.effective_spec, metric_results
        )
        result = self._classify_result(
            action_type=record.candidate.action_type,
            capability_context=snapshot.capability_context,
            primary_metric=snapshot.effective_spec.primary_metric,
            candidate_value=primary_result.value
            if primary_result is not None
            else None,
            best_value=snapshot.best_summary.primary_metric_value
            if snapshot.best_summary is not None
            else None,
            guardrail_failures=guardrail_failures,
            outcome_status=record.outcome.status,
        )
        secondary_metric_names = {
            metric.name for metric in snapshot.effective_spec.secondary_metrics
        }
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
            primary_metric_value=primary_result.value
            if primary_result is not None
            else None,
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
            outcome_status=record.outcome.status,
            failure_type=record.outcome.failure_type,
            failure_summary=record.outcome.failure_summary,
            recovery_actions=record.outcome.recovery_actions,
        )

    @staticmethod
    def _classify_result(
        action_type: str,
        capability_context: CapabilityContext,
        primary_metric: PrimaryMetric,
        candidate_value: float | None,
        best_value: float | None,
        guardrail_failures: list[str],
        outcome_status: str,
    ) -> IterationResult:
        if outcome_status != "success" or candidate_value is None:
            return "inconclusive"
        if guardrail_failures:
            return "regressed"
        inconclusive_actions = set(
            capability_context.environment_facts.get("inconclusive_actions", [])
        )
        if action_type in inconclusive_actions:
            return "inconclusive"
        if best_value is None:
            return "improved"
        if primary_metric.is_improvement(
            candidate=candidate_value, incumbent=best_value
        ):
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

    @staticmethod
    def _outcome_has_metrics(outcome: ExperimentOutcome, spec: ExperimentSpec) -> bool:
        resolved = outcome.resolved_metric_results(spec)
        primary = resolved.get(spec.primary_metric.name)
        return primary is not None and primary.value is not None

    @staticmethod
    def _merge_outcomes(
        base: ExperimentOutcome, continuation: ExperimentOutcome
    ) -> ExperimentOutcome:
        """Merge a continuation outcome into the base, preferring the continuation's metrics."""
        merged_metrics = dict(base.metric_results)
        merged_metrics.update(continuation.metric_results)
        return replace(
            continuation,
            metric_results=merged_metrics,
            notes=[*base.notes, *continuation.notes],
            artifacts=[*base.artifacts, *continuation.artifacts],
            next_ideas=continuation.next_ideas or base.next_ideas,
            execution_details={
                "base_details": base.execution_details,
                "continuation_details": continuation.execution_details,
            },
        )

    @staticmethod
    def _build_failure_outcome(exc: Exception) -> ExperimentOutcome:
        failure_type = exc.__class__.__name__
        failure_summary = str(exc).strip() or failure_type
        lower_summary = failure_summary.lower()
        recoverable = isinstance(
            exc, (ImportError, ModuleNotFoundError, FileNotFoundError)
        ) or any(
            token in lower_summary
            for token in (
                "no module named",
                "file not found",
                "path not found",
                "command not found",
            )
        )
        recovery_actions: list[str] = []
        if (
            "permissionerror" in lower_summary
            and "access is denied" in lower_summary
            and any(
                token in lower_summary
                for token in ("socketpair", "multiprocessing", "_ssock", "_csock")
            )
        ):
            recoverable = True
            failure_type = "MultiprocessingPermissionError"
            recovery_actions.extend(
                [
                    "Retry with multiprocessing disabled or worker count reduced to 1.",
                    "Switch to a serial execution path that avoids socketpair/resource setup if available.",
                ]
            )
        if "no module named" in lower_summary:
            recovery_actions.append(
                "Install or sync the missing Python dependency, then retry the iteration."
            )
        if "file not found" in lower_summary or "path not found" in lower_summary:
            recovery_actions.append(
                "Resolve the missing file or path from repo/config context, then retry."
            )
        if not recovery_actions:
            recovery_actions.append(
                "Inspect the execution error, make the smallest fix, and retry."
            )
        return ExperimentOutcome(
            status="recoverable_failure" if recoverable else "blocked",
            notes=[f"Execution failed during the iteration: {failure_summary}"],
            next_ideas=list(recovery_actions),
            failure_type=failure_type,
            failure_summary=failure_summary,
            recoverable=recoverable,
            recovery_actions=recovery_actions,
        )

    @staticmethod
    def _fallback_reflection(
        outcome: ExperimentOutcome, exc: Exception
    ) -> ReflectionSummary:
        if outcome.status == "success":
            return ReflectionSummary(
                assessment=f"The run completed, but reflection generation failed: {exc}",
                lessons=[
                    "Inspect the raw outcome directly because reflection generation failed."
                ],
                risks=[str(exc).strip()] if str(exc).strip() else [],
            )
        return ReflectionSummary(
            assessment=(
                f"The iteration hit a {outcome.status.replace('_', ' ')}. "
                "Treat the failure details as the next debugging target."
            ),
            lessons=[
                outcome.failure_summary or "Execution failed.",
                "Build directly on this failure instead of restarting the plan.",
            ],
            risks=[str(exc).strip()] if str(exc).strip() else [],
        )

    @staticmethod
    def _fallback_review(outcome: ExperimentOutcome, exc: Exception) -> ReviewDecision:
        if outcome.status == "success":
            return ReviewDecision(
                status="accepted",
                reason=f"Accepted by fallback because review generation failed: {exc}",
                should_update_memory=True,
            )
        if outcome.recoverable:
            return ReviewDecision(
                status="accepted",
                reason=(
                    "Accepted recoverable failure into memory so the next worker can continue from the concrete error."
                ),
                should_update_memory=True,
            )
        return ReviewDecision(
            status="pending_human",
            reason=f"Execution hit a non-recoverable blocker and review generation failed: {exc}",
            should_update_memory=False,
        )
