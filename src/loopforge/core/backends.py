from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Protocol

from loopforge.core.types import (
    AccessGuide,
    BootstrapTurn,
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentSpec,
    ExperimentSpecProposal,
    ExperimentOutcome,
    MemorySnapshot,
    MetricResult,
    RoleModelConfig,
    ReflectionSummary,
    RunnerAuthoringRequest,
    RunnerAuthoringResult,
    ReviewDecision,
    ProgressFn,
    SpecQuestion,
    StreamFn,
    _noop_progress,
)
from loopforge.core.runtime import is_generic_autonomous


RECENT_RECORD_DIGEST_LIMIT = 2
RECENT_SUMMARY_DIGEST_LIMIT = 3
MARKDOWN_NOTE_CHAR_LIMIT = 1200
MARKDOWN_NOTE_COUNT_LIMIT = 4
TEXT_PREVIEW_CHAR_LIMIT = 400


DATA_SCIENCE_METRICS_REASONING = (
    "HOW TO THINK ABOUT EXPERIMENT DESIGN:\n"
    "\n"
    "You are a skilled data scientist. Derive answers from the objective  - don't ask the user "
    "to confirm what they already told you.\n"
    "\n"
    "1. PRIMARY METRIC: The user's objective directly implies it. "
    "'Improve X for subset Y' means primary = X measured on Y. Just use it.\n"
    "\n"
    "2. GUARDRAILS: Think about what COULD BREAK when you change things. "
    "A guardrail is something you're not optimizing but could be harmed by your changes. "
    "The key test: 'If I change the model to improve the primary metric, could this other metric get worse?' "
    "If yes, it's a guardrail. If no causal connection exists, ignore it.\n"
    "\n"
    "Key pattern  - the COMPLEMENT RULE: when optimizing for a subset (e.g. bench players), "
    "the complement subset's performance (e.g. starters) is naturally a guardrail, because "
    "model changes that help one group can hurt the other. The aggregate (overall) follows from the subsets.\n"
    "\n"
    "3. DATA PIPELINE: Look at the discovered columns, data assets, and schema files in the capability context. "
    "Think through how your proposed metrics will actually be computed from the data that exists. "
    "If a field you need doesn't exist directly, reason about how to get it: "
    "derive it from existing columns, find it in a schema definition (pandera, SQL DDL, ORM models), "
    "join from another table, or compute it as a transformation. "
    "State your plan clearly. Only ask the user when you genuinely can't figure out the path to the data.\n"
    "\n"
    "4. METRIC GOAL DIRECTION: For every metric you propose, you MUST explicitly set goal to 'minimize' or 'maximize'. "
    "Reason from what the metric actually measures: if a perfect prediction yields 0 and worse predictions yield "
    "higher values, that is minimize. If a perfect prediction yields 1.0 and worse predictions yield lower, "
    "that is maximize. Never leave goal unspecified or assume a default  - there is no safe default. "
    "Think about the metric's semantics, not just its name.\n"
    "\n"
    "5. DON'T ASK WHAT YOU CAN DERIVE: "
    "If the objective implies the metric, use it. "
    "If the complement subset is obvious, make it a guardrail. "
    "If a column can be derived from existing data, propose the derivation. "
    "If a metric from the repo has no causal connection to your changes, exclude it silently. "
    "Make a qualified guess when the objective and repo context make one path much more plausible than the others. "
    "Only ask when there is genuine ambiguity you cannot resolve from the objective and repo context."
)

EXISTING_IMPLEMENTATION_GROUNDING = (
    "GROUNDING IN THE CURRENT IMPLEMENTATION:\n"
    "\n"
    "Before you propose changes, identify the current baseline implementation from the capability context. "
    "Treat capability_context notes, environment facts, file paths, and discovered actions as the current source of truth.\n"
    "When capability_context includes likely implementation files or baseline code paths, treat those as stronger evidence "
    "than generic repo-wide symbol lists.\n"
    "\n"
    "If the repo context already names baseline files, model types, scorers, feature generators, or CV strategy, "
    "use that information directly in your reasoning. Do not ask the user to tell you what model currently exists "
    "when the repo context already answers it.\n"
    "\n"
    "When the objective is to improve an existing model, anchor your proposal on how that model works today: "
    "what target formulation it uses, which features it builds, which utilities it reuses, and how it is evaluated. "
    "Do not defer this basic inspection to a future worker if the current repo context already contains it.\n"
    "\n"
    "Prefer extending or adapting the existing pipeline over inventing a brand-new formulation. "
    "If you believe the current baseline should change from regression to ordinal/distribution or classification-like modeling, "
    "justify that change by referencing the existing implementation and adjacent repo patterns rather than guessing.\n"
)


class WorkerBackend(Protocol):
    def propose_next_experiment(
        self, snapshot: MemorySnapshot
    ) -> ExperimentCandidate: ...


class SpecBackend(Protocol):
    def propose_spec(
        self,
        objective: str,
        capability_context: CapabilityContext,
        user_preferences: dict[str, Any] | None = None,
    ) -> ExperimentSpecProposal: ...


class BootstrapBackend(Protocol):
    def propose_bootstrap_turn(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        answer_history: dict[str, Any] | None = None,
        role_models: RoleModelConfig | None = None,
        bootstrap_memory: dict[str, Any] | None = None,
    ) -> BootstrapTurn: ...


class ReflectionBackend(Protocol):
    def reflect(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
    ) -> ReflectionSummary: ...


class AccessAdvisorBackend(Protocol):
    def build_access_guide(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        preflight_checks,
    ) -> AccessGuide: ...


class RunnerAuthoringBackend(Protocol):
    def author_runner(
        self, request: RunnerAuthoringRequest
    ) -> RunnerAuthoringResult: ...


class ReviewBackend(Protocol):
    def review(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
        reflection: ReflectionSummary,
    ) -> ReviewDecision: ...


class NarrationBackend(Protocol):
    def summarize_bootstrap(
        self,
        turn: BootstrapTurn,
        capability_context: CapabilityContext,
    ) -> str: ...

    def summarize_iteration(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
        reflection: ReflectionSummary,
        review: ReviewDecision,
        accepted_summary,
    ) -> str: ...


class _LiteLLMJsonBackend:
    def __init__(
        self,
        model: str,
        completion_fn: Any | None = None,
        temperature: float = 0.2,
        max_completion_tokens: int | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        stream_fn: StreamFn | None = None,
        progress_fn: ProgressFn | None = None,
        heartbeat_seconds: float | None = None,
        heartbeat_schedule_seconds: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0),
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}
        self.max_completion_tokens = max_completion_tokens
        self._completion_fn = completion_fn
        self._stream_fn = stream_fn
        self._progress_fn: ProgressFn = progress_fn or _noop_progress
        self._heartbeat_seconds = heartbeat_seconds
        self._heartbeat_schedule_seconds = tuple(heartbeat_schedule_seconds)

    def _completion_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs = dict(self.extra_kwargs)
        if (
            self.max_completion_tokens is not None
            and "max_completion_tokens" not in kwargs
            and "max_tokens" not in kwargs
        ):
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        kwargs.update(overrides)
        return kwargs

    def _stream_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Call litellm with stream=True, emitting tokens via stream_fn."""
        from litellm import completion as litellm_completion

        response = litellm_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            **self._completion_kwargs(**kwargs),
        )
        chunks: list[str] = []
        assert self._stream_fn is not None
        for part in response:
            choices = getattr(part, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            token = getattr(delta, "content", None) or ""
            if token:
                chunks.append(token)
                self._stream_fn(token)
        self._stream_fn("\n")
        return "".join(chunks)

    def _run_with_progress(
        self,
        *,
        progress_stage: str | None,
        progress_label: str | None,
        fn,
    ):
        if not progress_stage or not progress_label:
            return fn()
        stop_event = threading.Event()
        started_at = time.monotonic()

        def heartbeat() -> None:
            tick = 0
            if self._heartbeat_seconds is not None:
                if self._heartbeat_seconds <= 0:
                    return
                while not stop_event.wait(self._heartbeat_seconds):
                    tick += 1
                    elapsed = int(time.monotonic() - started_at)
                    self._progress_fn(
                        f"{progress_stage}_wait_{tick}",
                        f"[{self.model}] Still {progress_label.lower()} ({elapsed}s elapsed)...",
                    )
                return

            checkpoints = self._heartbeat_schedule_seconds or (15.0, 30.0, 60.0, 120.0)
            checkpoint_index = 0
            next_target = checkpoints[0]
            repeat_interval = checkpoints[-1]
            while True:
                remaining = max(0.0, next_target - (time.monotonic() - started_at))
                if stop_event.wait(remaining):
                    return
                tick += 1
                elapsed = int(time.monotonic() - started_at)
                self._progress_fn(
                    f"{progress_stage}_wait_{tick}",
                    f"[{self.model}] Still {progress_label.lower()} ({elapsed}s elapsed)...",
                )
                checkpoint_index += 1
                if checkpoint_index < len(checkpoints):
                    next_target = checkpoints[checkpoint_index]
                else:
                    next_target += repeat_interval

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            return fn()
        finally:
            stop_event.set()
            thread.join(timeout=0.01)

    def _complete_text(
        self,
        system_prompt: str,
        user_message: str,
        *,
        progress_stage: str | None = None,
        progress_label: str | None = None,
    ) -> str:
        """Lightweight text completion  - no JSON parsing, just a plain response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        def run() -> str:
            if self._stream_fn and self._completion_fn is None:
                return self._stream_completion(messages)
            completion_fn = self._completion_fn
            if completion_fn is None:
                from litellm import completion as litellm_completion

                completion_fn = litellm_completion
            response = completion_fn(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **self._completion_kwargs(),
            )
            return self._extract_content(response)

        return self._run_with_progress(
            progress_stage=progress_stage,
            progress_label=progress_label,
            fn=run,
        )

    def _complete_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        *,
        progress_stage: str | None = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt} Return a valid JSON object only.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload, indent=2, sort_keys=True, default=self._json_default
                ),
            },
        ]

        def run() -> dict[str, Any]:
            if self._stream_fn and self._completion_fn is None:
                try:
                    content = self._stream_completion(
                        messages, response_format={"type": "json_object"}
                    )
                    return json.loads(content)
                except Exception:
                    if progress_stage and progress_label:
                        self._progress_fn(
                            f"{progress_stage}_stream_fallback",
                            f"[{self.model}] Streaming raw output is unavailable while {progress_label.lower()}; falling back to the standard response path.",
                        )
            completion_fn = self._completion_fn
            if completion_fn is None:
                from litellm import completion as litellm_completion

                completion_fn = litellm_completion
            response = completion_fn(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                **self._completion_kwargs(),
            )
            return json.loads(self._extract_content(response))

        return self._run_with_progress(
            progress_stage=progress_stage,
            progress_label=progress_label,
            fn=run,
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, date | datetime):
            return value.isoformat()
        if callable(getattr(value, "isoformat", None)):
            return str(value.isoformat())
        if hasattr(value, "tolist") and not isinstance(value, str):
            return value.tolist()
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return str(value)

    @staticmethod
    def _extract_content(response: Any) -> str:
        if isinstance(response, Mapping):
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("LiteLLM response did not contain choices.")
            message = choices[0].get("message", {})
            return _LiteLLMJsonBackend._normalise_content(message.get("content"))
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("LiteLLM response did not contain choices.")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("LiteLLM response choice did not contain a message.")
        return _LiteLLMJsonBackend._normalise_content(getattr(message, "content", None))

    @staticmethod
    def _normalise_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            if chunks:
                return "".join(chunks)
        raise ValueError("Could not extract JSON text from LiteLLM response content.")


def build_iteration_policy(snapshot: MemorySnapshot) -> dict[str, Any]:
    allowed_actions = list(snapshot.effective_spec.allowed_actions)
    generic_autonomous = is_generic_autonomous(snapshot=snapshot)
    first_iteration = not snapshot.recent_records
    forced_next_action = snapshot.effective_spec.metadata.get("force_next_action")
    if forced_next_action not in allowed_actions:
        forced_next_action = None
    latest_record = snapshot.recent_records[-1] if snapshot.recent_records else None
    if (
        forced_next_action is None
        and latest_record is not None
        and latest_record.outcome.status == "recoverable_failure"
    ):
        if generic_autonomous and "fix_failure" in allowed_actions:
            recommended_next_action = "fix_failure"
        elif latest_record.candidate.action_type in allowed_actions:
            recommended_next_action = latest_record.candidate.action_type
        else:
            recommended_next_action = forced_next_action or (
                allowed_actions[0] if allowed_actions else "explore"
            )
    else:
        recommended_next_action = forced_next_action or (
            allowed_actions[0] if allowed_actions else "explore"
        )
    return {
        "name": "guided_observation",
        "allowed_actions": allowed_actions,
        "generic_autonomous": generic_autonomous,
        "first_iteration": first_iteration,
        "prefer_smallest_runnable_script": generic_autonomous and first_iteration,
        "must_finish_current_iteration_with_metrics_or_block": generic_autonomous,
        "recent_failures": [
            {
                "iteration_id": record.iteration_id,
                "action_type": record.candidate.action_type,
                "failure_type": record.outcome.failure_type,
                "failure_summary": record.outcome.failure_summary,
                "recovery_actions": record.outcome.recovery_actions,
            }
            for record in snapshot.recent_records
            if record.outcome.status != "success"
        ],
        "recent_action_types": [
            summary.action_type for summary in snapshot.recent_summaries
        ],
        "forced_next_action": forced_next_action,
        "recommended_next_action": recommended_next_action,
    }


def _markdown_content_by_name(snapshot: MemorySnapshot, filename: str) -> str | None:
    for item in snapshot.markdown_memory:
        if item.path == filename or item.path.endswith(f"/{filename}"):
            return item.content
    return None


def _truncate_text(value: Any, *, limit: int = TEXT_PREVIEW_CHAR_LIMIT) -> str | None:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    if not compact:
        return None
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _compact_step(step: ExecutionStep) -> dict[str, Any]:
    payload = {
        "kind": step.kind,
        "command": step.command if step.kind == "shell" else None,
        "path": step.path,
        "cwd": step.cwd,
        "timeout_seconds": step.timeout_seconds,
        "allow_failure": step.allow_failure,
        "rationale": _truncate_text(step.rationale, limit=240),
    }
    if step.content:
        payload["content_preview"] = _truncate_text(step.content, limit=240)
    return {key: value for key, value in payload.items() if value not in (None, [], {})}


def _compact_candidate(candidate: ExperimentCandidate) -> dict[str, Any]:
    return {
        "hypothesis": _truncate_text(candidate.hypothesis, limit=220),
        "action_type": candidate.action_type,
        "change_type": candidate.change_type,
        "instructions": _truncate_text(candidate.instructions, limit=280),
        "execution_steps": [
            _compact_step(step) for step in candidate.execution_steps[:3]
        ],
        "config_patch": candidate.config_patch,
        "metadata": {
            key: _truncate_text(value, limit=220) if isinstance(value, str) else value
            for key, value in candidate.metadata.items()
            if key
            not in {
                "_execution_repair_context",
            }
        },
    }


def _compact_metric_results(metric_results: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name, result in list(metric_results.items())[:8]:
        if isinstance(result, MetricResult):
            compact[name] = {
                "value": result.value,
                "passed": result.passed,
            }
        else:
            compact[name] = result
    return compact


def _compact_step_result(step_result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "index": step_result.get("index"),
            "kind": step_result.get("kind"),
            "command": step_result.get("command"),
            "cwd": step_result.get("cwd"),
            "returncode": step_result.get("returncode"),
            "status": step_result.get("status"),
            "stdout_preview": _truncate_text(step_result.get("stdout")),
            "stderr_preview": _truncate_text(step_result.get("stderr")),
        }.items()
        if value not in (None, "")
    }


def _compact_execution_details(execution_details: dict[str, Any]) -> dict[str, Any]:
    step_results = execution_details.get("step_results", [])
    attempts = execution_details.get("attempts", [])
    latest_step = (
        step_results[-1] if isinstance(step_results, list) and step_results else {}
    )
    compact = {
        "attempt_count": len(attempts) if isinstance(attempts, list) else None,
        "step_result_count": len(step_results)
        if isinstance(step_results, list)
        else None,
        "latest_step_result": (
            _compact_step_result(latest_step) if isinstance(latest_step, dict) else None
        ),
        "latest_stdout_preview": _truncate_text(
            execution_details.get("latest_stdout_preview")
        ),
        "intra_iteration_attempt_count": len(
            execution_details.get("intra_iteration_attempts", [])
        )
        if isinstance(execution_details.get("intra_iteration_attempts"), list)
        else None,
    }
    return {key: value for key, value in compact.items() if value not in (None, "", {})}


def _compact_outcome(outcome: ExperimentOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "failure_type": outcome.failure_type,
        "failure_summary": _truncate_text(outcome.failure_summary),
        "recoverable": outcome.recoverable,
        "recovery_actions": outcome.recovery_actions[:3],
        "primary_metric_value": outcome.primary_metric_value,
        "metric_results": _compact_metric_results(outcome.metric_results),
        "notes": [
            _truncate_text(note, limit=220) for note in outcome.notes[:4] if note
        ],
        "next_ideas": [
            _truncate_text(idea, limit=220) for idea in outcome.next_ideas[:3] if idea
        ],
        "execution_details": _compact_execution_details(outcome.execution_details),
    }


def _compact_iteration_record(record) -> dict[str, Any]:
    return {
        "iteration_id": record.iteration_id,
        "parent_iteration_id": record.parent_iteration_id,
        "candidate": _compact_candidate(record.candidate),
        "outcome": _compact_outcome(record.outcome),
        "reflection": {
            "assessment": _truncate_text(record.reflection.assessment, limit=280),
            "recommended_next_action": record.reflection.recommended_next_action,
        },
        "review": {
            "status": record.review.status,
            "reason": _truncate_text(record.review.reason, limit=280),
        },
    }


def _compact_recent_records(snapshot: MemorySnapshot) -> list[dict[str, Any]]:
    records = snapshot.recent_records[-RECENT_RECORD_DIGEST_LIMIT:]
    return [_compact_iteration_record(record) for record in records]


def _compact_recent_summaries(snapshot: MemorySnapshot) -> list[dict[str, Any]]:
    summaries = snapshot.recent_summaries[-RECENT_SUMMARY_DIGEST_LIMIT:]
    return [_compact_iteration_summary(summary) for summary in summaries]


def _compact_human_interventions(snapshot: MemorySnapshot) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in snapshot.recent_human_interventions[-3:]:
        compact.append(
            {
                "author": item.author,
                "type": item.type,
                "message": _truncate_text(item.message, limit=240),
                "effects": item.effects,
            }
        )
    return compact


def _compact_iteration_summary(summary) -> dict[str, Any]:
    return {
        "iteration_id": summary.iteration_id,
        "action_type": summary.action_type,
        "result": summary.result,
        "outcome_status": summary.outcome_status,
        "primary_metric_name": summary.primary_metric_name,
        "primary_metric_value": summary.primary_metric_value,
        "failure_type": summary.failure_type,
        "failure_summary": _truncate_text(summary.failure_summary),
        "lessons": [
            _truncate_text(item, limit=200) for item in summary.lessons[:3] if item
        ],
        "next_ideas": [
            _truncate_text(item, limit=200) for item in summary.next_ideas[:2] if item
        ],
    }


def _compact_markdown_text(
    content: str | None, *, limit: int = MARKDOWN_NOTE_CHAR_LIMIT
) -> str | None:
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def _compact_markdown_memory(
    snapshot: MemorySnapshot,
    *,
    preferred_suffixes: tuple[str, ...] | None = None,
    limit: int = MARKDOWN_NOTE_COUNT_LIMIT,
) -> list[dict[str, Any]]:
    notes = list(snapshot.markdown_memory)
    if preferred_suffixes is not None:
        preferred = [
            note
            for note in notes
            if isinstance(note.path, str) and note.path.endswith(preferred_suffixes)
        ]
        if preferred:
            notes = preferred
    compact_notes = []
    for item in notes[:limit]:
        compact_content = _compact_markdown_text(item.content)
        if compact_content is None:
            continue
        compact_notes.append(
            {
                "path": item.path,
                "content": compact_content,
            }
        )
    return compact_notes


def _compact_bootstrap_handoff(snapshot: MemorySnapshot, filename: str) -> str | None:
    return _compact_markdown_text(_markdown_content_by_name(snapshot, filename))


def _has_markdown_name(snapshot: MemorySnapshot, filename: str) -> bool:
    return _markdown_content_by_name(snapshot, filename) is not None


def _worker_markdown_handoff(
    snapshot: MemorySnapshot, *, iteration_policy: dict[str, Any]
) -> list[dict[str, Any]]:
    preferred_suffixes = (
        "bootstrap_handoff.md",
        "execution_runbook.md",
        "experiment_guide.md",
        "ops_access_guide.md",
    )
    if not (
        iteration_policy.get("generic_autonomous")
        and iteration_policy.get("first_iteration")
    ):
        return _compact_markdown_memory(
            snapshot,
            preferred_suffixes=preferred_suffixes,
            limit=3,
        )
    return _compact_markdown_memory(
        snapshot,
        preferred_suffixes=preferred_suffixes,
        limit=4,
    )


def build_execution_handoff(snapshot: MemorySnapshot) -> dict[str, Any]:
    env = snapshot.capability_context.environment_facts
    metadata = (
        snapshot.effective_spec.metadata
        if isinstance(snapshot.effective_spec.metadata, dict)
        else {}
    )
    return {
        "repo_root": env.get("repo_root"),
        "execution_shell": env.get("execution_shell"),
        "shell_family": env.get("shell_family"),
        "python_executable": env.get("python_executable"),
        "execution_backend_kind": env.get("execution_backend_kind"),
        "execution_contract": metadata.get("execution_contract"),
        "verified_execution_lane": bool(env.get("python_executable"))
        and bool(env.get("repo_root")),
        "must_reuse_verified_lane": True,
        "available_bootstrap_handoff_files": sorted(
            item.path for item in snapshot.markdown_memory
        ),
        "bootstrap_handoff": _compact_bootstrap_handoff(
            snapshot, "bootstrap_handoff.md"
        ),
        "execution_runbook": _compact_bootstrap_handoff(
            snapshot, "execution_runbook.md"
        ),
        "experiment_guide": _compact_bootstrap_handoff(snapshot, "experiment_guide.md"),
    }


class LiteLLMWorkerBackend(_LiteLLMJsonBackend):
    @staticmethod
    def _candidate_from_payload(payload: dict[str, Any]) -> ExperimentCandidate:
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        for source_key, target_key in (
            ("observation_summary", "observation_summary"),
            ("reasoning_summary", "reasoning_summary"),
            ("next_step_summary", "next_step_summary"),
        ):
            value = payload.get(source_key)
            if isinstance(value, str) and value.strip() and target_key not in metadata:
                metadata[target_key] = value.strip()
        return ExperimentCandidate(
            hypothesis=payload["hypothesis"],
            action_type=payload["action_type"],
            change_type=payload["change_type"],
            instructions=payload["instructions"],
            execution_steps=[
                ExecutionStep.from_dict(step)
                for step in payload.get("execution_steps", [])
            ],
            config_patch=payload.get("config_patch", {}),
            metadata=metadata,
        )

    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate:
        iteration_policy = build_iteration_policy(snapshot)
        execution_handoff = build_execution_handoff(snapshot)
        payload = self._complete_json(
            system_prompt=(
                "You are a fresh worker agent in an experimentation loop. "
                "Use only the provided effective spec, accepted memory, and human interventions. "
                "Respect the configured primary metric, secondary metrics, and guardrail metrics, "
                "including any scorer_ref or metric constraints. "
                "After bootstrap, execution is autonomous: do not wait for more user input during normal experimentation. "
                "Keep iterating until the configured stop condition, a true execution blocker, or a safety boundary is reached. "
                "Treat recoverable execution failures as normal iteration evidence. If the latest record failed, use it to decide the next fix/retry step. "
                "EACH ITERATION SHOULD AIM TO PRODUCE METRICS. An iteration is a full experiment cycle, not a single step. "
                "Combine inspection, code writing, and execution into ONE iteration's execution_steps. "
                "For example, a good iteration might: (1) read one key file to understand the API, (2) write a script that loads data/trains/evaluates, (3) run the script. "
                "All three happen in the same iteration. Do NOT spend an entire iteration just reading files. "
                "The only exception is the very first iteration when the repo is completely unknown  - one initial inspection is acceptable. "
                "After that, every iteration must write and run code that produces measurable results. "
                "For the first generic-autonomous iteration, prefer the smallest runnable script that tests the main path. "
                "If one script can load the data, fit/evaluate, and print metrics, do that instead of creating helper scaffolding or multiple setup-only commands. "
                "Prefer one self-contained script run that prints the configured metrics in a machine-readable way on the first attempt. "
                "If you emit execution_steps without parseable metric output, the runtime will treat the iteration as unfinished and ask for a repair. "
                "If execution_handoff.bootstrap_handoff, `execution_runbook.md`, or `experiment_guide.md` is present, treat that as the binding bootstrap handoff for how to run the repo and what to build first. "
                "If the bootstrap handoff names an existing script, framework, or baseline code path, start there instead of inventing a new pipeline. "
                "If execution_handoff.execution_contract.must_reference_baseline_paths is true, your execution_steps must explicitly inspect, copy, edit, or run one of execution_handoff.execution_contract.baseline_paths before branching into variants. "
                "Do not ignore that handoff and rediscover the same execution mechanics from scratch unless a concrete execution failure proves the handoff is wrong or incomplete. "
                "If the runbook says bootstrap already verified a repo root, Python executable, or command shape, reuse that verified lane instead of inventing activation commands, environment setup, or extra cd chains. "
                "When recovering from a prior failure, your first step must directly address the specific failure_summary and recovery_actions from the latest record. "
                "Do not repeat the same failing command unchanged. If you want to run a new repo-local script, create it in an earlier write_file step first. "
                "Do NOT dump entire source files with shell commands like type, cat, head, more, or Get-Content. "
                "If you need one API detail, use a short Python inspection or inspect it inside the runnable experiment script itself. "
                "In generic autonomous mode, action_type is a lightweight label. The real work is in execution_steps. "
                "In generic autonomous mode, an iteration is not complete until it produces the configured primary metric or hits a true non-recoverable blocker. "
                "Do not return a bare action label without concrete execution_steps. "
                "Read capability_context.environment_facts before you emit commands. Commands must match the actual execution shell. "
                "If execution_shell is cmd.exe, do not use Unix tools like head, grep, find -maxdepth, or ls -la. Prefer Python-native scripts or cmd-compatible commands. "
                "Use timeout_seconds on long-running shell steps when training is expected to take longer than a short diagnostic command. "
                "Surface concise operator-facing reasoning in metadata. "
                "Use metadata.observation_summary for what you observed from the repo/data/history, "
                "metadata.reasoning_summary for why this plan is the next best move, and "
                "metadata.next_step_summary for what the execution is trying to validate. "
                "These are short user-visible summaries, not private chain-of-thought. "
                "Return one bounded experiment that aims to produce metric results.\n\n"
                + DATA_SCIENCE_METRICS_REASONING
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "execution_handoff": execution_handoff,
                "best_summary": _compact_iteration_summary(snapshot.best_summary)
                if snapshot.best_summary is not None
                else None,
                "recent_records": _compact_recent_records(snapshot),
                "recent_summaries": _compact_recent_summaries(snapshot),
                "recent_human_interventions": _compact_human_interventions(snapshot),
                "lessons_learned": _truncate_text(snapshot.lessons_learned, limit=800),
                "markdown_memory": _worker_markdown_handoff(
                    snapshot, iteration_policy=iteration_policy
                ),
                "iteration_policy": iteration_policy,
                "next_iteration_id": snapshot.next_iteration_id,
                "instructions": {
                    "return_fields": [
                        "hypothesis",
                        "action_type",
                        "change_type",
                        "instructions",
                        "execution_steps",
                        "config_patch",
                        "metadata",
                    ],
                    "metadata_fields": [
                        "observation_summary",
                        "reasoning_summary",
                        "next_step_summary",
                    ],
                    "generic_autonomous_rules": [
                        "execution_steps are required for generic autonomous execution.",
                        "Reuse execution_handoff instead of inventing new activation or cd logic.",
                        "Treat execution_handoff.bootstrap_handoff as binding repo-specific guidance when present.",
                        "If execution_handoff.execution_contract.must_reference_baseline_paths is true, explicitly touch one baseline path in execution_steps.",
                        "If a repo-local script does not exist yet, add a write_file step before running it.",
                    ],
                },
            },
            progress_stage="worker_propose_llm",
            progress_label="planning the next experiment",
        )
        payload = self._normalise_candidate_payload(
            payload, snapshot, iteration_policy=iteration_policy
        )
        return self._candidate_from_payload(payload)

    def continue_experiment(
        self,
        snapshot: MemorySnapshot,
        previous_candidate: ExperimentCandidate,
        previous_outcome: ExperimentOutcome,
    ) -> ExperimentCandidate:
        """Ask the worker to extend an iteration that succeeded but produced no metrics."""
        execution_handoff = build_execution_handoff(snapshot)
        payload = self._complete_json(
            system_prompt=(
                "You are continuing an experiment iteration that succeeded but did NOT produce metrics. "
                "The previous steps completed (inspection, setup, etc.) but no model was trained or evaluated. "
                "You MUST now produce execution_steps that build on what was already done and: "
                "1. Write a script that trains/runs a model using the APIs and data you already inspected. "
                "2. Evaluate against the configured primary metric and guardrail metrics. "
                "3. Print the metric results to stdout so the runtime can capture them. "
                "Return a self-contained continuation that ends with parseable metric output, not another setup-only step. "
                "Do NOT re-inspect files or re-read APIs. Use what you already learned from markdown_memory, especially `bootstrap_handoff.md`, `execution_runbook.md`, and `experiment_guide.md`. "
                "If execution_handoff.execution_contract.must_reference_baseline_paths is true, keep the continuation anchored on one of those baseline paths instead of switching to a new from-scratch script. "
                "Reuse the verified execution lane from the runbook instead of inventing new shell setup or activation steps unless a concrete failure proved the runbook wrong. "
                "Commands must match capability_context.environment_facts.execution_shell. "
                "Do NOT dump files with shell commands like type, cat, head, more, or Get-Content. "
                "If execution_shell is cmd.exe, do not emit Unix tools like head, grep, find -maxdepth, or ls -la. "
                "Return JSON with: hypothesis, action_type, change_type, instructions, execution_steps, metadata. "
                "Use metadata.observation_summary, metadata.reasoning_summary, and metadata.next_step_summary "
                "for concise user-visible reasoning updates."
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "execution_handoff": execution_handoff,
                "previous_hypothesis": previous_candidate.hypothesis,
                "previous_instructions": previous_candidate.instructions,
                "previous_outcome_notes": previous_outcome.notes,
                "previous_step_results": previous_outcome.execution_details.get(
                    "step_results", []
                ),
                "instructions": {
                    "return_fields": [
                        "hypothesis",
                        "action_type",
                        "change_type",
                        "instructions",
                        "execution_steps",
                        "metadata",
                    ],
                    "metadata_fields": [
                        "observation_summary",
                        "reasoning_summary",
                        "next_step_summary",
                    ],
                    "generic_autonomous_rules": [
                        "execution_steps are required.",
                        "Reuse execution_handoff instead of inventing new activation or cd logic.",
                        "Treat execution_handoff.bootstrap_handoff as binding repo-specific guidance when present.",
                        "If execution_handoff.execution_contract.must_reference_baseline_paths is true, explicitly touch one baseline path in execution_steps.",
                    ],
                },
            },
            progress_stage="worker_continue_llm",
            progress_label="extending the current iteration",
        )
        payload = self._normalise_candidate_payload(
            payload, snapshot, iteration_policy=build_iteration_policy(snapshot)
        )
        return self._candidate_from_payload(payload)

    @staticmethod
    def _normalise_candidate_payload(
        payload: dict[str, Any] | None,
        snapshot: MemorySnapshot,
        *,
        iteration_policy: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_payload = dict(payload or {})
        generic_autonomous = bool(iteration_policy.get("generic_autonomous"))  # noqa: F841
        # Structural: ensure required string fields exist (visible if agent failed to specify)
        action_type = candidate_payload.get("action_type")
        if not isinstance(action_type, str) or not action_type.strip():
            candidate_payload["action_type"] = "unspecified"
        change_type = candidate_payload.get("change_type")
        if not isinstance(change_type, str) or not change_type.strip():
            candidate_payload["change_type"] = candidate_payload["action_type"]
        instructions = candidate_payload.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            candidate_payload["instructions"] = "[agent did not specify]"
        hypothesis = candidate_payload.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            candidate_payload["hypothesis"] = "[agent did not specify]"
        config_patch = candidate_payload.get("config_patch")
        if not isinstance(config_patch, dict):
            candidate_payload["config_patch"] = {}
        raw_steps = candidate_payload.get("execution_steps")
        if not isinstance(raw_steps, list):
            candidate_payload["execution_steps"] = []
        else:
            normalised_steps = []
            for item in raw_steps:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind", "shell"))
                if kind == "shell":
                    command = item.get("command")
                    if not isinstance(command, str) or not command.strip():
                        continue
                    normalised_steps.append(
                        {
                            "kind": "shell",
                            "command": command.strip(),
                            "cwd": item.get("cwd"),
                            "timeout_seconds": item.get("timeout_seconds"),
                            "allow_failure": bool(item.get("allow_failure", False)),
                            "rationale": str(item.get("rationale", "")),
                        }
                    )
                    continue
                if kind in {"write_file", "append_file"}:
                    path = item.get("path")
                    if not isinstance(path, str) or not path.strip():
                        continue
                    normalised_steps.append(
                        {
                            "kind": kind,
                            "command": "",
                            "path": path.strip(),
                            "content": str(item.get("content", "")),
                            "cwd": item.get("cwd"),
                            "timeout_seconds": item.get("timeout_seconds"),
                            "allow_failure": bool(item.get("allow_failure", False)),
                            "rationale": str(item.get("rationale", "")),
                        }
                    )
            candidate_payload["execution_steps"] = normalised_steps
        metadata = candidate_payload.get("metadata")
        if not isinstance(metadata, dict):
            candidate_payload["metadata"] = {}
        # Don't silently replace action_type  - let the executor validate and raise if invalid.
        # The agent's choice should be visible, even if wrong.
        return candidate_payload


class LiteLLMSpecBackend(_LiteLLMJsonBackend):
    def propose_spec(
        self,
        objective: str,
        capability_context: CapabilityContext,
        user_preferences: dict[str, Any] | None = None,
    ) -> ExperimentSpecProposal:
        payload = self._complete_json(
            system_prompt=(
                "You are planning an experimentation spec before the loop starts. "
                "Use the available metrics discovered in the capability context when possible. "
                "Recommend one primary metric, useful secondary metrics, and strict guardrail metrics. "
                "If the user may need to confirm a calculation choice, surface that as a question. "
                "Return JSON with objective, recommended_spec, questions, and notes."
            ),
            payload={
                "objective": objective,
                "capability_context": capability_context.to_dict(),
                "user_preferences": user_preferences or {},
                "instructions": {
                    "recommended_spec_must_include": [
                        "objective",
                        "primary_metric",
                        "secondary_metrics",
                        "guardrail_metrics",
                        "allowed_actions",
                        "constraints",
                        "search_space",
                        "stop_conditions",
                        "metadata",
                    ],
                    "question_fields": [
                        "key",
                        "prompt",
                        "rationale",
                        "required",
                        "suggested_answer",
                        "options",
                    ],
                },
            },
            progress_stage="runner_author_llm",
            progress_label="authoring the repo-specific runner",
        )
        raw_notes = payload.get("notes", [])
        if isinstance(raw_notes, str):
            raw_notes = [raw_notes] if raw_notes.strip() else []
        return ExperimentSpecProposal(
            objective=payload["objective"],
            recommended_spec=ExperimentSpec.from_dict(payload["recommended_spec"]),
            questions=[
                SpecQuestion.from_dict(item) for item in payload.get("questions", [])
            ],
            notes=raw_notes,
        )


class LiteLLMBootstrapBackend(_LiteLLMJsonBackend):
    def propose_bootstrap_turn(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        answer_history: dict[str, Any] | None = None,
        role_models: RoleModelConfig | None = None,
        bootstrap_memory: dict[str, Any] | None = None,
    ) -> BootstrapTurn:
        payload = self._complete_json(
            system_prompt=(
                "You are a skilled data scientist and the initial clarifying agent for an experimentation runtime. "
                "Read the user's goal, inspect the discovered repo capabilities, and align on the problem framing. "
                "You bring deep ML/stats expertise: reason about which metrics are relevant to the stated objective "
                "and filter out metrics that don't apply. "
                "You may suggest new metrics or evaluation approaches based on your domain knowledge, but always "
                "confirm metric choices with the user. "
                "Ask only the highest-value remaining questions, recommend a concrete experiment spec, "
                "and set ready_to_start=true only when the spec is specific enough to begin execution.\n\n"
                + EXISTING_IMPLEMENTATION_GROUNDING
                + "\n"
                + DATA_SCIENCE_METRICS_REASONING
            ),
            payload={
                "user_goal": user_goal,
                "capability_context": capability_context.to_dict(),
                "answer_history": answer_history or {},
                "bootstrap_memory": bootstrap_memory or {},
                "role_models": role_models.to_dict()
                if role_models is not None
                else None,
                "instructions": {
                    "return_fields": [
                        "assistant_message",
                        "proposal",
                        "role_models",
                        "ready_to_start",
                    ],
                    "proposal_fields": [
                        "objective",
                        "recommended_spec",
                        "questions",
                        "notes",
                    ],
                    "CRITICAL_recommended_spec_rules": [
                        "The recommended_spec JSON drives execution. You MUST populate all fields.",
                        "primary_metric MUST have 'name' and 'goal' ('minimize' or 'maximize').",
                        "guardrail_metrics MUST be an array of objects with 'name' and 'goal'.",
                        "EXECUTION CONTRACT - metadata MUST include these fields (the executor depends on them). "
                        "Read the source script in the capability notes to fill them out: "
                        "(1) metadata.source_script: file path the user referenced. "
                        "(2) metadata.baseline_function: the function/class that trains the baseline model. "
                        "(3) metadata.data_loading: how data is loaded (function name, file path, DB call). "
                        "(4) metadata.target_column: what column/variable is being predicted. "
                        "(5) metadata.baseline_metric_value: current metric value if visible in code/logs, else omit. "
                        "If you cannot determine any of these from the source script, set the field to null "
                        "and the system will ask the user.",
                    ],
                    "question_fields": [
                        "key",
                        "prompt",
                        "rationale",
                        "required",
                        "suggested_answer",
                        "options",
                    ],
                    "behavior": [
                        "Derive metrics, guardrails, and experiment design from the objective and repo context.",
                        "The data source has already been traced by the bootstrap code. Use it directly.",
                        "If the user references a specific file, use that file - do not substitute a different one.",
                        "Be direct. If something is unclear, ask one question per turn.",
                        "Set metadata.data_source if you know the exact file path.",
                        "Set metadata.baseline_metric_value if you find an existing metric value.",
                    ],
                },
            },
            progress_stage="bootstrap_plan_llm",
            progress_label="drafting the experiment plan",
        )
        proposal_payload = self._normalise_bootstrap_proposal(
            payload.get("proposal"),
            user_goal=user_goal,
            capability_context=capability_context,
        )
        resolved_role_models = role_models or RoleModelConfig(
            planner=self.model,
            worker=self.model,
            reflection=self.model,
            review=self.model,
            consultation=self.model,
            narrator=self.model,
        )
        return BootstrapTurn(
            assistant_message=payload.get(
                "assistant_message",
                "I scanned the repo and prepared an initial bootstrap proposal.",
            ),
            proposal=ExperimentSpecProposal.from_dict(proposal_payload),
            role_models=RoleModelConfig.from_dict(
                payload.get("role_models", resolved_role_models.to_dict())
            ),
            ready_to_start=payload.get("ready_to_start", False),
        )


class LiteLLMRunnerAuthoringBackend(_LiteLLMJsonBackend):
    def author_runner(self, request: RunnerAuthoringRequest) -> RunnerAuthoringResult:
        payload = self._complete_json(
            system_prompt=(
                "You are authoring a real repo-specific experiment runner module. "
                "Return a valid JSON object with fields module_source, summary, and notes. "
                "module_source must be plain Python code only. No markdown fences.\n\n"
                "The generated module must define build_adapter(spec, memory_root) and return AdapterSetup. "
                "It must expose real handlers, a discovery_provider or capability_provider, and a preflight_provider. "
                "Do not generate placeholders or NotImplementedError stubs. "
                "Do not intentionally block in preflight. Build the best runnable baseline path you can from the repo. "
                "Ground the runner in the existing implementation described in request.capability_context. "
                "If baseline files, feature generators, scorers, or model patterns are already identified there, "
                "reuse and adapt them instead of inventing a disconnected pipeline. "
                "When improving an existing model, start from the current code path and keep feature engineering on the repo's existing tooling stack. "
                "If request.bootstrap_answers contains high-level user hints about the data source or execution environment, "
                "use them to resolve the integration details yourself rather than asking for exact internal paths. "
                "Prefer baseline/eda/slice_analysis/targeted_tune actions over raw helper function names."
            ),
            payload={
                "request": request.to_dict(),
                "instructions": {
                    "module_must_import": [
                        "from pathlib import Path",
                        "from loopforge import AdapterSetup, CapabilityContext, ExperimentOutcome, MetricResult, PreflightCheck",
                    ],
                    "required_symbols": [
                        "build_adapter",
                    ],
                    "coding_rules": [
                        "Use ASCII only.",
                        "No markdown fences.",
                        "No ellipses or placeholder comments.",
                        "Do not reference .loopforge/generated.",
                        "Do not ask the user to manually edit Python files.",
                        "Use previous_errors to repair the last failed attempt when present.",
                    ],
                },
            },
            progress_stage="runner_author_llm",
            progress_label="authoring the repo-specific runner",
        )
        return RunnerAuthoringResult.from_dict(payload)

    def build_experiment_guide(
        self,
        turn: BootstrapTurn,
        capability_context: CapabilityContext,
        answers: dict[str, Any] | None = None,
    ) -> str:
        return self._complete_text(
            system_prompt=(
                "You are a senior data scientist writing a concise experiment guide for a worker agent "
                "that will execute this experiment. The worker has never seen the repo or the user conversation. "
                "Write only what the worker needs to get started. Be specific  - include file paths, function names, "
                "column names, and code patterns. No preamble."
            ),
            user_message=json.dumps(
                {
                    "experiment_spec": turn.proposal.recommended_spec.to_dict(),
                    "capability_context": capability_context.to_dict(),
                    "user_answers": answers or {},
                    "planner_notes": turn.proposal.notes,
                    "sections_to_cover": [
                        "Baseline: the current model's metric values if known (from logs, results files, model registry, or prior runs). If not known, the exact command to run to get them.",
                        "Data loading: how to load the training data (file path, function call, what it returns)",
                        "Schema: key columns, types, what they represent",
                        "Target: column to predict, any transformations needed",
                        "Features: what feature engineering to apply, existing utilities to reuse",
                        "Metrics: how each metric is computed, scorer references",
                        "Splits: CV strategy, grouping keys to prevent leakage",
                        "Derivations: any columns that need to be created from existing data",
                        "User decisions: anything confirmed during planning",
                    ],
                },
                indent=2,
            ),
            progress_stage="experiment_guide_llm",
            progress_label="writing the experiment guide",
        )

    @staticmethod
    def _normalise_bootstrap_proposal(
        payload: dict[str, Any] | None,
        *,
        user_goal: str,
        capability_context: CapabilityContext,
    ) -> dict[str, Any]:
        proposal_payload = dict(payload) if isinstance(payload, Mapping) else {}
        if (
            not isinstance(proposal_payload.get("objective"), str)
            or not proposal_payload.get("objective", "").strip()
        ):
            proposal_payload["objective"] = user_goal
        raw_recommended_spec = proposal_payload.get("recommended_spec")
        recommended_spec = (
            dict(raw_recommended_spec)
            if isinstance(raw_recommended_spec, Mapping)
            else {}
        )
        if (
            not isinstance(recommended_spec.get("objective"), str)
            or not recommended_spec.get("objective", "").strip()
        ):
            recommended_spec["objective"] = proposal_payload["objective"]
        # Normalise metrics  - structural only, never inject domain decisions
        recommended_spec["primary_metric"] = (
            LiteLLMBootstrapBackend._normalise_metric_payload(
                recommended_spec.get("primary_metric"),
            )
        )
        recommended_spec["secondary_metrics"] = [
            LiteLLMBootstrapBackend._normalise_metric_payload(metric_payload)
            for metric_payload in recommended_spec.get("secondary_metrics", [])
        ]
        recommended_spec["guardrail_metrics"] = [
            LiteLLMBootstrapBackend._normalise_metric_payload(metric_payload)
            for metric_payload in recommended_spec.get("guardrail_metrics", [])
        ]
        # Normalise allowed_actions  - keep what the agent proposed
        allowed_actions = recommended_spec.get("allowed_actions", [])
        if not isinstance(allowed_actions, list):
            allowed_actions = [str(allowed_actions)] if allowed_actions else []
        normalized_actions = [
            str(action) for action in allowed_actions if str(action).strip()
        ]
        if (
            capability_context.environment_facts.get("execution_backend_kind")
            == "generic_agentic"
        ):
            generic_actions = capability_context.environment_facts.get(
                "generic_autonomous_actions", []
            )
            if isinstance(generic_actions, list):
                merged_actions = [
                    str(action) for action in generic_actions if str(action).strip()
                ]
                for action in normalized_actions:
                    if action not in merged_actions:
                        merged_actions.append(action)
                normalized_actions = merged_actions
        recommended_spec["allowed_actions"] = normalized_actions
        # Structural defaults  - empty containers and infrastructure safety limits
        recommended_spec.setdefault("constraints", {})
        recommended_spec.setdefault("search_space", {})
        recommended_spec.setdefault("stop_conditions", {})
        # Infrastructure safety limits (not domain decisions)
        recommended_spec["stop_conditions"].setdefault("max_iterations", 30)
        recommended_spec["stop_conditions"].setdefault("max_autonomous_hours", 6)
        recommended_spec.setdefault("metadata", {})
        recommended_spec = LiteLLMBootstrapBackend._apply_capability_metric_defaults(
            recommended_spec,
            capability_context=capability_context,
        )
        proposal_payload["recommended_spec"] = recommended_spec
        proposal_payload.setdefault("questions", [])
        notes = proposal_payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes] if notes.strip() else []
        proposal_payload["notes"] = notes
        return proposal_payload

    @staticmethod
    def _normalise_metric_payload(
        payload: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        """Ensure structurally required fields exist so the dataclass won't crash."""
        if isinstance(payload, str):
            payload = {"name": payload}
        metric_payload = dict(payload or {})
        # name and goal are required by MetricSpec  - use visible sentinels if agent omitted them
        metric_payload.setdefault("name", "[unspecified]")
        metric_payload.setdefault("goal", "unspecified")
        # Structural defaults  - empty containers
        metric_payload.setdefault("params", {})
        metric_payload.setdefault("slice_by", [])
        metric_payload.setdefault("constraints", {})
        return metric_payload

    @staticmethod
    def _apply_capability_metric_defaults(
        recommended_spec: dict[str, Any],
        *,
        capability_context: CapabilityContext,
    ) -> dict[str, Any]:
        """Enrich agent-proposed metrics with repo metadata without overriding explicit agent choices."""
        metric_catalog = capability_context.available_metrics
        patched_spec = dict(recommended_spec)

        def enrich(metric_payload: dict[str, Any]) -> dict[str, Any]:
            metric = dict(metric_payload)
            metric_name = metric.get("name")
            if metric_name not in metric_catalog:
                return metric
            repo_meta = metric_catalog[metric_name]
            if metric.get("goal") not in {"maximize", "minimize"}:
                repo_goal = repo_meta.get("goal")
                if repo_goal in {"maximize", "minimize"}:
                    metric["goal"] = repo_goal
            if "scorer_ref" in repo_meta:
                metric.setdefault("scorer_ref", repo_meta["scorer_ref"])
            return metric

        patched_spec["primary_metric"] = enrich(
            dict(recommended_spec["primary_metric"])
        )
        patched_spec["secondary_metrics"] = [
            enrich(dict(metric))
            for metric in recommended_spec.get("secondary_metrics", [])
        ]
        patched_spec["guardrail_metrics"] = [
            enrich(dict(metric))
            for metric in recommended_spec.get("guardrail_metrics", [])
        ]
        return patched_spec


LiteLLMBootstrapBackend.build_experiment_guide = (
    LiteLLMRunnerAuthoringBackend.build_experiment_guide
)
LiteLLMBootstrapBackend._normalise_bootstrap_proposal = staticmethod(
    LiteLLMRunnerAuthoringBackend._normalise_bootstrap_proposal
)
LiteLLMBootstrapBackend._normalise_metric_payload = staticmethod(
    LiteLLMRunnerAuthoringBackend._normalise_metric_payload
)
LiteLLMBootstrapBackend._apply_capability_metric_defaults = staticmethod(
    LiteLLMRunnerAuthoringBackend._apply_capability_metric_defaults
)


class LiteLLMAccessAdvisorBackend(_LiteLLMJsonBackend):
    def build_access_guide(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        preflight_checks,
    ) -> AccessGuide:
        payload = self._complete_json(
            system_prompt=(
                "You are the access and permissions advisor for an experimentation runtime. "
                "Your job is to determine what credentials, permissions, environment variables, and commands "
                "a human needs in order to run safely. Write a direct markdown playbook that another coding "
                "agent can follow without improvising."
            ),
            payload={
                "user_goal": user_goal,
                "capability_context": capability_context.to_dict(),
                "preflight_checks": [item.to_dict() for item in preflight_checks],
                "instructions": {
                    "return_fields": [
                        "summary",
                        "required_env_vars",
                        "required_permissions",
                        "commands",
                        "steps",
                        "markdown",
                    ],
                    "behavior": [
                        "Be direct and operational.",
                        "Separate env vars, permissions, and commands.",
                        "If access is blocked, say exactly what is missing.",
                    ],
                },
            },
            progress_stage="access_guide_llm",
            progress_label="building the access guide",
        )
        return AccessGuide.from_dict(payload)


class LiteLLMReflectionBackend(_LiteLLMJsonBackend):
    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _coerce_recommended_next_action(value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, Mapping):
            for key in (
                "action_type",
                "recommended_next_action",
                "next_action",
                "action",
                "name",
            ):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
            candidate = value.get("candidate")
            if isinstance(candidate, Mapping):
                action_type = candidate.get("action_type")
                if isinstance(action_type, str) and action_type.strip():
                    return action_type.strip()
            reason = value.get("reason")
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
        return None

    @classmethod
    def _normalise_reflection_payload(
        cls,
        payload: Any,
        *,
        outcome: ExperimentOutcome,
    ) -> dict[str, Any]:
        raw_payload = payload if isinstance(payload, Mapping) else {}
        nested = raw_payload.get("reflection")
        if isinstance(nested, Mapping):
            reflection_payload: dict[str, Any] = {**dict(raw_payload), **dict(nested)}
        else:
            reflection_payload = dict(raw_payload)

        assessment = None
        for key in (
            "assessment",
            "summary",
            "analysis",
            "reflection",
            "message",
            "text",
        ):
            value = reflection_payload.get(key)
            if isinstance(value, str) and value.strip():
                assessment = value.strip()
                break

        lessons = cls._coerce_string_list(reflection_payload.get("lessons"))
        risks = cls._coerce_string_list(reflection_payload.get("risks"))
        recommended_next_action = cls._coerce_recommended_next_action(
            reflection_payload.get("recommended_next_action")
            or reflection_payload.get("next_action")
        )

        if assessment is None:
            if lessons:
                assessment = lessons[0]
            elif risks:
                assessment = risks[0]
            elif recommended_next_action is not None:
                assessment = (
                    f"The iteration {outcome.status.replace('_', ' ')}. "
                    f"Suggested next action: {recommended_next_action}."
                )
            else:
                assessment = (
                    "[fallback] Reflection agent did not provide an assessment."
                )

        return {
            "assessment": assessment,
            "lessons": lessons,
            "risks": risks,
            "recommended_next_action": recommended_next_action,
        }

    def reflect(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
    ) -> ReflectionSummary:
        payload = self._complete_json(
            system_prompt=(
                "You are a fresh reflection agent. Assess whether the experiment result is meaningful, "
                "what was learned, whether primary or guardrail metrics moved in the intended direction, "
                "and what action type should be considered next. "
                "If the iteration failed but the failure is recoverable, reflect on the failure as useful evidence and recommend the next repair or retry step. "
                "Treat the loop as autonomous after bootstrap: recommend the next experiment step directly unless execution is truly blocked.\n\n"
                + DATA_SCIENCE_METRICS_REASONING
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "candidate": _compact_candidate(candidate),
                "outcome": _compact_outcome(outcome),
                "recent_records": _compact_recent_records(snapshot),
                "best_summary": _compact_iteration_summary(snapshot.best_summary)
                if snapshot.best_summary is not None
                else None,
                "recent_human_interventions": _compact_human_interventions(snapshot),
                "markdown_memory": _compact_markdown_memory(snapshot, limit=3),
                "instructions": {
                    "return_fields": [
                        "assessment",
                        "lessons",
                        "risks",
                        "recommended_next_action",
                    ],
                    "assessment_requirements": [
                        "assessment is required and must be a non-empty string",
                        "recommended_next_action must be a short action label string, not an object",
                    ],
                },
            },
            progress_stage="reflect_llm",
            progress_label="reflecting on the latest outcome",
        )
        payload = self._normalise_reflection_payload(payload, outcome=outcome)
        return ReflectionSummary(
            assessment=payload["assessment"],
            lessons=payload.get("lessons", []),
            risks=payload.get("risks", []),
            recommended_next_action=payload.get("recommended_next_action"),
        )


class LiteLLMReviewBackend(_LiteLLMJsonBackend):
    def review(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
        reflection: ReflectionSummary,
    ) -> ReviewDecision:
        payload = self._complete_json(
            system_prompt=(
                "You are a fresh review agent. Decide whether this iteration should enter accepted memory. "
                "Use status accepted, rejected, or pending_human. Consider guardrail violations and "
                "whether the reported metric results match the configured scoring contract. "
                "Recoverable execution failures should usually still be accepted into memory if they materially help the next worker continue. "
                "Do not route work back to the user during normal experimentation unless a real blocker or safety boundary requires human intervention.\n\n"
                + DATA_SCIENCE_METRICS_REASONING
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "candidate": _compact_candidate(candidate),
                "outcome": _compact_outcome(outcome),
                "reflection": {
                    "assessment": _truncate_text(reflection.assessment, limit=280),
                    "lessons": [
                        _truncate_text(item, limit=200)
                        for item in reflection.lessons[:3]
                        if item
                    ],
                    "risks": [
                        _truncate_text(item, limit=200)
                        for item in reflection.risks[:3]
                        if item
                    ],
                    "recommended_next_action": reflection.recommended_next_action,
                },
                "recent_records": _compact_recent_records(snapshot),
                "best_summary": _compact_iteration_summary(snapshot.best_summary)
                if snapshot.best_summary is not None
                else None,
                "markdown_memory": _compact_markdown_memory(snapshot, limit=3),
            },
            progress_stage="review_llm",
            progress_label="reviewing whether to accept the iteration",
        )
        if payload.get("status") not in {"accepted", "rejected", "pending_human"}:
            payload["status"] = "pending_human"
        if (
            not isinstance(payload.get("reason"), str)
            or not payload.get("reason", "").strip()
        ):
            payload["reason"] = (
                "[fallback] Review agent returned incomplete response; escalating to human."
            )
        return ReviewDecision(
            status=payload["status"],
            reason=payload["reason"],
            should_update_memory=payload.get("should_update_memory", True),
        )


class LiteLLMNarrationBackend(_LiteLLMJsonBackend):
    @staticmethod
    def _coerce_message(payload: Any, *, fallback: str) -> str:
        if isinstance(payload, Mapping):
            for key in ("message", "summary", "narration", "text", "assistant_message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in payload.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(payload, str) and payload.strip():
            return payload.strip()
        return fallback

    def summarize_bootstrap(
        self,
        turn: BootstrapTurn,
        capability_context: CapabilityContext,
    ) -> str:
        payload = self._complete_json(
            system_prompt=(
                "You are the human-facing narrator for an experimentation runtime. "
                "Explain progress clearly and concisely so a human can follow what the system is doing."
            ),
            payload={
                "assistant_message": turn.assistant_message,
                "proposal": turn.proposal.to_dict(),
                "role_models": turn.role_models.to_dict(),
                "preflight_checks": [item.to_dict() for item in turn.preflight_checks],
                "ready_to_start": turn.ready_to_start,
                "missing_requirements": turn.missing_requirements,
                "capability_context": capability_context.to_dict(),
                "instructions": {
                    "return_fields": ["message"],
                    "style": [
                        "Write for a human following the run.",
                        "Mention blockers and next steps briefly.",
                        "Be direct and concise. Don't explain obvious decisions.",
                        "Don't mention things you excluded or decided against  - just focus on what matters.",
                        "If something wasn't found, say so plainly. Don't discuss unrelated things in the repo.",
                    ],
                },
            },
            progress_stage="narrate_bootstrap_llm",
            progress_label="preparing the bootstrap summary",
        )
        return self._coerce_message(
            payload,
            fallback=turn.assistant_message or "Bootstrap planning updated.",
        )

    def answer_question(
        self,
        question: str,
        turn: BootstrapTurn,
        capability_context: CapabilityContext,
    ) -> str:
        context = json.dumps(
            {
                "current_plan": turn.proposal.to_dict(),
                "capability_context": capability_context.to_dict(),
                "preflight_checks": [c.to_dict() for c in turn.preflight_checks],
            },
            indent=2,
        )
        return self._complete_text(
            system_prompt=(
                "You are a skilled data scientist discussing an experiment plan with the user. "
                "Answer their question directly and concisely based on the current context. "
                "Don't repeat the full plan  - just answer what they asked."
            ),
            user_message=f"Current context:\n{context}\n\nUser question: {question}",
            progress_stage="answer_question_llm",
            progress_label="answering the plan question",
        )

    def interpret_feedback(
        self,
        turn: BootstrapTurn,
        feedback: str,
        capability_context: CapabilityContext,
    ) -> dict[str, Any]:
        """Interpret user feedback and return spec updates or signal full replan."""
        return self._complete_json(
            system_prompt=(
                "You are interpreting user feedback on an experiment plan. "
                "The plan may have blockers. The user's feedback may resolve a blocker, adjust the spec, "
                "or fundamentally change the objective. "
                "If the feedback can be applied as a patch to the existing spec, return "
                '{"action": "patch", "spec_updates": {...}, "message": "what you changed"}. '
                "spec_updates should contain only the fields to change, using the same structure as the spec. "
                "For example, to fix the primary metric: "
                '{"action": "patch", "spec_updates": {"primary_metric": {"name": "ordinal_loss", "goal": "minimize"}}, '
                '"message": "Set primary metric to ordinal_loss (minimize)."}. '
                "If the feedback fundamentally changes the objective or requires re-analyzing the repo, return "
                '{"action": "replan", "message": "why a full replan is needed"}. '
                "Prefer patching over replanning. Most feedback is a small adjustment."
            ),
            payload={
                "current_spec": turn.proposal.recommended_spec.to_dict(),
                "blockers": [
                    c.to_dict() for c in turn.preflight_checks if c.status == "failed"
                ],
                "missing_requirements": turn.missing_requirements,
                "user_feedback": feedback,
                "capability_context": capability_context.to_dict(),
            },
            progress_stage="feedback_llm",
            progress_label="interpreting the feedback",
        )

    def fix_incomplete_metrics(
        self,
        current_spec: dict[str, Any],
        assistant_message: str,
        objective: str | None = None,
        capability_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask the agent to fix incomplete metrics by extracting info from its own prose."""
        return self._complete_json(
            system_prompt=(
                "You are fixing an experiment spec where the metric fields were left incomplete. "
                "Infer the missing metric names and goals from the objective, discovered metric catalog, "
                "and the assistant_message. The user's objective is valid source material even when the assistant_message "
                "is generic. Use the metric catalog whenever it already defines a goal. "
                "If the catalog does not settle it, reason from the metric semantics and the user's objective. "
                "If one metric is clearly named or strongly implied by the objective and repo context, choose it. "
                "Do not leave any metric goal unspecified if you can logically determine it. "
                "Do not return an empty object unless the objective and repo context genuinely provide no credible metric clue. "
                "Return JSON with: primary_metric (object with name and goal), "
                "guardrail_metrics (array of objects with name and goal), "
                "secondary_metrics (array of objects with name and goal). "
                "For goal: 'minimize' means lower-is-better metrics; 'maximize' means higher-is-better metrics. "
                "Only include metrics that are already present in the spec or clearly implied by the assistant_message."
            ),
            payload={
                "current_spec": current_spec,
                "objective": objective,
                "capability_context": capability_context,
                "assistant_message": assistant_message,
            },
            progress_stage="fix_metrics_llm",
            progress_label="inferring incomplete metric details",
        )

    def summarize_iteration(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
        reflection: ReflectionSummary,
        review: ReviewDecision,
        accepted_summary,
    ) -> str:
        payload = self._complete_json(
            system_prompt=(
                "You are the human-facing narrator for an experimentation runtime. "
                "Summarize what just happened, what changed, whether it was trusted, and what comes next. "
                "Make it clear the system will keep iterating autonomously after bootstrap unless it is genuinely blocked."
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "markdown_memory": _compact_markdown_memory(snapshot, limit=2),
                "candidate": _compact_candidate(candidate),
                "outcome": _compact_outcome(outcome),
                "reflection": {
                    "assessment": _truncate_text(reflection.assessment, limit=280),
                    "recommended_next_action": reflection.recommended_next_action,
                },
                "review": review.to_dict(),
                "accepted_summary": _compact_iteration_summary(accepted_summary)
                if accepted_summary is not None
                else None,
                "instructions": {
                    "return_fields": ["message"],
                    "style": [
                        "Write for a human operator.",
                        "Include the decision and the practical implication.",
                        "Keep it concise.",
                    ],
                },
            },
            progress_stage="narrate_iteration_llm",
            progress_label="writing the iteration summary",
        )
        fallback = (
            f"Iteration {snapshot.next_iteration_id} completed with review status {review.status}. "
            f"Outcome: {outcome.status}."
        )
        return self._coerce_message(payload, fallback=fallback)
