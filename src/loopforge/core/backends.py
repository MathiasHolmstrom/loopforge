from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from datetime import date, datetime
from dataclasses import replace
from pathlib import Path
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


DATA_SCIENCE_METRICS_REASONING = (
    "HOW TO THINK ABOUT EXPERIMENT DESIGN:\n"
    "\n"
    "You are a skilled data scientist. Derive answers from the objective â€” don't ask the user "
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
    "Key pattern â€” the COMPLEMENT RULE: when optimizing for a subset (e.g. bench players), "
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
    "that is maximize. Never leave goal unspecified or assume a default â€” there is no safe default. "
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
        """Lightweight text completion â€” no JSON parsing, just a plain response."""
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


def _has_markdown_name(snapshot: MemorySnapshot, filename: str) -> bool:
    return _markdown_content_by_name(snapshot, filename) is not None


def _worker_markdown_handoff(
    snapshot: MemorySnapshot, *, iteration_policy: dict[str, Any]
) -> list[dict[str, Any]]:
    notes = [item.to_dict() for item in snapshot.markdown_memory]
    if not (
        iteration_policy.get("generic_autonomous")
        and iteration_policy.get("first_iteration")
    ):
        return notes
    preferred_suffixes = (
        "bootstrap_handoff.md",
        "execution_runbook.md",
        "experiment_guide.md",
        "ops_access_guide.md",
    )
    filtered = [
        note
        for note in notes
        if isinstance(note.get("path"), str)
        and note["path"].endswith(preferred_suffixes)
    ]
    return filtered or notes


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
        "bootstrap_handoff": _markdown_content_by_name(
            snapshot, "bootstrap_handoff.md"
        ),
        "execution_runbook": _markdown_content_by_name(
            snapshot, "execution_runbook.md"
        ),
        "experiment_guide": _markdown_content_by_name(snapshot, "experiment_guide.md"),
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
                "The only exception is the very first iteration when the repo is completely unknown â€” one initial inspection is acceptable. "
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
                "best_summary": snapshot.best_summary.to_dict()
                if snapshot.best_summary is not None
                else None,
                "recent_records": [
                    record.to_dict() for record in snapshot.recent_records
                ],
                "recent_summaries": [
                    summary.to_dict() for summary in snapshot.recent_summaries
                ],
                "recent_human_interventions": [
                    item.to_dict() for item in snapshot.recent_human_interventions
                ],
                "lessons_learned": snapshot.lessons_learned,
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
        # Don't silently replace action_type â€” let the executor validate and raise if invalid.
        # The agent's choice should be visible, even if wrong.
        return candidate_payload


class ExecutionFixBackend(Protocol):
    def fix_execution_plan(
        self,
        candidate: ExperimentCandidate,
        failed_step_index: int,
        failure_summary: str,
        step_results: list[dict[str, Any]],
    ) -> list[ExecutionStep]: ...


class LiteLLMExecutionFixBackend(_LiteLLMJsonBackend):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_repair_summary: str | None = None

    def fix_execution_plan(
        self,
        candidate: ExperimentCandidate,
        failed_step_index: int,
        failure_summary: str,
        step_results: list[dict[str, Any]],
    ) -> list[ExecutionStep]:
        failed_step_result = step_results[-1] if step_results else {}
        payload = self._complete_json(
            system_prompt=(
                "You are the execution-repair helper for a coding agent. A previous execution plan failed. "
                "Read the exact failing command, stdout/stderr, and prior steps, then produce a materially revised "
                "execution_steps plan that directly fixes the failure. Think like a pragmatic senior engineer: "
                "preserve successful earlier steps when still useful, replace brittle steps when necessary, and do not "
                "repeat the same failing command unchanged unless an earlier step now changes its preconditions. "
                "Common fixes include: correcting cwd/import paths, creating a missing script before running it, "
                "rewriting a fragile inline python command into write_file + shell steps, installing/importing the "
                "right dependency, or adjusting the command for the current platform. "
                "Fix ONE concrete blocker at a time: directly address the first failing step, keep the rest of the plan as stable as possible, "
                "and avoid redesigning the whole iteration unless the failure proves the whole plan is wrong. "
                "Do NOT change the experiment objective or hypothesis. Only repair the execution plan. "
                "If the prior execution completed but did not report parseable metrics, revise the final run so it prints a machine-readable metric payload or explicit metric_name=value lines. "
                "Return JSON with exactly these fields: repair_summary (string) and execution_steps (array of step objects)."
            ),
            payload={
                "hypothesis": candidate.hypothesis,
                "instructions": candidate.instructions,
                "candidate_metadata": candidate.metadata,
                "current_steps": [step.to_dict() for step in candidate.execution_steps],
                "failed_step_index": failed_step_index,
                "failure_summary": failure_summary,
                "failed_step_result": failed_step_result,
                "step_results_so_far": step_results,
                "instructions_for_repair": {
                    "requirements": [
                        "Directly address the concrete failure shown in failure_summary and failed_step_result.",
                        "Return a revised plan, not commentary alone.",
                        "If a python -c command is failing because it is too brittle, prefer write_file plus a normal python script invocation.",
                        "If earlier steps already succeeded and remain necessary, keep them or explain their replacement in repair_summary.",
                    ]
                },
            },
            progress_stage="execution_fix_llm",
            progress_label="repairing the execution plan",
        )
        repair_summary = payload.get("repair_summary")
        self.last_repair_summary = (
            repair_summary.strip()
            if isinstance(repair_summary, str) and repair_summary.strip()
            else None
        )
        raw_steps = payload.get("execution_steps", [])
        if not isinstance(raw_steps, list):
            return []
        steps = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(ExecutionStep.from_dict(item))
            except Exception:
                continue
        return steps


class BaselineFirstWorkerBackend:
    def __init__(
        self,
        worker_backend: WorkerBackend,
        progress_fn: ProgressFn | None = None,
    ) -> None:
        self.worker_backend = worker_backend
        self.progress_fn: ProgressFn = progress_fn or _noop_progress

    @property
    def model(self) -> str | None:
        return getattr(self.worker_backend, "model", None)

    @property
    def max_completion_tokens(self) -> int | None:
        return getattr(self.worker_backend, "max_completion_tokens", None)

    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate:
        candidate = self._build_baseline_candidate(snapshot)
        if candidate is not None:
            self.progress_fn(
                "baseline_first",
                "Using deterministic baseline-first execution plan from bootstrap contract.",
            )
            return candidate
        return self.worker_backend.propose_next_experiment(snapshot)

    def continue_experiment(
        self,
        snapshot: MemorySnapshot,
        previous_candidate: ExperimentCandidate,
        previous_outcome: ExperimentOutcome,
    ) -> ExperimentCandidate:
        return self.worker_backend.continue_experiment(
            snapshot, previous_candidate, previous_outcome
        )

    @staticmethod
    def _build_baseline_candidate(
        snapshot: MemorySnapshot,
    ) -> ExperimentCandidate | None:
        metadata = (
            snapshot.effective_spec.metadata
            if isinstance(snapshot.effective_spec.metadata, dict)
            else {}
        )
        contract = metadata.get("execution_contract")
        if not isinstance(contract, dict):
            return None
        if not contract.get("must_reference_baseline_paths"):
            return None
        if snapshot.latest_summary is not None or snapshot.recent_records:
            return None
        baseline_paths = [
            str(item).strip()
            for item in contract.get("baseline_paths", [])
            if str(item).strip()
        ]
        if not baseline_paths:
            return None
        baseline_path = baseline_paths[0]
        env = snapshot.capability_context.environment_facts
        python_executable = str(env.get("python_executable") or "python")
        repo_root = str(env.get("repo_root") or ".")
        action_type = (
            "run_experiment"
            if "run_experiment" in snapshot.effective_spec.allowed_actions
            else (
                snapshot.effective_spec.allowed_actions[0]
                if snapshot.effective_spec.allowed_actions
                else "run_experiment"
            )
        )
        baseline_suffix = Path(baseline_path).suffix.lower()
        if baseline_suffix == ".py":
            execution_steps = [
                ExecutionStep(
                    kind="shell",
                    command=f'"{python_executable}" "{baseline_path}"',
                    cwd=repo_root,
                    timeout_seconds=120,
                    rationale=(
                        "Baseline-first contract: run the existing baseline path before inventing variants."
                    ),
                )
            ]
            instructions = (
                f"Run the existing baseline script at {baseline_path} exactly as the first iteration. "
                "Do not create a new standalone pipeline before this baseline path has been exercised."
            )
        else:
            safe_path = baseline_path.replace("\\", "\\\\").replace('"', '\\"')
            execution_steps = [
                ExecutionStep(
                    kind="shell",
                    command=(
                        f'"{python_executable}" -c "from pathlib import Path; '
                        f'p=Path(r\'{safe_path}\'); '
                        'print(p.read_text(encoding=\'utf-8\')[:2000])"'
                    ),
                    cwd=repo_root,
                    timeout_seconds=30,
                    rationale=(
                        "Baseline-first contract: inspect the existing baseline artifact before proposing changes."
                    ),
                )
            ]
            instructions = (
                f"Inspect the existing baseline artifact at {baseline_path} first. "
                "Do not branch into a new implementation before grounding on that baseline path."
            )
        return ExperimentCandidate(
            hypothesis=(
                f"Reproduce or ground on the existing baseline path {baseline_path} before any new experimentation."
            ),
            action_type=action_type,
            change_type="baseline_reuse",
            instructions=instructions,
            execution_steps=execution_steps,
            metadata={
                "observation_summary": f"Bootstrap identified baseline path {baseline_path}.",
                "reasoning_summary": (
                    "The first iteration is deterministic to enforce reuse of the existing framework."
                ),
                "next_step_summary": f"Exercise baseline path {baseline_path} before broader experimentation.",
                "baseline_first_contract": True,
            },
        )


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
                        "The recommended_spec JSON is what drives execution. Describing metrics in assistant_message "
                        "is NOT enough â€” you MUST populate the actual JSON fields.",
                        "primary_metric MUST have 'name' (string, the actual metric/scorer name) and 'goal' ('minimize' or 'maximize').",
                        "guardrail_metrics MUST be an array of metric objects, each with 'name' and 'goal'.",
                        "DO NOT leave primary_metric as an empty object â€” fill in the name and goal you decided on.",
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
                        "Think like a skilled data scientist. Derive metrics, guardrails, and experiment design "
                        "from the objective and repo context. Don't ask the user to confirm what they already told you.",
                        "Start by grounding on the current implementation described in capability_context. "
                        "If the repo context already names the baseline model, scorer, feature pipeline, or code paths, "
                        "reference that directly instead of asking the user what exists.",
                        "Use the user's initial instruction as a binding constraint, then refine it with repo evidence. "
                        "If the user says the model already exists in this repo and asks you to inspect it, do that mentally from the repo context before asking follow-up questions.",
                        "Prefer a qualified guess over a clarifying question when the repo evidence points strongly in one direction. "
                        "Reserve questions for cases where multiple materially different execution paths remain plausible.",
                        "You have creative freedom to suggest new metrics, evaluation approaches, or analysis "
                        "strategies based on your domain expertise â€” as long as you explain your reasoning.",
                        "Use prior run memory (summaries, lessons, markdown memory) to avoid repeating mistakes.",
                        "Only ask a question when there is genuine ambiguity you cannot resolve yourself. "
                        "Keep questions minimal, concrete, and high-value.",
                        "Be direct. If you can't find what the user asked for, say so plainly and ask. "
                        "Don't discuss unrelated things that happen to exist in the repo.",
                        "Don't explain obvious decisions. If you excluded an irrelevant metric, just exclude it â€” "
                        "don't write a note saying 'I intentionally excluded X because...'. The user already knows.",
                        "Ask at most ONE required question per turn. Don't bundle multiple questions or generate "
                        "fallback alternatives ('do you want X instead?') alongside the primary question. "
                        "The user's answer to the first question may make follow-ups irrelevant.",
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
                "Write only what the worker needs to get started. Be specific â€” include file paths, function names, "
                "column names, and code patterns. No preamble."
            ),
            user_message=json.dumps(
                {
                    "experiment_spec": turn.proposal.recommended_spec.to_dict(),
                    "capability_context": capability_context.to_dict(),
                    "user_answers": answers or {},
                    "planner_notes": turn.proposal.notes,
                    "sections_to_cover": [
                        "Data loading: exact function/file to call, what it returns",
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
        # Normalise metrics â€” structural only, never inject domain decisions
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
        # Normalise allowed_actions â€” keep what the agent proposed
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
        # Structural defaults â€” empty containers and infrastructure safety limits
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
        # name and goal are required by MetricSpec â€” use visible sentinels if agent omitted them
        metric_payload.setdefault("name", "[unspecified]")
        metric_payload.setdefault("goal", "unspecified")
        # Structural defaults â€” empty containers
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

        patched_spec["primary_metric"] = enrich(dict(recommended_spec["primary_metric"]))
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
                "candidate": candidate.to_dict(),
                "outcome": outcome.to_dict(),
                "recent_records": [
                    record.to_dict() for record in snapshot.recent_records
                ],
                "best_summary": snapshot.best_summary.to_dict()
                if snapshot.best_summary is not None
                else None,
                "recent_human_interventions": [
                    item.to_dict() for item in snapshot.recent_human_interventions
                ],
                "markdown_memory": [
                    item.to_dict() for item in snapshot.markdown_memory
                ],
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
                "candidate": candidate.to_dict(),
                "outcome": outcome.to_dict(),
                "reflection": reflection.to_dict(),
                "recent_records": [
                    record.to_dict() for record in snapshot.recent_records
                ],
                "best_summary": snapshot.best_summary.to_dict()
                if snapshot.best_summary is not None
                else None,
                "markdown_memory": [
                    item.to_dict() for item in snapshot.markdown_memory
                ],
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
                        "Don't mention things you excluded or decided against â€” just focus on what matters.",
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
                "Don't repeat the full plan â€” just answer what they asked."
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
                "markdown_memory": [
                    item.to_dict() for item in snapshot.markdown_memory
                ],
                "candidate": candidate.to_dict(),
                "outcome": outcome.to_dict(),
                "reflection": reflection.to_dict(),
                "review": review.to_dict(),
                "accepted_summary": accepted_summary.to_dict()
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

