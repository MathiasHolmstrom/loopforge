from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from dataclasses import replace
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
    OpsConsultation,
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
    "You are a skilled data scientist. Derive answers from the objective — don't ask the user "
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
    "Key pattern — the COMPLEMENT RULE: when optimizing for a subset (e.g. bench players), "
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
    "that is maximize. Never leave goal unspecified or assume a default — there is no safe default. "
    "Think about the metric's semantics, not just its name.\n"
    "\n"
    "5. DON'T ASK WHAT YOU CAN DERIVE: "
    "If the objective implies the metric, use it. "
    "If the complement subset is obvious, make it a guardrail. "
    "If a column can be derived from existing data, propose the derivation. "
    "If a metric from the repo has no causal connection to your changes, exclude it silently. "
    "Only ask when there is genuine ambiguity you cannot resolve from the objective and repo context."
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


class ConsultationBackend(Protocol):
    def consult(self, snapshot: MemorySnapshot) -> OpsConsultation: ...


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
        extra_kwargs: dict[str, Any] | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}
        self._completion_fn = completion_fn
        self._stream_fn = stream_fn

    def _stream_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Call litellm with stream=True, emitting tokens via stream_fn."""
        from litellm import completion as litellm_completion

        response = litellm_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            **kwargs,
            **self.extra_kwargs,
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

    def _complete_text(self, system_prompt: str, user_message: str) -> str:
        """Lightweight text completion — no JSON parsing, just a plain response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
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
            **self.extra_kwargs,
        )
        return self._extract_content(response)

    def _complete_json(
        self, system_prompt: str, payload: dict[str, Any]
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
        if self._stream_fn and self._completion_fn is None:
            content = self._stream_completion(
                messages, response_format={"type": "json_object"}
            )
            return json.loads(content)
        completion_fn = self._completion_fn
        if completion_fn is None:
            from litellm import completion as litellm_completion

            completion_fn = litellm_completion
        response = completion_fn(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            **self.extra_kwargs,
        )
        return json.loads(self._extract_content(response))

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


def build_execution_handoff(snapshot: MemorySnapshot) -> dict[str, Any]:
    env = snapshot.capability_context.environment_facts
    return {
        "repo_root": env.get("repo_root"),
        "execution_shell": env.get("execution_shell"),
        "shell_family": env.get("shell_family"),
        "python_executable": env.get("python_executable"),
        "execution_backend_kind": env.get("execution_backend_kind"),
        "verified_execution_lane": bool(env.get("python_executable"))
        and bool(env.get("repo_root")),
        "must_reuse_verified_lane": True,
        "available_bootstrap_handoff_files": sorted(
            item.path for item in snapshot.markdown_memory
        ),
        "execution_runbook": _markdown_content_by_name(
            snapshot, "execution_runbook.md"
        ),
        "experiment_guide": _markdown_content_by_name(snapshot, "experiment_guide.md"),
    }


class LiteLLMWorkerBackend(_LiteLLMJsonBackend):
    @staticmethod
    def _candidate_from_payload(payload: dict[str, Any]) -> ExperimentCandidate:
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
            metadata=payload.get("metadata", {}),
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
                "The only exception is the very first iteration when the repo is completely unknown — one initial inspection is acceptable. "
                "After that, every iteration must write and run code that produces measurable results. "
                "If markdown_memory contains `execution_runbook.md` or `experiment_guide.md`, treat them as the bootstrap handoff for how to run the repo and what to build first. "
                "Do not ignore that handoff and rediscover the same execution mechanics from scratch unless a concrete execution failure proves the handoff is wrong or incomplete. "
                "If the runbook says bootstrap already verified a repo root, Python executable, or command shape, reuse that verified lane instead of inventing activation commands, environment setup, or extra cd chains. "
                "When recovering from a prior failure, your first step must directly address the specific failure_summary and recovery_actions from the latest record. "
                "Do not repeat the same failing command unchanged. If you want to run a new repo-local script, create it in an earlier write_file step first. "
                "In generic autonomous mode, action_type is a lightweight label. The real work is in execution_steps. "
                "In generic autonomous mode, an iteration is not complete until it produces the configured primary metric or hits a true non-recoverable blocker. "
                "Do not return a bare action label without concrete execution_steps. "
                "Read capability_context.environment_facts before you emit commands. Commands must match the actual execution shell. "
                "If execution_shell is cmd.exe, do not use Unix tools like head, grep, find -maxdepth, or ls -la. Prefer Python-native scripts or cmd-compatible commands. "
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
                "markdown_memory": [
                    item.to_dict() for item in snapshot.markdown_memory
                ],
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
                    "generic_autonomous_rules": [
                        "execution_steps are required for generic autonomous execution.",
                        "Reuse execution_handoff instead of inventing new activation or cd logic.",
                        "If a repo-local script does not exist yet, add a write_file step before running it.",
                    ],
                },
            },
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
                "Do NOT re-inspect files or re-read APIs. Use what you already learned from markdown_memory, especially `execution_runbook.md` and `experiment_guide.md`. "
                "Reuse the verified execution lane from the runbook instead of inventing new shell setup or activation steps unless a concrete failure proved the runbook wrong. "
                "Commands must match capability_context.environment_facts.execution_shell. "
                "If execution_shell is cmd.exe, do not emit Unix tools like head, grep, find -maxdepth, or ls -la. "
                "Return JSON with: hypothesis, action_type, change_type, instructions, execution_steps."
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
                    ],
                    "generic_autonomous_rules": [
                        "execution_steps are required.",
                        "Reuse execution_handoff instead of inventing new activation or cd logic.",
                    ],
                },
            },
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
        # Don't silently replace action_type — let the executor validate and raise if invalid.
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


CONSULTATION_TRIGGER_TOKENS = (
    "databricks",
    "deploy",
    "deployment",
    "bundle",
    "workspace",
    "warehouse",
    "catalog",
    "secret",
    "token",
    "auth",
    "credential",
    "permission",
    "environment variable",
    "env var",
    "cli",
    "command",
    "external service",
    "service",
    "job",
    "cluster",
    "sdk",
    "api",
    "endpoint",
    "load data",
    "data access",
)


def should_request_ops_consult(snapshot: MemorySnapshot) -> bool:
    if snapshot.capability_context.environment_facts.get(
        "execution_backend_kind"
    ) == "generic_agentic" and _has_markdown_name(snapshot, "execution_runbook.md"):
        return False
    # Only trigger for genuine ops/infra topics — not generic data science work
    search_space = [
        snapshot.effective_spec.objective,
        " ".join(snapshot.effective_spec.allowed_actions),
        " ".join(snapshot.capability_context.notes),
        " ".join(snapshot.capability_context.available_data_assets),
        json.dumps(snapshot.capability_context.environment_facts, sort_keys=True),
        json.dumps(snapshot.effective_spec.metadata, sort_keys=True),
    ]
    haystack = " ".join(part.lower() for part in search_space if part)
    return any(token in haystack for token in CONSULTATION_TRIGGER_TOKENS)


def _inject_consultation(
    snapshot: MemorySnapshot, consultation: OpsConsultation
) -> MemorySnapshot:
    updated_notes = list(snapshot.capability_context.notes)
    updated_notes.append(f"Helper consult focus: {consultation.focus}")
    updated_notes.append(f"Helper consult guidance: {consultation.guidance}")
    updated_notes.append(f"Ops consult focus: {consultation.focus}")
    updated_notes.append(f"Ops consult guidance: {consultation.guidance}")
    if consultation.commands:
        updated_notes.append("Suggested commands: " + "; ".join(consultation.commands))
    if consultation.required_env_vars:
        updated_notes.append(
            "Required env vars: " + ", ".join(consultation.required_env_vars)
        )
    if consultation.risks:
        updated_notes.append("Operational risks: " + "; ".join(consultation.risks))
    updated_environment_facts = dict(snapshot.capability_context.environment_facts)
    updated_environment_facts["ops_consultation"] = consultation.to_dict()
    updated_environment_facts["helper_consultation"] = consultation.to_dict()
    updated_context = replace(
        snapshot.capability_context,
        notes=updated_notes,
        environment_facts=updated_environment_facts,
    )
    return replace(snapshot, capability_context=updated_context)


class ConsultingWorkerBackend:
    def __init__(
        self,
        worker_backend: WorkerBackend,
        consultation_backend: ConsultationBackend | None = None,
        progress_fn: ProgressFn | None = None,
        helper_label: str = "helper",
    ) -> None:
        self.worker_backend = worker_backend
        self.consultation_backend = consultation_backend
        self.progress_fn: ProgressFn = progress_fn or _noop_progress
        self.helper_label = helper_label

    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate:
        consultation = None
        delegated_snapshot = snapshot
        if self.consultation_backend is not None and should_request_ops_consult(
            snapshot
        ):
            self.progress_fn(
                "helper_consult",
                f"[{self.helper_label}] Codex is asking helper agent for advice...",
            )
            try:
                consultation = self.consultation_backend.consult(snapshot)
            except Exception as exc:
                consultation = OpsConsultation(
                    focus="ops consultation unavailable",
                    guidance=f"Consultation backend failed: {exc}",
                    should_consult=False,
                )
            if (
                consultation.should_consult
                and consultation.guidance.strip()
                and "unavailable" not in consultation.guidance.lower()
                and "no concrete guidance" not in consultation.guidance.lower()
            ):
                self.progress_fn(
                    "helper_consult_detail", self._format_consultation(consultation)
                )
            if consultation.should_consult:
                delegated_snapshot = _inject_consultation(snapshot, consultation)
        candidate = self.worker_backend.propose_next_experiment(delegated_snapshot)
        if consultation is None or not consultation.should_consult:
            return candidate
        merged_metadata = dict(candidate.metadata)
        merged_metadata["ops_consultation"] = consultation.to_dict()
        merged_metadata["helper_consultation"] = consultation.to_dict()
        return replace(candidate, metadata=merged_metadata)

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
    def _format_consultation(consultation: OpsConsultation) -> str:
        lines = [
            f"  Focus      : {consultation.focus}",
            f"  Guidance   : {consultation.guidance}",
        ]
        if consultation.commands:
            lines.append(f"  Commands   : {' ; '.join(consultation.commands[:3])}")
        if consultation.required_env_vars:
            lines.append(
                f"  Env        : {', '.join(consultation.required_env_vars[:5])}"
            )
        if consultation.risks:
            lines.append(f"  Risks      : {' ; '.join(consultation.risks[:3])}")
        return "\n".join(lines)


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
                        "is NOT enough — you MUST populate the actual JSON fields.",
                        "primary_metric MUST have 'name' (string, the actual metric/scorer name) and 'goal' ('minimize' or 'maximize').",
                        "guardrail_metrics MUST be an array of metric objects, each with 'name' and 'goal'.",
                        "DO NOT leave primary_metric as an empty object — fill in the name and goal you decided on.",
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
                        "You have creative freedom to suggest new metrics, evaluation approaches, or analysis "
                        "strategies based on your domain expertise — as long as you explain your reasoning.",
                        "Use prior run memory (summaries, lessons, markdown memory) to avoid repeating mistakes.",
                        "Only ask a question when there is genuine ambiguity you cannot resolve yourself. "
                        "Keep questions minimal, concrete, and high-value.",
                        "Be direct. If you can't find what the user asked for, say so plainly and ask. "
                        "Don't discuss unrelated things that happen to exist in the repo.",
                        "Don't explain obvious decisions. If you excluded an irrelevant metric, just exclude it — "
                        "don't write a note saying 'I intentionally excluded X because...'. The user already knows.",
                        "Ask at most ONE required question per turn. Don't bundle multiple questions or generate "
                        "fallback alternatives ('do you want X instead?') alongside the primary question. "
                        "The user's answer to the first question may make follow-ups irrelevant.",
                    ],
                },
            },
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
                "Write only what the worker needs to get started. Be specific — include file paths, function names, "
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
        # Normalise metrics — structural only, never inject domain decisions
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
        # Normalise allowed_actions — keep what the agent proposed
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
        # Structural defaults — empty containers and infrastructure safety limits
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
        # name and goal are required by MetricSpec — use visible sentinels if agent omitted them
        metric_payload.setdefault("name", "[unspecified]")
        metric_payload.setdefault("goal", "unspecified")
        # Structural defaults — empty containers
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
        """Enrich agent-proposed metrics with repo metadata (scorer_ref). Never override agent decisions."""
        metric_catalog = capability_context.available_metrics
        primary_metric = dict(recommended_spec["primary_metric"])
        metric_name = primary_metric.get("name", "primary_metric")
        if metric_name in metric_catalog:
            # Agent named a real metric — enrich with repo's scorer_ref if agent didn't specify
            repo_meta = metric_catalog[metric_name]
            if "scorer_ref" in repo_meta:
                primary_metric.setdefault("scorer_ref", repo_meta["scorer_ref"])
        recommended_spec["primary_metric"] = primary_metric
        # Don't auto-inject guardrails — the agent already has the full metric catalog
        # in the capability context and can propose guardrails if it decides they're needed.
        return recommended_spec


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


class LiteLLMConsultationBackend(_LiteLLMJsonBackend):
    def consult(self, snapshot: MemorySnapshot) -> OpsConsultation:
        try:
            payload = self._complete_json(
                system_prompt=(
                    "You are a specialist operations consult agent. "
                    "Help the primary coding agent with external services, CLI workflows, environment variables, "
                    "deployment steps, data access setup, and authentication issues. "
                    "Give concise, concrete guidance without taking ownership of the main task."
                ),
                payload={
                    "effective_spec": snapshot.effective_spec.to_dict(),
                    "capability_context": snapshot.capability_context.to_dict(),
                    "recent_human_interventions": [
                        item.to_dict() for item in snapshot.recent_human_interventions
                    ],
                    "best_summary": snapshot.best_summary.to_dict()
                    if snapshot.best_summary is not None
                    else None,
                    "markdown_memory": [
                        item.to_dict() for item in snapshot.markdown_memory
                    ],
                    "instructions": {
                        "return_fields": [
                            "focus",
                            "guidance",
                            "commands",
                            "required_env_vars",
                            "risks",
                            "should_consult",
                        ],
                        "behavior": [
                            "Prioritize exact operational steps.",
                            "Call out missing credentials, env vars, or permissions explicitly.",
                            "Prefer bounded commands over long prose.",
                        ],
                    },
                },
            )
        except Exception as exc:
            return OpsConsultation(
                focus="helper consultation unavailable",
                guidance=f"Consultation backend failed: {exc}",
                should_consult=False,
            )
        if isinstance(payload, Mapping):
            nested_payload = payload.get("consultation")
            if isinstance(nested_payload, Mapping):
                payload = dict(nested_payload)
        try:
            return OpsConsultation.from_dict(
                dict(payload) if isinstance(payload, Mapping) else {}
            )
        except Exception as exc:
            return OpsConsultation(
                focus="helper consultation unavailable",
                guidance=f"Consultation response could not be parsed: {exc}",
                should_consult=False,
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
                        "Don't mention things you excluded or decided against — just focus on what matters.",
                        "If something wasn't found, say so plainly. Don't discuss unrelated things in the repo.",
                    ],
                },
            },
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
                "Don't repeat the full plan — just answer what they asked."
            ),
            user_message=f"Current context:\n{context}\n\nUser question: {question}",
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
        )

    def fix_incomplete_metrics(
        self,
        current_spec: dict[str, Any],
        assistant_message: str,
    ) -> dict[str, Any]:
        """Ask the agent to fix incomplete metrics by extracting info from its own prose."""
        return self._complete_json(
            system_prompt=(
                "You are fixing an experiment spec where the metric fields were left incomplete. "
                "The agent already described the metrics in its message but forgot to fill them into the JSON. "
                "Extract the metric names and goals from the assistant_message and return the corrected spec fields. "
                "Return JSON with: primary_metric (object with name and goal), "
                "guardrail_metrics (array of objects with name and goal), "
                "secondary_metrics (array of objects with name and goal). "
                "For goal: 'minimize' means lower is better (losses, errors), 'maximize' means higher is better (scores, accuracies). "
                "Only include metrics the agent actually described."
            ),
            payload={
                "current_spec": current_spec,
                "assistant_message": assistant_message,
            },
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
        )
        fallback = (
            f"Iteration {snapshot.next_iteration_id} completed with review status {review.status}. "
            f"Outcome: {outcome.status}."
        )
        return self._coerce_message(payload, fallback=fallback)
