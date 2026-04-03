from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol

from loopforge.core.types import (
    AccessGuide,
    BootstrapTurn,
    CapabilityContext,
    ExperimentCandidate,
    ExperimentSpec,
    ExperimentSpecProposal,
    ExperimentOutcome,
    MemorySnapshot,
    OpsConsultation,
    RoleModelConfig,
    ReflectionSummary,
    ReviewDecision,
    SpecQuestion,
)


class WorkerBackend(Protocol):
    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate: ...


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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}
        self._completion_fn = completion_fn

    def _complete_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        completion_fn = self._completion_fn
        if completion_fn is None:
            from litellm import completion as litellm_completion

            completion_fn = litellm_completion
        response = completion_fn(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            **self.extra_kwargs,
        )
        return json.loads(self._extract_content(response))

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


class LiteLLMWorkerBackend(_LiteLLMJsonBackend):
    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate:
        payload = self._complete_json(
            system_prompt=(
                "You are a fresh worker agent in an experimentation loop. "
                "Use only the provided effective spec, accepted memory, and human interventions. "
                "Respect the configured primary metric, secondary metrics, and guardrail metrics, "
                "including any scorer_ref or metric constraints. Return one bounded next experiment."
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "best_summary": snapshot.best_summary.to_dict() if snapshot.best_summary is not None else None,
                "recent_summaries": [summary.to_dict() for summary in snapshot.recent_summaries],
                "recent_human_interventions": [item.to_dict() for item in snapshot.recent_human_interventions],
                "lessons_learned": snapshot.lessons_learned,
                "next_iteration_id": snapshot.next_iteration_id,
            },
        )
        return ExperimentCandidate(
            hypothesis=payload["hypothesis"],
            action_type=payload["action_type"],
            change_type=payload["change_type"],
            instructions=payload["instructions"],
            config_patch=payload.get("config_patch", {}),
            metadata=payload.get("metadata", {}),
        )


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


def _inject_consultation(snapshot: MemorySnapshot, consultation: OpsConsultation) -> MemorySnapshot:
    updated_notes = list(snapshot.capability_context.notes)
    updated_notes.append(f"Ops consult focus: {consultation.focus}")
    updated_notes.append(f"Ops consult guidance: {consultation.guidance}")
    if consultation.commands:
        updated_notes.append("Suggested commands: " + "; ".join(consultation.commands))
    if consultation.required_env_vars:
        updated_notes.append("Required env vars: " + ", ".join(consultation.required_env_vars))
    if consultation.risks:
        updated_notes.append("Operational risks: " + "; ".join(consultation.risks))
    updated_environment_facts = dict(snapshot.capability_context.environment_facts)
    updated_environment_facts["ops_consultation"] = consultation.to_dict()
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
    ) -> None:
        self.worker_backend = worker_backend
        self.consultation_backend = consultation_backend

    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate:
        consultation = None
        delegated_snapshot = snapshot
        if self.consultation_backend is not None and should_request_ops_consult(snapshot):
            try:
                consultation = self.consultation_backend.consult(snapshot)
            except Exception as exc:
                consultation = OpsConsultation(
                    focus="ops consultation unavailable",
                    guidance=f"Consultation backend failed: {exc}",
                    should_consult=False,
                )
            if consultation.should_consult:
                delegated_snapshot = _inject_consultation(snapshot, consultation)
        candidate = self.worker_backend.propose_next_experiment(delegated_snapshot)
        if consultation is None or not consultation.should_consult:
            return candidate
        merged_metadata = dict(candidate.metadata)
        merged_metadata["ops_consultation"] = consultation.to_dict()
        return replace(candidate, metadata=merged_metadata)


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
        return ExperimentSpecProposal(
            objective=payload["objective"],
            recommended_spec=ExperimentSpec.from_dict(payload["recommended_spec"]),
            questions=[SpecQuestion.from_dict(item) for item in payload.get("questions", [])],
            notes=payload.get("notes", []),
        )


class LiteLLMBootstrapBackend(_LiteLLMJsonBackend):
    def propose_bootstrap_turn(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        answer_history: dict[str, Any] | None = None,
        role_models: RoleModelConfig | None = None,
    ) -> BootstrapTurn:
        payload = self._complete_json(
            system_prompt=(
                "You are the initial clarifying agent for an experimentation runtime. "
                "Read the user's goal, inspect the discovered repo capabilities, and align on the problem framing. "
                "Ask only the highest-value remaining questions, recommend a concrete experiment spec, "
                "and set ready_to_start=true only when the spec is specific enough to begin execution."
            ),
            payload={
                "user_goal": user_goal,
                "capability_context": capability_context.to_dict(),
                "answer_history": answer_history or {},
                "role_models": role_models.to_dict() if role_models is not None else None,
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
                    "question_fields": [
                        "key",
                        "prompt",
                        "rationale",
                        "required",
                        "suggested_answer",
                        "options",
                    ],
                    "behavior": [
                        "Prefer discovered codebase metrics when available.",
                        "If the user already implied a metric calculation, reflect that in scorer_ref, params, or constraints.",
                        "Use guardrail metrics for non-negotiable regressions.",
                        "Keep required questions minimal and concrete.",
                    ],
                },
            },
        )
        proposal_payload = self._normalise_bootstrap_proposal(payload.get("proposal"), user_goal=user_goal)
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
            role_models=RoleModelConfig.from_dict(payload.get("role_models", resolved_role_models.to_dict())),
            ready_to_start=payload.get("ready_to_start", False),
        )

    @staticmethod
    def _normalise_bootstrap_proposal(
        payload: dict[str, Any] | None,
        *,
        user_goal: str,
    ) -> dict[str, Any]:
        proposal_payload = dict(payload or {})
        proposal_payload.setdefault("objective", user_goal)
        recommended_spec = dict(proposal_payload.get("recommended_spec", {}))
        recommended_spec.setdefault("objective", proposal_payload["objective"])
        recommended_spec["primary_metric"] = LiteLLMBootstrapBackend._normalise_metric_payload(
            recommended_spec.get("primary_metric"),
            default_name="primary_metric",
            default_goal="maximize",
        )
        recommended_spec["secondary_metrics"] = [
            LiteLLMBootstrapBackend._normalise_metric_payload(
                metric_payload,
                default_name=f"secondary_metric_{index + 1}",
                default_goal="maximize",
            )
            for index, metric_payload in enumerate(recommended_spec.get("secondary_metrics", []))
        ]
        recommended_spec["guardrail_metrics"] = [
            LiteLLMBootstrapBackend._normalise_metric_payload(
                metric_payload,
                default_name=f"guardrail_metric_{index + 1}",
                default_goal="maximize",
            )
            for index, metric_payload in enumerate(recommended_spec.get("guardrail_metrics", []))
        ]
        recommended_spec.setdefault("allowed_actions", [])
        recommended_spec.setdefault("constraints", {})
        recommended_spec.setdefault("search_space", {})
        recommended_spec.setdefault("stop_conditions", {})
        recommended_spec.setdefault("metadata", {})
        proposal_payload["recommended_spec"] = recommended_spec
        proposal_payload.setdefault("questions", [])
        proposal_payload.setdefault("notes", [])
        return proposal_payload

    @staticmethod
    def _normalise_metric_payload(
        payload: dict[str, Any] | None,
        *,
        default_name: str,
        default_goal: str,
    ) -> dict[str, Any]:
        metric_payload = dict(payload or {})
        metric_payload.setdefault("name", default_name)
        metric_payload.setdefault("goal", default_goal)
        metric_payload.setdefault("params", {})
        metric_payload.setdefault("aggregation", "scalar")
        metric_payload.setdefault("slice_by", [])
        metric_payload.setdefault("comparator", "value")
        metric_payload.setdefault("constraints", {})
        return metric_payload


class LiteLLMConsultationBackend(_LiteLLMJsonBackend):
    def consult(self, snapshot: MemorySnapshot) -> OpsConsultation:
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
                "recent_human_interventions": [item.to_dict() for item in snapshot.recent_human_interventions],
                "best_summary": snapshot.best_summary.to_dict() if snapshot.best_summary is not None else None,
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
        return OpsConsultation.from_dict(payload)


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
                "and what action type should be considered next."
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "candidate": candidate.to_dict(),
                "outcome": outcome.to_dict(),
                "best_summary": snapshot.best_summary.to_dict() if snapshot.best_summary is not None else None,
                "recent_human_interventions": [item.to_dict() for item in snapshot.recent_human_interventions],
            },
        )
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
                "whether the reported metric results match the configured scoring contract."
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "capability_context": snapshot.capability_context.to_dict(),
                "candidate": candidate.to_dict(),
                "outcome": outcome.to_dict(),
                "reflection": reflection.to_dict(),
                "best_summary": snapshot.best_summary.to_dict() if snapshot.best_summary is not None else None,
            },
        )
        return ReviewDecision(
            status=payload["status"],
            reason=payload["reason"],
            should_update_memory=payload.get("should_update_memory", True),
        )


class LiteLLMNarrationBackend(_LiteLLMJsonBackend):
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
                        "Mention blockers, next steps, and why they matter.",
                        "Keep it concise.",
                    ],
                },
            },
        )
        return payload["message"]

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
                "Summarize what just happened, what changed, whether it was trusted, and what comes next."
            ),
            payload={
                "effective_spec": snapshot.effective_spec.to_dict(),
                "candidate": candidate.to_dict(),
                "outcome": outcome.to_dict(),
                "reflection": reflection.to_dict(),
                "review": review.to_dict(),
                "accepted_summary": accepted_summary.to_dict() if accepted_summary is not None else None,
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
        return payload["message"]

