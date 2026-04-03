from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from loopforge.core.types import (
    ExperimentCandidate,
    ExperimentOutcome,
    MemorySnapshot,
    ReflectionSummary,
    ReviewDecision,
)


class WorkerBackend(Protocol):
    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate: ...


class ReflectionBackend(Protocol):
    def reflect(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
    ) -> ReflectionSummary: ...


class ReviewBackend(Protocol):
    def review(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
        reflection: ReflectionSummary,
    ) -> ReviewDecision: ...


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
                "Return one bounded next experiment."
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
                "what was learned, and what action type should be considered next."
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
                "Use status accepted, rejected, or pending_human."
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

