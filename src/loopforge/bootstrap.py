from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from loopforge.auto_adapter import synthesize_auto_adapter
from loopforge.core import (
    AdapterSetup,
    BootstrapTurn,
    CapabilityContext,
    ConsultingWorkerBackend,
    ExperimentOrchestrator,
    ExperimentSpec,
    FileMemoryStore,
    LiteLLMAccessAdvisorBackend,
    LiteLLMBootstrapBackend,
    LiteLLMConsultationBackend,
    LiteLLMNarrationBackend,
    LiteLLMReflectionBackend,
    LiteLLMReviewBackend,
    LiteLLMWorkerBackend,
    PreflightCheck,
    PrimaryMetric,
    RoleModelConfig,
    RoutingExperimentExecutor,
)


DEFAULT_OPENAI_MODEL = "openai/gpt-5.4"
DEFAULT_CLAUDE_MODEL = "anthropic/claude-sonnet-4-5"
DEFAULT_MODEL_PROFILE = "codex_with_claude_support"

MODEL_PROFILES: dict[str, dict[str, str]] = {
    "all_codex": {
        "planner": DEFAULT_OPENAI_MODEL,
        "worker": DEFAULT_OPENAI_MODEL,
        "reflection": DEFAULT_OPENAI_MODEL,
        "review": DEFAULT_OPENAI_MODEL,
        "consultation": DEFAULT_OPENAI_MODEL,
        "narrator": DEFAULT_OPENAI_MODEL,
    },
    "codex_with_claude_support": {
        "planner": DEFAULT_OPENAI_MODEL,
        "worker": DEFAULT_OPENAI_MODEL,
        "reflection": DEFAULT_OPENAI_MODEL,
        "review": DEFAULT_OPENAI_MODEL,
        "consultation": DEFAULT_CLAUDE_MODEL,
        "narrator": DEFAULT_CLAUDE_MODEL,
    },
}


def load_factory(factory_path: str) -> Any:
    module_name, attribute_name = factory_path.rsplit(":", maxsplit=1)
    module_path = Path(module_name)
    if module_name.endswith(".py") or module_path.exists():
        resolved_path = module_path.resolve()
        module_key = f"loopforge_generated_{resolved_path.stem}_{abs(hash(str(resolved_path)))}"
        if module_key in sys.modules:
            module = sys.modules[module_key]
        else:
            spec = importlib.util.spec_from_file_location(module_key, resolved_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {resolved_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def build_bootstrap_spec(objective: str) -> ExperimentSpec:
    return ExperimentSpec(
        objective=objective,
        primary_metric=PrimaryMetric(name="primary_metric", goal="maximize"),
        allowed_actions=[],
    )


def default_role_models(
    planner_model: str | None = None,
    worker_model: str | None = None,
    reflection_model: str | None = None,
    review_model: str | None = None,
    consultation_model: str | None = None,
    narrator_model: str | None = None,
    profile: str = DEFAULT_MODEL_PROFILE,
) -> RoleModelConfig:
    try:
        defaults = MODEL_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown model profile: {profile!r}") from exc
    resolved_planner = planner_model or defaults["planner"]
    resolved_worker = worker_model or defaults["worker"]
    resolved_reflection = reflection_model or defaults["reflection"] or resolved_worker
    resolved_review = review_model or defaults["review"] or resolved_reflection
    resolved_consultation = consultation_model or defaults["consultation"]
    resolved_narrator = narrator_model or defaults["narrator"] or resolved_reflection
    return RoleModelConfig(
        planner=resolved_planner,
        worker=resolved_worker,
        reflection=resolved_reflection,
        review=resolved_review,
        consultation=resolved_consultation,
        narrator=resolved_narrator,
    )


def load_adapter_setup(
    *,
    factory_path: str,
    spec: ExperimentSpec,
    memory_root: Path,
) -> AdapterSetup | None:
    adapter_result = load_factory(factory_path)(spec=spec, memory_root=memory_root)
    if isinstance(adapter_result, AdapterSetup):
        return adapter_result
    return None


def discover_capabilities_for_objective(
    *,
    objective: str,
    memory_root: Path | str,
    executor_factory_path: str,
) -> CapabilityContext:
    memory_root_path = Path(memory_root)
    adapter_setup = load_adapter_setup(
        factory_path=executor_factory_path,
        spec=build_bootstrap_spec(objective),
        memory_root=memory_root_path,
    )
    if adapter_setup is None:
        return CapabilityContext()
    if adapter_setup.discovery_provider is not None:
        return adapter_setup.discovery_provider(objective)
    if adapter_setup.capability_provider is not None:
        return adapter_setup.capability_provider(build_bootstrap_spec(objective))
    return CapabilityContext()


def run_preflight_checks(
    *,
    spec: ExperimentSpec,
    capability_context: CapabilityContext,
    memory_root: Path | str,
    executor_factory_path: str,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    memory_root_path = Path(memory_root)
    try:
        memory_root_path.mkdir(parents=True, exist_ok=True)
        probe_path = memory_root_path / ".loopforge-write-check"
        probe_path.write_text("ok\n", encoding="utf-8")
        probe_path.unlink()
        checks.append(
            PreflightCheck(
                name="memory_root_access",
                status="passed",
                detail=f"Memory root is writable: {memory_root_path}",
                scope="execution",
            )
        )
    except OSError as exc:
        checks.append(
            PreflightCheck(
                name="memory_root_access",
                status="failed",
                detail=f"Could not write to memory root {memory_root_path}: {exc}",
                scope="execution",
            )
        )

    adapter_setup = load_adapter_setup(
        factory_path=executor_factory_path,
        spec=spec,
        memory_root=memory_root_path,
    )
    if adapter_setup is not None and adapter_setup.preflight_provider is not None:
        checks.extend(adapter_setup.preflight_provider(spec, capability_context))
    elif capability_context.available_data_assets:
        checks.append(
            PreflightCheck(
                name="autonomous_execution_permissions",
                status="failed",
                detail=(
                    "Data assets were discovered, but the adapter did not expose a preflight_provider to "
                    "verify execution permissions. Autonomous execution is blocked until those checks exist."
                ),
                scope="execution",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="autonomous_execution_permissions",
                status="failed",
                detail=(
                    "The adapter did not expose a preflight_provider, so Loopforge cannot verify autonomous "
                    "execution permissions."
                ),
                scope="execution",
            )
        )
    return checks


def missing_requirements_from_bootstrap(
    *,
    questions,
    answers: dict[str, Any] | None,
    preflight_checks: list[PreflightCheck],
) -> list[str]:
    answered_keys = set((answers or {}).keys())
    missing_requirements = [
        f"answer:{question.key}"
        for question in questions
        if question.required and question.key not in answered_keys
    ]
    missing_requirements.extend(
        f"preflight:{check.name}"
        for check in preflight_checks
        if check.required and check.status == "failed"
    )
    return missing_requirements


def should_prepare_access_guide(
    *,
    capability_context: CapabilityContext,
    preflight_checks: list[PreflightCheck],
) -> bool:
    if capability_context.available_data_assets:
        return True
    access_tokens = ("permission", "auth", "credential", "token", "env", "secret", "databricks", "warehouse")
    if any(check.status != "passed" for check in preflight_checks):
        return True
    joined_notes = " ".join(capability_context.notes).lower()
    joined_env = str(capability_context.environment_facts).lower()
    return any(token in joined_notes or token in joined_env for token in access_tokens)


class Loopforge:
    def __init__(
        self,
        *,
        executor_factory_path: str | None = None,
        repo_root: Path | str = ".",
        memory_root: Path | str = ".loopforge",
        planner_model: str | None = None,
        worker_model: str | None = None,
        reflection_model: str | None = None,
        review_model: str | None = None,
        consultation_model: str | None = None,
        narrator_model: str | None = None,
        model_profile: str = DEFAULT_MODEL_PROFILE,
        temperature: float = 0.2,
        bootstrap_backend: Any | None = None,
        worker_backend: Any | None = None,
        consultation_backend: Any | None = None,
        access_advisor_backend: Any | None = None,
        reflection_backend: Any | None = None,
        review_backend: Any | None = None,
        narrator_backend: Any | None = None,
    ) -> None:
        self.executor_factory_path = executor_factory_path
        self.repo_root = Path(repo_root)
        self.memory_root = Path(memory_root)
        self.temperature = temperature
        self.role_models = default_role_models(
            planner_model=planner_model,
            worker_model=worker_model,
            reflection_model=reflection_model,
            review_model=review_model,
            consultation_model=consultation_model,
            narrator_model=narrator_model,
            profile=model_profile,
        )
        self.bootstrap_backend = bootstrap_backend or LiteLLMBootstrapBackend(
            model=self.role_models.planner,
            temperature=temperature,
        )
        primary_worker_backend = worker_backend or LiteLLMWorkerBackend(model=self.role_models.worker, temperature=temperature)
        self.consultation_backend = consultation_backend or LiteLLMConsultationBackend(
            model=self.role_models.consultation,
            temperature=temperature,
        )
        self.access_advisor_backend = access_advisor_backend or LiteLLMAccessAdvisorBackend(
            model=self.role_models.consultation,
            temperature=temperature,
        )
        self.worker_backend = ConsultingWorkerBackend(
            worker_backend=primary_worker_backend,
            consultation_backend=self.consultation_backend,
        )
        self.reflection_backend = reflection_backend or LiteLLMReflectionBackend(
            model=self.role_models.reflection,
            temperature=temperature,
        )
        self.review_backend = review_backend or LiteLLMReviewBackend(
            model=self.role_models.review,
            temperature=temperature,
        )
        self.narrator_backend = narrator_backend or LiteLLMNarrationBackend(
            model=self.role_models.narrator,
            temperature=temperature,
        )

    def resolve_executor_factory_path(self, objective: str) -> str:
        if self.executor_factory_path:
            return self.executor_factory_path
        synthesized_factory_path = synthesize_auto_adapter(
            repo_root=self.repo_root,
            memory_root=self.memory_root,
            objective=objective,
        )
        self.executor_factory_path = synthesized_factory_path
        return synthesized_factory_path

    def bootstrap(
        self,
        *,
        user_goal: str,
        answers: dict[str, Any] | None = None,
    ) -> BootstrapTurn:
        executor_factory_path = self.resolve_executor_factory_path(user_goal)
        capability_context = discover_capabilities_for_objective(
            objective=user_goal,
            memory_root=self.memory_root,
            executor_factory_path=executor_factory_path,
        )
        turn = self.bootstrap_backend.propose_bootstrap_turn(
            user_goal=user_goal,
            capability_context=capability_context,
            answer_history=answers,
            role_models=self.role_models,
        )
        preflight_checks = run_preflight_checks(
            spec=turn.proposal.recommended_spec,
            capability_context=capability_context,
            memory_root=self.memory_root,
            executor_factory_path=executor_factory_path,
        )
        missing_requirements = missing_requirements_from_bootstrap(
            questions=turn.proposal.questions,
            answers=answers,
            preflight_checks=preflight_checks,
        )
        ready_to_start = turn.ready_to_start and not missing_requirements
        access_guide_path = None
        if should_prepare_access_guide(
            capability_context=capability_context,
            preflight_checks=preflight_checks,
        ):
            access_guide = self.access_advisor_backend.build_access_guide(
                user_goal=user_goal,
                capability_context=capability_context,
                preflight_checks=preflight_checks,
            )
            self.memory_root.mkdir(parents=True, exist_ok=True)
            guide_path = self.memory_root / "ops_access_guide.md"
            guide_path.write_text(access_guide.markdown.rstrip() + "\n", encoding="utf-8")
            access_guide_path = str(guide_path)
        resolved_turn = replace(
            turn,
            preflight_checks=preflight_checks,
            ready_to_start=ready_to_start,
            missing_requirements=missing_requirements,
            access_guide_path=access_guide_path,
        )
        human_update = self.narrator_backend.summarize_bootstrap(resolved_turn, capability_context)
        if access_guide_path is not None:
            human_update = f"{human_update}\nAccess guide: {access_guide_path}"
        return replace(resolved_turn, human_update=human_update)

    def start(
        self,
        *,
        user_goal: str,
        answers: dict[str, Any] | None = None,
        iterations: int | None = None,
    ) -> dict[str, Any]:
        bootstrap_turn = self.bootstrap(user_goal=user_goal, answers=answers)
        if not bootstrap_turn.ready_to_start:
            return {"status": "needs_input", "bootstrap": bootstrap_turn.to_dict()}

        spec = bootstrap_turn.proposal.recommended_spec
        memory_store = FileMemoryStore(self.memory_root)
        adapter_result = load_factory(self.resolve_executor_factory_path(user_goal))(spec=spec, memory_root=self.memory_root)
        if isinstance(adapter_result, AdapterSetup):
            handlers = adapter_result.handlers
            capability_provider = adapter_result.capability_provider
        else:
            handlers = adapter_result
            capability_provider = None
        orchestrator = ExperimentOrchestrator(
            memory_store=memory_store,
            worker_backend=self.worker_backend,
            executor=RoutingExperimentExecutor(handlers=handlers),
            reflection_backend=self.reflection_backend,
            review_backend=self.review_backend,
            narrator_backend=self.narrator_backend,
            capability_provider=capability_provider,
        )
        orchestrator.initialize(spec=spec)
        results = orchestrator.run(iterations=iterations)
        return {
            "status": "started",
            "bootstrap": bootstrap_turn.to_dict(),
            "results": [
                {
                    "record": cycle_result.record.to_dict(),
                    "accepted_summary": (
                        cycle_result.accepted_summary.to_dict()
                        if cycle_result.accepted_summary is not None
                        else None
                    ),
                    "human_update": cycle_result.human_update,
                }
                for cycle_result in results
            ],
        }
