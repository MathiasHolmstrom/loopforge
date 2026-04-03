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
    ExperimentOrchestrator,
    ExperimentSpec,
    FileMemoryStore,
    LiteLLMBootstrapBackend,
    LiteLLMReflectionBackend,
    LiteLLMReviewBackend,
    LiteLLMWorkerBackend,
    PreflightCheck,
    PrimaryMetric,
    RoleModelConfig,
    RoutingExperimentExecutor,
)


DEFAULT_OPENAI_MODEL = "openai/gpt-5.4"


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
    planner_model: str = DEFAULT_OPENAI_MODEL,
    worker_model: str = DEFAULT_OPENAI_MODEL,
    reflection_model: str | None = None,
    review_model: str | None = None,
) -> RoleModelConfig:
    resolved_reflection = reflection_model or worker_model
    return RoleModelConfig(
        planner=planner_model,
        worker=worker_model,
        reflection=resolved_reflection,
        review=review_model or resolved_reflection,
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
            )
        )
    except OSError as exc:
        checks.append(
            PreflightCheck(
                name="memory_root_access",
                status="failed",
                detail=f"Could not write to memory root {memory_root_path}: {exc}",
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
                name="data_access_verification",
                status="warning",
                detail=(
                    "Data assets were discovered, but the adapter did not expose a preflight_provider to "
                    "verify access or permissions."
                ),
                required=False,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="data_access_verification",
                status="warning",
                detail="No data assets were discovered during bootstrap.",
                required=False,
            )
        )
    return checks


class Loopforge:
    def __init__(
        self,
        *,
        executor_factory_path: str | None = None,
        repo_root: Path | str = ".",
        memory_root: Path | str = ".loopforge",
        planner_model: str = DEFAULT_OPENAI_MODEL,
        worker_model: str = DEFAULT_OPENAI_MODEL,
        reflection_model: str | None = None,
        review_model: str | None = None,
        temperature: float = 0.2,
        bootstrap_backend: Any | None = None,
        worker_backend: Any | None = None,
        reflection_backend: Any | None = None,
        review_backend: Any | None = None,
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
        )
        self.bootstrap_backend = bootstrap_backend or LiteLLMBootstrapBackend(
            model=self.role_models.planner,
            temperature=temperature,
        )
        self.worker_backend = worker_backend or LiteLLMWorkerBackend(
            model=self.role_models.worker,
            temperature=temperature,
        )
        self.reflection_backend = reflection_backend or LiteLLMReflectionBackend(
            model=self.role_models.reflection,
            temperature=temperature,
        )
        self.review_backend = review_backend or LiteLLMReviewBackend(
            model=self.role_models.review,
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
        answered_keys = set((answers or {}).keys())
        missing_requirements = [
            f"answer:{question.key}"
            for question in turn.proposal.questions
            if question.required and question.key not in answered_keys
        ]
        missing_requirements.extend(
            f"preflight:{check.name}"
            for check in preflight_checks
            if check.required and check.status == "failed"
        )
        ready_to_start = turn.ready_to_start and not missing_requirements
        return replace(
            turn,
            preflight_checks=preflight_checks,
            ready_to_start=ready_to_start,
            missing_requirements=missing_requirements,
        )

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
                }
                for cycle_result in results
            ],
        }
