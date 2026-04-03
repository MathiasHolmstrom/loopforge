from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loopforge.bootstrap import (
    DEFAULT_MODEL_PROFILE,
    DEFAULT_OPENAI_MODEL,
    Loopforge,
    discover_capabilities_for_objective,
    default_role_models,
)
from loopforge.core import (
    AdapterSetup,
    ExperimentSpec,
    ExperimentSpecProposal,
    FileMemoryStore,
    HumanIntervention,
    LiteLLMSpecBackend,
)
from loopforge.bootstrap import load_factory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or steer the experimentation loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", required=True, help="Path to a JSON experiment spec.")
    run_parser.add_argument("--memory-root", default=".loopforge", help="Directory used for loop memory and summaries.")
    run_parser.add_argument("--executor-factory", required=True, help="Import path package.module:function_name")
    run_parser.add_argument("--model-profile", default=DEFAULT_MODEL_PROFILE)
    run_parser.add_argument(
        "--worker-model",
        default=None,
        help="LiteLLM model id for the worker agent.",
    )
    run_parser.add_argument("--reflection-model", help="LiteLLM model id for the reflection agent.")
    run_parser.add_argument("--review-model", help="LiteLLM model id for the review agent.")
    run_parser.add_argument("--consultation-model", help="LiteLLM model id for the operations consult agent.")
    run_parser.add_argument("--narrator-model", help="LiteLLM model id for the human-facing narrator agent.")
    run_parser.add_argument("--iterations", type=int, default=None, help="Override max iterations for this run.")
    run_parser.add_argument("--temperature", type=float, default=0.2)

    draft_parser = subparsers.add_parser("draft-spec")
    draft_parser.add_argument("--objective", required=True, help="Experiment objective to plan for.")
    draft_parser.add_argument("--repo-root", default=".", help="Repository root to scan when auto-synthesizing.")
    draft_parser.add_argument("--memory-root", default=".loopforge", help="Directory passed to the adapter factory.")
    draft_parser.add_argument("--executor-factory", help="Import path package.module:function_name")
    draft_parser.add_argument(
        "--planner-model",
        default=DEFAULT_OPENAI_MODEL,
        help="LiteLLM model id for the planning agent.",
    )
    draft_parser.add_argument("--preferences-json", default="{}")
    draft_parser.add_argument("--temperature", type=float, default=0.2)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--message", required=True, help="Initial description of the problem to solve.")
    start_parser.add_argument("--repo-root", default=".", help="Repository root to scan when auto-synthesizing.")
    start_parser.add_argument("--memory-root", default=".loopforge", help="Directory used for loop memory.")
    start_parser.add_argument("--executor-factory", help="Import path package.module:function_name")
    start_parser.add_argument("--model-profile", default=DEFAULT_MODEL_PROFILE)
    start_parser.add_argument("--planner-model", default=None)
    start_parser.add_argument("--worker-model", default=None)
    start_parser.add_argument("--reflection-model")
    start_parser.add_argument("--review-model")
    start_parser.add_argument("--consultation-model")
    start_parser.add_argument("--narrator-model")
    start_parser.add_argument("--answers-json", default="{}")
    start_parser.add_argument("--iterations", type=int, default=None)
    start_parser.add_argument("--temperature", type=float, default=0.2)

    interject_parser = subparsers.add_parser("interject")
    interject_parser.add_argument("--memory-root", required=True)
    interject_parser.add_argument("--message", required=True)
    interject_parser.add_argument("--effects-json", default="{}")
    interject_parser.add_argument("--author", default="human")
    interject_parser.add_argument("--type", default="note")
    return parser


def run_from_spec(
    *,
    spec_path: Path | str,
    memory_root: Path | str,
    executor_factory_path: str,
    worker_model: str | None,
    reflection_model: str | None = None,
    review_model: str | None = None,
    consultation_model: str | None = None,
    narrator_model: str | None = None,
    model_profile: str = DEFAULT_MODEL_PROFILE,
    iterations: int | None = None,
    temperature: float = 0.2,
) -> list[dict[str, Any]]:
    spec = ExperimentSpec.from_dict(json.loads(Path(spec_path).read_text(encoding="utf-8")))
    role_models = default_role_models(
        planner_model=worker_model,
        worker_model=worker_model,
        reflection_model=reflection_model,
        review_model=review_model,
        consultation_model=consultation_model,
        narrator_model=narrator_model,
        profile=model_profile,
    )
    app = Loopforge(
        executor_factory_path=executor_factory_path,
        memory_root=memory_root,
        model_profile=model_profile,
        planner_model=role_models.planner,
        worker_model=role_models.worker,
        reflection_model=role_models.reflection,
        review_model=role_models.review,
        consultation_model=role_models.consultation,
        narrator_model=role_models.narrator,
        temperature=temperature,
    )
    memory_store = FileMemoryStore(memory_root)
    adapter_result = load_factory(executor_factory_path)(spec=spec, memory_root=Path(memory_root))
    if isinstance(adapter_result, AdapterSetup):
        handlers = adapter_result.handlers
        capability_provider = adapter_result.capability_provider
    else:
        handlers = adapter_result
        capability_provider = None
    from loopforge.core import ExperimentOrchestrator, RoutingExperimentExecutor

    orchestrator = ExperimentOrchestrator(
        memory_store=memory_store,
        worker_backend=app.worker_backend,
        executor=RoutingExperimentExecutor(handlers=handlers),
        reflection_backend=app.reflection_backend,
        review_backend=app.review_backend,
        narrator_backend=app.narrator_backend,
        capability_provider=capability_provider,
    )
    orchestrator.initialize(spec=spec)
    return [
        {
            "record": cycle_result.record.to_dict(),
            "accepted_summary": (
                cycle_result.accepted_summary.to_dict() if cycle_result.accepted_summary is not None else None
            ),
            "human_update": cycle_result.human_update,
        }
        for cycle_result in orchestrator.run(iterations=iterations)
    ]

def draft_spec(
    *,
    objective: str,
    memory_root: Path | str,
    executor_factory_path: str | None,
    planner_model: str,
    repo_root: Path | str = ".",
    preferences: dict[str, Any] | None = None,
    temperature: float = 0.2,
) -> ExperimentSpecProposal:
    resolved_factory_path = executor_factory_path
    if resolved_factory_path is None:
        resolved_factory_path = Loopforge(
            executor_factory_path=None,
            repo_root=repo_root,
            memory_root=memory_root,
            planner_model=planner_model,
            worker_model=planner_model,
            temperature=temperature,
        ).resolve_executor_factory_path(objective)
    capability_context = discover_capabilities_for_objective(
        objective=objective,
        memory_root=memory_root,
        executor_factory_path=resolved_factory_path,
    )
    backend = LiteLLMSpecBackend(model=planner_model, temperature=temperature)
    return backend.propose_spec(
        objective=objective,
        capability_context=capability_context,
        user_preferences=preferences,
    )


def append_human_intervention(
    *,
    memory_root: Path | str,
    message: str,
    effects: dict[str, Any] | None = None,
    author: str = "human",
    type_: str = "note",
) -> dict[str, Any]:
    intervention = HumanIntervention(
        author=author,
        type=type_,
        message=message,
        timestamp=datetime.now(timezone.utc).isoformat(),
        effects=effects or {},
    )
    store = FileMemoryStore(memory_root)
    store.append_human_intervention(intervention)
    return intervention.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.command == "run":
        results = run_from_spec(
            spec_path=args.spec,
            memory_root=args.memory_root,
            executor_factory_path=args.executor_factory,
            worker_model=args.worker_model,
            reflection_model=args.reflection_model,
            review_model=args.review_model,
            consultation_model=args.consultation_model,
            narrator_model=args.narrator_model,
            model_profile=args.model_profile,
            iterations=args.iterations,
            temperature=args.temperature,
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    if args.command == "draft-spec":
        proposal = draft_spec(
            objective=args.objective,
            repo_root=args.repo_root,
            memory_root=args.memory_root,
            executor_factory_path=args.executor_factory,
            planner_model=args.planner_model,
            preferences=json.loads(args.preferences_json),
            temperature=args.temperature,
        )
        print(json.dumps(proposal.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "start":
        app = Loopforge(
            executor_factory_path=args.executor_factory,
            repo_root=args.repo_root,
            memory_root=args.memory_root,
            model_profile=args.model_profile,
            planner_model=args.planner_model,
            worker_model=args.worker_model,
            reflection_model=args.reflection_model,
            review_model=args.review_model,
            consultation_model=args.consultation_model,
            narrator_model=args.narrator_model,
            temperature=args.temperature,
        )
        result = app.start(
            user_goal=args.message,
            answers=json.loads(args.answers_json),
            iterations=args.iterations,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    intervention = append_human_intervention(
        memory_root=args.memory_root,
        message=args.message,
        effects=json.loads(args.effects_json),
        author=args.author,
        type_=args.type,
    )
    print(json.dumps(intervention, indent=2, sort_keys=True))
    return 0

