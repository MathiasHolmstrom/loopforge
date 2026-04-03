from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loopforge.core import (
    AdapterSetup,
    ExperimentOrchestrator,
    ExperimentSpec,
    FileMemoryStore,
    HumanIntervention,
    LiteLLMReflectionBackend,
    LiteLLMReviewBackend,
    LiteLLMWorkerBackend,
    RoutingExperimentExecutor,
)


def load_factory(factory_path: str) -> Any:
    module_name, attribute_name = factory_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or steer the experimentation loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", required=True, help="Path to a JSON experiment spec.")
    run_parser.add_argument("--memory-root", required=True, help="Directory used for loop memory and summaries.")
    run_parser.add_argument("--executor-factory", required=True, help="Import path package.module:function_name")
    run_parser.add_argument("--worker-model", required=True, help="LiteLLM model id for the worker agent.")
    run_parser.add_argument("--reflection-model", help="LiteLLM model id for the reflection agent.")
    run_parser.add_argument("--review-model", help="LiteLLM model id for the review agent.")
    run_parser.add_argument("--iterations", type=int, default=None, help="Override max iterations for this run.")
    run_parser.add_argument("--temperature", type=float, default=0.2)

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
    worker_model: str,
    reflection_model: str | None = None,
    review_model: str | None = None,
    iterations: int | None = None,
    temperature: float = 0.2,
) -> list[dict[str, Any]]:
    spec = ExperimentSpec.from_dict(json.loads(Path(spec_path).read_text(encoding="utf-8")))
    memory_root_path = Path(memory_root)
    memory_store = FileMemoryStore(memory_root_path)
    adapter_result = load_factory(executor_factory_path)(spec=spec, memory_root=memory_root_path)
    if isinstance(adapter_result, AdapterSetup):
        handlers = adapter_result.handlers
        capability_provider = adapter_result.capability_provider
    else:
        handlers = adapter_result
        capability_provider = None
    orchestrator = ExperimentOrchestrator(
        memory_store=memory_store,
        worker_backend=LiteLLMWorkerBackend(model=worker_model, temperature=temperature),
        executor=RoutingExperimentExecutor(handlers=handlers),
        reflection_backend=LiteLLMReflectionBackend(
            model=reflection_model or worker_model,
            temperature=temperature,
        ),
        review_backend=LiteLLMReviewBackend(
            model=review_model or reflection_model or worker_model,
            temperature=temperature,
        ),
        capability_provider=capability_provider,
    )
    orchestrator.initialize(spec=spec)
    return [
        {
            "record": cycle_result.record.to_dict(),
            "accepted_summary": (
                cycle_result.accepted_summary.to_dict() if cycle_result.accepted_summary is not None else None
            ),
        }
        for cycle_result in orchestrator.run(iterations=iterations)
    ]


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
            iterations=args.iterations,
            temperature=args.temperature,
        )
        print(json.dumps(results, indent=2, sort_keys=True))
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

