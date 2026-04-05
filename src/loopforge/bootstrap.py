from __future__ import annotations

import inspect
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from difflib import get_close_matches
from datetime import date, datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from loopforge.auto_adapter import build_repo_scan_context
from loopforge.core.bootstrap_contracts import (
    build_bootstrap_handoff,
    build_execution_runbook,
    resolve_repo_root_from_objective,
    should_prepare_access_guide,
)
from loopforge.core import (
    AdapterSetup,
    BootstrapTurn,
    CapabilityContext,
    DataAssetSchema,
    ExperimentCandidate,
    ExperimentInterrupted,
    ExperimentOrchestrator,
    ExperimentOutcome,
    ExperimentSpec,
    FileMemoryStore,
    LiteLLMAccessAdvisorBackend,
    LiteLLMBootstrapBackend,
    LiteLLMNarrationBackend,
    LiteLLMRunnerAuthoringBackend,
    PreflightCheck,
    ProgressFn,
    StreamFn,
    PrimaryMetric,
    RoleModelConfig,
    RoutingExperimentExecutor,
    MemorySnapshot,
    MetricSpec,
    RunnerAuthoringBackend,
    RunnerAuthoringRequest,
    RunnerValidationResult,
    SpecQuestion,
    ToolUseExecutor,
    ToolUsePlanner,
    ToolUseReviewer,
    _noop_progress,
)


DEFAULT_OPENAI_MODEL = "openai/gpt-5.4"
DEFAULT_CLAUDE_MODEL = "anthropic/claude-opus-4-6-v1"
DEFAULT_MODEL_PROFILE = "codex_with_claude_support"
ROLE_MODEL_NAMES = (
    "planner",
    "worker",
    "review",
    "consultation",
    "narrator",
)
REPO_MODEL_CONFIG_FILENAMES = (
    "loopforge.models.json",
    ".loopforge/models.json",
)
REPO_SETTINGS_FILENAMES = (
    "loopforge.settings.json",
    ".loopforge/settings.json",
)
ExecutionBackendKind = Literal["supported", "planning_only"]
GENERIC_AUTONOMOUS_ACTIONS = [
    "inspect_repo",
    "inspect_data",
    "edit_code",
    "run_experiment",
    "fix_failure",
]
EXECUTION_GUIDANCE_QUESTION_KEY = "execution_strategy_hint"
TRANSIENT_BOOTSTRAP_ANSWER_KEYS = {"user_feedback", "discussion"}
REMOTE_EXECUTION_HINT_TOKENS = (
    "s3",
    "dbfs",
    "databricks",
    "jdbc",
    "snowflake",
    "warehouse",
    "catalog",
    "spark",
    "delta",
    "bucket",
)
MODEL_PROFILES: dict[str, dict[str, str]] = {
    "all_codex": {
        "planner": DEFAULT_OPENAI_MODEL,
        "worker": DEFAULT_OPENAI_MODEL,
        "review": DEFAULT_OPENAI_MODEL,
        "consultation": DEFAULT_OPENAI_MODEL,
        "narrator": DEFAULT_OPENAI_MODEL,
    },
    "codex_with_claude_support": {
        "planner": DEFAULT_CLAUDE_MODEL,
        "worker": DEFAULT_OPENAI_MODEL,
        "review": DEFAULT_OPENAI_MODEL,
        "consultation": DEFAULT_CLAUDE_MODEL,
        "narrator": DEFAULT_CLAUDE_MODEL,
    },
}

DEFAULT_ROLE_MAX_COMPLETION_TOKENS: dict[str, int] = {
    "planner": 1800,
    "worker": 2200,
    "runner_authoring": 5000,
    "consultation": 1200,
    "access_advisor": 1200,
    "review": 700,
    "narrator": 700,
}


@dataclass(frozen=True)
class RepoRoleModelSettings:
    model_profile: str | None = None
    planner_model: str | None = None
    worker_model: str | None = None
    review_model: str | None = None
    consultation_model: str | None = None
    narrator_model: str | None = None


@dataclass(frozen=True)
class RepoLoopSettings:
    max_iterations: int | None = None
    max_autonomous_hours: float | None = None


def _can_use_openai_models() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _can_use_anthropic_helpers() -> bool:
    if os.getenv("ANTHROPIC_API_KEY"):
        return True
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True
    if os.getenv("ANTHROPIC_BEDROCK_BASE_URL") or os.getenv("CLAUDE_CODE_USE_BEDROCK"):
        return True
    if os.getenv("AWS_PROFILE"):
        return True
    return False


def _model_provider(model: str) -> str | None:
    if model.startswith("openai/"):
        return "openai"
    if model.startswith("anthropic/") or model.startswith("bedrock/us.anthropic."):
        return "anthropic"
    return None


def _default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        return DEFAULT_CLAUDE_MODEL
    raise ValueError(f"Unknown model provider: {provider!r}")


def _normalise_model_id(model: str) -> str:
    # Route through Bedrock proxy
    if model.startswith("anthropic/") and os.getenv("ANTHROPIC_BEDROCK_BASE_URL"):
        model_name = model.removeprefix("anthropic/")
        return f"bedrock/us.anthropic.{model_name}"
    return model


def _coerce_optional_string(
    payload: dict[str, Any], key: str, *, path: Path
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    raise ValueError(f"{path}: `{key}` must be a string when provided.")


def _parse_repo_role_model_settings(
    payload: dict[str, Any], *, path: Path
) -> RepoRoleModelSettings:
    roles_payload = payload.get("roles")
    if roles_payload is None:
        roles_payload = {}
    elif not isinstance(roles_payload, dict):
        raise ValueError(f"{path}: `roles` must be a JSON object when provided.")

    model_profile = _coerce_optional_string(payload, "model_profile", path=path)
    if model_profile is None:
        model_profile = _coerce_optional_string(payload, "profile", path=path)

    values: dict[str, str | None] = {"model_profile": model_profile}
    for role in ROLE_MODEL_NAMES:
        arg_name = f"{role}_model"
        role_value = payload.get(arg_name, payload.get(role, roles_payload.get(role)))
        if role_value is None:
            values[arg_name] = None
            continue
        if not isinstance(role_value, str):
            raise ValueError(f"{path}: `{arg_name}` must be a string when provided.")
        stripped = role_value.strip()
        values[arg_name] = stripped or None
    return RepoRoleModelSettings(**values)


def load_repo_role_model_settings(
    repo_root: Path | str,
) -> tuple[RepoRoleModelSettings, Path | None]:
    root = Path(repo_root)
    for relative_path in REPO_MODEL_CONFIG_FILENAMES:
        candidate = root / relative_path
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{candidate}: top-level JSON value must be an object.")
        return _parse_repo_role_model_settings(payload, path=candidate), candidate
    return RepoRoleModelSettings(), None


def _coerce_optional_positive_int(
    payload: dict[str, Any], key: str, *, path: Path
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path}: `{key}` must be an integer when provided.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{path}: `{key}` must be >= 0.")
        return value
    raise ValueError(f"{path}: `{key}` must be an integer when provided.")


def _coerce_optional_positive_float(
    payload: dict[str, Any], key: str, *, path: Path
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path}: `{key}` must be a number when provided.")
    if isinstance(value, int | float):
        numeric = float(value)
        if numeric < 0:
            raise ValueError(f"{path}: `{key}` must be >= 0.")
        return numeric
    raise ValueError(f"{path}: `{key}` must be a number when provided.")


def _parse_repo_loop_settings(payload: dict[str, Any], *, path: Path) -> RepoLoopSettings:
    stop_payload = payload.get("stop_conditions")
    if stop_payload is None:
        stop_payload = {}
    elif not isinstance(stop_payload, dict):
        raise ValueError(
            f"{path}: `stop_conditions` must be a JSON object when provided."
        )

    max_iterations = _coerce_optional_positive_int(
        stop_payload, "max_iterations", path=path
    )
    if max_iterations is None:
        max_iterations = _coerce_optional_positive_int(
            payload, "max_iterations", path=path
        )

    max_autonomous_hours = _coerce_optional_positive_float(
        stop_payload, "max_autonomous_hours", path=path
    )
    if max_autonomous_hours is None:
        max_autonomous_hours = _coerce_optional_positive_float(
            payload, "max_autonomous_hours", path=path
        )

    return RepoLoopSettings(
        max_iterations=max_iterations,
        max_autonomous_hours=max_autonomous_hours,
    )


def load_repo_loop_settings(repo_root: Path | str) -> tuple[RepoLoopSettings, Path | None]:
    root = Path(repo_root)
    for relative_path in REPO_SETTINGS_FILENAMES:
        candidate = root / relative_path
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{candidate}: top-level JSON value must be an object.")
        return _parse_repo_loop_settings(payload, path=candidate), candidate
    return RepoLoopSettings(), None


def _resolve_role_models(
    *,
    planner_model: str | None = None,
    worker_model: str | None = None,
    review_model: str | None = None,
    consultation_model: str | None = None,
    narrator_model: str | None = None,
    profile: str | None = None,
) -> tuple[RoleModelConfig, list[str]]:
    resolved_profile = profile or DEFAULT_MODEL_PROFILE
    try:
        defaults = MODEL_PROFILES[resolved_profile]
    except KeyError as exc:
        raise ValueError(f"Unknown model profile: {resolved_profile!r}") from exc

    desired_models = {
        "planner": planner_model or defaults["planner"],
        "worker": worker_model or defaults["worker"],
        "review": review_model or defaults["review"],
        "consultation": consultation_model or defaults["consultation"],
        "narrator": narrator_model or defaults["narrator"],
    }
    openai_available = _can_use_openai_models()
    anthropic_available = _can_use_anthropic_helpers()
    warnings: list[str] = []

    forced_provider: str | None = None
    if openai_available and not anthropic_available:
        forced_provider = "openai"
        warnings.append(
            "Anthropic credentials were not detected; routing all agent roles through OpenAI/Codex models."
        )
    elif anthropic_available and not openai_available:
        forced_provider = "anthropic"
        warnings.append(
            "OpenAI credentials were not detected; routing all agent roles through Claude models."
        )
    elif not openai_available and not anthropic_available:
        warnings.append(
            "No OpenAI or Anthropic credentials were detected. Model calls will fail until one provider is configured."
        )

    resolved_models: dict[str, str] = {}
    for role, desired_model in desired_models.items():
        desired_provider = _model_provider(desired_model)
        resolved_model = desired_model
        if forced_provider is not None and desired_provider not in (
            None,
            forced_provider,
        ):
            resolved_model = _default_model_for_provider(forced_provider)
            warnings.append(
                f"{role} requested `{desired_model}` but provider `{desired_provider}` is unavailable; "
                f"using `{resolved_model}` instead."
            )
        resolved_models[role] = resolved_model

    return (
        RoleModelConfig(
            planner=resolved_models["planner"],
            worker=resolved_models["worker"],
            review=resolved_models["review"],
            consultation=resolved_models["consultation"],
            narrator=resolved_models["narrator"],
        ),
        warnings,
    )


def summarise_runtime_exception(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if "PermissionError: [WinError 5] Access is denied" in text:
        return (
            "Execution reached the real experiment, but the downstream modeling stack hit a Windows "
            "permission error while creating multiprocessing resources."
        )
    return text


def _missing_python_dependency(exc: Exception) -> tuple[str, str] | None:
    package_name = None
    if isinstance(exc, ModuleNotFoundError):
        package_name = exc.name
    if not package_name:
        missing = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
        if missing:
            package_name = missing.group(1)
    if not package_name:
        return None
    module_name = package_name.split(".", maxsplit=1)[0]
    install_name = {
        "sklearn": "scikit-learn",
    }.get(module_name, module_name)
    return module_name, install_name


def load_factory(factory_path: str) -> Any:
    module_name, attribute_name = factory_path.rsplit(":", maxsplit=1)
    module_path = Path(module_name)
    if module_name.endswith(".py") or module_path.exists():
        resolved_path = module_path.resolve()
        module_key = (
            f"loopforge_generated_{resolved_path.stem}_{abs(hash(str(resolved_path)))}"
        )
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


def _invoke_adapter_factory(
    factory,
    *,
    spec: ExperimentSpec,
    memory_root: Path,
    repo_root: Path | None = None,
):
    kwargs: dict[str, Any] = {
        "spec": spec,
        "memory_root": memory_root,
    }
    if repo_root is not None:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "repo_root" in signature.parameters:
            kwargs["repo_root"] = repo_root
    return factory(**kwargs)


@dataclass(frozen=True)
class ExecutionBackendResolution:
    kind: ExecutionBackendKind
    factory_path: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RuntimeBinding:
    executor_factory_path: str | None
    handlers: dict[str, Any]
    capability_provider: Any
    capability_context: CapabilityContext


def _slugify_runner_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "repo"


def _authored_runner_paths(memory_root: Path, repo_root: Path) -> tuple[Path, Path]:
    runners_dir = memory_root / "runners"
    stem = _slugify_runner_name(repo_root.name)
    return (
        runners_dir / f"{stem}_runner.py",
        runners_dir / f"{stem}_runner_manifest.json",
    )


MAX_RUNNER_AUTHORING_ATTEMPTS = 3


def _coerce_preflight_checks(raw_checks: list[Any] | None) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for item in raw_checks or []:
        if isinstance(item, PreflightCheck):
            checks.append(item)
        elif isinstance(item, dict):
            checks.append(PreflightCheck.from_dict(item))
    return checks


def _validate_runner_factory(
    *,
    factory_path: str,
    objective: str,
    memory_root: Path,
) -> RunnerValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[PreflightCheck] = []
    try:
        spec = build_bootstrap_spec(objective)
        adapter_setup = load_adapter_setup(
            factory_path=factory_path, spec=spec, memory_root=memory_root
        )
    except Exception as exc:
        return RunnerValidationResult(
            success=False,
            factory_path=factory_path,
            errors=[f"Runner import failed: {exc}"],
        )
    if adapter_setup is None:
        return RunnerValidationResult(
            success=False,
            factory_path=factory_path,
            errors=["build_adapter did not return an AdapterSetup."],
        )
    if not adapter_setup.handlers:
        errors.append("Runner did not expose any executable handlers.")
    try:
        if adapter_setup.discovery_provider is not None:
            capability_context = adapter_setup.discovery_provider(objective)
        elif adapter_setup.capability_provider is not None:
            capability_context = adapter_setup.capability_provider(spec)
        else:
            capability_context = CapabilityContext()
            errors.append(
                "Runner did not expose a discovery_provider or capability_provider."
            )
    except Exception as exc:
        return RunnerValidationResult(
            success=False,
            factory_path=factory_path,
            errors=[f"Runner capability discovery failed: {exc}"],
        )
    if not adapter_setup.preflight_provider:
        errors.append("Runner did not expose a preflight_provider.")
    else:
        try:
            checks = _coerce_preflight_checks(
                adapter_setup.preflight_provider(spec, capability_context)
            )
        except Exception as exc:
            errors.append(f"Runner preflight failed to execute: {exc}")
        else:
            for check in checks:
                if check.status != "failed":
                    continue
                if check.name in {
                    "repo_execution_not_supported",
                    "auto_adapter_scaffold",
                    "autonomous_execution_permissions",
                }:
                    errors.append(check.detail)
                else:
                    warnings.append(check.detail)
    smoke_test_passed = False
    if not errors:
        smoke_error = _smoke_validate_runner(
            adapter_setup=adapter_setup,
            capability_context=capability_context,
            objective=objective,
        )
        if smoke_error is None:
            smoke_test_passed = True
        else:
            errors.append(smoke_error)
    return RunnerValidationResult(
        success=not errors,
        factory_path=factory_path,
        errors=errors,
        warnings=warnings,
        preflight_checks=checks,
        smoke_test_passed=smoke_test_passed,
    )


def _metric_is_incomplete(metric: PrimaryMetric) -> bool:
    return metric.name == "[unspecified]" or metric.goal not in (
        "maximize",
        "minimize",
    )


def _apply_metric_catalog_defaults(
    spec_dict: dict[str, Any], capability_context: CapabilityContext
) -> dict[str, Any]:
    metric_catalog = capability_context.available_metrics or {}

    def enrich(metric_payload: dict[str, Any] | None) -> dict[str, Any]:
        metric_data = dict(metric_payload or {})
        metric_name = metric_data.get("name")
        if not metric_name or metric_name not in metric_catalog:
            return metric_data
        repo_meta = metric_catalog[metric_name]
        if metric_data.get("goal") not in {"maximize", "minimize"}:
            repo_goal = repo_meta.get("goal")
            if repo_goal in {"maximize", "minimize"}:
                metric_data["goal"] = repo_goal
        if not metric_data.get("scorer_ref") and repo_meta.get("scorer_ref"):
            metric_data["scorer_ref"] = repo_meta["scorer_ref"]
        return metric_data

    patched = dict(spec_dict)
    patched["primary_metric"] = enrich(spec_dict.get("primary_metric"))
    patched["secondary_metrics"] = [
        enrich(metric) for metric in spec_dict.get("secondary_metrics", [])
    ]
    patched["guardrail_metrics"] = [
        enrich(metric) for metric in spec_dict.get("guardrail_metrics", [])
    ]
    return patched


def _normalise_metric_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _extract_primary_metric_from_feedback(
    feedback: str, capability_context: CapabilityContext
) -> str | None:
    text = " ".join(str(feedback).split())
    if not text:
        return None

    metric_catalog = capability_context.available_metrics or {}
    normalised_feedback = _normalise_metric_name(text)
    if "primarymetric" not in normalised_feedback:
        return None

    for metric_name in sorted(metric_catalog, key=len, reverse=True):
        if _is_generic_metric_placeholder(metric_name):
            continue
        if _normalise_metric_name(metric_name) in normalised_feedback:
            return metric_name

    patterns = (
        r"([A-Za-z0-9_][A-Za-z0-9_ \-]{0,80}?)\s+as\s+primary\s+metric\b",
        r"primary\s+metric\s*(?:to|=|is|should\s+be)?\s*[:\-]?\s*([A-Za-z0-9_][A-Za-z0-9_ \-]{0,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" .,:;!-")
        if candidate:
            normalised_candidate = _normalise_metric_name(candidate)
            for metric_name in metric_catalog:
                if _is_generic_metric_placeholder(metric_name):
                    continue
                if _normalise_metric_name(metric_name) == normalised_candidate:
                    return metric_name
            normalised_metric_map = {
                _normalise_metric_name(metric_name): metric_name
                for metric_name in metric_catalog
                if not _is_generic_metric_placeholder(metric_name)
            }
            close_match = get_close_matches(
                normalised_candidate,
                list(normalised_metric_map.keys()),
                n=1,
                cutoff=0.55,
            )
            if close_match:
                return normalised_metric_map[close_match[0]]
            return candidate
    return None


def _is_generic_metric_placeholder(name: str | None) -> bool:
    normalised = _normalise_metric_name(name or "")
    return normalised in {
        "",
        "metric",
        "primarymetric",
        "secondarymetric",
        "guardrailmetric",
        "scorer",
        "score",
        "loss",
    }


def _smoke_validate_runner(
    *,
    adapter_setup: AdapterSetup,
    capability_context: CapabilityContext,
    objective: str,
) -> str | None:
    if not adapter_setup.handlers:
        return "Runner smoke test failed: no handlers were exposed."
    handler_name = (
        "baseline"
        if "baseline" in adapter_setup.handlers
        else next(iter(adapter_setup.handlers))
    )
    spec = ExperimentSpec(
        objective=objective,
        primary_metric=PrimaryMetric(name="primary_metric", goal="maximize"),
        allowed_actions=[handler_name],
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=capability_context,
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )
    candidate = ExperimentCandidate(
        hypothesis="Smoke-test the authored runner baseline path.",
        action_type=handler_name,
        change_type="baseline" if handler_name == "baseline" else "validation",
        instructions=f"Smoke-test the {handler_name} handler.",
    )
    handler = adapter_setup.handlers[handler_name]
    try:
        outcome = handler.execute(candidate, snapshot)
    except Exception as exc:
        return f"Runner smoke test failed while executing {handler_name}: {exc}"
    if not isinstance(outcome, ExperimentOutcome):
        return f"Runner smoke test failed: handler {handler_name} did not return an ExperimentOutcome."
    return None


def build_bootstrap_spec(objective: str) -> ExperimentSpec:
    return ExperimentSpec(
        objective=objective,
        primary_metric=PrimaryMetric(name="primary_metric", goal="maximize"),
        allowed_actions=[],
    )


def cycle_results_to_payload(results: list[Any]) -> list[dict[str, Any]]:
    return [
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
    ]


def default_role_models(
    planner_model: str | None = None,
    worker_model: str | None = None,
    review_model: str | None = None,
    consultation_model: str | None = None,
    narrator_model: str | None = None,
    profile: str | None = None,
) -> RoleModelConfig:
    role_models, _ = _resolve_role_models(
        planner_model=planner_model,
        worker_model=worker_model,
        review_model=review_model,
        consultation_model=consultation_model,
        narrator_model=narrator_model,
        profile=profile,
    )
    return role_models


def load_adapter_setup(
    *,
    factory_path: str,
    spec: ExperimentSpec,
    memory_root: Path,
    repo_root: Path | None = None,
) -> AdapterSetup | None:
    adapter_result = _invoke_adapter_factory(
        load_factory(factory_path),
        spec=spec,
        memory_root=memory_root,
        repo_root=repo_root,
    )
    if isinstance(adapter_result, AdapterSetup):
        return adapter_result
    return None


DATA_PROBE_TIMEOUT_SECONDS = 5
DATA_PROBE_MAX_FILE_MB = 50
DATA_PROBE_SAMPLE_ROWS = 5
LOCAL_PROBEABLE_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}
_ACTIVE_DATA_PROBES: dict[str, Any] = {}


def _normalise_probe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if callable(getattr(value, "isoformat", None)):
        return str(value.isoformat())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalise_probe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_normalise_probe_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return _normalise_probe_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return _normalise_probe_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _looks_like_local_asset_path(asset_path: str) -> bool:
    stripped = asset_path.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if "://" in lowered or lowered.startswith(
        ("s3:", "dbfs:", "jdbc:", "snowflake:", "databricks:")
    ):
        return False
    if any(token in stripped for token in ("\n", "\r")):
        return False
    path = Path(stripped)
    if path.suffix.lower() in LOCAL_PROBEABLE_SUFFIXES:
        return True
    if path.is_absolute():
        return True
    if any(sep in stripped for sep in ("/", "\\")):
        return True
    return False


def _probe_rows_note(schema: DataAssetSchema) -> str:
    if schema.total_rows_verified is not None:
        return f"verified total rows={schema.total_rows_verified}"
    if schema.total_rows_estimate is not None:
        return f"estimated total rows={schema.total_rows_estimate}"
    if schema.sample_rows_loaded is not None:
        return f"sampled {schema.sample_rows_loaded} rows; total rows unknown"
    return "row count unknown"


def _probe_data_asset(
    asset_path: str,
    repo_root: Path,
    *,
    timeout_seconds: float | None = None,
    thread_factory=None,
) -> DataAssetSchema:
    """Quickly inspect a single data asset. Returns schema info or a load error."""
    import threading

    timeout_seconds = (
        DATA_PROBE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    if thread_factory is None:
        thread_factory = threading.Thread

    if not _looks_like_local_asset_path(asset_path):
        return DataAssetSchema(
            asset_path=asset_path,
            verification_level="not_local_file",
            load_error="Asset is not a local file-backed path; probe requires adapter-provided metadata.",
        )

    resolved = Path(asset_path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    probe_key = str(resolved.resolve())
    active_probe = _ACTIVE_DATA_PROBES.get(probe_key)
    if active_probe is not None and active_probe.is_alive():
        return DataAssetSchema(
            asset_path=asset_path,
            load_error="Probe skipped because a previous timed-out probe is still running for this asset.",
        )
    _ACTIVE_DATA_PROBES.pop(probe_key, None)

    def _do_probe() -> DataAssetSchema:
        if not resolved.exists():
            return DataAssetSchema(
                asset_path=asset_path, load_error=f"File not found: {resolved}"
            )

        size_mb = resolved.stat().st_size / (1024 * 1024)
        suffix = resolved.suffix.lower()
        probe_limit = DATA_PROBE_SAMPLE_ROWS * 10

        # Too large — report size but don't load
        if size_mb > DATA_PROBE_MAX_FILE_MB:
            return DataAssetSchema(
                asset_path=asset_path,
                load_error=f"File too large for quick probe ({size_mb:.1f} MB > {DATA_PROBE_MAX_FILE_MB} MB limit)",
            )

        try:
            import pandas as pd
        except ImportError:
            return DataAssetSchema(
                asset_path=asset_path, load_error="pandas not available for data probe"
            )

        try:
            total_rows_verified: int | None = None
            total_rows_estimate: int | None = None
            verification_level = "sample_only"
            if suffix == ".csv":
                df = pd.read_csv(resolved, nrows=probe_limit)
                if len(df) < probe_limit:
                    total_rows_verified = len(df)
            elif suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq
                except ImportError as exc:
                    return DataAssetSchema(
                        asset_path=asset_path,
                        verification_level="unsupported",
                        load_error=f"Parquet probing requires pyarrow for cheap metadata inspection: {exc}",
                    )
                parquet_file = pq.ParquetFile(resolved)
                total_rows_verified = int(parquet_file.metadata.num_rows)
                verification_level = "verified_total_rows"
                if parquet_file.num_row_groups == 0:
                    df = pd.DataFrame()
                else:
                    sample_table = parquet_file.read_row_group(0)
                    if sample_table.num_rows > probe_limit:
                        sample_table = sample_table.slice(0, probe_limit)
                    df = sample_table.to_pandas()
            elif suffix in (".json", ".jsonl"):
                df = pd.read_json(resolved, lines=suffix == ".jsonl", nrows=probe_limit)
                if len(df) < probe_limit:
                    total_rows_verified = len(df)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(resolved, nrows=probe_limit)
                if len(df) < probe_limit:
                    total_rows_verified = len(df)
            else:
                return DataAssetSchema(
                    asset_path=asset_path, load_error=f"Unsupported format: {suffix}"
                )
        except Exception as exc:
            return DataAssetSchema(
                asset_path=asset_path, load_error=f"Load failed: {exc}"
            )

        sample_values = {}
        for col in df.columns[:30]:
            unique = df[col].dropna().unique()[:DATA_PROBE_SAMPLE_ROWS]
            sample_values[col] = [_normalise_probe_value(value) for value in unique]

        return DataAssetSchema(
            asset_path=asset_path,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            sample_rows_loaded=len(df),
            total_rows_verified=total_rows_verified,
            total_rows_estimate=total_rows_estimate,
            verification_level=verification_level,
            sample_values=sample_values,
        )

    result_holder: dict[str, DataAssetSchema | Exception] = {}

    def _run_probe() -> None:
        try:
            result_holder["result"] = _do_probe()
        except Exception as exc:
            result_holder["error"] = exc

    worker = thread_factory(target=_run_probe, daemon=True)
    _ACTIVE_DATA_PROBES[probe_key] = worker
    worker.start()
    worker.join(timeout=timeout_seconds)
    if worker.is_alive():
        return DataAssetSchema(
            asset_path=asset_path,
            load_error=f"Probe timed out after {timeout_seconds}s",
        )
    _ACTIVE_DATA_PROBES.pop(probe_key, None)
    if "error" in result_holder:
        return DataAssetSchema(
            asset_path=asset_path, load_error=f"Probe error: {result_holder['error']}"
        )
    return result_holder.get("result") or DataAssetSchema(
        asset_path=asset_path,
        load_error="Probe failed without returning a schema result.",
    )


def probe_data_assets(
    capability_context: CapabilityContext, repo_root: Path
) -> CapabilityContext:
    """Run a quick schema probe on all discovered data assets. Non-blocking with hard timeout."""
    if not capability_context.available_data_assets:
        return capability_context
    schemas: list[DataAssetSchema] = []
    stop_after_timeout = False
    for asset in capability_context.available_data_assets:
        if stop_after_timeout:
            schemas.append(
                DataAssetSchema(
                    asset_path=asset,
                    load_error="Probe skipped because a previous asset probe timed out in this scan.",
                )
            )
            continue
        schema = _probe_data_asset(asset, repo_root)
        schemas.append(schema)
        if schema.load_error and "timed out" in schema.load_error.lower():
            stop_after_timeout = True
    notes = list(capability_context.notes)
    for schema in schemas:
        if schema.load_error:
            notes.append(f"Data probe ({schema.asset_path}): {schema.load_error}")
        elif schema.columns:
            notes.append(
                f"Data probe ({schema.asset_path}): {len(schema.columns)} columns, "
                f"{_probe_rows_note(schema)}. Columns: {', '.join(schema.columns[:15])}"
                + (
                    f" ... and {len(schema.columns) - 15} more"
                    if len(schema.columns) > 15
                    else ""
                )
            )
    return replace(capability_context, data_schemas=schemas, notes=notes)


def _detect_repo_python(repo_root: Path) -> str:
    """Find the Python executable for the target repo, not the running process."""
    # Check for .venv in the repo root
    if os.name == "nt":
        candidates = [
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            repo_root / ".venv" / "bin" / "python",
            repo_root / "venv" / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # Fallback to running process's Python
    return sys.executable


def discover_capabilities_for_objective(
    *,
    objective: str,
    memory_root: Path | str,
    executor_factory_path: str | None,
    repo_root: Path | str = ".",
) -> CapabilityContext:
    memory_root_path = Path(memory_root)
    from loopforge.auto_adapter import scan_repo

    scan_result = scan_repo(Path(repo_root))
    shell_family = "windows_cmd" if os.name == "nt" else "posix_sh"
    execution_shell = "cmd.exe" if os.name == "nt" else "/bin/sh"
    platform_facts = {
        "host_platform": sys.platform,
        "os_name": os.name,
        "execution_shell": execution_shell,
        "shell_family": shell_family,
        "python_executable": _detect_repo_python(Path(repo_root)),
        "repo_root": str(Path(repo_root).resolve()),
    }
    if executor_factory_path is None:
        context = build_repo_scan_context(
            repo_root,
            objective=objective,
            summary=scan_result,
        )
        context = replace(
            context,
            environment_facts={**context.environment_facts, **platform_facts},
            notes=[
                (
                    "Runtime platform: Windows commands run through cmd.exe. "
                    "Avoid Unix tools like head/grep/find -maxdepth/ls -la; prefer Python scripts or cmd-compatible commands."
                    if os.name == "nt"
                    else "Runtime platform: shell commands run through /bin/sh. Prefer portable repo-local commands."
                ),
                *context.notes,
            ],
        )
        return probe_data_assets(context, Path(repo_root))
    adapter_setup = load_adapter_setup(
        factory_path=executor_factory_path,
        spec=build_bootstrap_spec(objective),
        memory_root=memory_root_path,
        repo_root=Path(repo_root),
    )
    if adapter_setup is None:
        return CapabilityContext()
    if adapter_setup.discovery_provider is not None:
        context = adapter_setup.discovery_provider(objective)
    elif adapter_setup.capability_provider is not None:
        context = adapter_setup.capability_provider(build_bootstrap_spec(objective))
    else:
        context = CapabilityContext()
    discovered_assets = [
        asset
        for asset in scan_result.get("data_assets", [])
        if asset not in context.available_data_assets
    ]
    if discovered_assets:
        context = replace(
            context,
            available_data_assets=[*context.available_data_assets, *discovered_assets],
        )
    # Quick data probe — non-blocking, hard timeout per asset
    context = probe_data_assets(context, Path(repo_root))
    # Augment with fresh column scan so the agent always sees what columns exist in the code
    column_notes = [n for n in scan_result.get("notes", []) if "column" in n.lower()]
    for file_path, cols in scan_result.get("column_refs", {}).items():
        column_notes.append(f"  {file_path}: {', '.join(cols)}")
    if column_notes:
        context = replace(context, notes=[*context.notes, *column_notes])
    context = replace(
        context,
        environment_facts={**context.environment_facts, **platform_facts},
        notes=[
            (
                "Runtime platform: Windows commands run through cmd.exe. "
                "Avoid Unix tools like head/grep/find -maxdepth/ls -la; prefer Python scripts or cmd-compatible commands."
                if os.name == "nt"
                else "Runtime platform: shell commands run through /bin/sh. Prefer portable repo-local commands."
            ),
            *context.notes,
        ],
    )
    return context


def run_preflight_checks(
    *,
    spec: ExperimentSpec,
    capability_context: CapabilityContext,
    memory_root: Path | str,
    executor_factory_path: str | None,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    memory_root_path = Path(memory_root)

    invalid_metric_goals = [
        metric.name
        for metric in [
            spec.primary_metric,
            *spec.secondary_metrics,
            *spec.guardrail_metrics,
        ]
        if metric.goal not in {"maximize", "minimize"}
    ]
    if invalid_metric_goals:
        checks.append(
            PreflightCheck(
                name="metric_goal_unspecified",
                status="failed",
                detail=(
                    "Every metric must explicitly set goal to 'maximize' or 'minimize'. "
                    f"Invalid goal on: {', '.join(invalid_metric_goals)}"
                ),
                scope="bootstrap",
            )
        )

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

    guidance_required = capability_context.environment_facts.get(
        "execution_guidance_required"
    )
    if guidance_required:
        checks.append(
            PreflightCheck(
                name="execution_strategy_unresolved",
                status="failed",
                detail=str(
                    capability_context.environment_facts.get(
                        "execution_guidance_detail"
                    )
                    or "Loopforge still needs one high-level hint about how to reach the real data or execution backend."
                ),
                scope="bootstrap",
            )
        )
        return checks

    if (
        executor_factory_path is None
        or capability_context.environment_facts.get("autonomous_execution_supported")
        is False
    ):
        if (
            executor_factory_path is None
            and capability_context.environment_facts.get(
                "autonomous_execution_supported"
            )
            is not False
        ):
            repo_root_raw = capability_context.environment_facts.get("repo_root")
            repo_root_path = (
                Path(repo_root_raw)
                if isinstance(repo_root_raw, str) and repo_root_raw.strip()
                else Path.cwd()
            )
            python_executable = capability_context.environment_facts.get(
                "python_executable", sys.executable
            )
            try:
                probe = subprocess.run(
                    [
                        str(python_executable),
                        "-c",
                        "import os,sys;print('loopforge_execution_probe_ok');print(os.getcwd());print(sys.executable)",
                    ],
                    cwd=repo_root_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except Exception as exc:
                checks.append(
                    PreflightCheck(
                        name="generic_agentic_execution_probe",
                        status="failed",
                        detail=(
                            "Loopforge could not verify that the generic autonomous executor can run Python from the repo root: "
                            f"{exc}"
                        ),
                        scope="execution",
                    )
                )
                return checks
            if (
                probe.returncode != 0
                or "loopforge_execution_probe_ok" not in probe.stdout
            ):
                failure_summary = (
                    probe.stderr or probe.stdout or f"Exit code {probe.returncode}."
                ).strip()
                checks.append(
                    PreflightCheck(
                        name="generic_agentic_execution_probe",
                        status="failed",
                        detail=(
                            "Loopforge could not verify the generic autonomous execution lane from the repo root. "
                            f"Probe command failed: {failure_summary}"
                        ),
                        scope="execution",
                    )
                )
                return checks
            checks.append(
                PreflightCheck(
                    name="generic_agentic_execution",
                    status="passed",
                    detail="No repo-specific runner was found, so Loopforge will use the generic autonomous executor.",
                    required=False,
                    scope="execution",
                )
            )
            checks.append(
                PreflightCheck(
                    name="generic_agentic_execution_probe",
                    status="passed",
                    detail=(
                        "Verified the generic autonomous execution lane by running Python from the repo root "
                        f"with {python_executable}."
                    ),
                    required=False,
                    scope="execution",
                )
            )
            return checks
        checks.append(
            PreflightCheck(
                name="repo_execution_not_supported",
                status="failed",
                detail=(
                    "Loopforge can inspect this repo and plan the experiment, but this objective is not yet wired "
                    "to a real autonomous runner."
                ),
                scope="execution",
            )
        )
        return checks

    adapter_setup = load_adapter_setup(
        factory_path=executor_factory_path,
        spec=spec,
        memory_root=memory_root_path,
        repo_root=(
            Path(capability_context.environment_facts["repo_root"])
            if capability_context.environment_facts.get("repo_root")
            else None
        ),
    )
    if adapter_setup is not None and adapter_setup.preflight_provider is not None:
        checks.extend(
            _coerce_preflight_checks(
                adapter_setup.preflight_provider(spec, capability_context)
            )
        )
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


def _trace_data_source(
    *,
    user_goal: str,
    repo_root: Path,
    python_executable: str,
    progress_fn: ProgressFn,
) -> dict[str, Any]:
    """Deterministic data tracing: find files the user referenced, try loading data.

    Returns facts and generated questions based on real execution results.
    """
    results: dict[str, Any] = {
        "data_source": None,
        "file_content_preview": None,
        "data_loaded": False,
        "columns": [],
        "load_error": None,
        "generated_questions": [],
        "dependency_sync": {
            "attempted": False,
            "succeeded": False,
            "tool": None,
        },
    }

    # 1. Extract file/module references from user's goal
    progress_fn("data_trace", "Finding files referenced in your goal...")
    # Look for identifiers that could be filenames (words with underscores, .py suffix)
    import re as _re

    # Match explicit .py files or underscore-separated identifiers
    candidates: list[str] = []
    for match in _re.finditer(r"[\w/\\]+\.py\b", user_goal):
        candidates.append(match.group(0))
    for match in _re.finditer(
        r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", user_goal.lower()
    ):
        token = match.group(1)
        if token not in ("cross_validation", "primary_metric", "new_script"):
            candidates.append(token)

    # 2. Find matching files in the repo
    matched_file: str | None = None
    for candidate in candidates:
        # Try exact path first
        exact = repo_root / candidate
        if exact.exists() and exact.is_file():
            matched_file = str(exact.relative_to(repo_root))
            break
        # Try glob patterns
        patterns = [
            f"**/{candidate}",
            f"**/{candidate}.py",
            f"**/scripts/{candidate}.py",
            f"**/scripts/*{candidate}*.py",
            f"**/*{candidate}*.py",
        ]
        for pattern in patterns:
            matches = sorted(repo_root.glob(pattern))
            matches = [m for m in matches if m.is_file() and m.suffix == ".py"]
            if matches:
                matched_file = str(matches[0].relative_to(repo_root))
                break
        if matched_file:
            break

    if not matched_file:
        progress_fn("data_trace", "Could not find a matching file in the repo.")
        results["generated_questions"].append(
            SpecQuestion(
                key="data_source_file",
                prompt=(
                    "I couldn't find the file you referenced in the repo. "
                    "Which file contains the model training code? "
                    "(e.g. scripts/lol_train_models.py)"
                ),
                rationale="Need to identify the source file to trace the data pipeline.",
                required=True,
            )
        )
        return results

    results["data_source"] = matched_file
    progress_fn("data_trace", f"Found: {matched_file}")

    # 3. Read the file
    try:
        file_path = repo_root / matched_file
        content = file_path.read_text(encoding="utf-8", errors="replace")
        results["file_content_preview"] = content[:5000]
        progress_fn("data_trace", f"Read {matched_file} ({len(content)} chars)")
    except Exception as exc:
        results["load_error"] = f"Could not read {matched_file}: {exc}"
        return results

    # 4. Sync dependencies when uv is available. Otherwise keep using the
    # current environment instead of making uv a hard prerequisite.
    if shutil.which("uv") is None:
        progress_fn(
            "data_trace",
            "uv not found; using the current Python environment without auto-sync.",
        )
        return results

    progress_fn("data_trace", "Syncing dependencies with uv...")
    results["dependency_sync"] = {
        "attempted": True,
        "succeeded": False,
        "tool": "uv",
    }
    try:
        completed = subprocess.run(
            ["uv", "sync"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:
        progress_fn(
            "data_trace",
            f"Dependency sync unavailable ({exc}); continuing with the current environment.",
        )
        return results

    if completed.returncode == 0:
        results["dependency_sync"]["succeeded"] = True
        progress_fn("data_trace", "Dependencies synced with uv.")
    else:
        progress_fn(
            "data_trace",
            "uv sync failed; continuing with the current environment.",
        )

    return results


def _verify_execution_environment(
    *,
    repo_root: Path,
    capability_context: CapabilityContext,
    progress_fn: ProgressFn,
    dependency_sync: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run real checks to verify the experiment can actually execute.

    Returns a dict with verification results to include in the bootstrap handoff.
    """
    results: dict[str, Any] = {
        "dependency_sync": dependency_sync or {},
        "baseline_metric": None,
        "baseline_script": None,
        "baseline_output": None,
        "errors": [],
    }
    env = capability_context.environment_facts
    python_executable = str(env.get("python_executable") or sys.executable)

    # 1. Validate key imports work (deps already synced in data trace)
    import_check = subprocess.run(
        [
            python_executable,
            "-c",
            "import sys; "
            "failures = []; "
            "for pkg in ['polars', 'numpy', 'pandas', 'sklearn']: \n"
            "    try: __import__(pkg)\n"
            "    except ImportError: failures.append(pkg)\n"
            "if failures: print('MISSING:' + ','.join(failures)); sys.exit(1)\n"
            "print('imports_ok')",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if import_check.returncode == 0:
        results["imports_ok"] = True
        progress_fn("env_verify", "Key imports verified.")
    else:
        missing = import_check.stdout.strip()
        results["imports_ok"] = False
        results["errors"].append(f"Import check: {missing}")
        progress_fn("env_verify", f"Import check failed: {missing}")

    # 3. Find and try running the baseline script
    baseline_paths = [
        str(p).strip() for p in env.get("baseline_code_paths", []) if str(p).strip()
    ]
    if not baseline_paths:
        # Search for likely experiment scripts
        for pattern in [
            "scripts/*player*kills*.py",
            "scripts/*experiment*.py",
            "experiments/*.py",
        ]:
            candidates = sorted(repo_root.glob(pattern))
            if candidates:
                baseline_paths = [str(c.relative_to(repo_root)) for c in candidates[:3]]
                break

    if baseline_paths:
        baseline_script = baseline_paths[0]
        results["baseline_script"] = baseline_script
        script_path = repo_root / baseline_script

        if script_path.exists() and script_path.suffix == ".py":
            progress_fn("env_verify", f"Running baseline script: {baseline_script}...")
            try:
                run_result = subprocess.run(
                    [python_executable, str(script_path)],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env={**os.environ, "PYTHONPATH": str(repo_root)},
                )
                results["baseline_output"] = run_result.stdout[-4000:]
                if run_result.returncode == 0:
                    progress_fn("env_verify", "Baseline script completed successfully.")
                else:
                    error_preview = (run_result.stderr or run_result.stdout)[-500:]
                    results["errors"].append(
                        f"Baseline script failed (exit {run_result.returncode}): {error_preview}"
                    )
                    progress_fn(
                        "env_verify", f"Baseline script failed: {error_preview[:200]}"
                    )
            except subprocess.TimeoutExpired:
                results["errors"].append("Baseline script timed out after 300s")
                progress_fn("env_verify", "Baseline script timed out after 300s.")
            except Exception as exc:
                results["errors"].append(f"Could not run baseline: {exc}")
                progress_fn("env_verify", f"Could not run baseline: {exc}")

    return results


def _non_local_data_assets(capability_context: CapabilityContext) -> list[str]:
    return [
        asset
        for asset in capability_context.available_data_assets
        if isinstance(asset, str)
        and asset.strip()
        and not _looks_like_local_asset_path(asset)
    ]


def _summarise_bootstrap_answers(answers: dict[str, Any] | None) -> str:
    if not isinstance(answers, dict):
        return ""
    parts: list[str] = []
    for key, value in answers.items():
        if str(key) in TRANSIENT_BOOTSTRAP_ANSWER_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value if str(item).strip())
        else:
            rendered = str(value).strip()
        if rendered:
            parts.append(f"{key}={rendered}")
    return "\n".join(parts)


def _text_tokens(value: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    normalized = normalized.replace("_", " ")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized.lower())
        if len(token) >= 3
    }


def _compact_capability_context_for_metric_repair(
    *,
    objective: str,
    assistant_message: str,
    capability_context: CapabilityContext,
    limit: int = 24,
) -> dict[str, Any]:
    query_text = f"{objective}\n{assistant_message}"
    query_tokens = _text_tokens(query_text)
    metric_catalog = capability_context.available_metrics or {}
    ranked_metrics: list[tuple[int, str, dict[str, Any]]] = []

    for name, meta in metric_catalog.items():
        if not isinstance(name, str):
            continue
        metric_text = " ".join(
            [
                name,
                str(meta.get("scorer_ref", "")),
                str(meta.get("path", "")),
                str(meta.get("goal", "")),
                str(meta.get("preferred_role", "")),
            ]
        )
        metric_tokens = _text_tokens(metric_text)
        overlap = len(query_tokens & metric_tokens)
        exact_name_match = int(name.lower() in query_text.lower())
        score = overlap * 10 + exact_name_match * 100
        if score > 0:
            ranked_metrics.append((score, name, meta))

    ranked_metrics.sort(key=lambda item: (-item[0], item[1]))
    compact_metrics = {name: meta for _, name, meta in ranked_metrics[:limit]}
    if not compact_metrics:
        compact_metrics = dict(list(metric_catalog.items())[:limit])

    return {
        "available_metrics": compact_metrics,
        "environment_facts": capability_context.environment_facts,
        "notes": capability_context.notes[:20],
        "available_actions": capability_context.available_actions,
        "available_data_assets": capability_context.available_data_assets[:20],
    }


def _infer_local_execution_confidence(
    *,
    user_goal: str,
    capability_context: CapabilityContext,
    answers: dict[str, Any] | None,
) -> int:
    score = 0
    local_assets = [
        asset
        for asset in capability_context.available_data_assets
        if isinstance(asset, str)
        and asset.strip()
        and _looks_like_local_asset_path(asset)
    ]
    if local_assets:
        score += 1

    local_action_paths = [
        target
        for target in capability_context.available_actions.values()
        if isinstance(target, str)
        and target.strip()
        and (
            _looks_like_local_asset_path(target)
            or target.endswith(".py")
            or target.startswith(
                ("scripts/", "src/", "sports_predictions_orchestrator/")
            )
        )
    ]
    if local_action_paths:
        score += 1

    env_facts = capability_context.environment_facts or {}
    if env_facts.get("repo_root") or env_facts.get("python_executable"):
        score += 1
    baseline_paths = env_facts.get("baseline_code_paths")
    if isinstance(baseline_paths, list) and any(
        isinstance(path, str) and _looks_like_local_asset_path(path)
        for path in baseline_paths
    ):
        score += 1

    local_language = " ".join(
        [
            user_goal.lower(),
            _summarise_bootstrap_answers(answers).lower(),
            " ".join(capability_context.notes).lower(),
        ]
    )
    if any(
        phrase in local_language
        for phrase in (
            "this repo",
            "existing frame",
            "existing pipeline",
            "local python",
            "find where",
            "check lol data",
            "work within existing",
        )
    ):
        score += 1
    return score


def _detect_execution_guidance_gap(
    *,
    user_goal: str,
    capability_context: CapabilityContext,
    answers: dict[str, Any] | None,
) -> str | None:
    if isinstance(answers, dict):
        hint = answers.get(EXECUTION_GUIDANCE_QUESTION_KEY)
        if hint not in (None, ""):
            return None
    answer_text = _summarise_bootstrap_answers(answers).lower()
    if answer_text and any(
        token in answer_text for token in REMOTE_EXECUTION_HINT_TOKENS
    ):
        return None

    remote_assets = _non_local_data_assets(capability_context)
    env_blob = json.dumps(capability_context.environment_facts, sort_keys=True).lower()
    notes_blob = " ".join(capability_context.notes).lower()
    goal_blob = user_goal.lower()
    matched_tokens = [
        token
        for token in REMOTE_EXECUTION_HINT_TOKENS
        if token in env_blob or token in notes_blob or token in goal_blob
    ]
    matched_tokens = list(dict.fromkeys(matched_tokens))
    if not remote_assets and not matched_tokens:
        return None
    if (
        _infer_local_execution_confidence(
            user_goal=user_goal,
            capability_context=capability_context,
            answers=answers,
        )
        >= 2
    ):
        return None

    evidence: list[str] = []
    if remote_assets:
        preview = ", ".join(remote_assets[:2])
        if len(remote_assets) > 2:
            preview += f", and {len(remote_assets) - 2} more"
        evidence.append(f"non-local data assets were detected ({preview})")
    if matched_tokens:
        token_preview = ", ".join(matched_tokens[:4])
        evidence.append(
            f"the repo/objective points at an external backend ({token_preview})"
        )
    joined_evidence = " and ".join(evidence)
    return (
        "Loopforge inferred most of the repo structure, but it cannot yet verify the real execution lane because "
        f"{joined_evidence}. Provide one high-level hint about whether the baseline should run through local Python "
        "code or through the remote platform the repo points at. A short directional answer is enough; exact paths "
        "or internal module names are not required."
    )


def _has_execution_guidance_answer(answers: dict[str, Any] | None) -> bool:
    if not isinstance(answers, dict):
        return False
    hint = answers.get(EXECUTION_GUIDANCE_QUESTION_KEY)
    return isinstance(hint, str) and bool(hint.strip())


def _build_execution_guidance_question(
    *,
    capability_context: CapabilityContext,
    detail: str,
) -> SpecQuestion:
    remote_assets = _non_local_data_assets(capability_context)
    backend_examples: list[str] = []
    for asset in remote_assets[:2]:
        if isinstance(asset, str) and asset.strip():
            backend_examples.append(asset.strip())
    if not backend_examples:
        for token in REMOTE_EXECUTION_HINT_TOKENS:
            haystack = (
                " ".join(capability_context.notes).lower()
                + "\n"
                + json.dumps(
                    capability_context.environment_facts,
                    sort_keys=True,
                ).lower()
            )
            if token in haystack:
                backend_examples.append(token)
        backend_examples = backend_examples[:2]
    prompt = (
        "What high-level execution direction should Loopforge use to reach the real data/runtime? "
        "A short answer is enough, for example 'use the repo's local Python loader against S3' or "
        "'run the baseline through Databricks'."
    )
    if backend_examples:
        prompt += f" I detected: {', '.join(backend_examples)}."
    return SpecQuestion(
        key=EXECUTION_GUIDANCE_QUESTION_KEY,
        prompt=prompt,
        rationale=detail,
        required=True,
    )


def _ensure_execution_guidance_question(
    *,
    questions: list[SpecQuestion],
    capability_context: CapabilityContext,
    answers: dict[str, Any] | None,
    detail: str | None,
) -> list[SpecQuestion]:
    if detail is None:
        return questions
    if isinstance(answers, dict) and any(
        value not in (None, "") for value in answers.values()
    ):
        if any(question.required for question in questions):
            return questions
    if any(question.key == EXECUTION_GUIDANCE_QUESTION_KEY for question in questions):
        return questions
    if any(question.required for question in questions):
        return questions
    return [
        *questions,
        _build_execution_guidance_question(
            capability_context=capability_context, detail=detail
        ),
    ]


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


def apply_answers_to_bootstrap_turn(
    turn: BootstrapTurn,
    *,
    answers: dict[str, Any] | None,
) -> BootstrapTurn:
    answer_map = {
        str(key): value
        for key, value in (answers or {}).items()
        if str(key) not in TRANSIENT_BOOTSTRAP_ANSWER_KEYS and value not in (None, "")
    }
    spec = turn.proposal.recommended_spec
    merged_metadata = dict(spec.metadata)
    merged_metadata["bootstrap_answers"] = {
        **dict(merged_metadata.get("bootstrap_answers", {})),
        **answer_map,
    }
    answer_notes = list(turn.proposal.notes)
    for key, value in answer_map.items():
        note = f"User answer ({key}): {value}"
        if note not in answer_notes:
            answer_notes.append(note)
    updated_spec = replace(spec, metadata=merged_metadata)
    updated_proposal = replace(
        turn.proposal, recommended_spec=updated_spec, notes=answer_notes
    )
    missing_requirements = missing_requirements_from_bootstrap(
        questions=updated_proposal.questions,
        answers=answer_map,
        preflight_checks=turn.preflight_checks,
    )
    required_question_keys = {
        question.key for question in updated_proposal.questions if question.required
    }
    resolved_required_questions = bool(
        required_question_keys
    ) and required_question_keys.issubset(answer_map.keys())
    had_answer_blocker = any(
        req.startswith("answer:") for req in turn.missing_requirements
    )
    ready_to_start = (
        turn.ready_to_start or had_answer_blocker or resolved_required_questions
    ) and not missing_requirements
    return replace(
        turn,
        proposal=updated_proposal,
        missing_requirements=missing_requirements,
        ready_to_start=ready_to_start,
    )


def _is_internal_bootstrap_question(question: SpecQuestion) -> bool:
    joined = " ".join(
        part
        for part in (
            question.key,
            question.prompt,
            question.rationale or "",
            " ".join(str(o) for o in (question.options or [])),
            question.suggested_answer or "",
        )
        if part
    ).lower()
    internal_tokens = (
        "adapter",
        "executor factory",
        "execution entrypoint",
        "entrypoint",
        "module path",
        "factory path",
        ".loopforge/generated",
    )
    return any(token in joined for token in internal_tokens)


def sanitise_bootstrap_questions(questions: list[SpecQuestion]) -> list[SpecQuestion]:
    return [
        question
        for question in questions
        if not _is_internal_bootstrap_question(question)
    ]


def refine_bootstrap_questions(
    questions: list[SpecQuestion],
    *,
    answers: dict[str, Any] | None,
) -> list[SpecQuestion]:
    """Remove internal implementation questions but keep all domain questions the agent proposed."""
    return sanitise_bootstrap_questions(questions)


class Loopforge:
    def __init__(
        self,
        *,
        executor_factory_path: str | None = None,
        repo_root: Path | str = ".",
        memory_root: Path | str = ".loopforge",
        planner_model: str | None = None,
        worker_model: str | None = None,
        review_model: str | None = None,
        consultation_model: str | None = None,
        narrator_model: str | None = None,
        model_profile: str | None = None,
        temperature: float = 0.2,
        bootstrap_backend: Any | None = None,
        worker_backend: Any | None = None,
        runner_authoring_backend: RunnerAuthoringBackend | None = None,
        access_advisor_backend: Any | None = None,
        narrator_backend: Any | None = None,
        progress_fn: ProgressFn | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        self.explicit_executor_factory_path = executor_factory_path
        self.executor_factory_path = executor_factory_path
        self.repo_root = Path(repo_root)
        self.memory_root = Path(memory_root)
        self.temperature = temperature
        self.progress_fn: ProgressFn = progress_fn or _noop_progress
        self._stream_fn = stream_fn
        self._cached_capability_context: CapabilityContext | None = None
        self._cached_capability_key: tuple[str, str | None, str] | None = None
        self._cached_authoring_failure_reason: str | None = None
        self._attempted_dependency_installs: set[str] = set()
        repo_model_settings, repo_model_settings_path = load_repo_role_model_settings(
            self.repo_root
        )
        self.repo_loop_settings, repo_loop_settings_path = load_repo_loop_settings(
            self.repo_root
        )
        bedrock_base = os.getenv("ANTHROPIC_BEDROCK_BASE_URL")
        if bedrock_base:
            self._bedrock_kwargs: dict[str, Any] = {
                "api_base": bedrock_base,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "drop_params": True,
            }
        else:
            self._bedrock_kwargs: dict[str, Any] = {}
        resolved_profile = (
            model_profile or repo_model_settings.model_profile or DEFAULT_MODEL_PROFILE
        )
        self.role_models, model_routing_warnings = _resolve_role_models(
            planner_model=planner_model or repo_model_settings.planner_model,
            worker_model=worker_model or repo_model_settings.worker_model,
            review_model=review_model or repo_model_settings.review_model,
            consultation_model=consultation_model
            or repo_model_settings.consultation_model,
            narrator_model=narrator_model or repo_model_settings.narrator_model,
            profile=resolved_profile,
        )
        if repo_model_settings_path is not None:
            self.progress_fn(
                "model_config_loaded",
                f"Loaded role model settings from {repo_model_settings_path}.",
            )
        if repo_loop_settings_path is not None:
            self.progress_fn(
                "repo_settings_loaded",
                f"Loaded repo loop settings from {repo_loop_settings_path}.",
            )
        for warning_index, warning in enumerate(model_routing_warnings, start=1):
            self.progress_fn(
                f"model_routing_warning_{warning_index}",
                f"Warning: {warning}",
            )

        self._log_startup_summary()

        def _runtime_model(model: str) -> str:
            return _normalise_model_id(model)

        def _extra(model: str) -> dict[str, Any]:
            if _runtime_model(model).startswith("bedrock/") and self._bedrock_kwargs:
                return {"extra_kwargs": self._bedrock_kwargs}
            return {}

        self.bootstrap_backend = bootstrap_backend or LiteLLMBootstrapBackend(
            model=_runtime_model(self.role_models.planner),
            request_model=_runtime_model(self.role_models.planner),
            temperature=temperature,
            max_completion_tokens=DEFAULT_ROLE_MAX_COMPLETION_TOKENS["planner"],
            stream_fn=stream_fn,
            progress_fn=self.progress_fn,
            **_extra(self.role_models.planner),
        )
        self.runner_authoring_backend = (
            runner_authoring_backend
            or LiteLLMRunnerAuthoringBackend(
                model=_runtime_model(self.role_models.worker),
                request_model=_runtime_model(self.role_models.worker),
                temperature=temperature,
                max_completion_tokens=DEFAULT_ROLE_MAX_COMPLETION_TOKENS[
                    "runner_authoring"
                ],
                stream_fn=stream_fn,
                progress_fn=self.progress_fn,
                **_extra(self.role_models.worker),
            )
        )
        self.access_advisor_backend = (
            access_advisor_backend
            or LiteLLMAccessAdvisorBackend(
                model=_runtime_model(self.role_models.consultation),
                request_model=_runtime_model(self.role_models.consultation),
                temperature=temperature,
                max_completion_tokens=DEFAULT_ROLE_MAX_COMPLETION_TOKENS[
                    "access_advisor"
                ],
                stream_fn=stream_fn,
                progress_fn=self.progress_fn,
                **_extra(self.role_models.consultation),
            )
        )
        self.narrator_backend = narrator_backend or LiteLLMNarrationBackend(
            model=_runtime_model(self.role_models.narrator),
            request_model=_runtime_model(self.role_models.narrator),
            temperature=temperature,
            max_completion_tokens=DEFAULT_ROLE_MAX_COMPLETION_TOKENS["narrator"],
            stream_fn=stream_fn,
            progress_fn=self.progress_fn,
            **_extra(self.role_models.narrator),
        )
        self._extra = _extra
        self._runtime_model = _runtime_model

    def _log_startup_summary(self) -> None:
        """Log a human-readable summary of the effective configuration."""
        models = self.role_models
        loop = self.repo_loop_settings
        # Deduplicate models: group roles that share the same model
        model_to_roles: dict[str, list[str]] = {}
        for role, model in [
            ("planner", models.planner),
            ("worker", models.worker),
            ("reflection", models.worker),
            ("review", models.review),
            ("consultation", models.consultation),
            ("narrator", models.narrator),
        ]:
            model_to_roles.setdefault(model, []).append(role)
        if len(model_to_roles) == 1:
            only_model = next(iter(model_to_roles))
            model_summary = f"all roles -> {only_model}"
        else:
            parts = []
            for model, roles in model_to_roles.items():
                parts.append(f"{', '.join(roles)} -> {model}")
            model_summary = " | ".join(parts)
        max_iter = loop.max_iterations if loop.max_iterations is not None else 30
        max_hours = loop.max_autonomous_hours if loop.max_autonomous_hours is not None else 6
        lines = [
            "Configuration:",
            f"  Models     : {model_summary}",
            f"  Iterations : up to {max_iter}",
            f"  Time limit : {max_hours}h",
        ]
        self.progress_fn("startup_summary_detail", "\n".join(lines))

    def resolve_execution_backend(self, objective: str) -> ExecutionBackendResolution:
        if self.explicit_executor_factory_path:
            self.executor_factory_path = self.explicit_executor_factory_path
            return ExecutionBackendResolution(
                kind="supported", factory_path=self.explicit_executor_factory_path
            )
        cached_authored_runner = self._load_cached_authored_runner(objective)
        if cached_authored_runner is not None:
            return cached_authored_runner
        self.executor_factory_path = None
        return ExecutionBackendResolution(
            kind="planning_only",
            reason=(
                "No repo-specific runner was found. Loopforge will use the generic autonomous executor for this objective."
            ),
        )

    def resolve_executor_factory_path(self, objective: str) -> str | None:
        return self.resolve_execution_backend(objective).factory_path

    def _fallback_capability_provider(
        self, _effective_spec: ExperimentSpec
    ) -> CapabilityContext:
        return self._cached_capability_context or CapabilityContext()

    def resolve_runtime_binding(
        self,
        *,
        spec: ExperimentSpec,
        objective: str,
        executor_factory_path: str | None = None,
    ) -> RuntimeBinding:
        resolved_factory_path = executor_factory_path
        if resolved_factory_path is None:
            resolved_factory_path = self.resolve_executor_factory_path(objective)
        if resolved_factory_path is None:
            capability_context = self._fallback_capability_provider(spec)
            return RuntimeBinding(
                executor_factory_path=None,
                handlers={},
                capability_provider=self._fallback_capability_provider,
                capability_context=capability_context,
            )

        adapter_result = _invoke_adapter_factory(
            load_factory(resolved_factory_path),
            spec=spec,
            memory_root=self.memory_root,
            repo_root=self.repo_root,
        )
        if isinstance(adapter_result, AdapterSetup):
            handlers = adapter_result.handlers
            capability_provider = adapter_result.capability_provider
        else:
            handlers = adapter_result
            capability_provider = None
        capability_context = (
            capability_provider(spec)
            if capability_provider is not None
            else CapabilityContext()
        )
        return RuntimeBinding(
            executor_factory_path=resolved_factory_path,
            handlers=handlers,
            capability_provider=capability_provider,
            capability_context=capability_context,
        )

    def build_orchestrator(
        self,
        *,
        spec: ExperimentSpec,
        objective: str,
        memory_store: FileMemoryStore | None = None,
        executor_factory_path: str | None = None,
    ) -> tuple[ExperimentOrchestrator, RuntimeBinding]:
        runtime = self.resolve_runtime_binding(
            spec=spec,
            objective=objective,
            executor_factory_path=executor_factory_path,
        )
        extra_kwargs_worker = self._extra(self.role_models.worker).get("extra_kwargs")
        extra_kwargs_review = self._extra(self.role_models.review).get("extra_kwargs")
        tool_use_agent = ToolUseExecutor(
            model=self.role_models.worker,
            request_model=self._runtime_model(self.role_models.worker),
            repo_root=self.repo_root,
            temperature=self.temperature,
            progress_fn=self.progress_fn,
            extra_kwargs=extra_kwargs_worker,
        )
        reviewer = ToolUseReviewer(
            model=self.role_models.review,
            request_model=self._runtime_model(self.role_models.review),
            repo_root=self.repo_root,
            temperature=self.temperature,
            progress_fn=self.progress_fn,
            extra_kwargs=extra_kwargs_review,
        )
        orchestrator = ExperimentOrchestrator(
            memory_store=memory_store or FileMemoryStore(self.memory_root),
            worker_backend=tool_use_agent,
            executor=RoutingExperimentExecutor(
                handlers=runtime.handlers,
                plan_executor=tool_use_agent,
                recovery_handler=self._make_execution_recovery_handler(),
            ),
            reviewer=reviewer,
            narrator_backend=self.narrator_backend,
            capability_provider=runtime.capability_provider,
            progress_fn=self.progress_fn,
            role_models=self.role_models,
        )
        return orchestrator, runtime

    def _load_cached_authored_runner(
        self, objective: str
    ) -> ExecutionBackendResolution | None:
        runner_path, manifest_path = _authored_runner_paths(
            self.memory_root, self.repo_root
        )
        if not runner_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if Path(manifest.get("repo_root", "")).resolve() != self.repo_root.resolve():
            return None
        if str(manifest.get("user_goal", "")).strip() != objective.strip():
            return None
        validation = _validate_runner_factory(
            factory_path=f"{runner_path}:build_adapter",
            objective=objective,
            memory_root=self.memory_root,
        )
        if not validation.success:
            try:
                runner_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        self.executor_factory_path = f"{runner_path}:build_adapter"
        return ExecutionBackendResolution(
            kind="supported", factory_path=self.executor_factory_path
        )

    def _author_runner(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        answers: dict[str, Any] | None = None,
    ) -> ExecutionBackendResolution:
        cached = self._load_cached_authored_runner(user_goal)
        if cached is not None:
            return cached
        runner_path, manifest_path = _authored_runner_paths(
            self.memory_root, self.repo_root
        )
        runner_path.parent.mkdir(parents=True, exist_ok=True)
        previous_errors: list[str] = []
        last_validation: RunnerValidationResult | None = None
        temp_paths: list[Path] = []
        try:
            for attempt_number in range(1, MAX_RUNNER_AUTHORING_ATTEMPTS + 1):
                temp_path = runner_path.with_name(
                    f"{runner_path.stem}_candidate_{attempt_number}.py"
                )
                temp_paths.append(temp_path)
                request = RunnerAuthoringRequest(
                    user_goal=user_goal,
                    repo_root=str(self.repo_root.resolve()),
                    capability_context=capability_context,
                    target_module_path=str(runner_path),
                    attempt_number=attempt_number,
                    previous_errors=list(previous_errors),
                    bootstrap_answers=dict(answers or {}),
                )
                try:
                    authored = self.runner_authoring_backend.author_runner(request)
                except Exception as exc:
                    return ExecutionBackendResolution(
                        kind="planning_only",
                        reason=f"Runner authoring failed before validation: {exc}",
                    )
                temp_path.write_text(
                    authored.module_source.rstrip() + "\n", encoding="utf-8"
                )
                validation = _validate_runner_factory(
                    factory_path=f"{temp_path}:build_adapter",
                    objective=user_goal,
                    memory_root=self.memory_root,
                )
                last_validation = validation
                if validation.success:
                    runner_path.write_text(
                        temp_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "repo_root": str(self.repo_root.resolve()),
                                "user_goal": user_goal,
                                "factory_path": f"{runner_path}:build_adapter",
                                "summary": authored.summary,
                                "notes": authored.notes,
                                "validation": validation.to_dict(),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    self.executor_factory_path = f"{runner_path}:build_adapter"
                    return ExecutionBackendResolution(
                        kind="supported", factory_path=self.executor_factory_path
                    )
                previous_errors = list(validation.errors)
            reason = "Loopforge tried to author a repo-specific runner, but validation failed."
            if last_validation is not None and last_validation.errors:
                reason += " " + "; ".join(last_validation.errors[:3])
            return ExecutionBackendResolution(kind="planning_only", reason=reason)
        finally:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _repair_incomplete_metrics(
        self,
        *,
        spec: ExperimentSpec,
        capability_context: CapabilityContext,
        assistant_message: str,
    ) -> ExperimentSpec:
        all_metrics = [
            spec.primary_metric,
            *spec.secondary_metrics,
            *spec.guardrail_metrics,
        ]
        if not any(_metric_is_incomplete(metric) for metric in all_metrics):
            return spec

        spec_dict = _apply_metric_catalog_defaults(spec.to_dict(), capability_context)
        try:
            patched_spec = ExperimentSpec.from_dict(spec_dict)
        except Exception as exc:
            self.progress_fn(
                "fix_metrics_warning_catalog",
                f"Metric catalog defaults could not be applied cleanly: {exc}",
            )
            patched_spec = spec

        all_metrics = [
            patched_spec.primary_metric,
            *patched_spec.secondary_metrics,
            *patched_spec.guardrail_metrics,
        ]
        if not any(_metric_is_incomplete(metric) for metric in all_metrics):
            return patched_spec

        self.progress_fn(
            "fix_metrics",
            "Inferring incomplete metric details from planning context...",
        )
        compact_context = _compact_capability_context_for_metric_repair(
            objective=patched_spec.objective,
            assistant_message=assistant_message,
            capability_context=capability_context,
        )
        try:
            try:
                fixes = self.narrator_backend.fix_incomplete_metrics(
                    current_spec=patched_spec.to_dict(),
                    assistant_message=assistant_message,
                    objective=patched_spec.objective,
                    capability_context=compact_context,
                )
            except TypeError:
                fixes = self.narrator_backend.fix_incomplete_metrics(
                    current_spec=patched_spec.to_dict(),
                    assistant_message=assistant_message,
                )
        except Exception as exc:
            self.progress_fn(
                "fix_metrics_warning_llm",
                f"Metric repair inference failed; keeping the current spec unchanged: {exc}",
            )
        return patched_spec

    def _apply_repo_stop_condition_defaults(self, spec: ExperimentSpec) -> ExperimentSpec:
        stop_conditions = dict(spec.stop_conditions)
        changed = False
        if (
            "max_iterations" not in stop_conditions
            and self.repo_loop_settings.max_iterations is not None
        ):
            stop_conditions["max_iterations"] = self.repo_loop_settings.max_iterations
            changed = True
        if (
            "max_autonomous_hours" not in stop_conditions
            and self.repo_loop_settings.max_autonomous_hours is not None
        ):
            stop_conditions["max_autonomous_hours"] = (
                self.repo_loop_settings.max_autonomous_hours
            )
            changed = True
        if not changed:
            return spec
        return replace(spec, stop_conditions=stop_conditions)

        if not fixes:
            return patched_spec

        spec_dict = patched_spec.to_dict()
        for key in ("primary_metric", "secondary_metrics", "guardrail_metrics"):
            if key in fixes:
                spec_dict[key] = fixes[key]
        spec_dict = _apply_metric_catalog_defaults(spec_dict, capability_context)
        try:
            return ExperimentSpec.from_dict(spec_dict)
        except Exception as exc:
            self.progress_fn(
                "fix_metrics_warning_patch",
                f"Metric repair returned an invalid spec patch; keeping the current spec unchanged: {exc}",
            )
            return patched_spec

    def _resolve_explicit_primary_metric_goal(
        self,
        *,
        spec: ExperimentSpec,
        capability_context: CapabilityContext,
        assistant_message: str,
    ) -> ExperimentSpec:
        primary_metric = spec.primary_metric
        if (
            not primary_metric.name
            or primary_metric.name == "[unspecified]"
            or primary_metric.goal in {"maximize", "minimize"}
        ):
            return spec

        focused_metrics = {}
        metric_catalog = capability_context.available_metrics or {}
        if primary_metric.name in metric_catalog:
            focused_metrics[primary_metric.name] = metric_catalog[primary_metric.name]
        compact_context = {
            "available_metrics": focused_metrics,
            "environment_facts": capability_context.environment_facts,
            "notes": capability_context.notes[:20],
            "available_actions": capability_context.available_actions,
            "available_data_assets": capability_context.available_data_assets[:20],
        }
        try:
            fixes = self.narrator_backend.fix_incomplete_metrics(
                current_spec=spec.to_dict(),
                assistant_message=assistant_message,
                objective=spec.objective,
                capability_context=compact_context,
            )
        except TypeError:
            try:
                fixes = self.narrator_backend.fix_incomplete_metrics(
                    current_spec=spec.to_dict(),
                    assistant_message=assistant_message,
                )
            except Exception as exc:
                self.progress_fn(
                    "fix_metrics_warning_primary_goal",
                    f"Primary metric goal inference failed; keeping the current goal unchanged: {exc}",
                )
                return spec
        except Exception as exc:
            self.progress_fn(
                "fix_metrics_warning_primary_goal",
                f"Primary metric goal inference failed; keeping the current goal unchanged: {exc}",
            )
            return spec

        if not isinstance(fixes, dict):
            return spec
        primary_payload = fixes.get("primary_metric")
        if not isinstance(primary_payload, dict):
            return spec
        if primary_payload.get("name") != primary_metric.name:
            primary_payload = {**primary_payload, "name": primary_metric.name}
        if primary_payload.get("goal") not in {"maximize", "minimize"}:
            return spec
        spec_dict = spec.to_dict()
        spec_dict["primary_metric"] = {
            **spec_dict.get("primary_metric", {}),
            **primary_payload,
        }
        try:
            return ExperimentSpec.from_dict(spec_dict)
        except Exception as exc:
            self.progress_fn(
                "fix_metrics_warning_primary_patch",
                f"Primary metric goal patch was invalid; keeping the current goal unchanged: {exc}",
            )
            return spec

    def bootstrap(
        self,
        *,
        user_goal: str,
        answers: dict[str, Any] | None = None,
    ) -> BootstrapTurn:
        # Resolve target repo from objective if it mentions a specific repo
        resolved_root = resolve_repo_root_from_objective(user_goal, self.repo_root)
        if resolved_root != self.repo_root:
            self.repo_root = resolved_root
            self._cached_capability_context = None  # Force re-scan of the correct repo
            self._cached_capability_key = None
            self._cached_authoring_failure_reason = None
            if self.explicit_executor_factory_path is None:
                self.executor_factory_path = None
        self.progress_fn("resolve_backend", "Detecting execution environment...")
        resolution = self.resolve_execution_backend(user_goal)
        executor_factory_path = resolution.factory_path
        capability_key = (
            str(self.repo_root.resolve()),
            executor_factory_path,
            user_goal.strip(),
        )
        if (
            self._cached_capability_context is None
            or self._cached_capability_key != capability_key
        ):
            self.progress_fn(
                "discover_capabilities", "Scanning repository capabilities..."
            )
            self._cached_capability_context = discover_capabilities_for_objective(
                objective=user_goal,
                memory_root=self.memory_root,
                executor_factory_path=executor_factory_path,
                repo_root=self.repo_root,
            )
            self._cached_capability_key = capability_key
        capability_context = self._cached_capability_context
        guidance_gap_detail = None
        if (
            resolution.kind == "planning_only"
            and self.explicit_executor_factory_path is None
            and _has_execution_guidance_answer(answers)
        ):
            self.progress_fn(
                "author_runner",
                "Authoring repo-specific runner from discovered context...",
            )
            authored_resolution = self._author_runner(
                user_goal,
                capability_context,
                answers=answers,
            )
            if authored_resolution.kind == "supported":
                resolution = authored_resolution
                executor_factory_path = authored_resolution.factory_path
                capability_context = discover_capabilities_for_objective(
                    objective=user_goal,
                    memory_root=self.memory_root,
                    executor_factory_path=executor_factory_path,
                    repo_root=self.repo_root,
                )
                capability_key = (
                    str(self.repo_root.resolve()),
                    executor_factory_path,
                    user_goal.strip(),
                )
                self._cached_capability_context = capability_context
                self._cached_capability_key = capability_key
            elif authored_resolution.reason:
                self._cached_authoring_failure_reason = authored_resolution.reason
                resolution = authored_resolution
        if resolution.kind == "planning_only":
            guidance_gap_detail = _detect_execution_guidance_gap(
                user_goal=user_goal,
                capability_context=capability_context,
                answers=answers,
            )
        if (
            resolution.kind == "planning_only"
            and resolution.reason
            and guidance_gap_detail is None
        ):
            merged_actions = dict(capability_context.available_actions)
            for action in GENERIC_AUTONOMOUS_ACTIONS:
                merged_actions.setdefault(action, "generic_autonomous_executor")
            capability_context = replace(
                capability_context,
                available_actions=merged_actions,
                environment_facts={
                    **capability_context.environment_facts,
                    "execution_resolution_reason": resolution.reason,
                    "autonomous_execution_supported": True,
                    "execution_guidance_required": False,
                    "execution_backend_kind": "generic_agentic",
                    "runner_kind": "generic_agentic_executor",
                    "generic_autonomous_actions": list(GENERIC_AUTONOMOUS_ACTIONS),
                },
                notes=[
                    resolution.reason,
                    "Generic autonomous execution is available after bootstrap. The worker can inspect the repo, edit files, run commands, and recover from ordinary failures without a repo-specific runner.",
                    *capability_context.notes,
                ],
            )
            self._cached_capability_context = capability_context
        elif resolution.kind == "planning_only" and guidance_gap_detail is not None:
            capability_context = replace(
                capability_context,
                environment_facts={
                    **capability_context.environment_facts,
                    "execution_resolution_reason": resolution.reason,
                    "execution_guidance_required": True,
                    "execution_guidance_detail": guidance_gap_detail,
                    "autonomous_execution_supported": False,
                },
                notes=[
                    guidance_gap_detail,
                    *capability_context.notes,
                ],
            )
            self._cached_capability_context = capability_context
        # Deterministic data tracing — find files the user referenced, try loading
        env = capability_context.environment_facts
        data_trace = _trace_data_source(
            user_goal=user_goal,
            repo_root=self.repo_root,
            python_executable=str(env.get("python_executable") or sys.executable),
            progress_fn=self.progress_fn,
        )

        # Use the tool-use planner to read the repo and fill out the contract
        self.progress_fn(
            "propose_plan",
            f"[{self.role_models.planner}] Analyzing repo and planning experiment...",
        )
        planner_extra = self._extra(self.role_models.planner).get("extra_kwargs")
        planner = ToolUsePlanner(
            model=self.role_models.planner,
            request_model=self._runtime_model(self.role_models.planner),
            repo_root=self.repo_root,
            temperature=self.temperature,
            progress_fn=self.progress_fn,
            extra_kwargs=planner_extra,
        )
        plan_result = planner.plan(
            user_goal=user_goal,
            source_file_hint=data_trace.get("data_source"),
            answers=answers,
        )
        contract = plan_result.get("contract", {})
        planner_questions = plan_result.get("questions", [])

        # Build ExperimentSpec from contract
        primary_metric_name = contract.get("primary_metric", "[unspecified]")
        primary_metric_goal = contract.get("primary_metric_goal", "unspecified")
        raw_guardrails = contract.get("guardrail_metrics") or []
        raw_secondaries = contract.get("secondary_metrics") or []

        spec = ExperimentSpec(
            objective=user_goal,
            primary_metric=MetricSpec(
                name=primary_metric_name, goal=primary_metric_goal
            ),
            allowed_actions=list(
                capability_context.environment_facts.get(
                    "generic_autonomous_actions", ["run_experiment"]
                )
            ),
            guardrail_metrics=[
                MetricSpec(
                    name=m["name"] if isinstance(m, dict) else str(m),
                    goal=m.get("goal", "unspecified")
                    if isinstance(m, dict)
                    else "unspecified",
                )
                for m in raw_guardrails
            ],
            secondary_metrics=[
                MetricSpec(
                    name=m["name"] if isinstance(m, dict) else str(m),
                    goal=m.get("goal", "unspecified")
                    if isinstance(m, dict)
                    else "unspecified",
                )
                for m in raw_secondaries
            ],
            metadata={
                "source_script": contract.get("source_script")
                or data_trace.get("data_source"),
                "baseline_function": contract.get("baseline_function"),
                "data_loading": contract.get("data_loading"),
                "target_column": contract.get("target_column"),
                "baseline_metric_value": contract.get("baseline_value"),
            },
        )
        spec = self._apply_repo_stop_condition_defaults(spec)

        # Convert planner questions to SpecQuestion objects
        spec_questions = [
            SpecQuestion(
                key=f"planner_q_{i}",
                prompt=q.get("question", ""),
                rationale=q.get("reason", ""),
                required=True,
            )
            for i, q in enumerate(planner_questions)
            if q.get("question")
        ]

        # Build a BootstrapTurn from the planner results
        from loopforge.core.types import ExperimentSpecProposal, BootstrapTurn

        turn = BootstrapTurn(
            assistant_message="I analyzed the repo and built an experiment plan.",
            proposal=ExperimentSpecProposal(
                objective=user_goal,
                recommended_spec=spec,
                questions=spec_questions,
                notes=[],
            ),
            role_models=self.role_models,
            ready_to_start=len(spec_questions) == 0
            and primary_metric_goal != "unspecified",
        )
        # Add execution metadata
        turn = replace(
            turn,
            proposal=replace(
                turn.proposal,
                recommended_spec=replace(
                    turn.proposal.recommended_spec,
                    metadata={
                        **turn.proposal.recommended_spec.metadata,
                        "execution_backend_kind": capability_context.environment_facts.get(
                            "execution_backend_kind"
                        ),
                    },
                ),
            ),
        )

        # Check contract fields — generate questions for anything the LLM left empty
        contract_questions: list[SpecQuestion] = []
        metadata = turn.proposal.recommended_spec.metadata
        if not metadata.get("source_script") and not data_trace.get("data_source"):
            contract_questions.append(
                SpecQuestion(
                    key="source_script",
                    prompt="Which file contains the model training code?",
                    rationale="Could not find the file you referenced in the repo.",
                    required=True,
                )
            )
        if not metadata.get("baseline_function"):
            contract_questions.append(
                SpecQuestion(
                    key="baseline_function",
                    prompt=(
                        f"I read {data_trace.get('data_source', 'the source script')} but couldn't identify "
                        "the baseline model function. Which function trains the model?"
                    ),
                    rationale="Need to know which function to use as baseline.",
                    required=True,
                )
            )
        if not metadata.get("data_loading"):
            contract_questions.append(
                SpecQuestion(
                    key="data_loading",
                    prompt="How is the training data loaded? (e.g. function name, file path, or describe the pipeline)",
                    rationale="Need to understand the data pipeline for the executor.",
                    required=True,
                )
            )
        if not metadata.get("target_column"):
            contract_questions.append(
                SpecQuestion(
                    key="target_column",
                    prompt="What column/variable is the model predicting?",
                    rationale="Need to know the prediction target.",
                    required=True,
                )
            )

        if contract_questions:
            turn = replace(
                turn,
                proposal=replace(
                    turn.proposal,
                    questions=[*turn.proposal.questions, *contract_questions],
                ),
                ready_to_start=False,
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
        ready_to_start = not missing_requirements

        # Verify the execution environment can actually run experiments
        env_verification: dict[str, Any] = {}
        if ready_to_start:
            env_verification = _verify_execution_environment(
                repo_root=self.repo_root,
                capability_context=capability_context,
                progress_fn=self.progress_fn,
                dependency_sync=data_trace.get("dependency_sync"),
            )

        bootstrap_warnings: list[str] = []

        def add_bootstrap_warning(stage: str, message: str) -> None:
            if message not in bootstrap_warnings:
                bootstrap_warnings.append(message)
            self.progress_fn(stage, message)

        access_guide_path = None
        if should_prepare_access_guide(
            capability_context=capability_context,
            preflight_checks=preflight_checks,
        ):
            try:
                self.progress_fn("access_guide", "Building access guide...")
                access_guide = self.access_advisor_backend.build_access_guide(
                    user_goal=user_goal,
                    capability_context=capability_context,
                    preflight_checks=preflight_checks,
                )
                agent_markdown_dir = self.memory_root / "agent_markdown"
                agent_markdown_dir.mkdir(parents=True, exist_ok=True)
                guide_path = agent_markdown_dir / "ops_access_guide.md"
                guide_path.write_text(
                    access_guide.markdown.rstrip() + "\n", encoding="utf-8"
                )
                access_guide_path = str(guide_path)
            except Exception as exc:
                add_bootstrap_warning(
                    "access_guide_warning",
                    f"Could not build the access guide: {exc}",
                )
        resolved_turn = replace(
            turn,
            preflight_checks=preflight_checks,
            ready_to_start=ready_to_start,
            missing_requirements=missing_requirements,
            access_guide_path=access_guide_path,
        )
        generic_autonomous = (
            capability_context.environment_facts.get("execution_backend_kind")
            == "generic_agentic"
        )
        try:
            self.progress_fn("execution_runbook", "Writing execution runbook...")
            runbook_text = build_execution_runbook(
                repo_root=self.repo_root,
                capability_context=capability_context,
                turn=resolved_turn,
                preflight_checks=preflight_checks,
            )
            agent_markdown_dir = self.memory_root / "agent_markdown"
            agent_markdown_dir.mkdir(parents=True, exist_ok=True)
            runbook_path = agent_markdown_dir / "execution_runbook.md"
            runbook_path.write_text(runbook_text, encoding="utf-8")
        except Exception as exc:
            add_bootstrap_warning(
                "execution_runbook_warning",
                f"Could not write execution runbook: {exc}",
            )
        try:
            self.progress_fn("bootstrap_handoff", "Writing bootstrap handoff...")
            handoff_text = build_bootstrap_handoff(
                capability_context=capability_context,
                turn=resolved_turn,
                answers=answers,
                env_verification=env_verification,
            )
            agent_markdown_dir = self.memory_root / "agent_markdown"
            agent_markdown_dir.mkdir(parents=True, exist_ok=True)
            handoff_path = agent_markdown_dir / "bootstrap_handoff.md"
            handoff_path.write_text(handoff_text, encoding="utf-8")
        except Exception as exc:
            add_bootstrap_warning(
                "bootstrap_handoff_warning",
                f"Could not write bootstrap handoff: {exc}",
            )
        if ready_to_start or not [
            r for r in missing_requirements if r.startswith("answer:")
        ]:
            try:
                self.progress_fn("experiment_guide", "Writing experiment guide...")
                guide_text = self.bootstrap_backend.build_experiment_guide(
                    resolved_turn,
                    capability_context,
                    answers,
                )
                agent_markdown_dir = self.memory_root / "agent_markdown"
                agent_markdown_dir.mkdir(parents=True, exist_ok=True)
                guide_path = agent_markdown_dir / "experiment_guide.md"
                guide_path.write_text(guide_text.rstrip() + "\n", encoding="utf-8")
            except Exception as exc:
                add_bootstrap_warning(
                    "experiment_guide_warning",
                    f"Could not write experiment guide: {exc}",
                )
        if generic_autonomous:
            human_update = resolved_turn.assistant_message
        else:
            self.progress_fn(
                "narrate", f"[{self.role_models.narrator}] Preparing summary..."
            )
            try:
                human_update = self.narrator_backend.summarize_bootstrap(
                    resolved_turn, capability_context
                )
            except Exception as exc:
                add_bootstrap_warning(
                    "bootstrap_summary_warning",
                    f"Could not prepare the bootstrap summary; falling back to the planner message: {exc}",
                )
                human_update = resolved_turn.assistant_message
        if bootstrap_warnings:
            resolved_turn = replace(
                resolved_turn,
                proposal=replace(
                    resolved_turn.proposal,
                    notes=[*resolved_turn.proposal.notes, *bootstrap_warnings],
                ),
            )
            warning_block = "Warnings:\n" + "\n".join(
                f"- {warning}" for warning in bootstrap_warnings
            )
            human_update = (
                f"{human_update}\n\n{warning_block}" if human_update else warning_block
            )
        if resolved_turn.access_guide_path:
            human_update = (
                f"{human_update}\n\nAccess guide: {resolved_turn.access_guide_path}"
                if human_update
                else f"Access guide: {resolved_turn.access_guide_path}"
            )
        return replace(resolved_turn, human_update=human_update)

    @staticmethod
    def _apply_metric_goal_fixes(
        turn: BootstrapTurn, fixes: dict[str, str]
    ) -> BootstrapTurn:
        """Apply goal fixes from the LLM to metrics in the spec."""
        spec = turn.proposal.recommended_spec
        primary = spec.primary_metric
        if primary.name in fixes:
            primary = replace(primary, goal=fixes[primary.name])
        secondary = [
            replace(m, goal=fixes[m.name]) if m.name in fixes else m
            for m in spec.secondary_metrics
        ]
        guardrails = [
            replace(m, goal=fixes[m.name]) if m.name in fixes else m
            for m in spec.guardrail_metrics
        ]
        patched_spec = replace(
            spec,
            primary_metric=primary,
            secondary_metrics=secondary,
            guardrail_metrics=guardrails,
        )
        return replace(
            turn, proposal=replace(turn.proposal, recommended_spec=patched_spec)
        )

    def apply_feedback(
        self, turn: BootstrapTurn, feedback: str
    ) -> BootstrapTurn | None:
        """Interpret user feedback and patch the existing plan. Returns None if full replan needed."""
        cap_ctx = self._cached_capability_context or CapabilityContext()
        try:
            result = self.narrator_backend.interpret_feedback(turn, feedback, cap_ctx)
        except Exception:
            result = None
        fallback_primary_metric = _extract_primary_metric_from_feedback(
            feedback, cap_ctx
        )
        if not result and not fallback_primary_metric:
            return None  # Fall through to full replan
        action = (result or {}).get("action", "replan")
        spec_updates = (result or {}).get("spec_updates", {})
        if action == "replan" and fallback_primary_metric:
            current_primary = turn.proposal.recommended_spec.primary_metric
            existing_metadata = dict(turn.proposal.recommended_spec.metadata)
            existing_bootstrap_answers = dict(
                existing_metadata.get("bootstrap_answers", {})
            )
            existing_operator_guidance = existing_metadata.get("operator_guidance", [])
            if not isinstance(existing_operator_guidance, list):
                existing_operator_guidance = [str(existing_operator_guidance)]
            merged_operator_guidance = [
                item for item in existing_operator_guidance if str(item).strip()
            ]
            if feedback not in merged_operator_guidance:
                merged_operator_guidance.append(feedback)
            spec_updates = {
                **spec_updates,
                "primary_metric": {
                    "name": fallback_primary_metric,
                    "goal": current_primary.goal,
                    **(
                        {"display_name": current_primary.display_name}
                        if current_primary.display_name
                        else {}
                    ),
                },
                "metadata": {
                    "operator_guidance": merged_operator_guidance,
                    "bootstrap_answers": existing_bootstrap_answers,
                },
            }
            action = "patch"
            result = {
                "action": "patch",
                "spec_updates": spec_updates,
                "message": f"Updated the primary metric to {fallback_primary_metric} and carried your guidance into execution.",
            }
        action = result.get("action", "replan")
        if action == "replan":
            return None
        spec_updates = result.get("spec_updates", {})
        if fallback_primary_metric:
            patched_primary = (
                spec_updates.get("primary_metric")
                if isinstance(spec_updates.get("primary_metric"), dict)
                else None
            )
            patched_primary_name = (
                patched_primary.get("name") if patched_primary is not None else None
            )
            metric_catalog = cap_ctx.available_metrics or {}
            if _is_generic_metric_placeholder(patched_primary_name) or (
                isinstance(patched_primary_name, str)
                and patched_primary_name not in metric_catalog
                and fallback_primary_metric in metric_catalog
            ):
                current_primary = turn.proposal.recommended_spec.primary_metric
                spec_updates = {
                    **spec_updates,
                    "primary_metric": {
                        **(patched_primary or {}),
                        "name": fallback_primary_metric,
                        "goal": (
                            patched_primary.get("goal")
                            if patched_primary
                            and patched_primary.get("goal") in {"minimize", "maximize"}
                            else current_primary.goal
                        ),
                    },
                }
                result = {
                    **result,
                    "spec_updates": spec_updates,
                    "message": (
                        f"Updated the primary metric to {fallback_primary_metric} and carried your guidance into execution."
                    ),
                }
        if not spec_updates:
            return None
        # Apply the patch to the existing spec
        spec = turn.proposal.recommended_spec
        spec_dict = spec.to_dict()
        for key, value in spec_updates.items():
            if isinstance(value, dict) and isinstance(spec_dict.get(key), dict):
                spec_dict[key] = {**spec_dict[key], **value}
            else:
                spec_dict[key] = value
        try:
            patched_spec = ExperimentSpec.from_dict(spec_dict)
        except Exception:
            return None  # Malformed patch — fall through to replan
        if fallback_primary_metric:
            patched_spec = self._resolve_explicit_primary_metric_goal(
                spec=patched_spec,
                capability_context=cap_ctx,
                assistant_message=result.get("message", feedback),
            )
        patched_spec = self._repair_incomplete_metrics(
            spec=patched_spec,
            capability_context=cap_ctx,
            assistant_message=result.get("message", turn.assistant_message),
        )
        patched_proposal = replace(turn.proposal, recommended_spec=patched_spec)
        # Re-run preflight checks on the patched spec
        executor_factory_path = self.resolve_executor_factory_path(
            patched_spec.objective
        )
        # Clear stale execution_guidance_required if the hint was already answered
        bootstrap_answers = patched_spec.metadata.get("bootstrap_answers", {})
        if not isinstance(bootstrap_answers, dict):
            bootstrap_answers = {}
        if bootstrap_answers.get(EXECUTION_GUIDANCE_QUESTION_KEY):
            cap_ctx = replace(
                cap_ctx,
                environment_facts={
                    **cap_ctx.environment_facts,
                    "execution_guidance_required": False,
                },
            )
        preflight_checks = run_preflight_checks(
            spec=patched_spec,
            capability_context=cap_ctx,
            memory_root=self.memory_root,
            executor_factory_path=executor_factory_path,
        )
        missing_requirements = missing_requirements_from_bootstrap(
            questions=patched_proposal.questions,
            answers=bootstrap_answers,
            preflight_checks=preflight_checks,
        )
        ready_to_start = not missing_requirements
        message = result.get("message", "Plan updated.")
        return replace(
            turn,
            proposal=patched_proposal,
            preflight_checks=preflight_checks,
            missing_requirements=missing_requirements,
            ready_to_start=ready_to_start,
            human_update=message,
        )

    def start(
        self,
        *,
        user_goal: str,
        answers: dict[str, Any] | None = None,
        iterations: int | None = None,
        max_autonomous_hours: float | None = None,
    ) -> dict[str, Any]:
        bootstrap_turn = self.bootstrap(user_goal=user_goal, answers=answers)
        return self.start_from_bootstrap_turn(
            bootstrap_turn=bootstrap_turn,
            user_goal=user_goal,
            iterations=iterations,
            max_autonomous_hours=max_autonomous_hours,
            reset_state=True,
        )

    def _install_python_dependency(
        self, package_name: str
    ) -> tuple[bool, dict[str, Any]]:
        attempts = [
            ["uv", "pip", "install", package_name],
            [sys.executable, "-m", "pip", "install", package_name],
        ]
        last_result: dict[str, Any] = {}
        for command in attempts:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:
                last_result = {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": str(exc),
                }
                continue
            last_result = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            if completed.returncode == 0:
                return True, last_result
        return False, last_result

    def _make_execution_recovery_handler(self):
        def recover(handler, candidate, snapshot, exc):
            dependency = _missing_python_dependency(exc)
            if dependency is None:
                return None
            module_name, install_name = dependency
            if install_name in self._attempted_dependency_installs:
                return None
            self._attempted_dependency_installs.add(install_name)
            self.progress_fn(
                "dependency_recovery",
                f"Installing missing dependency {install_name}...",
            )
            installed, install_result = self._install_python_dependency(install_name)
            if not installed:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[
                        f"Execution failed because Python dependency {module_name!r} is missing.",
                        f"Automatic install attempt failed for {install_name}.",
                    ],
                    next_ideas=[
                        "Inspect the environment, install the dependency manually if needed, then retry."
                    ],
                    failure_type=exc.__class__.__name__,
                    failure_summary=str(exc).strip() or exc.__class__.__name__,
                    recoverable=True,
                    recovery_actions=[
                        f"Install dependency {install_name} and retry the same step."
                    ],
                    execution_details={"install_attempt": install_result},
                )
            try:
                return handler.execute(candidate, snapshot)
            except Exception as retry_exc:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[
                        f"Installed dependency {install_name} automatically, but the retried execution still failed.",
                        f"Retry failure: {retry_exc}",
                    ],
                    next_ideas=[
                        "Inspect the new failure and continue from it in the next iteration."
                    ],
                    failure_type=retry_exc.__class__.__name__,
                    failure_summary=str(retry_exc).strip()
                    or retry_exc.__class__.__name__,
                    recoverable=True,
                    recovery_actions=[
                        "Use the updated environment and new failure details to choose the next fix."
                    ],
                    execution_details={"install_attempt": install_result},
                )

        return recover

    def start_from_bootstrap_turn(
        self,
        *,
        bootstrap_turn: BootstrapTurn,
        user_goal: str,
        iterations: int | None = None,
        max_autonomous_hours: float | None = None,
        iteration_callback=None,
        reset_state: bool = False,
    ) -> dict[str, Any]:
        if not bootstrap_turn.ready_to_start:
            return {"status": "needs_input", "bootstrap": bootstrap_turn.to_dict()}

        spec = self._apply_repo_stop_condition_defaults(
            bootstrap_turn.proposal.recommended_spec
        )
        if spec != bootstrap_turn.proposal.recommended_spec:
            bootstrap_turn = replace(
                bootstrap_turn,
                proposal=replace(bootstrap_turn.proposal, recommended_spec=spec),
            )
        memory_store = FileMemoryStore(self.memory_root)
        if not reset_state and memory_store.has_persisted_state():
            try:
                stored_spec = memory_store.load_spec()
            except Exception as exc:
                return {
                    "status": "blocked",
                    "bootstrap": bootstrap_turn.to_dict(),
                    "error": {
                        "type": "PersistedStateLoadFailed",
                        "message": (
                            "Existing loop state could not be read. Refusing to reset "
                            "persisted memory automatically."
                        ),
                        "detail": str(exc),
                        "cause_type": exc.__class__.__name__,
                    },
                }
            else:
                if stored_spec.to_dict() != spec.to_dict():
                    self.progress_fn(
                        "state_reset_warning",
                        "Stored experiment spec differs from the current bootstrap plan; resetting persisted loop state for a fresh run.",
                    )
                    reset_state = True
        orchestrator, _runtime = self.build_orchestrator(
            spec=spec,
            objective=user_goal,
            memory_store=memory_store,
        )
        orchestrator.initialize(spec=spec, reset_state=reset_state)
        try:
            results = orchestrator.run(
                iterations=iterations,
                max_autonomous_hours=max_autonomous_hours,
                iteration_callback=iteration_callback,
            )
        except (ExperimentInterrupted, KeyboardInterrupt):
            raise
        except Exception as exc:
            return {
                "status": "blocked",
                "bootstrap": bootstrap_turn.to_dict(),
                "error": {
                    "type": exc.__class__.__name__,
                    "message": summarise_runtime_exception(exc),
                    "detail": str(exc),
                },
            }
        if results and results[-1].record.outcome.status == "blocked":
            blocked_outcome = results[-1].record.outcome
            return {
                "status": "blocked",
                "bootstrap": bootstrap_turn.to_dict(),
                "results": cycle_results_to_payload(results),
                "error": {
                    "type": blocked_outcome.failure_type or "ExecutionBlocked",
                    "message": summarise_runtime_exception(
                        RuntimeError(
                            blocked_outcome.failure_summary
                            or "Execution hit a non-recoverable blocker."
                        )
                    ),
                    "detail": blocked_outcome.failure_summary or "",
                },
            }
        return {
            "status": "started",
            "bootstrap": bootstrap_turn.to_dict(),
            "results": cycle_results_to_payload(results),
        }
