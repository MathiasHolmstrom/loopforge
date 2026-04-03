from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from loopforge.auto_adapter import build_repo_scan_context
from loopforge.core import (
    AdapterSetup,
    BootstrapTurn,
    CapabilityContext,
    ConsultingWorkerBackend,
    DataAssetSchema,
    ExperimentCandidate,
    ExperimentInterrupted,
    ExperimentOrchestrator,
    ExperimentOutcome,
    ExperimentSpec,
    FileMemoryStore,
    LiteLLMAccessAdvisorBackend,
    LiteLLMBootstrapBackend,
    LiteLLMConsultationBackend,
    LiteLLMNarrationBackend,
    LiteLLMRunnerAuthoringBackend,
    LiteLLMReflectionBackend,
    LiteLLMExecutionFixBackend,
    LiteLLMReviewBackend,
    LiteLLMWorkerBackend,
    PreflightCheck,
    ProgressFn,
    StreamFn,
    PrimaryMetric,
    RoleModelConfig,
    RoutingExperimentExecutor,
    MemorySnapshot,
    MarkdownMemoryNote,
    RunnerAuthoringBackend,
    RunnerAuthoringRequest,
    RunnerAuthoringResult,
    RunnerValidationResult,
    SpecQuestion,
    _noop_progress,
)
from loopforge.core.agentic_execution import GenericExecutionPlanExecutor


DEFAULT_OPENAI_MODEL = "openai/gpt-5.4"
DEFAULT_CLAUDE_MODEL = "anthropic/claude-opus-4-6-v1"
DEFAULT_MODEL_PROFILE = "codex_with_claude_support"
ExecutionBackendKind = Literal["supported", "planning_only"]
GENERIC_AUTONOMOUS_ACTIONS = [
    "inspect_repo",
    "inspect_data",
    "edit_code",
    "run_experiment",
    "fix_failure",
]
EXECUTION_GUIDANCE_QUESTION_KEY = "execution_strategy_hint"
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
    "table",
    "bucket",
)

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
        "planner": DEFAULT_CLAUDE_MODEL,
        "worker": DEFAULT_OPENAI_MODEL,
        "reflection": DEFAULT_OPENAI_MODEL,
        "review": DEFAULT_OPENAI_MODEL,
        "consultation": DEFAULT_CLAUDE_MODEL,
        "narrator": DEFAULT_CLAUDE_MODEL,
    },
}


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


def _resolve_helper_model(preferred_model: str, fallback_model: str) -> str:
    if preferred_model.startswith("anthropic/") and not _can_use_anthropic_helpers():
        return fallback_model
    # Route through Bedrock proxy
    if preferred_model.startswith("anthropic/") and os.getenv("ANTHROPIC_BEDROCK_BASE_URL"):
        model_name = preferred_model.removeprefix("anthropic/")
        return f"bedrock/us.anthropic.{model_name}"
    return preferred_model


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
    return runners_dir / f"{stem}_runner.py", runners_dir / f"{stem}_runner_manifest.json"


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
        adapter_setup = load_adapter_setup(factory_path=factory_path, spec=spec, memory_root=memory_root)
    except Exception as exc:
        return RunnerValidationResult(success=False, factory_path=factory_path, errors=[f"Runner import failed: {exc}"])
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
            errors.append("Runner did not expose a discovery_provider or capability_provider.")
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
            checks = _coerce_preflight_checks(adapter_setup.preflight_provider(spec, capability_context))
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


def _smoke_validate_runner(
    *,
    adapter_setup: AdapterSetup,
    capability_context: CapabilityContext,
    objective: str,
) -> str | None:
    if not adapter_setup.handlers:
        return "Runner smoke test failed: no handlers were exposed."
    handler_name = "baseline" if "baseline" in adapter_setup.handlers else next(iter(adapter_setup.handlers))
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
    fallback = defaults["worker"]
    resolved_planner = _resolve_helper_model(planner_model or defaults["planner"], fallback)
    resolved_worker = _resolve_helper_model(worker_model or defaults["worker"], fallback)
    resolved_reflection = _resolve_helper_model(reflection_model or defaults["reflection"] or resolved_worker, resolved_worker)
    resolved_review = _resolve_helper_model(review_model or defaults["review"] or resolved_reflection, resolved_reflection)
    resolved_consultation = _resolve_helper_model(consultation_model or defaults["consultation"], resolved_worker)
    resolved_narrator = _resolve_helper_model(narrator_model or defaults["narrator"] or resolved_reflection, resolved_reflection)
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
    if "://" in lowered or lowered.startswith(("s3:", "dbfs:", "jdbc:", "snowflake:", "databricks:")):
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


def _probe_data_asset(asset_path: str, repo_root: Path) -> DataAssetSchema:
    """Quickly inspect a single data asset. Returns schema info or a load error."""
    import threading

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
            return DataAssetSchema(asset_path=asset_path, load_error=f"File not found: {resolved}")

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
            return DataAssetSchema(asset_path=asset_path, load_error="pandas not available for data probe")

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
                return DataAssetSchema(asset_path=asset_path, load_error=f"Unsupported format: {suffix}")
        except Exception as exc:
            return DataAssetSchema(asset_path=asset_path, load_error=f"Load failed: {exc}")

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

    worker = threading.Thread(target=_run_probe, daemon=True)
    _ACTIVE_DATA_PROBES[probe_key] = worker
    worker.start()
    worker.join(timeout=DATA_PROBE_TIMEOUT_SECONDS)
    if worker.is_alive():
        return DataAssetSchema(
            asset_path=asset_path,
            load_error=f"Probe timed out after {DATA_PROBE_TIMEOUT_SECONDS}s",
        )
    _ACTIVE_DATA_PROBES.pop(probe_key, None)
    if "error" in result_holder:
        return DataAssetSchema(asset_path=asset_path, load_error=f"Probe error: {result_holder['error']}")
    return result_holder.get("result") or DataAssetSchema(
        asset_path=asset_path,
        load_error="Probe failed without returning a schema result.",
    )


def probe_data_assets(capability_context: CapabilityContext, repo_root: Path) -> CapabilityContext:
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
                + (f" ... and {len(schema.columns) - 15} more" if len(schema.columns) > 15 else "")
            )
    return replace(capability_context, data_schemas=schemas, notes=notes)


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
        "python_executable": sys.executable,
        "repo_root": str(Path(repo_root).resolve()),
    }
    if executor_factory_path is None:
        context = build_repo_scan_context(repo_root)
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
    )
    if adapter_setup is None:
        return CapabilityContext()
    if adapter_setup.discovery_provider is not None:
        context = adapter_setup.discovery_provider(objective)
    elif adapter_setup.capability_provider is not None:
        context = adapter_setup.capability_provider(build_bootstrap_spec(objective))
    else:
        context = CapabilityContext()
    discovered_assets = [asset for asset in scan_result.get("data_assets", []) if asset not in context.available_data_assets]
    if discovered_assets:
        context = replace(context, available_data_assets=[*context.available_data_assets, *discovered_assets])
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
        for metric in [spec.primary_metric, *spec.secondary_metrics, *spec.guardrail_metrics]
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

    guidance_required = capability_context.environment_facts.get("execution_guidance_required")
    if guidance_required:
        checks.append(
            PreflightCheck(
                name="execution_strategy_unresolved",
                status="failed",
                detail=str(
                    capability_context.environment_facts.get("execution_guidance_detail")
                    or "Loopforge still needs one high-level hint about how to reach the real data or execution backend."
                ),
                scope="bootstrap",
            )
        )
        return checks

    if executor_factory_path is None or capability_context.environment_facts.get("autonomous_execution_supported") is False:
        if executor_factory_path is None and capability_context.environment_facts.get("autonomous_execution_supported") is not False:
            repo_root_raw = capability_context.environment_facts.get("repo_root")
            repo_root_path = Path(repo_root_raw) if isinstance(repo_root_raw, str) and repo_root_raw.strip() else Path.cwd()
            python_executable = capability_context.environment_facts.get("python_executable", sys.executable)
            try:
                probe = subprocess.run(
                    [str(python_executable), "-c", "import os,sys;print('loopforge_execution_probe_ok');print(os.getcwd());print(sys.executable)"],
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
            if probe.returncode != 0 or "loopforge_execution_probe_ok" not in probe.stdout:
                failure_summary = (probe.stderr or probe.stdout or f"Exit code {probe.returncode}.").strip()
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
    )
    if adapter_setup is not None and adapter_setup.preflight_provider is not None:
        checks.extend(_coerce_preflight_checks(adapter_setup.preflight_provider(spec, capability_context)))
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


def _non_local_data_assets(capability_context: CapabilityContext) -> list[str]:
    return [
        asset
        for asset in capability_context.available_data_assets
        if isinstance(asset, str) and asset.strip() and not _looks_like_local_asset_path(asset)
    ]


def _summarise_bootstrap_answers(answers: dict[str, Any] | None) -> str:
    if not isinstance(answers, dict):
        return ""
    parts: list[str] = []
    for key, value in answers.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value if str(item).strip())
        else:
            rendered = str(value).strip()
        if rendered:
            parts.append(f"{key}={rendered}")
    return "\n".join(parts)


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
    if answer_text and any(token in answer_text for token in REMOTE_EXECUTION_HINT_TOKENS):
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

    evidence: list[str] = []
    if remote_assets:
        preview = ", ".join(remote_assets[:2])
        if len(remote_assets) > 2:
            preview += f", and {len(remote_assets) - 2} more"
        evidence.append(f"non-local data assets were detected ({preview})")
    if matched_tokens:
        token_preview = ", ".join(matched_tokens[:4])
        evidence.append(f"the repo/objective points at an external backend ({token_preview})")
    joined_evidence = " and ".join(evidence)
    return (
        "Loopforge inferred most of the repo structure, but it cannot yet verify the real execution lane because "
        f"{joined_evidence}. Provide one high-level hint about whether the baseline should run through local Python "
        "code or through the remote platform the repo points at. A short directional answer is enough; exact paths "
        "or internal module names are not required."
    )


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
            haystack = " ".join(capability_context.notes).lower() + "\n" + json.dumps(
                capability_context.environment_facts,
                sort_keys=True,
            ).lower()
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
    if isinstance(answers, dict) and any(value not in (None, "") for value in answers.values()):
        if any(question.required for question in questions):
            return questions
    if any(question.key == EXECUTION_GUIDANCE_QUESTION_KEY for question in questions):
        return questions
    if any(question.required for question in questions):
        return questions
    return [*questions, _build_execution_guidance_question(capability_context=capability_context, detail=detail)]


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
    answer_map = {str(key): value for key, value in (answers or {}).items() if value not in (None, "")}
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
    updated_proposal = replace(turn.proposal, recommended_spec=updated_spec, notes=answer_notes)
    missing_requirements = missing_requirements_from_bootstrap(
        questions=updated_proposal.questions,
        answers=answer_map,
        preflight_checks=turn.preflight_checks,
    )
    required_question_keys = {
        question.key
        for question in updated_proposal.questions
        if question.required
    }
    resolved_required_questions = bool(required_question_keys) and required_question_keys.issubset(answer_map.keys())
    had_answer_blocker = any(req.startswith("answer:") for req in turn.missing_requirements)
    ready_to_start = (turn.ready_to_start or had_answer_blocker or resolved_required_questions) and not missing_requirements
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
    return [question for question in questions if not _is_internal_bootstrap_question(question)]



def refine_bootstrap_questions(
    questions: list[SpecQuestion],
    *,
    answers: dict[str, Any] | None,
) -> list[SpecQuestion]:
    """Remove internal implementation questions but keep all domain questions the agent proposed."""
    return sanitise_bootstrap_questions(questions)


def should_prepare_access_guide(
    *,
    capability_context: CapabilityContext,
    preflight_checks: list[PreflightCheck],
) -> bool:
    import re

    if capability_context.available_data_assets:
        return True
    if any(check.status != "passed" for check in preflight_checks):
        return True
    joined_notes = " ".join(capability_context.notes).lower()
    joined_env = str(capability_context.environment_facts).lower()
    access_patterns = (
        r"\bpermission(s)?\b",
        r"\bauth\b",
        r"\bauthentication\b",
        r"\bcredential(s)?\b",
        r"\btoken(s)?\b",
        r"\benv\b",
        r"\benvironment variable(s)?\b",
        r"\bsecret(s)?\b",
        r"\bdatabricks\b",
        r"\bwarehouse\b",
    )
    haystack = f"{joined_notes}\n{joined_env}"
    return any(re.search(pattern, haystack) for pattern in access_patterns)


def build_execution_runbook(
    *,
    repo_root: Path,
    capability_context: CapabilityContext,
    turn: BootstrapTurn,
    preflight_checks: list[PreflightCheck],
) -> str:
    env = capability_context.environment_facts
    shell_name = str(env.get("execution_shell", "unknown"))
    python_executable = str(env.get("python_executable", sys.executable))
    runner_kind = str(env.get("runner_kind", "unknown"))
    probe_check = next((check for check in preflight_checks if check.name == "generic_agentic_execution_probe"), None)
    lines = [
        "# Execution Runbook",
        "",
        "## Environment",
        f"- Repo root: {repo_root.resolve()}",
        f"- Execution shell for shell steps: {shell_name}",
        f"- Python executable: {python_executable}",
        f"- Runner kind: {runner_kind}",
        "",
    ]
    if probe_check is not None:
        lines.extend(
            [
                "## Verified Execution Lane",
                f"- Bootstrap verification: {probe_check.status}",
                f"- Detail: {probe_check.detail}",
                f"- Preferred command shape: \"{python_executable}\" path\\to\\script.py",
                "",
            ]
        )
    lines.extend(
        [
        "## Ground Rules",
        "- Treat this file as the execution handoff from bootstrap. Prefer these instructions over rediscovering obvious repo mechanics.",
        "- Bootstrap has already verified the execution lane above. Reuse that verified repo root and Python executable instead of inventing new activation commands, shell setup, or extra cd chains.",
        "- Aim for one end-to-end iteration that writes/runs code and prints metrics, not a long chain of disconnected inspection-only retries.",
        "- Fix one blocker at a time. Do not redesign the whole plan when a single command or import fails.",
        "- If you want to run a repo-local script that does not exist yet, create it first with a write step, then run it.",
        ]
    )
    if env.get("shell_family") == "windows_cmd":
        lines.extend(
            [
                "- Shell commands run through cmd.exe on Windows. Do not use Unix tools like `head`, `grep`, `find -maxdepth`, or `ls -la`.",
                "- Prefer `write_file` + `\"<python_executable>\" script.py`, or short cmd-compatible commands. Python-native inspection is usually the safest choice.",
            ]
        )
    else:
        lines.append("- Shell commands should stay portable and repo-local. Prefer Python-native inspection when shell portability is uncertain.")

    lines.extend(
        [
            "",
            "## Immediate Objective",
            f"- Goal: {turn.proposal.recommended_spec.objective}",
            f"- Primary metric: {turn.proposal.recommended_spec.primary_metric.name} ({turn.proposal.recommended_spec.primary_metric.goal})",
        ]
    )
    if capability_context.available_data_assets:
        lines.append("")
        lines.append("## Known Data Assets")
        for asset in capability_context.available_data_assets[:10]:
            lines.append(f"- {asset}")
    if turn.proposal.notes:
        lines.append("")
        lines.append("## Planner Notes")
        for note in turn.proposal.notes[:12]:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## First Iteration Guidance",
            "- Reuse the experiment guide and discovered repo paths to build the smallest runnable script that loads data, computes the configured metrics, and prints them clearly.",
            "- If repo APIs are still uncertain, do one bounded inspection step first, then immediately write and run the experiment script in the same iteration whenever possible.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"



def resolve_repo_root_from_objective(objective: str, current_root: Path) -> Path:
    """If the objective mentions a repo name, try to find it as a sibling or child directory."""
    import re
    resolved = current_root.resolve()
    # Extract potential repo names — hyphenated or underscored multi-word names
    candidates = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{2,}(?:-[a-zA-Z0-9_]+)+', objective)
    # Also try single words that look like repo names (before "repo" keyword)
    repo_match = re.search(r'(?:inside|in|from|of)\s+(\S+?)(?:\s+repo|\s+repository)?(?:\s|$)', objective, re.IGNORECASE)
    if repo_match:
        candidates.insert(0, repo_match.group(1).strip().rstrip('.'))
    for name in candidates:
        # Check sibling directory
        sibling = resolved.parent / name
        if sibling.is_dir() and sibling.resolve() != resolved:
            return sibling
        # Check child directory
        child = resolved / name
        if child.is_dir():
            return child
    return current_root


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
        runner_authoring_backend: RunnerAuthoringBackend | None = None,
        consultation_backend: Any | None = None,
        access_advisor_backend: Any | None = None,
        reflection_backend: Any | None = None,
        review_backend: Any | None = None,
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
        bedrock_base = os.getenv("ANTHROPIC_BEDROCK_BASE_URL")
        if bedrock_base:
            self._bedrock_kwargs: dict[str, Any] = {
                "api_base": bedrock_base,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "drop_params": True,
            }
        else:
            self._bedrock_kwargs: dict[str, Any] = {}
        self.role_models = default_role_models(
            planner_model=planner_model,
            worker_model=worker_model,
            reflection_model=reflection_model,
            review_model=review_model,
            consultation_model=consultation_model,
            narrator_model=narrator_model,
            profile=model_profile,
        )
        def _extra(model: str) -> dict[str, Any]:
            if model.startswith("bedrock/") and self._bedrock_kwargs:
                return {"extra_kwargs": self._bedrock_kwargs}
            return {}

        self.bootstrap_backend = bootstrap_backend or LiteLLMBootstrapBackend(
            model=self.role_models.planner, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.planner),
        )
        primary_worker_backend = worker_backend or LiteLLMWorkerBackend(
            model=self.role_models.worker, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.worker),
        )
        self.runner_authoring_backend = runner_authoring_backend or LiteLLMRunnerAuthoringBackend(
            model=self.role_models.worker, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.worker),
        )
        self.consultation_backend = consultation_backend or LiteLLMConsultationBackend(
            model=self.role_models.consultation, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.consultation),
        )
        self.access_advisor_backend = access_advisor_backend or LiteLLMAccessAdvisorBackend(
            model=self.role_models.consultation, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.consultation),
        )
        self.worker_backend = ConsultingWorkerBackend(
            worker_backend=primary_worker_backend,
            consultation_backend=self.consultation_backend,
            progress_fn=self.progress_fn,
            helper_label=self.role_models.consultation,
        )
        self.reflection_backend = reflection_backend or LiteLLMReflectionBackend(
            model=self.role_models.reflection, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.reflection),
        )
        self.review_backend = review_backend or LiteLLMReviewBackend(
            model=self.role_models.review, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.review),
        )
        self.narrator_backend = narrator_backend or LiteLLMNarrationBackend(
            model=self.role_models.narrator, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.narrator),
        )
        self.execution_fix_backend = LiteLLMExecutionFixBackend(
            model=self.role_models.consultation, temperature=temperature, stream_fn=stream_fn, **_extra(self.role_models.consultation),
        )

    def resolve_execution_backend(self, objective: str) -> ExecutionBackendResolution:
        if self.explicit_executor_factory_path:
            self.executor_factory_path = self.explicit_executor_factory_path
            return ExecutionBackendResolution(kind="supported", factory_path=self.explicit_executor_factory_path)
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

    def _fallback_capability_provider(self, _effective_spec: ExperimentSpec) -> CapabilityContext:
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

        adapter_result = load_factory(resolved_factory_path)(spec=spec, memory_root=self.memory_root)
        if isinstance(adapter_result, AdapterSetup):
            handlers = adapter_result.handlers
            capability_provider = adapter_result.capability_provider
        else:
            handlers = adapter_result
            capability_provider = None
        capability_context = capability_provider(spec) if capability_provider is not None else CapabilityContext()
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
        orchestrator = ExperimentOrchestrator(
            memory_store=memory_store or FileMemoryStore(self.memory_root),
            worker_backend=self.worker_backend,
            executor=RoutingExperimentExecutor(
                handlers=runtime.handlers,
                plan_executor=GenericExecutionPlanExecutor(
                    repo_root=self.repo_root,
                    fix_backend=self.execution_fix_backend,
                    progress_fn=self.progress_fn,
                ),
                recovery_handler=self._make_execution_recovery_handler(),
            ),
            reflection_backend=self.reflection_backend,
            review_backend=self.review_backend,
            narrator_backend=self.narrator_backend,
            capability_provider=runtime.capability_provider,
            progress_fn=self.progress_fn,
            role_models=self.role_models,
        )
        return orchestrator, runtime

    def _load_cached_authored_runner(self, objective: str) -> ExecutionBackendResolution | None:
        runner_path, manifest_path = _authored_runner_paths(self.memory_root, self.repo_root)
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
        return ExecutionBackendResolution(kind="supported", factory_path=self.executor_factory_path)

    def _author_runner(
        self,
        user_goal: str,
        capability_context: CapabilityContext,
        answers: dict[str, Any] | None = None,
    ) -> ExecutionBackendResolution:
        cached = self._load_cached_authored_runner(user_goal)
        if cached is not None:
            return cached
        runner_path, manifest_path = _authored_runner_paths(self.memory_root, self.repo_root)
        runner_path.parent.mkdir(parents=True, exist_ok=True)
        previous_errors: list[str] = []
        last_validation: RunnerValidationResult | None = None
        temp_paths: list[Path] = []
        try:
            for attempt_number in range(1, MAX_RUNNER_AUTHORING_ATTEMPTS + 1):
                temp_path = runner_path.with_name(f"{runner_path.stem}_candidate_{attempt_number}.py")
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
                temp_path.write_text(authored.module_source.rstrip() + "\n", encoding="utf-8")
                validation = _validate_runner_factory(
                    factory_path=f"{temp_path}:build_adapter",
                    objective=user_goal,
                    memory_root=self.memory_root,
                )
                last_validation = validation
                if validation.success:
                    runner_path.write_text(temp_path.read_text(encoding="utf-8"), encoding="utf-8")
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
                    return ExecutionBackendResolution(kind="supported", factory_path=self.executor_factory_path)
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
        # Don't load prior session memory during bootstrap — it pollutes new objectives
        # with irrelevant context from previous runs. Memory is only used during iterations.
        bootstrap_memory = {}
        capability_key = (str(self.repo_root.resolve()), executor_factory_path, user_goal.strip())
        if self._cached_capability_context is None or self._cached_capability_key != capability_key:
            self.progress_fn("discover_capabilities", "Scanning repository capabilities...")
            self._cached_capability_context = discover_capabilities_for_objective(
                objective=user_goal,
                memory_root=self.memory_root,
                executor_factory_path=executor_factory_path,
                repo_root=self.repo_root,
            )
            self._cached_capability_key = capability_key
        capability_context = self._cached_capability_context
        if resolution.kind == "planning_only" and self.explicit_executor_factory_path is None:
            self.progress_fn("author_runner", "Authoring repo-specific runner from discovered context...")
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
                capability_key = (str(self.repo_root.resolve()), executor_factory_path, user_goal.strip())
                self._cached_capability_context = capability_context
                self._cached_capability_key = capability_key
            elif authored_resolution.reason:
                self._cached_authoring_failure_reason = authored_resolution.reason
                resolution = authored_resolution
        guidance_gap_detail = None
        if resolution.kind == "planning_only":
            guidance_gap_detail = _detect_execution_guidance_gap(
                user_goal=user_goal,
                capability_context=capability_context,
                answers=answers,
            )
        if resolution.kind == "planning_only" and resolution.reason and guidance_gap_detail is None:
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
        self.progress_fn("propose_plan", f"[{self.role_models.planner}] Drafting experiment plan...")
        try:
            turn = self.bootstrap_backend.propose_bootstrap_turn(
                user_goal=user_goal,
                capability_context=capability_context,
                answer_history=answers,
                role_models=self.role_models,
                bootstrap_memory=bootstrap_memory,
            )
        except TypeError:
            turn = self.bootstrap_backend.propose_bootstrap_turn(
                user_goal=user_goal,
                capability_context=capability_context,
                answer_history=answers,
                role_models=self.role_models,
            )
        turn = replace(
            turn,
            proposal=replace(
                turn.proposal,
                recommended_spec=replace(
                    turn.proposal.recommended_spec,
                    metadata={
                        **turn.proposal.recommended_spec.metadata,
                        "execution_backend_kind": capability_context.environment_facts.get("execution_backend_kind"),
                        "autonomous_execution_supported": capability_context.environment_facts.get(
                            "autonomous_execution_supported"
                        ),
                    },
                ),
                questions=_ensure_execution_guidance_question(
                    questions=refine_bootstrap_questions(turn.proposal.questions, answers=answers),
                    capability_context=capability_context,
                    answers=answers,
                    detail=guidance_gap_detail,
                ),
            ),
        )
        # Auto-fix incomplete metrics — the agent often describes metrics in prose but
        # leaves the JSON fields empty. Extract from the agent's own message.
        spec = turn.proposal.recommended_spec
        all_metrics = [spec.primary_metric, *spec.secondary_metrics, *spec.guardrail_metrics]
        has_incomplete = any(
            m.name == "[unspecified]" or m.goal not in ("maximize", "minimize")
            for m in all_metrics
        )
        if has_incomplete:
            self.progress_fn("fix_metrics", "Extracting metric details from agent's analysis...")
            try:
                fixes = self.narrator_backend.fix_incomplete_metrics(
                    current_spec=spec.to_dict(),
                    assistant_message=turn.assistant_message,
                )
                if fixes:
                    spec_dict = spec.to_dict()
                    for key in ("primary_metric", "secondary_metrics", "guardrail_metrics"):
                        if key in fixes:
                            spec_dict[key] = fixes[key]
                    try:
                        patched_spec = ExperimentSpec.from_dict(spec_dict)
                        turn = replace(turn, proposal=replace(turn.proposal, recommended_spec=patched_spec))
                    except Exception:
                        pass
            except Exception:
                pass  # Preflight check will catch remaining issues

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
                guide_path.write_text(access_guide.markdown.rstrip() + "\n", encoding="utf-8")
                access_guide_path = str(guide_path)
            except Exception:
                pass
        resolved_turn = replace(
            turn,
            preflight_checks=preflight_checks,
            ready_to_start=ready_to_start,
            missing_requirements=missing_requirements,
            access_guide_path=access_guide_path,
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
        except Exception:
            pass
        # Write experiment guide when ready (or close to ready)
        if ready_to_start or not [r for r in missing_requirements if r.startswith("answer:")]:
            try:
                self.progress_fn("experiment_guide", "Writing experiment guide...")
                guide_text = self.bootstrap_backend.build_experiment_guide(
                    resolved_turn, capability_context, answers,
                )
                agent_markdown_dir = self.memory_root / "agent_markdown"
                agent_markdown_dir.mkdir(parents=True, exist_ok=True)
                guide_path = agent_markdown_dir / "experiment_guide.md"
                guide_path.write_text(guide_text.rstrip() + "\n", encoding="utf-8")
            except Exception:
                pass
        self.progress_fn("narrate", f"[{self.role_models.narrator}] Preparing summary...")
        try:
            human_update = self.narrator_backend.summarize_bootstrap(resolved_turn, capability_context)
        except Exception:
            human_update = resolved_turn.assistant_message
        if resolved_turn.access_guide_path:
            human_update = (
                f"{human_update}\n\nAccess guide: {resolved_turn.access_guide_path}"
                if human_update
                else f"Access guide: {resolved_turn.access_guide_path}"
            )
        return replace(resolved_turn, human_update=human_update)

    @staticmethod
    def _apply_metric_goal_fixes(turn: BootstrapTurn, fixes: dict[str, str]) -> BootstrapTurn:
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
        patched_spec = replace(spec, primary_metric=primary, secondary_metrics=secondary, guardrail_metrics=guardrails)
        return replace(turn, proposal=replace(turn.proposal, recommended_spec=patched_spec))

    def apply_feedback(self, turn: BootstrapTurn, feedback: str) -> BootstrapTurn | None:
        """Interpret user feedback and patch the existing plan. Returns None if full replan needed."""
        cap_ctx = self._cached_capability_context or CapabilityContext()
        try:
            result = self.narrator_backend.interpret_feedback(turn, feedback, cap_ctx)
        except Exception:
            return None  # Fall through to full replan
        action = result.get("action", "replan")
        if action == "replan":
            return None
        spec_updates = result.get("spec_updates", {})
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
        patched_proposal = replace(turn.proposal, recommended_spec=patched_spec)
        # Re-run preflight checks on the patched spec
        executor_factory_path = self.resolve_executor_factory_path(patched_spec.objective)
        preflight_checks = run_preflight_checks(
            spec=patched_spec,
            capability_context=cap_ctx,
            memory_root=self.memory_root,
            executor_factory_path=executor_factory_path,
        )
        bootstrap_answers = patched_spec.metadata.get("bootstrap_answers", {})
        if not isinstance(bootstrap_answers, dict):
            bootstrap_answers = {}
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
    ) -> dict[str, Any]:
        bootstrap_turn = self.bootstrap(user_goal=user_goal, answers=answers)
        return self.start_from_bootstrap_turn(
            bootstrap_turn=bootstrap_turn,
            user_goal=user_goal,
            iterations=iterations,
            reset_state=True,
        )

    def _install_python_dependency(self, package_name: str) -> tuple[bool, dict[str, Any]]:
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
            self.progress_fn("dependency_recovery", f"Installing missing dependency {install_name}...")
            installed, install_result = self._install_python_dependency(install_name)
            if not installed:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[
                        f"Execution failed because Python dependency {module_name!r} is missing.",
                        f"Automatic install attempt failed for {install_name}.",
                    ],
                    next_ideas=["Inspect the environment, install the dependency manually if needed, then retry."],
                    failure_type=exc.__class__.__name__,
                    failure_summary=str(exc).strip() or exc.__class__.__name__,
                    recoverable=True,
                    recovery_actions=[f"Install dependency {install_name} and retry the same step."],
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
                    next_ideas=["Inspect the new failure and continue from it in the next iteration."],
                    failure_type=retry_exc.__class__.__name__,
                    failure_summary=str(retry_exc).strip() or retry_exc.__class__.__name__,
                    recoverable=True,
                    recovery_actions=["Use the updated environment and new failure details to choose the next fix."],
                    execution_details={"install_attempt": install_result},
                )

        return recover

    def start_from_bootstrap_turn(
        self,
        *,
        bootstrap_turn: BootstrapTurn,
        user_goal: str,
        iterations: int | None = None,
        iteration_callback=None,
        reset_state: bool = False,
    ) -> dict[str, Any]:
        if not bootstrap_turn.ready_to_start:
            return {"status": "needs_input", "bootstrap": bootstrap_turn.to_dict()}

        spec = bootstrap_turn.proposal.recommended_spec
        memory_store = FileMemoryStore(self.memory_root)
        if not reset_state and memory_store.has_persisted_state():
            try:
                stored_spec = memory_store.load_spec()
            except Exception:
                reset_state = True
            else:
                if stored_spec.to_dict() != spec.to_dict():
                    reset_state = True
        orchestrator, _runtime = self.build_orchestrator(
            spec=spec,
            objective=user_goal,
            memory_store=memory_store,
        )
        orchestrator.initialize(spec=spec, reset_state=reset_state)
        try:
            results = orchestrator.run(iterations=iterations, iteration_callback=iteration_callback)
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
                        RuntimeError(blocked_outcome.failure_summary or "Execution hit a non-recoverable blocker.")
                    ),
                    "detail": blocked_outcome.failure_summary or "",
                },
            }
        return {
            "status": "started",
            "bootstrap": bootstrap_turn.to_dict(),
            "results": cycle_results_to_payload(results),
        }
