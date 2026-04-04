from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

from loopforge.core.runtime import is_generic_autonomous
from loopforge.core.types import (
    ExecutionStep,
    ExperimentCandidate,
    ExperimentOutcome,
    ExperimentSpec,
    MemorySnapshot,
    MetricResult,
)


STEP_PROGRESS_CHECKPOINTS = (10.0, 30.0, 60.0, 120.0)


def _step_workdir(step: ExecutionStep, repo_root: Path) -> Path:
    if not step.cwd:
        return repo_root
    candidate = Path(step.cwd)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _ensure_within_repo(path: Path, repo_root: Path) -> Path:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    resolved_path.relative_to(resolved_root)
    return resolved_path


def _referenced_python_scripts(command: str) -> list[str]:
    matches: list[str] = []
    pattern = re.compile(
        r"""(?ix)
        (?:^|&&|\|\|)
        \s*
        (?:"[^"]*python(?:\.exe)?"|'[^']*python(?:\.exe)?'|[^\s&|]*python(?:\.exe)?|py(?:\.exe)?)
        \s+
        (?:"([^"]+\.py)"|'([^']+\.py)'|([^\s&|]+\.py))
        """
    )
    for match in pattern.finditer(command):
        script = next((group for group in match.groups() if group), None)
        if script:
            matches.append(script)
    return matches


def _execution_contract(snapshot: MemorySnapshot) -> dict[str, Any]:
    metadata = (
        snapshot.effective_spec.metadata
        if isinstance(snapshot.effective_spec.metadata, dict)
        else {}
    )
    contract = metadata.get("execution_contract")
    return contract if isinstance(contract, dict) else {}


def _baseline_reference_tokens(path_str: str) -> set[str]:
    normalized = path_str.replace("\\", "/").strip().lower()
    if not normalized:
        return set()
    path = Path(normalized)
    tokens = {
        normalized,
        normalized.replace("/", "\\"),
        path.name.lower(),
    }
    if path.suffix == ".py":
        module_token = ".".join(part for part in path.with_suffix("").parts if part)
        if module_token:
            tokens.add(module_token.lower())
    return {token for token in tokens if token}


def _step_mentions_baseline(step: ExecutionStep, baseline_paths: list[str]) -> bool:
    haystacks = [step.command, step.path or "", step.content or ""]
    lowered_haystacks = [item.lower() for item in haystacks if item]
    for baseline_path in baseline_paths:
        for token in _baseline_reference_tokens(baseline_path):
            if any(token in haystack for haystack in lowered_haystacks):
                return True
    return False


def _validate_baseline_reuse_contract(
    steps: list[ExecutionStep],
    snapshot: MemorySnapshot,
) -> tuple[bool, int, str]:
    contract = _execution_contract(snapshot)
    if not contract.get("must_reference_baseline_paths"):
        return True, 0, ""
    if contract.get("enforcement_scope") == "until_first_successful_iteration" and (
        snapshot.latest_summary is not None
    ):
        return True, 0, ""
    baseline_paths = [
        str(item).strip()
        for item in contract.get("baseline_paths", [])
        if str(item).strip()
    ]
    if not baseline_paths:
        return True, 0, ""
    if any(_step_mentions_baseline(step, baseline_paths) for step in steps):
        return True, 0, ""
    preview = ", ".join(baseline_paths[:3])
    return (
        False,
        1,
        "Bootstrap contract requires the first execution plan to explicitly inspect, copy, edit, or run one of the existing baseline paths before branching into variants: "
        f"{preview}. Re-anchor on the existing framework instead of inventing a new from-scratch script.",
    )


def _short_command(command: str, limit: int = 100) -> str:
    compact = " ".join(command.split())
    if len(compact) > limit:
        return compact[: limit - 3] + "..."
    return compact


def _short_path_label(path_str: str, keep_parts: int = 3) -> str:
    normalized = path_str.replace("/", "\\")
    parts = [part for part in normalized.split("\\") if part]
    if len(parts) <= keep_parts:
        return normalized
    return "\\".join(parts[-keep_parts:])


def _effective_shell_segment(command: str) -> str:
    segments = [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|\|\|)\s*", command)
        if segment.strip()
    ]
    if not segments:
        return command.strip()
    for segment in reversed(segments):
        lowered = segment.lower()
        if lowered.startswith("cd ") or lowered.startswith("cd /d "):
            continue
        return segment
    return segments[-1]


def _strip_cmd_prefix(command: str) -> str:
    stripped = command.strip()
    lowered = stripped.lower()
    for prefix in ("cmd /c ", "cmd.exe /c "):
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _looks_like_raw_file_dump_command(command: str) -> bool:
    effective = _strip_cmd_prefix(_effective_shell_segment(command))
    lowered = effective.lower()
    if lowered.startswith("get-content "):
        return True
    for prefix in ("type ", "cat ", "head ", "more "):
        if lowered.startswith(prefix):
            return True
    return False


def _describe_inline_python(command: str) -> str:
    args = _inline_python_command_args(command)
    if args is None:
        return "run inline Python"
    code = args[args.index("-c") + 1].lower()
    if any(
        token in code
        for token in ("exists(", "is_file(", "is_dir(", "pathlib.path", "resolve(")
    ):
        return "check file/path with inline Python"
    if any(
        token in code
        for token in ("read_parquet", "read_csv", "read_json", "read_pickle")
    ):
        return "inspect data with inline Python"
    if "metric_results" in code or "primary_metric" in code:
        return "emit metric payload via inline Python"
    if "print(" in code:
        return "inspect values with inline Python"
    return "run inline Python"


def _describe_shell_command(command: str) -> str:
    effective = _effective_shell_segment(command)
    if _inline_python_command_args(effective) is not None:
        return _describe_inline_python(effective)
    scripts = _referenced_python_scripts(effective)
    if len(scripts) == 1:
        return f"run {_short_path_label(scripts[0])}"
    lowered = effective.lower()
    if "pytest" in lowered:
        return "run pytest"
    if lowered.startswith("git diff"):
        return "inspect git diff"
    if lowered.startswith("git status"):
        return "inspect git status"
    return _short_command(effective)


def _describe_step(step: ExecutionStep) -> str:
    if step.kind == "write_file" and step.path:
        return f"write {_short_path_label(step.path)}"
    if step.kind == "append_file" and step.path:
        return f"append {_short_path_label(step.path)}"
    if step.kind == "shell":
        return _describe_shell_command(step.command)
    return step.kind


def _step_progress_label(step: ExecutionStep) -> str:
    label = _describe_step(step)
    rationale = " ".join(step.rationale.split())
    if not rationale:
        return label
    if len(rationale) > 140:
        rationale = rationale[:137] + "..."
    return f"{label} - {rationale}"


def _replace_first_python_script(command: str, new_script_path: str) -> str:
    pattern = re.compile(
        r"""(?ix)
        (?P<prefix>
            (?:^|&&|\|\|)\s*
            (?:"[^"]*python(?:\.exe)?"|'[^']*python(?:\.exe)?'|[^\s&|]*python(?:\.exe)?|py(?:\.exe)?)
            \s+
        )
        (?:"[^"]+\.py"|'[^']+\.py'|[^\s&|]+\.py)
        """
    )
    replacement = lambda match: f'{match.group("prefix")}"{new_script_path}"'
    return pattern.sub(replacement, command, count=1)


def _planned_python_paths(steps: list[ExecutionStep], repo_root: Path) -> list[Path]:
    planned: list[Path] = []
    for step in steps:
        if step.kind not in {"write_file", "append_file"} or not step.path:
            continue
        if not step.path.lower().endswith(".py"):
            continue
        try:
            planned.append(_ensure_within_repo(repo_root / step.path, repo_root))
        except Exception:
            continue
    return planned


def _local_fast_repair_missing_script(
    *,
    current_steps: list[ExecutionStep],
    repo_root: Path,
    failed_step_index: int,
) -> list[ExecutionStep] | None:
    if failed_step_index <= 0 or failed_step_index > len(current_steps):
        return None
    failed_step = current_steps[failed_step_index - 1]
    if failed_step.kind != "shell":
        return None
    scripts = _referenced_python_scripts(failed_step.command)
    if len(scripts) != 1:
        return None
    missing_script = Path(scripts[0])
    try:
        workdir = _ensure_within_repo(_step_workdir(failed_step, repo_root), repo_root)
    except Exception:
        return None

    replacement_target: Path | None = None
    direct_repo_candidate = (
        (repo_root / missing_script).resolve()
        if not missing_script.is_absolute()
        else missing_script.resolve()
    )
    if direct_repo_candidate.exists():
        replacement_target = direct_repo_candidate
    else:
        planned_paths = _planned_python_paths(current_steps, repo_root)
        basename_matches = [
            path
            for path in planned_paths
            if path.name.lower() == missing_script.name.lower()
        ]
        if len(basename_matches) == 1:
            replacement_target = basename_matches[0]
        elif len(planned_paths) == 1:
            replacement_target = planned_paths[0]

    if replacement_target is None:
        return None

    rewritten_command = _replace_first_python_script(
        failed_step.command, str(replacement_target)
    )
    if rewritten_command == failed_step.command:
        return None

    revised_steps = list(current_steps)
    revised_steps[failed_step_index - 1] = replace(
        failed_step,
        command=rewritten_command,
        rationale=(
            failed_step.rationale
            or f"Run the repaired script path {replacement_target.relative_to(repo_root) if replacement_target.is_relative_to(repo_root) else replacement_target}."
        ),
    )
    return revised_steps


def _strip_outer_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _inline_python_command_args(command: str) -> list[str] | None:
    """Parse simple python -c invocations so they can run without shell quoting bugs."""
    try:
        raw_tokens = shlex.split(command, posix=False)
    except ValueError:
        return None
    if len(raw_tokens) < 3:
        return None
    tokens = [_strip_outer_quotes(token) for token in raw_tokens]
    executable = Path(tokens[0]).name.lower()
    if executable not in {"python", "python.exe", "py", "py.exe"}:
        return None
    try:
        command_index = tokens.index("-c", 1)
    except ValueError:
        return None
    if command_index >= len(tokens) - 1:
        return None
    return [
        tokens[0],
        *tokens[1:command_index],
        "-c",
        tokens[command_index + 1],
        *tokens[command_index + 2 :],
    ]


def _classify_shell_failure(stdout: str, stderr: str) -> tuple[str, bool, list[str]]:
    lower_error = f"{stdout}\n{stderr}".lower()
    if (
        "permissionerror" in lower_error
        and "access is denied" in lower_error
        and any(
            token in lower_error
            for token in ("socketpair", "multiprocessing", "_ssock", "_csock")
        )
    ):
        return (
            "MultiprocessingPermissionError",
            True,
            [
                "Retry with multiprocessing disabled or worker count reduced to 1.",
                "Switch to a serial execution path that avoids socketpair/resource-tracker setup if the repo supports it.",
            ],
        )
    if "can't open file" in lower_error or (
        "no such file or directory" in lower_error and ".py" in lower_error
    ):
        return (
            "MissingScriptFile",
            True,
            [
                "Create the referenced script file before running it.",
                "Or replace the script invocation with inline Python if the script is only a temporary probe.",
            ],
        )
    if "is not recognized as an internal or external command" in lower_error:
        return (
            "PlatformCommandMismatch",
            True,
            [
                "Replace Unix-style utilities with commands compatible with the active shell.",
                "On Windows cmd.exe, prefer Python scripts or cmd-compatible commands over head/grep/find -maxdepth/ls -la.",
            ],
        )
    blocked = "access is denied" in lower_error or "permission denied" in lower_error
    if blocked:
        return (
            "ShellPermissionDenied",
            True,
            [
                "Retry from a writable repo path or adjust the working directory and file permissions.",
                "If the repo is in a restricted temp/system directory, move it under a normal workspace path and retry.",
            ],
        )
    return (
        "ShellCommandFailed",
        True,
        ["Inspect the error output, adjust the command or environment, and retry."],
    )


def _run_subprocess_with_progress(
    *,
    command: str | list[str],
    cwd: Path,
    shell: bool,
    timeout_seconds: int,
    progress_fn,
    step_index: int,
    total_steps: int,
    step_description: str,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=shell,
    )
    started_at = time.monotonic()
    checkpoints = STEP_PROGRESS_CHECKPOINTS
    checkpoint_index = 0
    next_target = checkpoints[0]
    repeat_interval = checkpoints[-1]
    tick = 0

    while True:
        elapsed = time.monotonic() - started_at
        remaining_timeout = timeout_seconds - elapsed
        if remaining_timeout <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=timeout_seconds,
                output=stdout,
                stderr=stderr,
            )

        time_until_checkpoint = max(0.0, next_target - elapsed)
        if time_until_checkpoint > 0:
            wait_for = min(1.0, remaining_timeout, time_until_checkpoint)
        else:
            wait_for = min(0.1, remaining_timeout)
        try:
            stdout, stderr = process.communicate(timeout=wait_for)
            return subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started_at
            if progress_fn is not None and elapsed >= next_target:
                tick += 1
                progress_fn(
                    f"step_{step_index}_wait_{tick}",
                    f"Still running step {step_index}/{total_steps}: {step_description} ({int(elapsed)}s elapsed, waiting for process exit)...",
                )
                checkpoint_index += 1
                if checkpoint_index < len(checkpoints):
                    next_target = checkpoints[checkpoint_index]
                else:
                    next_target += repeat_interval


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
        return numeric if isfinite(numeric) else None
    if isinstance(value, str):
        stripped = value.strip().rstrip(",")
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        return numeric if isfinite(numeric) else None
    return None


def _known_metric_specs(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        metric.name: metric
        for metric in (
            spec.primary_metric,
            *spec.secondary_metrics,
            *spec.guardrail_metrics,
        )
    }


def _normalise_metric_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _metric_name_aliases(spec: ExperimentSpec) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for metric in (
        spec.primary_metric,
        *spec.secondary_metrics,
        *spec.guardrail_metrics,
    ):
        for candidate in (metric.name, metric.display_name):
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            aliases.setdefault(_normalise_metric_name(candidate), metric.name)
    aliases.setdefault(
        _normalise_metric_name("primary_metric"), spec.primary_metric.name
    )
    aliases.setdefault(
        _normalise_metric_name("primary metric"), spec.primary_metric.name
    )
    return aliases


def _resolve_metric_name(name: str, spec: ExperimentSpec) -> str | None:
    known_metrics = _known_metric_specs(spec)
    if name in known_metrics:
        return name
    aliases = _metric_name_aliases(spec)
    return aliases.get(_normalise_metric_name(name))


def _coerce_metric_result(
    name: str, raw_value: Any, spec: ExperimentSpec
) -> MetricResult | None:
    known_metrics = _known_metric_specs(spec)
    resolved_name = _resolve_metric_name(name, spec)
    metric_name = resolved_name or name
    metric_spec = known_metrics.get(metric_name)
    if isinstance(raw_value, dict):
        value = _coerce_float(raw_value.get("value"))
        passed = raw_value.get("passed")
        if not isinstance(passed, bool):
            passed = None
        scorer_ref = raw_value.get("scorer_ref")
        details = raw_value.get("details", {})
        if value is None and passed is None:
            return None
        return MetricResult(
            name=metric_name,
            value=value,
            passed=passed,
            scorer_ref=str(scorer_ref)
            if isinstance(scorer_ref, str) and scorer_ref.strip()
            else (metric_spec.scorer_ref if metric_spec is not None else None),
            details=details if isinstance(details, dict) else {},
        )
    value = _coerce_float(raw_value)
    if value is None:
        return None
    return MetricResult(
        name=metric_name,
        value=value,
        scorer_ref=metric_spec.scorer_ref if metric_spec is not None else None,
    )


def _extract_metric_payload_from_mapping(
    payload: dict[str, Any], spec: ExperimentSpec
) -> dict[str, Any]:
    metric_results: dict[str, MetricResult] = {}
    secondary_metrics: dict[str, float] = {}
    guardrail_metrics: dict[str, float] = {}
    known_metrics = _known_metric_specs(spec)
    secondary_names = {metric.name for metric in spec.secondary_metrics}
    guardrail_names = {metric.name for metric in spec.guardrail_metrics}

    raw_metric_results = payload.get("metric_results")
    if not isinstance(raw_metric_results, dict):
        raw_metric_results = (
            payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        )
    for name, raw_value in raw_metric_results.items():
        result = _coerce_metric_result(str(name), raw_value, spec)
        if result is not None:
            metric_results[result.name] = result
            if result.value is not None and result.name in secondary_names:
                secondary_metrics[result.name] = result.value
            if result.value is not None and result.name in guardrail_names:
                guardrail_metrics[result.name] = result.value

    for section_name, target in (
        ("secondary_metrics", secondary_metrics),
        ("guardrail_metrics", guardrail_metrics),
    ):
        section_payload = payload.get(section_name)
        if not isinstance(section_payload, dict):
            continue
        for name, raw_value in section_payload.items():
            value = _coerce_float(raw_value)
            if value is None:
                continue
            target[str(name)] = value
            metric_results.setdefault(
                str(name),
                MetricResult(
                    name=str(name),
                    value=value,
                    scorer_ref=known_metrics.get(str(name)).scorer_ref
                    if str(name) in known_metrics
                    else None,
                ),
            )

    if not metric_results:
        for name in known_metrics:
            if name in payload:
                result = _coerce_metric_result(name, payload.get(name), spec)
                if result is not None:
                    metric_results[name] = result

    primary_metric_value = _coerce_float(payload.get("primary_metric_value"))
    if primary_metric_value is None:
        primary_metric = metric_results.get(spec.primary_metric.name)
        if primary_metric is not None:
            primary_metric_value = primary_metric.value

    secondary_names = {metric.name for metric in spec.secondary_metrics}
    guardrail_names = {metric.name for metric in spec.guardrail_metrics}
    for name, result in metric_results.items():
        if result.value is None:
            continue
        if name in secondary_names:
            secondary_metrics.setdefault(name, result.value)
        if name in guardrail_names:
            guardrail_metrics.setdefault(name, result.value)

    return {
        "metric_results": metric_results,
        "primary_metric_value": primary_metric_value,
        "secondary_metrics": secondary_metrics,
        "guardrail_metrics": guardrail_metrics,
    }


def _extract_metric_payload_from_text(
    text: str, spec: ExperimentSpec
) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {
            "metric_results": {},
            "primary_metric_value": None,
            "secondary_metrics": {},
            "guardrail_metrics": {},
        }

    json_candidates = [
        stripped,
        *[line.strip() for line in stripped.splitlines() if line.strip()],
    ]
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            metric_payload = _extract_metric_payload_from_mapping(parsed, spec)
            if (
                metric_payload["metric_results"]
                or metric_payload["primary_metric_value"] is not None
            ):
                return metric_payload

    metric_results: dict[str, MetricResult] = {}
    secondary_metrics: dict[str, float] = {}
    guardrail_metrics: dict[str, float] = {}
    secondary_names = {metric.name for metric in spec.secondary_metrics}
    guardrail_names = {metric.name for metric in spec.guardrail_metrics}
    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        for separator in ("=", ":"):
            if separator not in candidate:
                continue
            name, _, value = candidate.partition(separator)
            metric_name = name.strip()
            resolved_name = _resolve_metric_name(metric_name, spec)
            if resolved_name is None and _normalise_metric_name(
                metric_name
            ) != _normalise_metric_name("primary_metric_value"):
                continue
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            if _normalise_metric_name(metric_name) == _normalise_metric_name(
                "primary_metric_value"
            ):
                return {
                    "metric_results": metric_results,
                    "primary_metric_value": numeric,
                    "secondary_metrics": secondary_metrics,
                    "guardrail_metrics": guardrail_metrics,
                }
            result = _coerce_metric_result(resolved_name or metric_name, numeric, spec)
            if result is None:
                continue
            metric_results[result.name] = result
            if result.name in secondary_names:
                secondary_metrics[result.name] = numeric
            if result.name in guardrail_names:
                guardrail_metrics[result.name] = numeric
            break

    primary_metric = metric_results.get(spec.primary_metric.name)
    return {
        "metric_results": metric_results,
        "primary_metric_value": primary_metric.value
        if primary_metric is not None
        else None,
        "secondary_metrics": secondary_metrics,
        "guardrail_metrics": guardrail_metrics,
    }


def _extract_metric_payload(
    step_results: list[dict[str, Any]], spec: ExperimentSpec
) -> dict[str, Any]:
    metric_results: dict[str, MetricResult] = {}
    secondary_metrics: dict[str, float] = {}
    guardrail_metrics: dict[str, float] = {}
    primary_metric_value: float | None = None

    for step_result in step_results:
        stdout = step_result.get("stdout")
        if not isinstance(stdout, str) or not stdout.strip():
            continue
        parsed = _extract_metric_payload_from_text(stdout, spec)
        metric_results.update(parsed["metric_results"])
        secondary_metrics.update(parsed["secondary_metrics"])
        guardrail_metrics.update(parsed["guardrail_metrics"])
        if parsed["primary_metric_value"] is not None:
            primary_metric_value = parsed["primary_metric_value"]

    if primary_metric_value is None:
        primary_result = metric_results.get(spec.primary_metric.name)
        if primary_result is not None:
            primary_metric_value = primary_result.value

    return {
        "metric_results": metric_results,
        "primary_metric_value": primary_metric_value,
        "secondary_metrics": secondary_metrics,
        "guardrail_metrics": guardrail_metrics,
    }


def _validate_steps_pre_execution(
    steps: list[ExecutionStep],
    repo_root: Path,
    snapshot: MemorySnapshot | None = None,
) -> tuple[bool, int, str]:
    """Quick pre-validation. Returns (ok, failed_step_index, reason)."""
    if snapshot is not None and is_generic_autonomous(snapshot=snapshot):
        contract_ok, contract_failed_idx, contract_reason = (
            _validate_baseline_reuse_contract(steps, snapshot)
        )
        if not contract_ok:
            return contract_ok, contract_failed_idx, contract_reason
    planned_files: set[Path] = set()
    for index, step in enumerate(steps, start=1):
        if step.kind in {"write_file", "append_file"}:
            if step.path:
                target = (repo_root / step.path).resolve()
                planned_files.add(target)
            continue
        if step.kind == "shell":
            try:
                workdir = _ensure_within_repo(_step_workdir(step, repo_root), repo_root)
            except Exception:
                return (
                    False,
                    index,
                    f"Step {index} uses cwd outside the repo root: {step.cwd}",
                )
            if _looks_like_raw_file_dump_command(step.command):
                return (
                    False,
                    index,
                    (
                        f"Step {index} uses a raw file-dump shell command ({_effective_shell_segment(step.command)}). "
                        "Do not dump source files with shell commands like type/cat/head/more/Get-Content; "
                        "use one bounded Python inspection or go straight to write_file plus a runnable script."
                    ),
                )
            for script in _referenced_python_scripts(step.command):
                script_path = Path(script)
                candidate_path = (
                    (workdir / script_path)
                    if not script_path.is_absolute()
                    else script_path
                )
                if (
                    not candidate_path.exists()
                    and candidate_path.resolve() not in planned_files
                ):
                    return (
                        False,
                        index,
                        (
                            f"Step {index} references {candidate_path} which does not exist "
                            f"and is not created by an earlier write_file step."
                        ),
                    )
    return True, 0, ""


def _run_steps(
    steps: list[ExecutionStep],
    repo_root: Path,
    default_timeout_seconds: int,
    max_captured_chars: int,
    progress_fn=None,
) -> tuple[ExperimentOutcome | None, list[dict[str, Any]]]:
    """Execute steps sequentially. Returns (failure_outcome_or_None, step_results)."""
    step_results: list[dict[str, Any]] = []
    planned_files: set[Path] = set()
    total_steps = len(steps)

    for index, step in enumerate(steps, start=1):
        step_description = _step_progress_label(step)
        if progress_fn is not None:
            progress_fn(
                f"step_{index}",
                f"Running step {index}/{total_steps}: {step_description}",
            )
        try:
            workdir = _ensure_within_repo(_step_workdir(step, repo_root), repo_root)
        except Exception:
            return ExperimentOutcome(
                status="blocked",
                notes=[
                    f"Step {index} targeted a working directory outside the repo root."
                ],
                failure_type="UnsafePath",
                failure_summary=f"Refused to run outside the repo root via cwd: {step.cwd}",
                recoverable=False,
                execution_details={"step_results": step_results},
            ), step_results
        timeout_seconds = step.timeout_seconds or default_timeout_seconds

        if step.kind in {"write_file", "append_file"}:
            if not step.path:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[f"Step {index} is missing a target path."],
                    failure_type="MissingStepPath",
                    failure_summary=f"Step {index} did not include a path.",
                    recoverable=True,
                    recovery_actions=[
                        "Provide a repo-relative path for the file step and retry."
                    ],
                    execution_details={"step_results": step_results},
                ), step_results
            try:
                target_path = _ensure_within_repo((repo_root / step.path), repo_root)
            except Exception:
                return ExperimentOutcome(
                    status="blocked",
                    notes=[f"Step {index} targeted a path outside the repo root."],
                    failure_type="UnsafePath",
                    failure_summary=f"Refused to write outside the repo root: {step.path}",
                    recoverable=False,
                    execution_details={"step_results": step_results},
                ), step_results
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if step.kind == "write_file":
                target_path.write_text(step.content or "", encoding="utf-8")
            else:
                with target_path.open("a", encoding="utf-8") as handle:
                    handle.write(step.content or "")
            step_results.append(
                {
                    "index": index,
                    "kind": step.kind,
                    "path": str(target_path),
                    "bytes_written": len((step.content or "").encode("utf-8")),
                }
            )
            planned_files.add(target_path.resolve())
            continue

        # Shell step — check for missing scripts first
        for script in _referenced_python_scripts(step.command):
            script_path = Path(script)
            candidate_path = (
                (workdir / script_path)
                if not script_path.is_absolute()
                else script_path
            )
            if (
                not candidate_path.exists()
                and candidate_path.resolve() not in planned_files
            ):
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[f"Step {index} references missing script: {candidate_path}"],
                    failure_type="MissingScriptFile",
                    failure_summary=f"Script {candidate_path} does not exist.",
                    recoverable=True,
                    recovery_actions=[
                        "Create the script in a write_file step before running it.",
                        "Or use inline python -c instead.",
                    ],
                    execution_details={"step_results": step_results},
                ), step_results

        try:
            inline_python_args = _inline_python_command_args(step.command)
            if inline_python_args is not None:
                completed = _run_subprocess_with_progress(
                    command=inline_python_args,
                    cwd=workdir,
                    shell=False,
                    timeout_seconds=timeout_seconds,
                    progress_fn=progress_fn,
                    step_index=index,
                    total_steps=total_steps,
                    step_description=step_description,
                )
            else:
                completed = _run_subprocess_with_progress(
                    command=step.command,
                    cwd=workdir,
                    shell=True,
                    timeout_seconds=timeout_seconds,
                    progress_fn=progress_fn,
                    step_index=index,
                    total_steps=total_steps,
                    step_description=step_description,
                )
        except subprocess.TimeoutExpired as exc:
            step_results.append(
                {
                    "index": index,
                    "kind": step.kind,
                    "command": step.command,
                    "cwd": str(workdir),
                    "timeout_seconds": timeout_seconds,
                    "status": "timeout",
                }
            )
            return ExperimentOutcome(
                status="recoverable_failure",
                notes=[f"Step {index} timed out: {step.command}"],
                failure_type="TimeoutExpired",
                failure_summary=str(exc),
                recoverable=True,
                recovery_actions=["Shorten or narrow the command, then retry."],
                execution_details={"step_results": step_results},
            ), step_results
        except PermissionError as exc:
            step_results.append(
                {
                    "index": index,
                    "kind": step.kind,
                    "command": step.command,
                    "cwd": str(workdir),
                    "timeout_seconds": timeout_seconds,
                    "status": "permission_error",
                    "stderr": str(exc),
                }
            )
            return ExperimentOutcome(
                status="recoverable_failure",
                notes=[f"Step {index} could not start because the OS denied access."],
                failure_type="ShellPermissionDenied",
                failure_summary=str(exc),
                recoverable=True,
                recovery_actions=[
                    "Retry from a writable repo path or adjust the working directory and file permissions.",
                    "If the repo is in a restricted temp/system directory, move it under a normal workspace path and retry.",
                ],
                execution_details={"step_results": step_results},
            ), step_results

        stdout = completed.stdout[-max_captured_chars:]
        stderr = completed.stderr[-max_captured_chars:]
        step_results.append(
            {
                "index": index,
                "kind": step.kind,
                "command": step.command,
                "cwd": str(workdir),
                "timeout_seconds": timeout_seconds,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )

        if completed.returncode != 0 and not step.allow_failure:
            failure_type, recoverable, recovery_actions = _classify_shell_failure(
                stdout, stderr
            )
            return ExperimentOutcome(
                status="recoverable_failure" if recoverable else "blocked",
                notes=[f"Step {index} failed: {step.command}"],
                failure_type=failure_type,
                failure_summary=stderr.strip()
                or stdout.strip()
                or f"Exit code {completed.returncode}.",
                recoverable=recoverable,
                recovery_actions=recovery_actions,
                execution_details={"step_results": step_results},
            ), step_results

    # All steps succeeded
    return None, step_results


class GenericExecutionPlanExecutor:
    def __init__(
        self,
        *,
        repo_root: Path | str,
        fix_backend=None,
        max_retries: int = 8,
        default_timeout_seconds: int = 120,
        max_captured_chars: int = 4000,
        progress_fn=None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.fix_backend = fix_backend
        self.max_retries = max_retries
        self.default_timeout_seconds = default_timeout_seconds
        self.max_captured_chars = max_captured_chars
        self.progress_fn = progress_fn or (lambda stage, msg: None)
        self.last_fix_summary: str | None = None

    def _fix_backend_label(self) -> str:
        model = getattr(self.fix_backend, "model", None)
        if isinstance(model, str) and model.strip():
            return model.strip()
        return "execution_fixer"

    @staticmethod
    def _preview_steps(steps: list[ExecutionStep], limit: int = 3) -> str:
        previews: list[str] = []
        for step in steps[:limit]:
            previews.append(_describe_step(step))
        return " | ".join(previews)

    @staticmethod
    def _preview_repair_summary(summary: str | None, limit: int = 220) -> str | None:
        if not isinstance(summary, str):
            return None
        compact = " ".join(summary.split())
        if not compact:
            return None
        if len(compact) > limit:
            return compact[: limit - 3] + "..."
        return compact

    @staticmethod
    def _steps_equal(left: list[ExecutionStep], right: list[ExecutionStep]) -> bool:
        return [step.to_dict() for step in left] == [step.to_dict() for step in right]

    @staticmethod
    def _metrics_required_for_success(
        candidate: ExperimentCandidate, snapshot: MemorySnapshot
    ) -> bool:
        return bool(candidate.execution_steps) and is_generic_autonomous(
            snapshot=snapshot
        )

    @staticmethod
    def _build_metricless_failure(
        *,
        candidate: ExperimentCandidate,
        snapshot: MemorySnapshot,
        step_results: list[dict[str, Any]],
        attempts: list[dict[str, Any]],
    ) -> ExperimentOutcome:
        primary_metric_name = snapshot.effective_spec.primary_metric.name
        latest_stdout_preview = ""
        if step_results:
            latest_stdout_preview = str(step_results[-1].get("stdout", "")).strip()[
                :500
            ]
        notes = [
            "Execution steps completed, but the runtime could not capture the configured metrics from stdout."
        ]
        if latest_stdout_preview:
            notes.append(f"Latest stdout preview: {latest_stdout_preview}")
        return ExperimentOutcome(
            status="recoverable_failure",
            notes=notes,
            failure_type="MetricsNotReported",
            failure_summary=(
                f"Execution succeeded but did not print a parseable value for the primary metric "
                f"{primary_metric_name!r} or a machine-readable metric payload."
            ),
            recoverable=True,
            recovery_actions=[
                "Revise the execution steps so the final run prints the configured metrics to stdout.",
                "Prefer a single JSON payload with metric_results or explicit metric_name=value lines.",
            ],
            execution_details={
                "step_results": step_results,
                "latest_stdout_preview": latest_stdout_preview,
                "attempts": attempts,
                "candidate": candidate.to_dict(),
            },
        )

    def execute(
        self, candidate: ExperimentCandidate, snapshot: MemorySnapshot
    ) -> ExperimentOutcome:
        if not candidate.execution_steps:
            return ExperimentOutcome(
                status="recoverable_failure",
                notes=["No execution steps were provided for the agentic executor."],
                failure_type="MissingExecutionSteps",
                failure_summary="The worker proposed an agentic step without concrete execution steps.",
                recoverable=True,
                recovery_actions=[
                    "Propose concrete shell steps for the next iteration."
                ],
                execution_details={"step_results": []},
            )

        current_steps = list(candidate.execution_steps)
        all_attempt_results: list[dict[str, Any]] = []

        for attempt in range(1 + self.max_retries):
            # Pre-validate before running
            valid, failed_idx, reason = _validate_steps_pre_execution(
                current_steps, self.repo_root, snapshot
            )
            if not valid:
                local_fixed_steps = _local_fast_repair_missing_script(
                    current_steps=current_steps,
                    repo_root=self.repo_root,
                    failed_step_index=failed_idx,
                )
                if local_fixed_steps is not None:
                    self.progress_fn(
                        f"local_fix_attempt_{attempt}",
                        f"Applied local fast repair for MissingScriptFile: {self._preview_steps(local_fixed_steps)}",
                    )
                    current_steps = local_fixed_steps
                    continue
            if (
                not valid
                and self.fix_backend is not None
                and attempt < self.max_retries
            ):
                fixer_label = self._fix_backend_label()
                self.progress_fn(
                    f"fix_attempt_{attempt}",
                    f"[{fixer_label}] Pre-validation failed at step {failed_idx}: {reason}",
                )
                fixed_steps = self._ask_for_fix(
                    snapshot,
                    candidate,
                    current_steps,
                    failed_idx,
                    reason,
                    [],
                    attempt + 1,
                )
                if fixed_steps:
                    repair_summary = self._preview_repair_summary(self.last_fix_summary)
                    self.progress_fn(
                        f"fix_attempt_{attempt}_detail",
                        f"[{fixer_label}] Revised plan with {len(fixed_steps)} step(s): {self._preview_steps(fixed_steps)}",
                    )
                    if repair_summary:
                        self.progress_fn(
                            f"fix_attempt_{attempt}_summary",
                            f"[{fixer_label}] Repair reasoning: {repair_summary}",
                        )
                    current_steps = fixed_steps
                    continue
                self.progress_fn(
                    f"fix_attempt_{attempt}_detail",
                    f"[{fixer_label}] Did not return a revised execution plan.",
                )
            if not valid:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[
                        f"Execution plan pre-validation failed at step {failed_idx}."
                    ],
                    failure_type="InvalidExecutionPlan",
                    failure_summary=reason,
                    recoverable=True,
                    recovery_actions=[
                        "Replace shell-only file inspection with one bounded Python inspection or a write_file + python script run.",
                        "Keep the iteration end-to-end so the final step prints metrics.",
                    ],
                    execution_details={
                        "step_results": [],
                        "attempts": all_attempt_results,
                        "candidate": candidate.to_dict(),
                    },
                )

            # Execute
            if attempt > 0:
                self.progress_fn(
                    f"retry_{attempt}",
                    f"Retry attempt {attempt + 1}: {self._preview_steps(current_steps)}",
                )

            failure_outcome, step_results = _run_steps(
                current_steps,
                self.repo_root,
                self.default_timeout_seconds,
                self.max_captured_chars,
                self.progress_fn,
            )
            all_attempt_results.append(
                {
                    "attempt": attempt + 1,
                    "steps": [s.to_dict() for s in current_steps],
                    "step_results": step_results,
                    "success": failure_outcome is None,
                }
            )

            if failure_outcome is None:
                metric_payload = _extract_metric_payload(
                    step_results, snapshot.effective_spec
                )
                metrics_captured = bool(
                    metric_payload["metric_results"]
                    or metric_payload["primary_metric_value"] is not None
                )
                if not metrics_captured and self._metrics_required_for_success(
                    candidate, snapshot
                ):
                    failure_outcome = self._build_metricless_failure(
                        candidate=candidate,
                        snapshot=snapshot,
                        step_results=step_results,
                        attempts=all_attempt_results,
                    )
                    if attempt < self.max_retries and self.fix_backend is not None:
                        fixer_label = self._fix_backend_label()
                        self.progress_fn(
                            f"fix_attempt_{attempt}",
                            f"[{fixer_label}] Execution produced no parseable metrics: {failure_outcome.failure_summary[:160]}",
                        )
                        fixed_steps = self._ask_for_fix(
                            snapshot,
                            candidate,
                            current_steps,
                            len(step_results),
                            failure_outcome.failure_summary
                            or "Metrics were not reported",
                            step_results,
                            attempt + 1,
                        )
                        if fixed_steps:
                            self.progress_fn(
                                f"fix_attempt_{attempt}_detail",
                                f"[{fixer_label}] Revised plan with {len(fixed_steps)} step(s): {self._preview_steps(fixed_steps)}",
                            )
                            current_steps = fixed_steps
                            continue
                        self.progress_fn(
                            f"fix_attempt_{attempt}_detail",
                            f"[{fixer_label}] Did not return a revised execution plan.",
                        )
                    return failure_outcome
                notes = [
                    f"Executed {len(current_steps)} step(s) successfully"
                    + (f" (after {attempt} fix(es))" if attempt > 0 else "")
                    + "."
                ]
                if metrics_captured:
                    notes.append("Captured metric output from execution stdout.")
                # Success!
                return ExperimentOutcome(
                    status="success",
                    metric_results=metric_payload["metric_results"],
                    primary_metric_value=metric_payload["primary_metric_value"],
                    secondary_metrics=metric_payload["secondary_metrics"],
                    guardrail_metrics=metric_payload["guardrail_metrics"],
                    notes=notes,
                    execution_details={
                        "step_results": step_results,
                        "attempts": all_attempt_results,
                    },
                )

            # Failed — can we retry?
            if failure_outcome.status == "blocked":
                # Non-recoverable — don't retry
                return failure_outcome

            local_fixed_steps = self._try_local_fast_repair(
                failure_outcome=failure_outcome,
                current_steps=current_steps,
                failed_step_index=len(step_results),
            )
            if local_fixed_steps is not None:
                self.progress_fn(
                    f"local_fix_attempt_{attempt}",
                    f"Applied local fast repair for {failure_outcome.failure_type}: {self._preview_steps(local_fixed_steps)}",
                )
                current_steps = local_fixed_steps
                continue

            if attempt < self.max_retries and self.fix_backend is not None:
                fixer_label = self._fix_backend_label()
                self.progress_fn(
                    f"fix_attempt_{attempt}",
                    f"[{fixer_label}] Step {len(step_results)} failed ({failure_outcome.failure_type or failure_outcome.status}): {failure_outcome.failure_summary[:160]}",
                )
                fixed_steps = self._ask_for_fix(
                    snapshot,
                    candidate,
                    current_steps,
                    len(step_results),
                    failure_outcome.failure_summary or "Unknown failure",
                    step_results,
                    attempt + 1,
                )
                if fixed_steps:
                    repair_summary = self._preview_repair_summary(self.last_fix_summary)
                    self.progress_fn(
                        f"fix_attempt_{attempt}_detail",
                        f"[{fixer_label}] Revised plan with {len(fixed_steps)} step(s): {self._preview_steps(fixed_steps)}",
                    )
                    if repair_summary:
                        self.progress_fn(
                            f"fix_attempt_{attempt}_summary",
                            f"[{fixer_label}] Repair reasoning: {repair_summary}",
                        )
                    current_steps = fixed_steps
                    continue
                self.progress_fn(
                    f"fix_attempt_{attempt}_detail",
                    f"[{fixer_label}] Did not return a revised execution plan.",
                )

            # No more retries or fix backend unavailable
            failure_outcome = replace(
                failure_outcome,
                execution_details={
                    **failure_outcome.execution_details,
                    "attempts": all_attempt_results,
                },
            )
            return failure_outcome

        # Should not reach here, but just in case
        return ExperimentOutcome(
            status="recoverable_failure",
            notes=["All retry attempts exhausted."],
            failure_type="RetriesExhausted",
            failure_summary="Execution failed after all retry attempts.",
            recoverable=True,
            execution_details={"attempts": all_attempt_results},
        )

    def _ask_for_fix(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        current_steps: list[ExecutionStep],
        failed_step_index: int,
        failure_summary: str,
        step_results: list[dict[str, Any]],
        attempt_number: int,
    ) -> list[ExecutionStep] | None:
        try:
            self.last_fix_summary = None
            repair_candidate = replace(
                candidate,
                execution_steps=list(current_steps),
                metadata={
                    **candidate.metadata,
                    "_execution_repair_context": {
                        "attempt_number": attempt_number,
                        "max_retries": self.max_retries,
                        "environment_facts": snapshot.capability_context.environment_facts,
                        "repo_root": str(self.repo_root.resolve()),
                    },
                },
            )
            fixed = self.fix_backend.fix_execution_plan(
                candidate=repair_candidate,
                failed_step_index=failed_step_index,
                failure_summary=failure_summary,
                step_results=step_results,
            )
            repair_summary = getattr(self.fix_backend, "last_repair_summary", None)
            self.last_fix_summary = (
                repair_summary.strip()
                if isinstance(repair_summary, str) and repair_summary.strip()
                else None
            )
            if not fixed or self._steps_equal(fixed, current_steps):
                return None
            return fixed
        except Exception:
            return None

    def _try_local_fast_repair(
        self,
        *,
        failure_outcome: ExperimentOutcome,
        current_steps: list[ExecutionStep],
        failed_step_index: int,
    ) -> list[ExecutionStep] | None:
        if failure_outcome.failure_type == "MissingScriptFile":
            return _local_fast_repair_missing_script(
                current_steps=current_steps,
                repo_root=self.repo_root,
                failed_step_index=failed_step_index,
            )
        return None
