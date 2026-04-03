from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

from loopforge.core.types import (
    ExecutionStep,
    ExperimentCandidate,
    ExperimentOutcome,
    ExperimentSpec,
    MemorySnapshot,
    MetricResult,
)


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
        (?:python|py)(?:\.exe)?
        \s+
        (?:"([^"]+\.py)"|'([^']+\.py)'|([^\s&|]+\.py))
        """
    )
    for match in pattern.finditer(command):
        script = next((group for group in match.groups() if group), None)
        if script:
            matches.append(script)
    return matches


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


def _coerce_metric_result(
    name: str, raw_value: Any, spec: ExperimentSpec
) -> MetricResult | None:
    known_metrics = _known_metric_specs(spec)
    metric_spec = known_metrics.get(name)
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
            name=name,
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
        name=name,
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
    known_metrics = set(_known_metric_specs(spec))
    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        for separator in ("=", ":"):
            if separator not in candidate:
                continue
            name, _, value = candidate.partition(separator)
            metric_name = name.strip()
            if (
                metric_name not in known_metrics
                and metric_name != "primary_metric_value"
            ):
                continue
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            if metric_name == "primary_metric_value":
                return {
                    "metric_results": metric_results,
                    "primary_metric_value": numeric,
                    "secondary_metrics": secondary_metrics,
                    "guardrail_metrics": guardrail_metrics,
                }
            result = _coerce_metric_result(metric_name, numeric, spec)
            if result is None:
                continue
            metric_results[metric_name] = result
            if metric_name in secondary_names:
                secondary_metrics[metric_name] = numeric
            if metric_name in guardrail_names:
                guardrail_metrics[metric_name] = numeric
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
) -> tuple[bool, int, str]:
    """Quick pre-validation. Returns (ok, failed_step_index, reason)."""
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

    for index, step in enumerate(steps, start=1):
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
                completed = subprocess.run(
                    inline_python_args,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=timeout_seconds,
                    check=False,
                )
            else:
                completed = subprocess.run(
                    step.command,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=timeout_seconds,
                    check=False,
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

    def _fix_backend_label(self) -> str:
        model = getattr(self.fix_backend, "model", None)
        if isinstance(model, str) and model.strip():
            return model.strip()
        return "execution_fixer"

    @staticmethod
    def _preview_steps(steps: list[ExecutionStep], limit: int = 3) -> str:
        previews: list[str] = []
        for step in steps[:limit]:
            if step.kind == "shell":
                previews.append(f"$ {step.command[:120]}")
            elif step.kind in {"write_file", "append_file"}:
                previews.append(f"{step.kind} {step.path}")
            else:
                previews.append(step.kind)
        return " | ".join(previews)

    @staticmethod
    def _steps_equal(left: list[ExecutionStep], right: list[ExecutionStep]) -> bool:
        return [step.to_dict() for step in left] == [step.to_dict() for step in right]

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
                current_steps, self.repo_root
            )
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
                # Fix backend returned nothing — fall through to try anyway

            # Execute
            if attempt > 0:
                self.progress_fn(f"retry_{attempt}", f"Retry attempt {attempt + 1}...")

            failure_outcome, step_results = _run_steps(
                current_steps,
                self.repo_root,
                self.default_timeout_seconds,
                self.max_captured_chars,
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
                notes = [
                    f"Executed {len(current_steps)} step(s) successfully"
                    + (f" (after {attempt} fix(es))" if attempt > 0 else "")
                    + "."
                ]
                if (
                    metric_payload["metric_results"]
                    or metric_payload["primary_metric_value"] is not None
                ):
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
            if not fixed or self._steps_equal(fixed, current_steps):
                return None
            return fixed
        except Exception:
            return None
