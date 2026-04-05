"""Interactive tool-use agents.

All LLM agents in loopforge use the same pattern: litellm + tools in a loop.
The model reads files, runs commands, thinks, and reports structured results.

Three agents:
- ToolUsePlanner: reads repo, fills execution contract, asks user questions
- ToolUseExecutor: writes code, runs experiments, reports metrics
- ToolUseReviewer: analyzes results, extracts lessons, decides accept/reject
"""

from __future__ import annotations

import json
import re
import subprocess
import time
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
    ProgressFn,
    ReflectionSummary,
    ReviewDecision,
    _noop_progress,
)


# ---------------------------------------------------------------------------
# Utilities (previously in agentic_execution.py)
# ---------------------------------------------------------------------------

STEP_PROGRESS_CHECKPOINTS = (10.0, 30.0, 60.0, 120.0)


def _ensure_within_repo(path: Path, repo_root: Path) -> Path:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    resolved_path.relative_to(resolved_root)
    return resolved_path


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

    while True:
        elapsed = time.monotonic() - started_at
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=timeout_seconds,
                output=stdout,
                stderr=stderr,
            )
        try:
            stdout, stderr = process.communicate(timeout=min(2.0, remaining))
            return subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started_at
            if progress_fn is not None and elapsed >= 120 and int(elapsed) % 120 < 3:
                progress_fn("waiting", f"Still running ({int(elapsed)}s)...")


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


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format, used by litellm)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository. Returns the file content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative file path to read",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to return (default 300)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the repo matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.csv'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents for a regex pattern. Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob to restrict which files to search (e.g. '*.py')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the repo. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative file path to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the repo root. Returns stdout, stderr, and exit code. "
                "Use this to install dependencies, run scripts, train models, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Log your reasoning. Call this before major actions to explain "
                "what you observed, what you plan to do next, and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your current thinking and plan",
                    },
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_metrics",
            "description": (
                "Report experiment metric results. Call this when you have trained/evaluated "
                "a model and have numeric results. This signals the experiment is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "object",
                        "description": 'Mapping of metric name to numeric value, e.g. {"rmse": 0.42, "r2": 0.87}',
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief description of what was done and what the results mean",
                    },
                },
                "required": ["metrics"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

_MAX_GLOB_RESULTS = 80
_MAX_SEARCH_RESULTS = 60


def _execute_read_file(args: dict[str, Any], repo_root: Path, max_chars: int) -> str:
    path_str = args.get("path", "")
    max_lines = int(args.get("max_lines", 10000))
    try:
        target = _ensure_within_repo(repo_root / path_str, repo_root)
    except (ValueError, OSError):
        return f"Error: path is outside repo root: {path_str}"
    if not target.exists():
        return f"Error: file not found: {path_str}"
    if not target.is_file():
        return f"Error: not a file: {path_str}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        truncated = "".join(lines[:max_lines])
        return truncated[:max_chars] + f"\n... ({len(lines) - max_lines} more lines)"
    return text[:max_chars]


def _execute_list_files(args: dict[str, Any], repo_root: Path) -> str:
    pattern = args.get("pattern", "**/*")
    try:
        matches = sorted(repo_root.glob(pattern))
    except Exception as exc:
        return f"Error: invalid glob pattern: {exc}"
    # Filter to files only, exclude hidden/build dirs
    files = [
        str(m.relative_to(repo_root))
        for m in matches
        if m.is_file()
        and not any(
            part.startswith(".") or part in {"__pycache__", "node_modules"}
            for part in m.relative_to(repo_root).parts
        )
    ]
    if not files:
        return "No files matched the pattern."
    if len(files) > _MAX_GLOB_RESULTS:
        return (
            "\n".join(files[:_MAX_GLOB_RESULTS])
            + f"\n... ({len(files) - _MAX_GLOB_RESULTS} more)"
        )
    return "\n".join(files)


def _execute_search_files(args: dict[str, Any], repo_root: Path, max_chars: int) -> str:
    pattern_str = args.get("pattern", "")
    glob_filter = args.get("glob", "**/*")
    try:
        regex = re.compile(pattern_str, re.IGNORECASE)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
    results: list[str] = []
    try:
        candidates = sorted(repo_root.glob(glob_filter))
    except Exception as exc:
        return f"Error: invalid glob: {exc}"
    for path in candidates:
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_root))
        if any(
            part.startswith(".") or part in {"__pycache__", "node_modules"}
            for part in path.relative_to(repo_root).parts
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{rel}:{lineno}: {line.rstrip()}")
                if len(results) >= _MAX_SEARCH_RESULTS:
                    break
        if len(results) >= _MAX_SEARCH_RESULTS:
            break
    if not results:
        return "No matches found."
    output = "\n".join(results)
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"
    return output


def _execute_write_file(args: dict[str, Any], repo_root: Path) -> str:
    path_str = args.get("path", "")
    content = args.get("content", "")
    try:
        target = _ensure_within_repo(repo_root / path_str, repo_root)
    except Exception:
        return f"Error: path is outside repo root: {path_str}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} bytes to {path_str}"


def _execute_run_command(
    args: dict[str, Any],
    repo_root: Path,
    default_timeout: int,
    max_chars: int,
    progress_fn: ProgressFn,
    turn: int,
) -> str:
    command = args.get("command", "")
    timeout = int(args.get("timeout", default_timeout))
    try:
        completed = _run_subprocess_with_progress(
            command=command,
            cwd=repo_root,
            shell=True,
            timeout_seconds=timeout,
            progress_fn=progress_fn,
            step_index=turn,
            total_steps=0,
            step_description=command[:80],
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except PermissionError as exc:
        return f"Error: permission denied: {exc}"
    stdout = completed.stdout[-max_chars:] if completed.stdout else ""
    stderr = completed.stderr[-max_chars:] if completed.stderr else ""
    parts = [f"exit_code={completed.returncode}"]
    if stdout.strip():
        parts.append(f"stdout:\n{stdout}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)


def _execute_report_metrics(
    args: dict[str, Any], spec: ExperimentSpec
) -> tuple[str, dict[str, MetricResult], float | None]:
    raw_metrics = args.get("metrics", {})
    summary = args.get("summary", "")
    metric_results: dict[str, MetricResult] = {}
    primary_value: float | None = None

    known = {
        m.name: m
        for m in (spec.primary_metric, *spec.secondary_metrics, *spec.guardrail_metrics)
    }
    for name, raw_value in raw_metrics.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not isfinite(value):
            continue
        metric_spec = known.get(name)
        metric_results[name] = MetricResult(
            name=name,
            value=value,
            scorer_ref=metric_spec.scorer_ref if metric_spec else None,
        )
        if name == spec.primary_metric.name:
            primary_value = value

    response = f"Metrics recorded: {json.dumps(raw_metrics)}"
    if summary:
        response += f"\nSummary: {summary}"
    return response, metric_results, primary_value


# ---------------------------------------------------------------------------
# Experiment history formatting
# ---------------------------------------------------------------------------


def _format_iteration_history(
    snapshot: MemorySnapshot,
    *,
    full_detail_count: int = 3,
) -> str:
    """Format experiment history: one-line summaries of all, detail on recent."""
    spec = snapshot.effective_spec
    records = snapshot.recent_records
    if not records:
        return ""

    parts = ["=== EXPERIMENT HISTORY ==="]

    # One-line summary of ALL iterations
    parts.append("All iterations:")
    for record in records:
        metric_val = "no metrics"
        if record.outcome.primary_metric_value is not None:
            metric_val = (
                f"{spec.primary_metric.name}={record.outcome.primary_metric_value:.4g}"
            )
        elif record.outcome.metric_results:
            primary = record.outcome.metric_results.get(spec.primary_metric.name)
            if primary and primary.value is not None:
                metric_val = f"{spec.primary_metric.name}={primary.value:.4g}"
        status = record.review.status if record.review else record.outcome.status
        hypothesis = record.candidate.hypothesis[:80]
        parts.append(
            f"  #{record.iteration_id}: {hypothesis} | {metric_val} | {status}"
        )

    # Full detail on recent iterations
    recent = records[-full_detail_count:]
    if recent:
        parts.append("")
        parts.append("Recent iteration details:")
        for record in recent:
            parts.append(f"  --- Iteration {record.iteration_id} ---")
            parts.append(f"  Hypothesis: {record.candidate.hypothesis}")
            parts.append(f"  Outcome: {record.outcome.status}")
            # Metrics
            resolved = record.outcome.resolved_metric_results(spec)
            for name, result in resolved.items():
                if result.value is not None:
                    parts.append(f"    {name} = {result.value:.4g}")
            # Failure info
            if record.outcome.failure_summary:
                parts.append(f"  Failure: {record.outcome.failure_summary[:300]}")
            # Lessons from reflection
            if record.reflection and record.reflection.lessons:
                for lesson in record.reflection.lessons[:3]:
                    parts.append(f"  Lesson: {lesson}")
            # Reviewer's recommendation
            if record.reflection and record.reflection.recommended_next_action:
                parts.append(
                    f"  Reviewer suggested: {record.reflection.recommended_next_action}"
                )
            # Files changed
            if record.outcome.code_or_config_changes:
                parts.append(
                    f"  Files changed: {', '.join(record.outcome.code_or_config_changes[:5])}"
                )

    parts.append("=== END HISTORY ===")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _build_system_prompt(
    candidate: ExperimentCandidate,
    snapshot: MemorySnapshot,
) -> str:
    spec = snapshot.effective_spec
    env = snapshot.capability_context.environment_facts

    guardrails = ", ".join(m.name for m in spec.guardrail_metrics) or "none"
    secondary = ", ".join(m.name for m in spec.secondary_metrics) or "none"

    parts = [
        "You are an ML experiment worker with interactive access to a code repository.",
        "",
        "=== USER'S INSTRUCTION (this is what they asked for — follow it precisely) ===",
        spec.objective,
        "=============================================================================",
        "",
        f"Primary metric: {spec.primary_metric.name} (goal: {spec.primary_metric.goal})",
        f"Secondary metrics: {secondary}",
        f"Guardrail metrics: {guardrails}",
        "",
        f"Repo root: {env.get('repo_root', '.')}",
        f"Python: {env.get('python_executable', 'python')}",
        "",
        "IMPORTANT: The bootstrap handoff below may contain environment verification results.",
        "If the handoff shows a baseline script ERROR, do NOT run that same script — read the error and adapt.",
        "If the handoff shows successful baseline output with metrics, call report_metrics immediately.",
        "",
        "Rules:",
        "- Call think() before major actions to explain your reasoning and plan.",
        "- ONE experiment idea per iteration. Fix small errors but do not redesign the whole script.",
        "- If after a few fixes the core approach is wrong, call report_metrics with what you learned and stop.",
        "- The outer loop gives you more iterations with lessons from this one.",
        "- NEVER use bare `pip install`. Use `uv pip install` or `uv sync` for dependencies.",
        "- Do not install new packages unless absolutely necessary — the bootstrapper already verified deps.",
    ]

    # Phase-specific guidance: baseline first, then improve
    if snapshot.best_summary is None and not snapshot.recent_records:
        parts.append("")
        parts.append("PHASE: BASELINE (no metrics exist yet)")
        parts.append(
            "Your ONLY job this iteration is to get the EXISTING model's baseline metric."
        )
        parts.append(
            "Do NOT add features, change the model, or try to improve anything yet."
        )
        parts.append("")
        parts.append(
            "The bootstrap handoff below already contains the baseline script path and possibly its output."
        )
        parts.append(
            "READ IT FIRST. If it contains metric values, call report_metrics immediately."
        )
        parts.append(
            "If it contains a baseline script path, run that script AS-IS — do not rewrite it."
        )
        parts.append("Only search the repo if the handoff doesn't answer the question.")
        parts.append("")
        parts.append(
            "The next iteration will focus on improvements — this one just establishes the number to beat."
        )
    elif snapshot.best_summary is not None:
        parts.append("")
        parts.append("PHASE: IMPROVE (baseline exists)")
        parts.append(
            f"Best result so far: {spec.primary_metric.name} = {snapshot.best_summary.primary_metric_value} (iteration {snapshot.best_summary.iteration_id})"
        )
        parts.append(
            "Your job: make ONE focused change to beat that number, then report the new metric."
        )
    else:
        parts.append("")
        parts.append("PHASE: RETRY (prior attempts failed to produce metrics)")
        parts.append(
            "Focus on getting a metric out — even the baseline. Read prior errors and fix them."
        )

    parts.append("")
    parts.append("Your workflow:")
    parts.append("1. Read the repo to find the existing model/experiment code")
    parts.append("2. Write or adapt ONE script")
    parts.append("3. Run it (use timeout=300 for training)")
    parts.append("4. Fix obvious errors if needed and retry")
    parts.append("5. After getting metrics, do quick post-prediction analysis:")
    parts.append(
        "   - Add feature importance to your script (e.g. model.feature_importances_)"
    )
    parts.append("   - Check where the model fails worst (error by segment/category)")
    parts.append("   - Note any patterns in residuals")
    parts.append(
        "6. Call report_metrics with numeric results AND your analysis findings in the summary"
    )
    parts.append("   The reviewer uses your findings to decide what to try next.")

    # Add context from bootstrap handoff / experiment guide if present
    for note in snapshot.markdown_memory:
        if note.path.endswith(
            (
                "bootstrap_handoff.md",
                "experiment_guide.md",
                "execution_runbook.md",
            )
        ):
            parts.append("")
            parts.append(f"--- {note.path} ---")
            content = note.content
            parts.append(content)

    # Add experiment history
    history = _format_iteration_history(snapshot)
    if history:
        parts.append("")
        parts.append(history)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main executor class
# ---------------------------------------------------------------------------


class ToolUseExecutor:
    """Executes experiments via an interactive LLM tool-use loop.

    The model can read files, write files, run commands, and report metrics.
    Replaces the static GenericExecutionPlanExecutor.
    """

    def __init__(
        self,
        *,
        model: str,
        repo_root: Path | str,
        max_errors: int = 15,
        max_tool_calls: int = 200,
        default_timeout_seconds: int = 300,
        max_captured_chars: int = 100_000,
        temperature: float = 0.2,
        progress_fn: ProgressFn | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.repo_root = Path(repo_root)
        self.max_errors = max_errors
        self.max_tool_calls = max_tool_calls
        self.default_timeout_seconds = default_timeout_seconds
        self.max_captured_chars = max_captured_chars
        self.temperature = temperature
        self.progress_fn: ProgressFn = progress_fn or _noop_progress
        self.extra_kwargs: dict[str, Any] = extra_kwargs or {}

    # -- WorkerBackend protocol ------------------------------------------

    def propose_next_experiment(self, snapshot: MemorySnapshot) -> ExperimentCandidate:
        """Return a minimal candidate — the real work happens in execute()."""
        iteration = snapshot.next_iteration_id
        if snapshot.best_summary is not None:
            hypothesis = f"Iteration {iteration}: improve on best {snapshot.effective_spec.primary_metric.name} = {snapshot.best_summary.primary_metric_value}"
        elif iteration == 1:
            hypothesis = f"Iteration {iteration}: establish baseline for {snapshot.effective_spec.primary_metric.name}"
        else:
            hypothesis = f"Iteration {iteration}: continue experimenting on {snapshot.effective_spec.primary_metric.name}"
        return ExperimentCandidate(
            hypothesis=hypothesis,
            action_type="run_experiment",
            change_type="interactive",
            instructions="Interactive agent will read the repo, write code, and run experiments.",
            execution_steps=[
                ExecutionStep(kind="shell", command="interactive tool-use agent")
            ],
            metadata={"interactive_agent": True},
        )

    def continue_experiment(
        self,
        snapshot: MemorySnapshot,
        previous_candidate: ExperimentCandidate,
        previous_outcome: ExperimentOutcome,
    ) -> ExperimentCandidate:
        """Return a minimal candidate for the executor to handle."""
        return self.propose_next_experiment(snapshot)

    # -- ActionExecutor protocol -------------------------------------------

    def execute(
        self, candidate: ExperimentCandidate, snapshot: MemorySnapshot
    ) -> ExperimentOutcome:
        """Run an interactive tool-use loop to execute the experiment."""
        from litellm import completion as litellm_completion

        spec = snapshot.effective_spec
        system_prompt = _build_system_prompt(candidate, snapshot)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Begin the experiment. Hypothesis: {candidate.hypothesis}",
            },
        ]

        # State accumulated during the loop
        reported_metrics: dict[str, MetricResult] = {}
        reported_primary_value: float | None = None
        command_outputs: list[str] = []  # stdout from run_command calls
        tool_call_log: list[dict[str, Any]] = []
        files_written: list[str] = []
        metrics_reported = False
        reported_summary = ""
        error_count = 0
        turn = 0

        self.progress_fn(
            "tool_use_start", f"[{self.model}] Starting interactive execution..."
        )

        while turn < self.max_tool_calls:
            turn += 1
            try:
                response = litellm_completion(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=self.temperature,
                    **self.extra_kwargs,
                )
            except KeyboardInterrupt:
                self.progress_fn("interrupted", "Interrupted by user.")
                return self._build_outcome(
                    spec=spec,
                    reported_metrics=reported_metrics,
                    reported_primary_value=reported_primary_value,
                    reported_summary=reported_summary,
                    command_outputs=command_outputs,
                    tool_call_log=tool_call_log,
                    files_written=files_written,
                    metrics_reported=metrics_reported,
                    turns_used=turn,
                )
            except Exception as exc:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    notes=[f"LLM call failed on turn {turn}: {exc}"],
                    failure_type="LLMCallFailed",
                    failure_summary=str(exc),
                    recoverable=True,
                    recovery_actions=[
                        "Retry the iteration or check model availability."
                    ],
                    execution_details={"tool_call_log": tool_call_log, "turn": turn},
                )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Show the agent's reasoning if it produced text
            if message.content and message.content.strip():
                reasoning = message.content.strip()
                if len(reasoning) > 500:
                    reasoning = reasoning[:500] + "..."
                self.progress_fn(f"agent_thought_{turn}", f"[{self.model}] {reasoning}")

            # Append assistant message to conversation
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if message.content:
                assistant_msg["content"] = message.content
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # If no tool calls, model is done
            if not message.tool_calls:
                if finish_reason == "stop":
                    break
                # Some models return "length" if max_tokens hit
                break

            # Execute each tool call
            interrupted = False
            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                self.progress_fn(
                    f"tool_{turn}",
                    f"[{self.model}] {tool_name}({_brief_args(tool_args)})",
                )

                try:
                    tool_result = self._dispatch_tool(tool_name, tool_args, spec, turn)
                except KeyboardInterrupt:
                    self.progress_fn(
                        "interrupted", "Interrupted by user during tool execution."
                    )
                    interrupted = True
                    break

                tool_call_log.append(
                    {
                        "turn": turn,
                        "tool": tool_name,
                        "args": tool_args,
                        "result_preview": tool_result[:500],
                    }
                )

                # Track side effects
                if tool_name == "run_command":
                    command_outputs.append(tool_result)
                    if tool_result.startswith(
                        "exit_code="
                    ) and not tool_result.startswith("exit_code=0"):
                        error_count += 1
                        if error_count >= self.max_errors:
                            self.progress_fn(
                                "error_limit",
                                f"Hit {self.max_errors} failed commands — stopping.",
                            )
                            interrupted = True
                            break
                elif tool_name == "write_file":
                    files_written.append(tool_args.get("path", ""))
                elif tool_name == "report_metrics":
                    _, metrics, primary = _execute_report_metrics(tool_args, spec)
                    reported_metrics.update(metrics)
                    if primary is not None:
                        reported_primary_value = primary
                    reported_summary = tool_args.get("summary", "")
                    metrics_reported = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

            # If interrupted or metrics reported, stop the loop
            if interrupted or metrics_reported:
                break

        # Build outcome
        return self._build_outcome(
            spec=spec,
            reported_metrics=reported_metrics,
            reported_primary_value=reported_primary_value,
            reported_summary=reported_summary,
            command_outputs=command_outputs,
            tool_call_log=tool_call_log,
            files_written=files_written,
            metrics_reported=metrics_reported,
            turns_used=turn,
        )

    def _dispatch_tool(
        self,
        name: str,
        args: dict[str, Any],
        spec: ExperimentSpec,
        turn: int,
    ) -> str:
        if name == "read_file":
            return _execute_read_file(args, self.repo_root, self.max_captured_chars)
        if name == "list_files":
            return _execute_list_files(args, self.repo_root)
        if name == "search_files":
            return _execute_search_files(args, self.repo_root, self.max_captured_chars)
        if name == "write_file":
            return _execute_write_file(args, self.repo_root)
        if name == "run_command":
            return _execute_run_command(
                args,
                self.repo_root,
                self.default_timeout_seconds,
                self.max_captured_chars,
                self.progress_fn,
                turn,
            )
        if name == "think":
            return "OK"
        if name == "report_metrics":
            response, _, _ = _execute_report_metrics(args, spec)
            return response
        return f"Error: unknown tool '{name}'"

    def _build_outcome(
        self,
        *,
        spec: ExperimentSpec,
        reported_metrics: dict[str, MetricResult],
        reported_primary_value: float | None,
        reported_summary: str = "",
        command_outputs: list[str],
        tool_call_log: list[dict[str, Any]],
        files_written: list[str],
        metrics_reported: bool,
        turns_used: int,
    ) -> ExperimentOutcome:
        # If metrics were explicitly reported, use those
        metric_results = dict(reported_metrics)
        primary_value = reported_primary_value

        # Fallback: try to extract metrics from command stdout
        if not metric_results:
            for output in reversed(command_outputs):
                parsed = _extract_metric_payload_from_text(output, spec)
                if parsed["metric_results"]:
                    metric_results.update(parsed["metric_results"])
                    if parsed["primary_metric_value"] is not None:
                        primary_value = parsed["primary_metric_value"]
                    break

        # Classify secondary/guardrail
        secondary_names = {m.name for m in spec.secondary_metrics}
        guardrail_names = {m.name for m in spec.guardrail_metrics}
        secondary_metrics = {
            name: r.value
            for name, r in metric_results.items()
            if name in secondary_names and r.value is not None
        }
        guardrail_metrics = {
            name: r.value
            for name, r in metric_results.items()
            if name in guardrail_names and r.value is not None
        }

        has_metrics = bool(metric_results) or primary_value is not None

        if has_metrics:
            return ExperimentOutcome(
                status="success",
                metric_results=metric_results,
                primary_metric_value=primary_value,
                secondary_metrics=secondary_metrics,
                guardrail_metrics=guardrail_metrics,
                notes=[
                    f"Interactive execution completed in {turns_used} turns.",
                    f"Metrics reported via {'report_metrics tool' if metrics_reported else 'stdout parsing'}.",
                ]
                + ([f"Executor analysis: {reported_summary}"] if reported_summary else []),
                code_or_config_changes=files_written,
                execution_details={
                    "tool_call_log": tool_call_log,
                    "turns_used": turns_used,
                },
            )

        # No metrics produced
        return ExperimentOutcome(
            status="recoverable_failure",
            notes=[
                f"Interactive execution used {turns_used} turns but did not produce metrics.",
                "The agent may have explored the repo or encountered errors without completing the experiment.",
            ],
            failure_type="MetricsNotReported",
            failure_summary=(
                f"The agent completed {turns_used} tool-use turns but did not call "
                f"report_metrics or print parseable values for {spec.primary_metric.name!r}."
            ),
            recoverable=True,
            recovery_actions=[
                "The next iteration should focus on actually running the model and reporting results.",
            ],
            code_or_config_changes=files_written,
            execution_details={
                "tool_call_log": tool_call_log,
                "turns_used": turns_used,
            },
        )


def _brief_args(args: dict[str, Any], limit: int = 300) -> str:
    """Short preview of tool arguments for progress display."""
    if "reasoning" in args:
        r = args["reasoning"]
        return r[:limit] + "..." if len(r) > limit else r
    if "path" in args:
        return args["path"]
    if "command" in args:
        cmd = args["command"]
        return cmd[:limit] + "..." if len(cmd) > limit else cmd
    if "pattern" in args:
        return args["pattern"]
    if "metrics" in args:
        return ", ".join(f"{k}={v}" for k, v in list(args["metrics"].items())[:3])
    return str(args)[:limit]


# ---------------------------------------------------------------------------
# Shared tool-use loop
# ---------------------------------------------------------------------------


def _run_tool_loop(
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict[str, Any]],
    tool_dispatcher,
    stop_tool: str,
    max_errors: int = 10,
    max_tool_calls: int = 100,
    temperature: float = 0.2,
    progress_fn: ProgressFn,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generic tool-use agentic loop. Returns the stop_tool's arguments when called."""
    from litellm import completion as litellm_completion

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    error_count = 0
    turn = 0
    result: dict[str, Any] = {}

    while turn < max_tool_calls:
        turn += 1
        try:
            response = litellm_completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                **(extra_kwargs or {}),
            )
        except KeyboardInterrupt:
            progress_fn("interrupted", "Interrupted by user.")
            return result
        except Exception as exc:
            progress_fn("error", f"LLM call failed: {exc}")
            return result

        message = response.choices[0].message

        # Show reasoning
        if message.content and message.content.strip():
            reasoning = message.content.strip()
            if len(reasoning) > 500:
                reasoning = reasoning[:500] + "..."
            progress_fn(f"agent_thought_{turn}", f"[{model}] {reasoning}")

        # Build assistant message
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            # Model responded with text instead of calling a tool.
            # If we haven't got the stop_tool result yet, nudge it to call it.
            if not result and turn < max_tool_calls - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Please call {stop_tool} now with the information you have.",
                    }
                )
                continue
            break

        # Execute tools
        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            progress_fn(
                f"tool_{turn}", f"[{model}] {tool_name}({_brief_args(tool_args)})"
            )

            # Check if this is the stop tool
            if tool_name == stop_tool:
                result = tool_args
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": "OK"}
                )
                return result

            try:
                tool_result = tool_dispatcher(tool_name, tool_args, turn)
            except KeyboardInterrupt:
                progress_fn("interrupted", "Interrupted by user.")
                return result

            if (
                tool_name == "run_command"
                and tool_result.startswith("exit_code=")
                and not tool_result.startswith("exit_code=0")
            ):
                error_count += 1
                if error_count >= max_errors:
                    progress_fn("error_limit", f"Hit {max_errors} failed commands.")
                    return result

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
            )

    return result


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------

PLANNER_TOOLS = [
    t
    for t in TOOLS
    if t["function"]["name"]
    in ("read_file", "list_files", "search_files", "think", "run_command")
] + [
    {
        "type": "function",
        "function": {
            "name": "fill_contract",
            "description": (
                "Fill out the execution contract. Call this when you understand the codebase well enough. "
                "Every field you can determine should be filled. Fields you're uncertain about, set to null."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_script": {
                        "type": "string",
                        "description": "File path the user referenced",
                    },
                    "baseline_function": {
                        "type": "string",
                        "description": "Function/class that trains the baseline model",
                    },
                    "data_loading": {
                        "type": "string",
                        "description": "How training data is loaded (function, file path, pipeline description)",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Column/variable being predicted",
                    },
                    "primary_metric": {
                        "type": "string",
                        "description": "Primary metric name",
                    },
                    "primary_metric_goal": {
                        "type": "string",
                        "enum": ["minimize", "maximize"],
                        "description": "Whether to minimize or maximize the metric",
                    },
                    "guardrail_metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "goal": {
                                    "type": "string",
                                    "enum": ["minimize", "maximize"],
                                },
                            },
                            "required": ["name", "goal"],
                        },
                        "description": "Guardrail metrics with name and goal",
                    },
                    "secondary_metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "goal": {
                                    "type": "string",
                                    "enum": ["minimize", "maximize"],
                                },
                            },
                            "required": ["name", "goal"],
                        },
                        "description": "Secondary metrics to track with name and goal",
                    },
                    "baseline_value": {
                        "type": "number",
                        "description": "Known baseline metric value if found",
                    },
                },
                "required": [
                    "source_script",
                    "baseline_function",
                    "data_loading",
                    "target_column",
                    "primary_metric",
                    "primary_metric_goal",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a question when you're genuinely uncertain about something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you need to ask this",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


class ToolUsePlanner:
    """Plans experiments by reading the repo and filling out an execution contract."""

    def __init__(
        self,
        *,
        model: str,
        repo_root: Path | str,
        temperature: float = 0.2,
        progress_fn: ProgressFn | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.repo_root = Path(repo_root)
        self.temperature = temperature
        self.progress_fn: ProgressFn = progress_fn or _noop_progress
        self.extra_kwargs: dict[str, Any] = extra_kwargs or {}

    def plan(
        self,
        *,
        user_goal: str,
        source_file_hint: str | None = None,
        answers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read the repo and fill out the execution contract.

        Returns dict with:
        - contract: dict of filled contract fields
        - questions: list of questions for the user
        """
        system_prompt = (
            "You are an experiment planner. The user wants to improve an ML model.\n"
            "Your job: read the source code, understand the data flow, and fill out "
            "the execution contract by calling fill_contract.\n\n"
            "Guidelines:\n"
            "- Call think() to reason about what you find\n"
            "- Read the file the user mentioned FIRST\n"
            "- Trace the data loading: what function loads data? what does it return?\n"
            "- Identify the baseline model function\n"
            "- Identify the target column and metrics\n"
            "- Think about ITERATION EFFICIENCY: if the model depends on slow upstream steps "
            "(other models, preprocessing, feature generation), consider how the executor can "
            "avoid re-running them every iteration. E.g. save upstream outputs to a file first, "
            "or split into a setup step and an iteration step. Include your recommendation in "
            "the fill_contract data_loading field.\n"
            "- If you're genuinely uncertain about something, call ask_user()\n"
            "- When you have enough info, call fill_contract with all fields\n"
            f"\nRepo root: {self.repo_root}\n"
        )

        user_msg_parts = [f"User goal: {user_goal}"]
        if source_file_hint:
            user_msg_parts.append(
                f"\nI found this file in the repo matching your description: {source_file_hint}"
            )
            user_msg_parts.append(
                "Start by reading it to understand the model training code."
            )
        if answers:
            user_msg_parts.append(
                f"\nUser has already answered these questions: {json.dumps(answers, indent=2)}"
            )

        contract: dict[str, Any] = {}
        questions: list[dict[str, str]] = []

        def dispatch(name: str, args: dict[str, Any], turn: int) -> str:
            if name == "read_file":
                return _execute_read_file(args, self.repo_root, 100_000)
            if name == "list_files":
                return _execute_list_files(args, self.repo_root)
            if name == "search_files":
                return _execute_search_files(args, self.repo_root, 100_000)
            if name == "think":
                return "OK"
            if name == "run_command":
                return _execute_run_command(
                    args,
                    self.repo_root,
                    30,
                    100_000,
                    self.progress_fn,
                    turn,
                )
            if name == "ask_user":
                questions.append(
                    {
                        "question": args.get("question", ""),
                        "reason": args.get("reason", ""),
                    }
                )
                return "Question recorded. Continue analyzing or call fill_contract."
            return f"Unknown tool: {name}"

        result = _run_tool_loop(
            model=self.model,
            system_prompt=system_prompt,
            user_message="\n".join(user_msg_parts),
            tools=PLANNER_TOOLS,
            tool_dispatcher=dispatch,
            stop_tool="fill_contract",
            max_tool_calls=50,
            temperature=self.temperature,
            progress_fn=self.progress_fn,
            extra_kwargs=self.extra_kwargs,
        )

        # result is the fill_contract args (or empty if loop ended without calling it)
        contract = result if result else {}

        return {
            "contract": contract,
            "questions": questions,
        }


# ---------------------------------------------------------------------------
# Reviewer agent
# ---------------------------------------------------------------------------

REVIEWER_TOOLS = [
    t for t in TOOLS if t["function"]["name"] in ("read_file", "search_files", "think")
] + [
    {
        "type": "function",
        "function": {
            "name": "report_review",
            "description": "Report your review of this experiment iteration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["accepted", "rejected"],
                        "description": "Accept into memory or reject",
                    },
                    "reason": {"type": "string", "description": "Why this decision"},
                    "lessons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What was learned from this iteration",
                    },
                    "next_experiment": {
                        "type": "string",
                        "description": "What to try in the next iteration",
                    },
                },
                "required": ["status", "reason", "lessons", "next_experiment"],
            },
        },
    },
]


class ToolUseReviewer:
    """Reviews experiment results: analyzes, extracts lessons, decides accept/reject, proposes next."""

    def __init__(
        self,
        *,
        model: str,
        repo_root: Path | str,
        temperature: float = 0.2,
        progress_fn: ProgressFn | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.repo_root = Path(repo_root)
        self.temperature = temperature
        self.progress_fn: ProgressFn = progress_fn or _noop_progress
        self.extra_kwargs: dict[str, Any] = extra_kwargs or {}

    def review(
        self,
        snapshot: MemorySnapshot,
        candidate: ExperimentCandidate,
        outcome: ExperimentOutcome,
    ) -> tuple[ReflectionSummary, ReviewDecision]:
        """Analyze the iteration and return both reflection and review."""
        spec = snapshot.effective_spec

        # Build context for the reviewer
        metrics_str = ""
        resolved = outcome.resolved_metric_results(spec)
        for name, result in resolved.items():
            if result.value is not None:
                metrics_str += f"  {name} = {result.value:.4g}\n"
        if not metrics_str and outcome.primary_metric_value is not None:
            metrics_str = (
                f"  {spec.primary_metric.name} = {outcome.primary_metric_value:.4g}\n"
            )

        best_str = "None yet"
        if (
            snapshot.best_summary is not None
            and snapshot.best_summary.primary_metric_value is not None
        ):
            best_str = f"{spec.primary_metric.name} = {snapshot.best_summary.primary_metric_value:.4g}"

        system_prompt = (
            "You are an expert data scientist reviewing ML experiments. Your job:\n"
            "1. Evaluate whether the experiment produced valid, trustworthy metrics\n"
            "2. Check for data science red flags (leakage, bad splits, unrealistic results)\n"
            "3. Compare metrics against experiment history\n"
            "4. Propose a SPECIFIC, ACTIONABLE next experiment\n\n"
            "ACCEPT if: experiment produced valid metrics with sound methodology.\n"
            "REJECT only for: data leakage, broken evaluation, unrealistic metrics, guardrail violations.\n\n"
            "CRITICAL — your next_experiment recommendation drives the next iteration.\n"
            "Be SPECIFIC. Not 'add features' but 'try adding position as a categorical feature "
            "because the current model treats all positions equally but kills distribution varies "
            "significantly by role (ADC vs Support).'\n"
            "Read the experiment script to understand what features exist, what the model does, "
            "and reason about what specific change would most likely improve the metric.\n\n"
            "DIAGNOSTIC THINKING: Think like a domain expert reviewing this model.\n"
            "- Where might this model struggle? (certain segments, categories, time periods, edge cases)\n"
            "- What biases could exist in the predictions?\n"
            "- What segmented metrics would reveal blind spots?\n"
            "Your next_experiment can ask the executor to add DIAGNOSTIC METRICS — the primary metric "
            "broken down by relevant categories in the data. Once you see segmented results, "
            "you can make informed feature engineering suggestions based on actual weaknesses.\n\n"
            "Call think() to reason deeply, then call report_review.\n"
        )

        # Build comprehensive context
        metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
        user_parts = [
            f"Objective: {spec.objective}",
            f"Primary metric: {spec.primary_metric.name} (goal: {spec.primary_metric.goal})",
            f"Best so far: {best_str}",
        ]
        # Include planner contract so reviewer's recommendations are consistent
        if metadata.get("source_script"):
            user_parts.append(f"Source script: {metadata['source_script']}")
        if metadata.get("data_loading"):
            user_parts.append(f"Data pipeline: {metadata['data_loading']}")
        if metadata.get("baseline_function"):
            user_parts.append(f"Baseline function: {metadata['baseline_function']}")
        user_parts.extend(
            [
                "",
                f"CURRENT ITERATION ({snapshot.next_iteration_id}):",
                f"  Hypothesis: {candidate.hypothesis}",
                f"  Status: {outcome.status}",
                f"  Metrics:\n{metrics_str or '  (none reported)'}",
            ]
        )
        if outcome.failure_summary:
            user_parts.append(f"  Failure: {outcome.failure_summary[:500]}")
        if outcome.notes:
            for note in outcome.notes[:5]:
                if note.startswith("Executor analysis:"):
                    user_parts.append(f"  {note}")
                else:
                    user_parts.append(f"  Note: {note}")
        if outcome.code_or_config_changes:
            user_parts.append(
                f"  Files changed: {', '.join(outcome.code_or_config_changes[:5])}"
            )

        # Add full experiment history
        history = _format_iteration_history(snapshot)
        if history:
            user_parts.append("")
            user_parts.append(history)

        user_message = "\n".join(user_parts)

        def dispatch(name: str, args: dict[str, Any], turn: int) -> str:
            if name == "read_file":
                return _execute_read_file(args, self.repo_root, 100_000)
            if name == "search_files":
                return _execute_search_files(args, self.repo_root, 100_000)
            if name == "think":
                return "OK"
            return f"Unknown tool: {name}"

        result = _run_tool_loop(
            model=self.model,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=REVIEWER_TOOLS,
            tool_dispatcher=dispatch,
            stop_tool="report_review",
            max_tool_calls=15,
            temperature=self.temperature,
            progress_fn=self.progress_fn,
            extra_kwargs=self.extra_kwargs,
        )

        # Parse result
        status = result.get(
            "status", "accepted" if outcome.status == "success" else "rejected"
        )
        reason = result.get("reason", "Review completed.")
        lessons = result.get("lessons", [])
        next_experiment = result.get("next_experiment")

        reflection = ReflectionSummary(
            assessment=reason,
            lessons=lessons,
            recommended_next_action=next_experiment,
        )
        review = ReviewDecision(
            status=status,
            reason=reason,
            should_update_memory=True,
        )
        return reflection, review
