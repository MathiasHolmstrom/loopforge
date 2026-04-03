from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from loopforge.core.types import ExecutionStep, ExperimentCandidate, ExperimentOutcome, MemorySnapshot


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


def _classify_shell_failure(stdout: str, stderr: str) -> tuple[str, bool, list[str]]:
    lower_error = f"{stdout}\n{stderr}".lower()
    if (
        "permissionerror" in lower_error
        and "access is denied" in lower_error
        and any(token in lower_error for token in ("socketpair", "multiprocessing", "_ssock", "_csock"))
    ):
        return (
            "MultiprocessingPermissionError",
            True,
            [
                "Retry with multiprocessing disabled or worker count reduced to 1.",
                "Switch to a serial execution path that avoids socketpair/resource-tracker setup if the repo supports it.",
            ],
        )
    if "can't open file" in lower_error or ("no such file or directory" in lower_error and ".py" in lower_error):
        return (
            "MissingScriptFile",
            True,
            [
                "Create the referenced script file before running it.",
                "Or replace the script invocation with inline Python if the script is only a temporary probe.",
            ],
        )
    blocked = "access is denied" in lower_error or "permission denied" in lower_error
    return (
        "ShellCommandFailed",
        not blocked,
        ([] if blocked else ["Inspect the error output, adjust the command or environment, and retry."]),
    )


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
                return False, index, f"Step {index} uses cwd outside the repo root: {step.cwd}"
            for script in _referenced_python_scripts(step.command):
                script_path = Path(script)
                candidate_path = (workdir / script_path) if not script_path.is_absolute() else script_path
                if not candidate_path.exists() and candidate_path.resolve() not in planned_files:
                    return False, index, (
                        f"Step {index} references {candidate_path} which does not exist "
                        f"and is not created by an earlier write_file step."
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
                notes=[f"Step {index} targeted a working directory outside the repo root."],
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
                    recovery_actions=["Provide a repo-relative path for the file step and retry."],
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
            step_results.append({
                "index": index,
                "kind": step.kind,
                "path": str(target_path),
                "bytes_written": len((step.content or "").encode("utf-8")),
            })
            planned_files.add(target_path.resolve())
            continue

        # Shell step — check for missing scripts first
        for script in _referenced_python_scripts(step.command):
            script_path = Path(script)
            candidate_path = (workdir / script_path) if not script_path.is_absolute() else script_path
            if not candidate_path.exists() and candidate_path.resolve() not in planned_files:
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
            step_results.append({
                "index": index, "kind": step.kind, "command": step.command,
                "cwd": str(workdir), "timeout_seconds": timeout_seconds, "status": "timeout",
            })
            return ExperimentOutcome(
                status="recoverable_failure",
                notes=[f"Step {index} timed out: {step.command}"],
                failure_type="TimeoutExpired",
                failure_summary=str(exc),
                recoverable=True,
                recovery_actions=["Shorten or narrow the command, then retry."],
                execution_details={"step_results": step_results},
            ), step_results

        stdout = completed.stdout[-max_captured_chars:]
        stderr = completed.stderr[-max_captured_chars:]
        step_results.append({
            "index": index, "kind": step.kind, "command": step.command,
            "cwd": str(workdir), "timeout_seconds": timeout_seconds,
            "returncode": completed.returncode, "stdout": stdout, "stderr": stderr,
        })

        if completed.returncode != 0 and not step.allow_failure:
            failure_type, recoverable, recovery_actions = _classify_shell_failure(stdout, stderr)
            return ExperimentOutcome(
                status="recoverable_failure" if recoverable else "blocked",
                notes=[f"Step {index} failed: {step.command}"],
                failure_type=failure_type,
                failure_summary=stderr.strip() or stdout.strip() or f"Exit code {completed.returncode}.",
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
        max_retries: int = 2,
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

    def execute(self, candidate: ExperimentCandidate, snapshot: MemorySnapshot) -> ExperimentOutcome:
        if not candidate.execution_steps:
            return ExperimentOutcome(
                status="recoverable_failure",
                notes=["No execution steps were provided for the agentic executor."],
                failure_type="MissingExecutionSteps",
                failure_summary="The worker proposed an agentic step without concrete execution steps.",
                recoverable=True,
                recovery_actions=["Propose concrete shell steps for the next iteration."],
                execution_details={"step_results": []},
            )

        current_steps = list(candidate.execution_steps)
        all_attempt_results: list[dict[str, Any]] = []

        for attempt in range(1 + self.max_retries):
            # Pre-validate before running
            valid, failed_idx, reason = _validate_steps_pre_execution(current_steps, self.repo_root)
            if not valid and self.fix_backend is not None and attempt < self.max_retries:
                self.progress_fn(
                    f"fix_attempt_{attempt}",
                    f"Pre-validation failed (step {failed_idx}): {reason}. Asking agent to fix...",
                )
                fixed_steps = self._ask_for_fix(
                    candidate, failed_idx, reason, [],
                )
                if fixed_steps:
                    current_steps = fixed_steps
                    continue
                # Fix backend returned nothing — fall through to try anyway

            # Execute
            if attempt > 0:
                self.progress_fn(f"retry_{attempt}", f"Retry attempt {attempt + 1}...")

            failure_outcome, step_results = _run_steps(
                current_steps, self.repo_root,
                self.default_timeout_seconds, self.max_captured_chars,
            )
            all_attempt_results.append({
                "attempt": attempt + 1,
                "steps": [s.to_dict() for s in current_steps],
                "step_results": step_results,
                "success": failure_outcome is None,
            })

            if failure_outcome is None:
                # Success!
                return ExperimentOutcome(
                    status="success",
                    notes=[f"Executed {len(current_steps)} step(s) successfully"
                           + (f" (after {attempt} fix(es))" if attempt > 0 else "") + "."],
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
                self.progress_fn(
                    f"fix_attempt_{attempt}",
                    f"Step failed: {failure_outcome.failure_summary[:100]}. Asking agent to fix...",
                )
                fixed_steps = self._ask_for_fix(
                    candidate,
                    len(step_results),
                    failure_outcome.failure_summary or "Unknown failure",
                    step_results,
                )
                if fixed_steps:
                    current_steps = fixed_steps
                    continue

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
        candidate: ExperimentCandidate,
        failed_step_index: int,
        failure_summary: str,
        step_results: list[dict[str, Any]],
    ) -> list[ExecutionStep] | None:
        try:
            fixed = self.fix_backend.fix_execution_plan(
                candidate=candidate,
                failed_step_index=failed_step_index,
                failure_summary=failure_summary,
                step_results=step_results,
            )
            return fixed if fixed else None
        except Exception:
            return None
