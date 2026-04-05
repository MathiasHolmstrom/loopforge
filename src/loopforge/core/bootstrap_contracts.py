from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from loopforge.core.types import (
    BootstrapTurn,
    CapabilityContext,
    ExperimentSpec,
    PreflightCheck,
)

TRANSIENT_BOOTSTRAP_ANSWER_KEYS = {"user_feedback", "discussion"}


EXISTING_FRAMEWORK_HINT_TOKENS = (
    "existing script",
    "existing framework",
    "existing pipeline",
    "existing model",
    "current script",
    "current framework",
    "current pipeline",
    "copy the script",
    "copy my script",
    "work from there",
    "work from the existing",
    "reuse the existing",
    "use the existing",
    "keep the existing",
)


def should_prepare_access_guide(
    *,
    capability_context: CapabilityContext,
    preflight_checks: list[PreflightCheck],
) -> bool:
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


def normalise_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped:
                result.append(stripped)
        return result
    return []


def _should_enforce_baseline_reuse(*parts: str) -> bool:
    combined = " ".join(part.strip().lower() for part in parts if part and part.strip())
    if not combined:
        return False
    if any(token in combined for token in EXISTING_FRAMEWORK_HINT_TOKENS):
        return True
    words = {token for token in re.findall(r"[a-z0-9_]+", combined) if token}
    mentions_existing_artifact = "existing" in words and bool(
        words & {"script", "framework", "pipeline", "model", "baseline"}
    )
    mentions_copying_existing = "copy" in words and bool(
        words & {"script", "model", "framework", "pipeline", "baseline", "paste"}
    )
    mentions_repo_baseline = "baseline" in words or (
        "exists" in words and "here" in words
    )
    return (
        mentions_existing_artifact
        or mentions_copying_existing
        or mentions_repo_baseline
    )


def apply_bootstrap_execution_contract(
    *,
    spec: ExperimentSpec,
    capability_context: CapabilityContext,
    user_goal: str,
    assistant_message: str,
    answers: dict[str, Any] | None,
    answer_summary: str,
) -> ExperimentSpec:
    metadata = dict(spec.metadata)
    operator_guidance = normalise_text_list(metadata.get("operator_guidance", []))
    baseline_paths = normalise_text_list(
        capability_context.environment_facts.get("baseline_code_paths", [])
    )
    metadata["execution_contract"] = {
        "baseline_paths": baseline_paths,
        "must_reference_baseline_paths": bool(baseline_paths)
        and _should_enforce_baseline_reuse(
            user_goal,
            assistant_message,
            answer_summary,
            " ".join(operator_guidance),
        ),
        "enforcement_scope": "until_first_successful_iteration",
    }
    return replace(spec, metadata=metadata)


def build_bootstrap_handoff(
    *,
    capability_context: CapabilityContext,
    turn: BootstrapTurn,
    answers: dict[str, Any] | None,
    env_verification: dict[str, Any] | None = None,
) -> str:
    spec = turn.proposal.recommended_spec
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    bootstrap_answers = dict(metadata.get("bootstrap_answers", {}))
    for key, value in (answers or {}).items():
        if str(key) in TRANSIENT_BOOTSTRAP_ANSWER_KEYS:
            continue
        if value not in (None, ""):
            bootstrap_answers[str(key)] = value
    operator_guidance = normalise_text_list(metadata.get("operator_guidance", []))
    planner_notes = [
        str(note).strip() for note in turn.proposal.notes if str(note).strip()
    ]
    capability_notes = [
        str(note).strip() for note in capability_context.notes if str(note).strip()
    ]
    baseline_paths = normalise_text_list(
        capability_context.environment_facts.get("baseline_code_paths", [])
    )
    # Prefer source_script from contract over auto-discovered baseline paths
    source_script = metadata.get("source_script")
    if source_script and source_script not in ("not specified",):
        baseline_paths = [source_script]

    lines = [
        "# Bootstrap Handoff",
        "",
        "## Planner Summary",
        turn.assistant_message.strip()
        or "Bootstrap completed without an explicit summary.",
        "",
        "## Binding Execution Intent",
        f"- Objective: {spec.objective}",
        f"- Primary metric: {spec.primary_metric.name} ({spec.primary_metric.goal})",
        f"- Source script: {metadata.get('source_script', 'not specified')}",
        f"- Baseline function: {metadata.get('baseline_function', 'not specified')}",
        f"- Data loading: {metadata.get('data_loading', 'not specified')}",
        f"- Target column: {metadata.get('target_column', 'not specified')}",
        "- Treat the current repo implementation as the default starting point.",
        "- Reuse the existing script, baseline path, and framework whenever the repo context identifies them.",
        "- Do not replace the baseline with a new pipeline unless a concrete execution failure proves the existing path cannot be used.",
    ]
    if baseline_paths:
        lines.extend(["", "## Baseline Code Paths"])
        lines.extend(f"- {path}" for path in baseline_paths[:12])
    if bootstrap_answers:
        lines.extend(["", "## Confirmed User Answers"])
        lines.extend(f"- {key}: {value}" for key, value in bootstrap_answers.items())
    if operator_guidance:
        lines.extend(["", "## Operator Guidance"])
        lines.extend(f"- {item}" for item in operator_guidance[-8:])
    if planner_notes:
        lines.extend(["", "## Planner Notes"])
        lines.extend(f"- {note}" for note in planner_notes[:16])
    if capability_notes:
        lines.extend(["", "## Repo Grounding"])
        lines.extend(f"- {note}" for note in capability_notes[:16])
    if env_verification:
        lines.extend(
            ["", "## Environment Verification (bootstrapper ran these checks)"]
        )
        if env_verification.get("deps_synced"):
            lines.append("- Dependencies: synced (uv sync succeeded)")
        else:
            lines.append("- Dependencies: NOT synced — use `uv sync` before running")
        if env_verification.get("baseline_script"):
            lines.append(f"- Baseline script: {env_verification['baseline_script']}")
        if env_verification.get("baseline_output"):
            output_preview = env_verification["baseline_output"].strip()[-2000:]
            lines.extend(
                [
                    "",
                    "### Baseline Script Output (last 2000 chars)",
                    "```",
                    output_preview,
                    "```",
                ]
            )
        for error in env_verification.get("errors", []):
            lines.append(f"- ERROR: {error}")
    return "\n".join(lines).rstrip() + "\n"


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
    probe_check = next(
        (
            check
            for check in preflight_checks
            if check.name == "generic_agentic_execution_probe"
        ),
        None,
    )
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
                f'- Preferred command shape: "{python_executable}" path\\to\\script.py',
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
                '- Prefer `write_file` + `"<python_executable>" script.py`, or short cmd-compatible commands. Python-native inspection is usually the safest choice.',
            ]
        )
    else:
        lines.append(
            "- Shell commands should stay portable and repo-local. Prefer Python-native inspection when shell portability is uncertain."
        )

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
    resolved = current_root.resolve()
    candidates = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}(?:-[a-zA-Z0-9_]+)+", objective)
    repo_match = re.search(
        r"(?:inside|in|from|of)\s+(\S+?)(?:\s+repo|\s+repository)?(?:\s|$)",
        objective,
        re.IGNORECASE,
    )
    if repo_match:
        candidates.insert(0, repo_match.group(1).strip().rstrip("."))
    for name in candidates:
        sibling = resolved.parent / name
        if sibling.is_dir() and sibling.resolve() != resolved:
            return sibling
        child = resolved / name
        if child.is_dir():
            return child
    return current_root
