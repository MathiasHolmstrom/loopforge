from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loopforge.bootstrap import (
    Loopforge,
    apply_answers_to_bootstrap_turn,
    cycle_results_to_payload,
    discover_capabilities_for_objective,
    run_preflight_checks,
)
from loopforge.core import (
    BootstrapTurn,
    CapabilityContext,
    ExperimentInterrupted,
    ExperimentSpec,
    ExperimentSpecProposal,
    FileMemoryStore,
    HumanIntervention,
    LiteLLMSpecBackend,
)


def _flush_stdin() -> None:
    """Drain any buffered stdin so the next input() waits for fresh keystrokes."""
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        import select

        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or steer the experimentation loop."
    )
    parser.add_argument(
        "--stream-llm-output",
        action="store_true",
        help="Stream raw model output during interactive planning and execution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--spec", required=True, help="Path to a JSON experiment spec."
    )
    run_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative execution steps.",
    )
    run_parser.add_argument(
        "--memory-root",
        default=".loopforge",
        help="Directory used for loop memory and summaries.",
    )
    run_parser.add_argument(
        "--executor-factory",
        required=True,
        help="Import path package.module:function_name",
    )
    run_parser.add_argument("--model-profile", default=None)
    run_parser.add_argument(
        "--worker-model",
        default=None,
        help="LiteLLM model id for the worker agent.",
    )
    run_parser.add_argument(
        "--review-model", help="LiteLLM model id for the review agent."
    )
    run_parser.add_argument(
        "--consultation-model",
        help="LiteLLM model id for the operations consult agent.",
    )
    run_parser.add_argument(
        "--narrator-model", help="LiteLLM model id for the human-facing narrator agent."
    )
    run_parser.add_argument(
        "--iterations",
        "--max-iterations",
        dest="iterations",
        type=int,
        default=None,
        help="Override max iterations for this run.",
    )
    run_parser.add_argument(
        "--max-autonomous-hours",
        type=float,
        default=None,
        help="Override max autonomous runtime hours for this run.",
    )
    run_parser.add_argument("--temperature", type=float, default=0.2)

    draft_parser = subparsers.add_parser("draft-spec")
    draft_parser.add_argument(
        "--objective", required=True, help="Experiment objective to plan for."
    )
    draft_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to scan when auto-synthesizing.",
    )
    draft_parser.add_argument(
        "--memory-root",
        default=".loopforge",
        help="Directory passed to the adapter factory.",
    )
    draft_parser.add_argument(
        "--executor-factory", help="Import path package.module:function_name"
    )
    draft_parser.add_argument(
        "--planner-model",
        default=None,
        help="LiteLLM model id for the planning agent.",
    )
    draft_parser.add_argument("--preferences-json", default="{}")
    draft_parser.add_argument("--temperature", type=float, default=0.2)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument(
        "--message", required=False, help="Initial description of the problem to solve."
    )
    start_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to scan when auto-synthesizing.",
    )
    start_parser.add_argument(
        "--memory-root", default=".loopforge", help="Directory used for loop memory."
    )
    start_parser.add_argument(
        "--executor-factory", help="Import path package.module:function_name"
    )
    start_parser.add_argument("--model-profile", default=None)
    start_parser.add_argument("--planner-model", default=None)
    start_parser.add_argument("--worker-model", default=None)
    start_parser.add_argument("--review-model")
    start_parser.add_argument("--consultation-model")
    start_parser.add_argument("--narrator-model")
    start_parser.add_argument("--answers-json", default="{}")
    start_parser.add_argument(
        "--iterations",
        "--max-iterations",
        dest="iterations",
        type=int,
        default=None,
    )
    start_parser.add_argument("--max-autonomous-hours", type=float, default=None)
    start_parser.add_argument("--temperature", type=float, default=0.2)
    start_parser.add_argument(
        "--stream-llm-output",
        action="store_true",
        help="Stream raw model output during interactive planning and execution.",
    )

    interject_parser = subparsers.add_parser("interject")
    interject_parser.add_argument("--memory-root", required=True)
    interject_parser.add_argument("--message", required=True)
    interject_parser.add_argument("--effects-json", default="{}")
    interject_parser.add_argument("--author", default="human")
    interject_parser.add_argument("--type", default="note")
    return parser


def _prompt_non_empty(prompt: str, *, input_fn=input, print_fn=print) -> str:
    while True:
        value = input_fn(prompt).strip()
        if value:
            return value
        print_fn("A problem statement is required.")


def _apply_suggested_answers(turn, answers: dict[str, Any], *, print_fn=print) -> int:
    # Don't auto-fill — let the user answer their own questions.
    # Suggested answers are shown as defaults in the prompt instead.
    return 0


def _sanitize_human_text(text: str) -> str:
    replacements = {
        "auto_adapter_scaffold": "repo execution setup",
        "adapter scaffold": "repo execution setup",
        "generated adapter module": "generated execution file",
        "adapter module": "execution file",
        ".loopforge/generated": ".loopforge/generated",
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
    }
    sanitized = text
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return sanitized.encode(encoding, errors="replace").decode(
        encoding, errors="replace"
    )


def _looks_like_accidental_ready_prompt_feedback(
    response: str, *, user_goal: str, turn: BootstrapTurn
) -> bool:
    cleaned = response.strip().lower()
    if len(cleaned) < 20:
        return False
    goal = user_goal.strip().lower()
    objective = (turn.proposal.recommended_spec.objective or "").strip().lower()
    if cleaned == goal or cleaned == objective:
        return True
    if cleaned in goal or cleaned in objective:
        return True
    if goal and (goal in cleaned or objective in cleaned):
        return True
    return False


def _ready_plan_signature(turn: BootstrapTurn) -> str:
    payload = {
        "objective": turn.proposal.recommended_spec.objective,
        "primary_metric": turn.proposal.recommended_spec.primary_metric.to_dict(),
        "secondary_metrics": [
            metric.to_dict()
            for metric in turn.proposal.recommended_spec.secondary_metrics
        ],
        "guardrail_metrics": [
            metric.to_dict()
            for metric in turn.proposal.recommended_spec.guardrail_metrics
        ],
        "allowed_actions": list(turn.proposal.recommended_spec.allowed_actions),
        "human_update": turn.human_update,
        "ready_to_start": turn.ready_to_start,
    }
    return json.dumps(payload, sort_keys=True)


def _friendly_requirement(requirement: str) -> str:
    if requirement == "preflight:auto_adapter_scaffold":
        return "Loopforge still needs a runnable execution binding for the detected experiment path."
    if requirement == "preflight:repo_execution_not_supported":
        return "This repo is not yet wired with a real autonomous runner for the requested experiment."
    if requirement.startswith("preflight:"):
        return requirement.removeprefix("preflight:")
    if requirement.startswith("answer:"):
        return requirement.removeprefix("answer:")
    return requirement


def _friendly_check_detail(check) -> str:
    if check.name == "auto_adapter_scaffold":
        return (
            "Loopforge found a likely experiment path, but it does not yet have a runnable repo-specific "
            "execution binding for it."
        )
    if check.name == "repo_execution_not_supported":
        return (
            "Loopforge can inspect this repo and plan the work, but it cannot execute experiments here yet "
            "because there is no supported repo-specific runner."
        )
    return _sanitize_human_text(check.detail)


def _friendly_metric_label(metric) -> str:
    name = getattr(metric, "display_name", None) or metric.name
    words = name.replace("_", " ").split()
    if not words:
        return name
    return " ".join(
        word.upper() if word.isupper() else word.capitalize() for word in words
    )


def _summarize_actions(actions: list[str], *, generic_autonomous: bool = False) -> str:
    if generic_autonomous:
        return "agent will autonomously inspect, edit, run, and recover as needed"
    if not actions:
        return "no runnable actions were discovered yet"
    preview = ", ".join(actions[:3])
    if len(actions) > 3:
        preview += f", and {len(actions) - 3} more"
    return f"choose among the discovered actions: {preview}"


def _print_plan_summary(turn, *, print_fn=print) -> None:
    spec = turn.proposal.recommended_spec
    generic_autonomous = (
        spec.metadata.get("execution_backend_kind") == "generic_agentic"
    )
    operator_guidance = spec.metadata.get("operator_guidance", [])
    if isinstance(operator_guidance, str):
        operator_guidance = [operator_guidance]
    print_fn("\n--- Experiment plan ---")
    print_fn(f"  Goal       : {_sanitize_human_text(spec.objective)}")
    print_fn(
        f"  Metric     : {_friendly_metric_label(spec.primary_metric)} ({spec.primary_metric.goal})"
    )
    if spec.guardrail_metrics:
        print_fn(
            f"  Guardrails : {', '.join(_friendly_metric_label(metric) for metric in spec.guardrail_metrics[:3])}"
        )
    if spec.secondary_metrics:
        print_fn(
            f"  Also track : {', '.join(_friendly_metric_label(metric) for metric in spec.secondary_metrics[:3])}"
        )
    # Show execution contract
    source_script = spec.metadata.get("source_script", "not specified")
    print_fn(f"  Script     : {source_script}")
    baseline_fn = spec.metadata.get("baseline_function")
    if baseline_fn:
        print_fn(f"  Model func : {baseline_fn}")
    data_loading = spec.metadata.get("data_loading")
    if data_loading:
        print_fn(f"  Data       : {data_loading}")
    target = spec.metadata.get("target_column")
    if target:
        print_fn(f"  Target     : {target}")
    # Show baseline if the planner found one
    baseline_value = spec.metadata.get("baseline_metric_value")
    if baseline_value is not None:
        print_fn(f"  Baseline   : {spec.primary_metric.name} = {baseline_value}")
    else:
        print_fn("  Baseline   : not found - first iteration will establish one")
    print_fn(
        f"  Next       : {_summarize_actions(spec.allowed_actions, generic_autonomous=generic_autonomous)}"
    )
    if operator_guidance:
        latest_guidance = _sanitize_human_text(
            " ".join(str(operator_guidance[-1]).split())
        )
        if len(latest_guidance) > 120:
            latest_guidance = latest_guidance[:117] + "..."
        print_fn(f"  Guidance   : {latest_guidance}")
    print_fn("")


def _print_blocked_summary(turn, *, print_fn=print) -> None:
    """Print a human-readable summary when the loop cannot start."""
    failed_checks = [c for c in (turn.preflight_checks or []) if c.status == "failed"]
    required_failures = [check for check in failed_checks if check.required]
    setup_failures = [check for check in failed_checks if not check.required]
    if required_failures:
        print_fn("\nBlocked — the following checks failed:")
        for check in required_failures:
            print_fn(f"  - {_friendly_check_detail(check)}")
    if setup_failures:
        print_fn("\nSetup still needed:")
        for check in setup_failures:
            print_fn(f"  - {_friendly_check_detail(check)}")
    if turn.missing_requirements:
        print_fn("\nStill needed:")
        for req in turn.missing_requirements:
            print_fn(f"  - {_friendly_requirement(req)}")
    notes = getattr(turn.proposal, "notes", None) or []
    if notes:
        print_fn("\nNotes:")
        for note in notes:
            print_fn(f"  - {_sanitize_human_text(note)}")


def _print_result_summary(result: dict[str, Any], *, print_fn=print) -> None:
    """Print a human-readable summary of the experiment result."""
    status = result.get("status", "unknown")
    if status == "blocked":
        error = result.get("error", {})
        print_fn(f"\nExperiment blocked: {error.get('message', 'unknown error')}")
        return
    if status == "needs_input":
        print_fn("\nExperiment still needs input before it can run.")
        return
    cycle_results = result.get("results", [])
    print_fn(f"\nExperiment finished — {len(cycle_results)} iteration(s) completed.")
    for i, cycle in enumerate(cycle_results, 1):
        update = cycle.get("human_update")
        if update:
            print_fn(f"\n--- Iteration {i} ---")
            print_fn(update)


def _confirm_or_feedback(prompt: str, *, input_fn=input) -> str | None:
    """Ask for confirmation or feedback. Returns None for yes, feedback string otherwise."""
    answer = input_fn(prompt).strip()
    if answer.lower() in ("y", "yes", ""):
        return None
    return answer


def _make_stream_fn():
    """Build a stream callback that writes tokens to stdout in real-time."""

    def stream(token: str) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

    return stream


def _make_progress_fn(print_fn=print):
    """Build a progress callback that prints stage updates and detail blocks."""
    last_stage = [None]

    def progress(stage: str, message: str) -> None:
        if stage == last_stage[0]:
            return
        last_stage[0] = stage
        # Detail stages contain multi-line content — print as-is
        if stage.endswith("_detail"):
            print_fn(message)
            return
        print_fn(f"\n  ... {message}")

    return progress


def _make_iteration_callback(print_fn=print):
    """Build a callback that prints rich per-iteration details."""
    iteration_count = [0]

    def callback(cycle_result) -> None:
        iteration_count[0] += 1
        record = cycle_result.record
        candidate = record.candidate
        outcome = record.outcome
        reflection = record.reflection
        review = record.review

        print_fn(f"\n{'=' * 60}")
        print_fn(f"  Iteration {record.iteration_id} — {candidate.action_type}")
        print_fn(f"{'=' * 60}")
        print_fn(f"  Hypothesis : {candidate.hypothesis}")

        # Outcome
        metrics_parts = []
        for name, result in outcome.metric_results.items():
            if result.value is not None:
                metrics_parts.append(f"{name}={result.value:.4g}")
        if outcome.primary_metric_value is not None and not metrics_parts:
            metrics_parts.append(f"primary={outcome.primary_metric_value:.4g}")
        metrics_str = " | ".join(metrics_parts) if metrics_parts else "no metrics"
        print_fn(f"  Outcome    : {outcome.status} | {metrics_str}")
        if outcome.notes:
            for note in outcome.notes[:3]:
                print_fn(f"               {note}")

        # Reflection
        assessment = reflection.assessment
        if len(assessment) > 200:
            assessment = assessment[:200] + "..."
        print_fn(f"  Reflection : {assessment}")
        if reflection.lessons:
            for lesson in reflection.lessons[:3]:
                print_fn(f"    - {lesson}")

        # Review
        print_fn(f"  Review     : {review.status} — {review.reason}")
        if reflection.recommended_next_action:
            print_fn(f"  Next       : {reflection.recommended_next_action}")

        # Accepted summary result
        if cycle_result.accepted_summary is not None:
            print_fn(f"  Result     : {cycle_result.accepted_summary.result}")

        # Narrator update
        if cycle_result.human_update:
            print_fn(f"\n  {cycle_result.human_update}")
        print_fn("")

    return callback


def _planning_status_message(
    planner_tag: str, *, first_run: bool, replan_reason: str
) -> str | None:
    if first_run:
        return f"\n{planner_tag} Analysing repository and planning experiment..."
    if replan_reason == "feedback":
        return f"\n{planner_tag} Updating plan..."
    return None


def _resolve_question_answer(question, raw: str) -> str:
    if question.options:
        if not raw and question.suggested_answer:
            return question.suggested_answer
        if raw.isdigit() and 1 <= int(raw) <= len(question.options):
            return question.options[int(raw) - 1]
        return raw or question.suggested_answer or ""
    if not raw and question.suggested_answer:
        return question.suggested_answer
    return raw or ""


def _is_start_intent(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if normalized in {"y", "yes", "go", "start", "run", "proceed", "continue", "cont"}:
        return True
    start_phrases = (
        "go ahead",
        "go ahead start",
        "go ahead and start",
        "start it",
        "start now",
        "run it",
        "run now",
        "let's start",
        "lets start",
        "start man",
        "go ahead man",
    )
    return any(phrase in normalized for phrase in start_phrases)


def _extract_start_with_context(text: str) -> str | None:
    """Detect 'Y but also X' or 'yes, also X' patterns. Returns the extra context or None."""
    normalized = " ".join(text.strip().split())
    lower = normalized.lower()
    prefixes = (
        "y but ",
        "y, but ",
        "y but also ",
        "y, but also ",
        "yes but ",
        "yes, but ",
        "yes but also ",
        "yes, but also ",
        "yes, also ",
        "y, also ",
        "yes also ",
        "go but ",
        "go, but ",
        "go but also ",
        "start but ",
        "start, but ",
        "y, ",
        "yes, ",
        "go, ",
    )
    for prefix in prefixes:
        if lower.startswith(prefix):
            extra = normalized[len(prefix) :].strip()
            if extra:
                return extra
    return None


def run_interactive_start(
    *,
    repo_root: Path | str = ".",
    memory_root: Path | str = ".loopforge",
    executor_factory: str | None = None,
    model_profile: str | None = None,
    planner_model: str | None = None,
    worker_model: str | None = None,
    review_model: str | None = None,
    consultation_model: str | None = None,
    narrator_model: str | None = None,
    iterations: int | None = None,
    max_autonomous_hours: float | None = None,
    temperature: float = 0.2,
    stream_llm_output: bool = False,
    input_fn=input,
    print_fn=print,
) -> int:
    progress_fn = _make_progress_fn(print_fn)
    stream_fn = _make_stream_fn() if stream_llm_output else None
    app = Loopforge(
        executor_factory_path=executor_factory,
        repo_root=repo_root,
        memory_root=memory_root,
        model_profile=model_profile,
        planner_model=planner_model,
        worker_model=worker_model,
        review_model=review_model,
        consultation_model=consultation_model,
        narrator_model=narrator_model,
        temperature=temperature,
        progress_fn=progress_fn,
        stream_fn=stream_fn,
    )
    user_goal = _prompt_non_empty(
        "What problem are we solving? ", input_fn=input_fn, print_fn=print_fn
    )
    answers: dict[str, Any] = {}
    replan_reason = "initial"
    first_run = True
    resume_existing_run = False
    pending_resume_start = False
    turn = None
    last_ready_plan_signature: str | None = None
    planner_tag = f"[{getattr(app, 'role_models', None) and app.role_models.planner or 'planner'}]"
    narrator_tag = f"[{getattr(app, 'role_models', None) and app.role_models.narrator or 'narrator'}]"
    while True:
        response = (
            "y"
            if pending_resume_start and turn is not None and turn.ready_to_start
            else ""
        )
        pending_resume_start = False
        deferred_start_response: str | None = None
        # Only call the LLM when we actually need a new plan
        if replan_reason:
            planning_message = _planning_status_message(
                planner_tag,
                first_run=first_run,
                replan_reason=replan_reason,
            )
            if planning_message:
                print_fn(planning_message)
            first_run = False
            try:
                turn = app.bootstrap(user_goal=user_goal, answers=answers)
                last_ready_plan_signature = None
            except KeyboardInterrupt:
                print_fn("\n\n--- Interrupted during planning ---")
                try:
                    redirect = input_fn(
                        "Feedback for current plan (or 'restart: ...' / 'quit'): "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    print_fn("\nAborted.")
                    return 0
                if redirect.lower() in ("quit", "exit"):
                    print_fn("Aborted.")
                    return 0
                if redirect.lower().startswith("restart:"):
                    restarted_goal = redirect.partition(":")[2].strip()
                    if restarted_goal:
                        user_goal = restarted_goal
                    answers = {}
                    first_run = True
                    replan_reason = "initial"
                    continue
                if redirect.lower().startswith("restart "):
                    restarted_goal = redirect[len("restart ") :].strip()
                    if restarted_goal:
                        user_goal = restarted_goal
                    answers = {}
                    first_run = True
                    replan_reason = "initial"
                    continue
                pending_required = (
                    [
                        q
                        for q in turn.proposal.questions
                        if q.required and q.key not in answers
                    ]
                    if turn is not None
                    else []
                )
                if redirect and len(pending_required) == 1:
                    answers[pending_required[0].key] = _resolve_question_answer(
                        pending_required[0], redirect
                    )
                    turn = apply_answers_to_bootstrap_turn(turn, answers=answers)
                    replan_reason = ""
                    continue
                pending_optional = (
                    [
                        q
                        for q in turn.proposal.questions
                        if not q.required and q.key not in answers
                    ]
                    if turn is not None
                    else []
                )
                if redirect and len(pending_optional) == 1:
                    answers[pending_optional[0].key] = _resolve_question_answer(
                        pending_optional[0], redirect
                    )
                    turn = apply_answers_to_bootstrap_turn(turn, answers=answers)
                    replan_reason = ""
                    continue
                if redirect:
                    answers["user_feedback"] = redirect
                    replan_reason = "feedback"
                else:
                    replan_reason = "feedback"
                continue
            replan_reason = ""

            # Show the human-friendly narrative
            narrative = turn.human_update or turn.assistant_message or ""
            if narrative:
                print_fn(f"\n{narrator_tag} {_sanitize_human_text(narrative)}")
            if turn.access_guide_path:
                print_fn(f"\n(Access guide written to {turn.access_guide_path})")

            # Show blockers if any
            failed_checks = [
                c for c in (turn.preflight_checks or []) if c.status == "failed"
            ]
            if failed_checks:
                _print_blocked_summary(turn, print_fn=print_fn)

            # Auto-fill safe defaults, then re-plan with them
            if _apply_suggested_answers(turn, answers, print_fn=print_fn):
                replan_reason = "answers"
                continue

        # Ask any remaining required questions — re-plan after answers
        required_questions = [q for q in turn.proposal.questions if q.required]
        pending_questions = [q for q in required_questions if q.key not in answers]
        if pending_questions:
            for question in pending_questions:
                if question.options:
                    print_fn(f"\n{question.prompt}")
                    for i, opt in enumerate(question.options, 1):
                        print_fn(f"  {i}. {opt}")
                    print_fn("  Or type your own answer.")
                    if question.suggested_answer:
                        print_fn(f"  (default: {question.suggested_answer})")
                    raw = input_fn("Your choice: ").strip()
                    answers[question.key] = _resolve_question_answer(question, raw)
                else:
                    print_fn(f"\n{question.prompt}")
                    if question.suggested_answer:
                        print_fn(f"  (default: {question.suggested_answer})")
                    raw = input_fn("> ").strip()
                    answers[question.key] = _resolve_question_answer(question, raw)
            # Re-run the full pipeline with answers so data trace re-checks
            replan_reason = "answers"
            continue

        optional_questions = [
            q
            for q in turn.proposal.questions
            if not q.required and q.key not in answers
        ]
        if optional_questions:
            question = optional_questions[0]
            if question.options:
                print_fn(f"\n{question.prompt}")
                for i, opt in enumerate(question.options, 1):
                    print_fn(f"  {i}. {opt}")
                print_fn("  Or press Enter to skip.")
                raw = input_fn("Your choice: ").strip()
                if turn.ready_to_start and raw and _is_start_intent(raw):
                    deferred_start_response = raw
                else:
                    answers[question.key] = _resolve_question_answer(question, raw)
            else:
                print_fn(f"\n{question.prompt}")
                raw = input_fn("> ").strip()
                if turn.ready_to_start and raw and _is_start_intent(raw):
                    deferred_start_response = raw
                else:
                    answers[question.key] = raw
            if deferred_start_response is not None:
                response = deferred_start_response
            elif answers[question.key]:
                turn = apply_answers_to_bootstrap_turn(turn, answers=answers)
                # Re-show blockers if still blocked
                failed_checks = [
                    c for c in (turn.preflight_checks or []) if c.status == "failed"
                ]
                if failed_checks and not turn.ready_to_start:
                    _print_blocked_summary(turn, print_fn=print_fn)
                continue
            else:
                response = ""

        # Show the plan summary if ready
        if turn.ready_to_start:
            plan_signature = _ready_plan_signature(turn)
            if plan_signature != last_ready_plan_signature:
                _print_plan_summary(turn, print_fn=print_fn)
                last_ready_plan_signature = plan_signature

        # Always prompt the user — whether ready, blocked, or anything else
        if not response:
            if turn.ready_to_start:
                print_fn(
                    "Start experiment? [Y = run / type feedback to revise / quit = exit]"
                )
                prompt_text = "> "
            else:
                prompt_text = "> "
            _flush_stdin()
            response = input_fn(prompt_text).strip()
        if not response:
            continue
        if response.lower() in ("quit", "exit"):
            print_fn("Aborted.")
            return 0
        # Check for "Y but also..." — start with extra context
        if turn.ready_to_start:
            extra_context = _extract_start_with_context(response)
            if extra_context:
                print_fn(f"\n  Extra context noted: {extra_context}")
                # Store in spec metadata so it flows to the executor
                current_guidance = list(
                    turn.proposal.recommended_spec.metadata.get("operator_guidance", [])
                )
                current_guidance.append(extra_context)
                turn = replace(
                    turn,
                    proposal=replace(
                        turn.proposal,
                        recommended_spec=replace(
                            turn.proposal.recommended_spec,
                            metadata={
                                **turn.proposal.recommended_spec.metadata,
                                "operator_guidance": current_guidance,
                            },
                        ),
                    ),
                )
                response = "y"  # Treat as start intent
        if turn.ready_to_start and _is_start_intent(response):
            print_fn("\nStarting experiment loop...\n")
            try:
                iteration_cb = _make_iteration_callback(print_fn)
                result = app.start_from_bootstrap_turn(
                    bootstrap_turn=turn,
                    user_goal=user_goal,
                    iterations=iterations,
                    max_autonomous_hours=max_autonomous_hours,
                    iteration_callback=iteration_cb,
                    reset_state=not resume_existing_run,
                )
            except ExperimentInterrupted as exc:
                print_fn(
                    f"\n--- Interrupted after {len(exc.results_so_far)} iteration(s) ---"
                )
                for i, cycle in enumerate(exc.results_so_far, 1):
                    if cycle.human_update:
                        print_fn(f"\n  Iteration {i}: {cycle.human_update}")
                try:
                    redirect = input_fn(
                        "\nNew instructions (or 'quit' to stop): "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    print_fn("\nAborted.")
                    return 0
                if not redirect or redirect.lower() in ("quit", "exit"):
                    print_fn("Stopped.")
                    return 0
                append_human_intervention(
                    memory_root=app.memory_root,
                    message=redirect,
                    effects={"user_redirect": True},
                    author="human",
                    type_="override",
                )
                reopened = FileMemoryStore(app.memory_root).reopen_last_iteration()
                if reopened is not None:
                    print_fn(
                        f"  Reopened iteration {reopened.iteration_id} with the new reviewer feedback."
                    )
                answers.pop("user_feedback", None)
                resume_existing_run = True
                replan_reason = ""
                pending_resume_start = True
                continue
            except KeyboardInterrupt:
                print_fn("\n\n--- Interrupted during experiment ---")
                try:
                    redirect = input_fn(
                        "New instructions (or 'quit' to stop): "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    print_fn("\nAborted.")
                    return 0
                if not redirect or redirect.lower() in ("quit", "exit"):
                    print_fn("Stopped.")
                    return 0
                append_human_intervention(
                    memory_root=app.memory_root,
                    message=redirect,
                    effects={"user_redirect": True},
                    author="human",
                    type_="override",
                )
                answers.pop("user_feedback", None)
                resume_existing_run = True
                replan_reason = ""
                pending_resume_start = True
                continue
            _print_result_summary(result, print_fn=print_fn)
            return 0
        # If it looks like a question, answer it directly — no full re-plan
        if response.rstrip().endswith("?") or response.lower().startswith(
            ("what ", "why ", "how ", "can ", "do ", "is ", "are ")
        ):
            answers.setdefault("discussion", [])
            answers["discussion"].append(response)
            try:
                cap_ctx = app._cached_capability_context or CapabilityContext()
                quick_reply = app.narrator_backend.answer_question(
                    response, turn, cap_ctx
                )
                print_fn(f"\n{narrator_tag} {quick_reply}\n")
            except Exception as exc:
                print_fn(
                    f"\n{narrator_tag} I couldn't answer that question directly because the narrator failed: {exc}\n"
                )
            continue
        # Try to patch the existing plan with the feedback (no full replan)
        self_progress = _make_progress_fn(print_fn)
        self_progress("interpret_feedback", "Interpreting feedback...")
        updated_turn = app.apply_feedback(turn, response)
        if updated_turn is not None:
            turn = updated_turn
            answers.pop("user_feedback", None)
            answers.pop("discussion", None)
            if turn.human_update:
                print_fn(f"\n{narrator_tag} {turn.human_update}")
            # Re-show blockers or plan
            failed_checks = [
                c for c in (turn.preflight_checks or []) if c.status == "failed"
            ]
            if failed_checks:
                _print_blocked_summary(turn, print_fn=print_fn)
            elif turn.ready_to_start:
                _print_plan_summary(turn, print_fn=print_fn)
                last_ready_plan_signature = _ready_plan_signature(turn)
            continue
        # Feedback requires full replan
        answers["user_feedback"] = response
        replan_reason = "feedback"
        continue


def run_from_spec(
    *,
    spec_path: Path | str,
    repo_root: Path | str = ".",
    memory_root: Path | str,
    executor_factory_path: str,
    worker_model: str | None,
    review_model: str | None = None,
    consultation_model: str | None = None,
    narrator_model: str | None = None,
    model_profile: str | None = None,
    iterations: int | None = None,
    max_autonomous_hours: float | None = None,
    temperature: float = 0.2,
) -> list[dict[str, Any]]:
    spec = ExperimentSpec.from_dict(
        json.loads(Path(spec_path).read_text(encoding="utf-8"))
    )
    app = Loopforge(
        executor_factory_path=executor_factory_path,
        repo_root=repo_root,
        memory_root=memory_root,
        model_profile=model_profile,
        worker_model=worker_model,
        review_model=review_model,
        consultation_model=consultation_model,
        narrator_model=narrator_model,
        temperature=temperature,
    )
    spec = app._apply_repo_stop_condition_defaults(spec)
    memory_store = FileMemoryStore(memory_root)
    orchestrator, runtime = app.build_orchestrator(
        spec=spec,
        objective=spec.objective,
        memory_store=memory_store,
        executor_factory_path=executor_factory_path,
    )
    preflight_checks = run_preflight_checks(
        spec=spec,
        capability_context=runtime.capability_context,
        memory_root=memory_root,
        executor_factory_path=executor_factory_path,
    )
    blocking_checks = [
        check.detail
        for check in preflight_checks
        if check.required and check.status == "failed"
    ]
    if blocking_checks:
        raise ValueError("; ".join(blocking_checks))
    orchestrator.initialize(spec=spec, reset_state=True)
    return cycle_results_to_payload(
        orchestrator.run(
            iterations=iterations, max_autonomous_hours=max_autonomous_hours
        )
    )


def draft_spec(
    *,
    objective: str,
    memory_root: Path | str,
    executor_factory_path: str | None,
    planner_model: str | None,
    repo_root: Path | str = ".",
    preferences: dict[str, Any] | None = None,
    temperature: float = 0.2,
) -> ExperimentSpecProposal:
    probe_app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=memory_root,
        planner_model=planner_model,
        temperature=temperature,
    )
    resolved_factory_path = executor_factory_path
    if resolved_factory_path is None:
        resolved_factory_path = probe_app.resolve_executor_factory_path(objective)
    capability_context = discover_capabilities_for_objective(
        objective=objective,
        memory_root=memory_root,
        executor_factory_path=resolved_factory_path,
        repo_root=repo_root,
    )
    backend = LiteLLMSpecBackend(
        model=probe_app.role_models.planner,
        temperature=temperature,
    )
    proposal = backend.propose_spec(
        objective=objective,
        capability_context=capability_context,
        user_preferences=preferences,
    )
    return replace(
        proposal,
        recommended_spec=probe_app._apply_repo_stop_condition_defaults(
            proposal.recommended_spec
        ),
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
    if args.command is None:
        return run_interactive_start(stream_llm_output=args.stream_llm_output)
    if args.command == "run":
        try:
            results = run_from_spec(
                spec_path=args.spec,
                repo_root=args.repo_root,
                memory_root=args.memory_root,
                executor_factory_path=args.executor_factory,
                worker_model=args.worker_model,
                review_model=args.review_model,
                consultation_model=args.consultation_model,
                narrator_model=args.narrator_model,
                model_profile=args.model_profile,
                iterations=args.iterations,
                max_autonomous_hours=args.max_autonomous_hours,
                temperature=args.temperature,
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {"status": "blocked", "error": str(exc)}, indent=2, sort_keys=True
                )
            )
            return 1
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
        if not args.message:
            return run_interactive_start(
                repo_root=args.repo_root,
                memory_root=args.memory_root,
                executor_factory=args.executor_factory,
                model_profile=args.model_profile,
                planner_model=args.planner_model,
                worker_model=args.worker_model,
                review_model=args.review_model,
                consultation_model=args.consultation_model,
                narrator_model=args.narrator_model,
                iterations=args.iterations,
                max_autonomous_hours=args.max_autonomous_hours,
                temperature=args.temperature,
                stream_llm_output=args.stream_llm_output,
            )
        app = Loopforge(
            executor_factory_path=args.executor_factory,
            repo_root=args.repo_root,
            memory_root=args.memory_root,
            model_profile=args.model_profile,
            planner_model=args.planner_model,
            worker_model=args.worker_model,
            review_model=args.review_model,
            consultation_model=args.consultation_model,
            narrator_model=args.narrator_model,
            temperature=args.temperature,
        )
        result = app.start(
            user_goal=args.message,
            answers=json.loads(args.answers_json),
            iterations=args.iterations,
            max_autonomous_hours=args.max_autonomous_hours,
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


if __name__ == "__main__":
    raise SystemExit(main())
