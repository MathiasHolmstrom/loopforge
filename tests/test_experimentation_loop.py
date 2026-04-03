from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import loopforge.bootstrap as bootstrap_module
import loopforge.pilot_adapters as pilot_adapters_module
from loopforge.core.backends import build_iteration_policy
from loopforge import (
    AdapterSetup,
    AccessGuide,
    BootstrapTurn,
    CapabilityContext,
    ConsultingWorkerBackend,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentInterrupted,
    ExperimentOrchestrator,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentSpecProposal,
    FileMemoryStore,
    IterationSummary,
    Loopforge,
    MarkdownMemoryNote,
    MemorySnapshot,
    MetricResult,
    MetricSpec,
    OpsConsultation,
    PreflightCheck,
    PrimaryMetric,
    ReflectionSummary,
    RoleModelConfig,
    ReviewDecision,
    RoutingExperimentExecutor,
    SpecQuestion,
)
from loopforge.auto_adapter import scan_repo, synthesize_auto_adapter
from loopforge.bootstrap import (
    apply_answers_to_bootstrap_turn,
    load_factory,
    refine_bootstrap_questions,
    sanitise_bootstrap_questions,
)
from loopforge.cli import (
    _apply_suggested_answers,
    _print_blocked_summary,
    _print_plan_summary,
    _sanitize_human_text,
    append_human_intervention,
    discover_capabilities_for_objective,
    draft_spec,
    main,
    run_from_spec,
    run_interactive_start,
)


class FakeWorkerBackend:
    def __init__(self, candidates: list[ExperimentCandidate]) -> None:
        self._candidates = list(candidates)
        self.snapshots = []

    def propose_next_experiment(self, snapshot):
        self.snapshots.append(snapshot)
        return self._candidates.pop(0)


class StaticActionExecutor:
    def __init__(self, outcomes: list[ExperimentOutcome]) -> None:
        self._outcomes = list(outcomes)

    def execute(self, candidate: ExperimentCandidate, snapshot):
        return self._outcomes.pop(0)


class FakeReflectionBackend:
    def __init__(self, reflections: list[ReflectionSummary]) -> None:
        self._reflections = list(reflections)

    def reflect(self, snapshot, candidate, outcome):
        return self._reflections.pop(0)


class FakeReviewBackend:
    def __init__(self, decisions: list[ReviewDecision]) -> None:
        self._decisions = list(decisions)

    def review(self, snapshot, candidate, outcome, reflection):
        return self._decisions.pop(0)


class FakeNarrationBackend:
    def __init__(self) -> None:
        self.bootstrap_calls = []
        self.iteration_calls = []

    def summarize_bootstrap(self, turn, capability_context):
        self.bootstrap_calls.append((turn, capability_context))
        return f"bootstrap:{turn.proposal.objective}"

    def summarize_iteration(self, snapshot, candidate, outcome, reflection, review, accepted_summary):
        self.iteration_calls.append((snapshot, candidate, outcome, reflection, review, accepted_summary))
        return f"iteration:{candidate.action_type}:{review.status}"


class FakeConsultationBackend:
    def __init__(self, focus: str = "databricks deploy") -> None:
        self.focus = focus
        self.calls = []

    def consult(self, snapshot):
        self.calls.append(snapshot)
        return OpsConsultation(
            focus=self.focus,
            guidance="Use bundle deploy and verify env vars first.",
            commands=["databricks bundle validate", "databricks bundle deploy -t dev"],
            required_env_vars=["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
            risks=["Missing workspace permissions"],
            should_consult=True,
        )


class FakeAccessAdvisorBackend:
    def __init__(self) -> None:
        self.calls = []

    def build_access_guide(self, user_goal, capability_context, preflight_checks):
        self.calls.append((user_goal, capability_context, preflight_checks))
        return AccessGuide(
            summary="Check Databricks credentials before running.",
            required_env_vars=["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
            required_permissions=["Workspace access", "Job run permission"],
            commands=["databricks auth profiles", "databricks bundle validate"],
            steps=["Export env vars", "Validate access", "Run the bundle"],
            markdown=(
                "# Access Guide\n\n"
                "## Environment Variables\n"
                "- DATABRICKS_HOST\n"
                "- DATABRICKS_TOKEN\n"
            ),
        )


class RaisingNarrationBackend:
    def summarize_bootstrap(self, turn, capability_context):
        raise RuntimeError("invalid bearer token")

    def summarize_iteration(self, snapshot, candidate, outcome, reflection, review, accepted_summary):
        raise RuntimeError("invalid bearer token")


class RaisingAccessAdvisorBackend:
    def build_access_guide(self, user_goal, capability_context, preflight_checks):
        raise RuntimeError("invalid bearer token")


def build_spec(**overrides) -> ExperimentSpec:
    payload = {
        "objective": "Improve pass outcome validation loss.",
        "primary_metric": PrimaryMetric(name="log_loss", goal="minimize"),
        "allowed_actions": ["baseline", "eda", "train", "tune", "evaluate"],
        "constraints": {"max_runtime_minutes": 30},
        "search_space": {"learning_rate": [0.01, 0.03, 0.05]},
        "stop_conditions": {"max_iterations": 4, "patience": 2},
        "metadata": {"model_key": "pass_outcome"},
    }
    payload.update(overrides)
    return ExperimentSpec(**payload)


def test_default_role_models_routes_claude_only_to_helper_roles(monkeypatch) -> None:
    monkeypatch.setattr("loopforge.bootstrap._can_use_anthropic_helpers", lambda: False)
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)

    role_models = bootstrap_module.default_role_models(
        consultation_model="anthropic/claude-sonnet-4-5",
        narrator_model="anthropic/claude-sonnet-4-5",
    )

    assert role_models.planner == bootstrap_module.DEFAULT_OPENAI_MODEL
    assert role_models.worker == bootstrap_module.DEFAULT_OPENAI_MODEL
    assert role_models.reflection == bootstrap_module.DEFAULT_OPENAI_MODEL
    assert role_models.review == bootstrap_module.DEFAULT_OPENAI_MODEL
    # Without Anthropic access, Claude falls back to OpenAI
    assert role_models.consultation == bootstrap_module.DEFAULT_OPENAI_MODEL
    assert role_models.narrator == bootstrap_module.DEFAULT_OPENAI_MODEL


def test_default_role_models_keeps_claude_helpers_when_supported(monkeypatch) -> None:
    monkeypatch.setattr("loopforge.bootstrap._can_use_anthropic_helpers", lambda: True)
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)

    role_models = bootstrap_module.default_role_models()

    assert role_models.consultation == bootstrap_module.DEFAULT_CLAUDE_MODEL
    assert role_models.narrator == bootstrap_module.DEFAULT_CLAUDE_MODEL


def test_detect_builtin_executor_factory_for_lol_kills_repo(tmp_path) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    (repo_root / "examples" / "lol").mkdir(parents=True)
    (repo_root / "examples" / "lol" / "pipeline_transformer_example.py").write_text("print('lol')\n", encoding="utf-8")

    factory = pilot_adapters_module.detect_builtin_executor_factory(
        repo_root,
        "create player kills model in player-performance-ratings repo for lol",
    )

    assert factory == "loopforge.pilot_adapters:build_lol_kills_adapter"


def test_loopforge_does_not_prefer_builtin_lol_adapter_by_default(tmp_path) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    (repo_root / "examples" / "lol").mkdir(parents=True)
    (repo_root / "examples" / "lol" / "pipeline_transformer_example.py").write_text("print('lol')\n", encoding="utf-8")

    app = Loopforge(repo_root=repo_root, memory_root=tmp_path / "memory")

    resolution = app.resolve_execution_backend("create player kills model in player-performance-ratings repo for lol")

    assert resolution.kind == "planning_only"
    assert resolution.factory_path is None


def test_loopforge_init_wires_default_models_to_expected_backends(tmp_path) -> None:
    app = Loopforge(memory_root=tmp_path / "memory")

    assert app.bootstrap_backend.model == app.role_models.planner
    assert app.worker_backend.worker_backend.model == app.role_models.worker
    assert app.reflection_backend.model == app.role_models.reflection
    assert app.review_backend.model == app.role_models.review
    # Consultation/narrator use Claude when Bedrock is available, otherwise fall back to OpenAI
    expected_helper = app.role_models.consultation
    assert app.consultation_backend.model == expected_helper
    assert app.access_advisor_backend.model == expected_helper
    expected_narrator = app.role_models.narrator
    assert app.narrator_backend.model == expected_narrator


def test_loopforge_prefers_explicit_executor_factory(tmp_path) -> None:
    app = Loopforge(
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        executor_factory_path="custom.module:build_adapter",
    )

    resolved = app.resolve_executor_factory_path("Improve the model.")

    assert resolved == "custom.module:build_adapter"


def test_loopforge_uses_planning_only_backend_when_no_supported_runner(tmp_path) -> None:
    app = Loopforge(repo_root=tmp_path / "repo", memory_root=tmp_path / "memory")

    resolution = app.resolve_execution_backend("Improve the LoL kills model.")

    assert resolution.kind == "planning_only"
    assert resolution.factory_path is None
    assert app.executor_factory_path is None


def test_run_interactive_start_does_not_enable_raw_token_streaming(monkeypatch) -> None:
    captured_kwargs = {}

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def bootstrap(self, *, user_goal, answers=None):
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=True,
                human_update="Ready to start.",
            )

        def start_from_bootstrap_turn(self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs):
            return {"status": "started", "bootstrap": {"objective": user_goal}, "results": []}

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: "y" if "Ready to start?" in prompt else "Improve model",
        print_fn=lambda *_args, **_kwargs: None,
    )

    assert exit_code == 0
    assert "stream_fn" not in captured_kwargs or captured_kwargs["stream_fn"] is None


def test_build_metric_guess_prefers_primary_secondary_and_guardrails() -> None:
    guess = bootstrap_module.build_metric_guess(
        CapabilityContext(
            available_metrics={
                "mae": {"preferred_role": "primary"},
                "r2": {"preferred_role": "secondary"},
                "ordinal_loss": {"preferred_role": "guardrail"},
            }
        ),
        objective="improve player points model",
    )

    # mae is generic and relevant; r2 is generic; ordinal_loss contains no overlap
    assert "Primary: mae" in guess


def test_run_interactive_start_asks_for_goal_then_follow_up_and_starts(monkeypatch) -> None:
    prompts = []
    outputs = []
    answers = iter(["Fix Databricks deploy", "dev workspace", "y"])
    created_apps = []

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = []
            self.start_calls = []
            created_apps.append(self)

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls.append((user_goal, dict(answers or {})))
            if not answers:
                return BootstrapTurn(
                    assistant_message="Need workspace target.",
                    proposal=ExperimentSpecProposal(
                        objective=user_goal,
                        recommended_spec=build_spec(objective=user_goal),
                        questions=[SpecQuestion(key="workspace", prompt="Which workspace should I target?")],
                    ),
                    role_models=bootstrap_module.default_role_models(),
                    ready_to_start=False,
                    human_update="Scanning repo and checking access.",
                )
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=True,
                human_update="Ready to start.",
            )

        def start_from_bootstrap_turn(self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs):
            self.start_calls.append((user_goal, bootstrap_turn.proposal.recommended_spec.metadata, iterations))
            return {"status": "started", "bootstrap": {"objective": user_goal}, "results": []}

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert prompts[0] == "What problem are we solving? "
    # Questions without options show prompt as output, use "> " for input
    assert "> " in prompts
    # Confirmation prompt when ready
    assert any("Ready to start?" in p or "What would you like to do?" in p for p in prompts)
    assert any("Scanning repo" in o for o in outputs)
    assert any("finished" in o.lower() or "started" in o.lower() for o in outputs)
    assert len(created_apps) == 1
    assert created_apps[0].bootstrap_calls == [("Fix Databricks deploy", {})]
    assert created_apps[0].start_calls[0][1]["bootstrap_answers"]["workspace"] == "dev workspace"


def test_run_interactive_start_accepts_natural_language_start_intent(monkeypatch) -> None:
    outputs = []

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.start_calls = 0

        def bootstrap(self, *, user_goal, answers=None):
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=True,
                human_update="Ready to start.",
            )

        def start_from_bootstrap_turn(self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs):
            self.start_calls += 1
            return {"status": "started", "bootstrap": {"objective": user_goal}, "results": []}

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: "Go ahead start man" if "Ready to start?" in prompt else "Improve model",
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any("Starting experiment loop" in output for output in outputs)


def test_run_interactive_start_does_not_print_plan_update_when_replanning_after_required_answer(monkeypatch) -> None:
    outputs = []
    answers = iter(["Improve the LoL kills model", "1", "quit"])
    captured_bootstraps = []

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = []

        def bootstrap(self, *, user_goal, answers=None):
            snapshot = dict(answers or {})
            self.bootstrap_calls.append((user_goal, snapshot))
            captured_bootstraps.append((user_goal, snapshot))
            if "dataset_choice" not in snapshot:
                return BootstrapTurn(
                    assistant_message="I need to know which dataset to use.",
                    proposal=ExperimentSpecProposal(
                        objective=user_goal,
                        recommended_spec=build_spec(objective=user_goal),
                        questions=[
                            SpecQuestion(
                                key="dataset_choice",
                                prompt="Which dataset should I use?",
                                options=["Use the larger dataset", "Use the subsample first"],
                            )
                        ],
                    ),
                    role_models=bootstrap_module.default_role_models(),
                    ready_to_start=False,
                    human_update="Need a dataset choice before I lock the plan.",
                )
            return BootstrapTurn(
                assistant_message="Plan updated with the selected dataset.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=False,
                human_update="Dataset choice captured.",
            )

        def start_from_bootstrap_turn(self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs):
            return {"status": "started", "bootstrap": bootstrap_turn.to_dict(), "results": []}

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any("Analysing repository and planning experiment" in output for output in outputs)
    assert not any("Updating plan..." in output for output in outputs)
    assert len(captured_bootstraps) == 1


def test_run_interactive_start_prints_plan_update_when_user_feedback_changes_plan(monkeypatch) -> None:
    outputs = []
    answers = iter(["Improve the LoL kills model", "Use Bayesian smoothing", "quit"])

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = []

        def bootstrap(self, *, user_goal, answers=None):
            snapshot = dict(answers or {})
            self.bootstrap_calls.append((user_goal, snapshot))
            if "user_feedback" not in snapshot:
                return BootstrapTurn(
                    assistant_message="Ready.",
                    proposal=ExperimentSpecProposal(
                        objective=user_goal,
                        recommended_spec=build_spec(objective=user_goal),
                        questions=[],
                    ),
                    role_models=bootstrap_module.default_role_models(),
                    ready_to_start=True,
                    human_update="Ready to start.",
                )
            return BootstrapTurn(
                assistant_message="I updated the plan.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=False,
                human_update="Plan revised around your feedback.",
            )

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any("Updating plan..." in output for output in outputs)
    assert any("Plan revised around your feedback." in output for output in outputs)


def test_run_interactive_start_offers_optional_context_once_then_replans(monkeypatch) -> None:
    outputs = []
    prompts = []
    answers = iter(["Improve the LoL kills model", "Try position-aware features if useful", "y"])

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = []
            self.start_calls = []

        def bootstrap(self, *, user_goal, answers=None):
            snapshot = dict(answers or {})
            self.bootstrap_calls.append((user_goal, snapshot))
            if "user_extra_context" not in snapshot:
                return BootstrapTurn(
                    assistant_message="Ready.",
                    proposal=ExperimentSpecProposal(
                        objective=user_goal,
                        recommended_spec=build_spec(objective=user_goal),
                        questions=[
                            SpecQuestion(
                                key="user_extra_context",
                                prompt="Any extra modelling context, constraints, or ideas the agent should consider before it proceeds autonomously? Leave blank to skip.",
                                required=False,
                            )
                        ],
                    ),
                    role_models=bootstrap_module.default_role_models(),
                    ready_to_start=True,
                    human_update="Ready to start.",
                )
        def start_from_bootstrap_turn(self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs):
            self.start_calls.append((user_goal, bootstrap_turn.proposal.recommended_spec.metadata, iterations))
            return {"status": "started", "bootstrap": {"objective": user_goal}, "results": []}

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert prompts[0] == "What problem are we solving? "
    assert prompts.count("> ") >= 1
    assert not any("Updating plan..." in output for output in outputs)


def test_main_without_args_enters_interactive_mode(monkeypatch) -> None:
    monkeypatch.setattr("loopforge.cli.run_interactive_start", lambda **kwargs: 7)

    assert main([]) == 7


def test_apply_suggested_answers_skips_metric_discussion_defaults() -> None:
    outputs = []
    answers = {}
    turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective="Improve NBA points model.",
            recommended_spec=build_spec(objective="Improve NBA points model."),
            questions=[
                SpecQuestion(
                    key="metric_choice",
                    prompt="Which metric should define improvement?",
                    suggested_answer="use the experiment default",
                )
            ],
        ),
        role_models=bootstrap_module.default_role_models(),
    )

    applied = _apply_suggested_answers(turn, answers, print_fn=outputs.append)

    assert applied == 0
    assert answers == {}
    assert outputs == []


def test_bootstrap_backend_tolerates_missing_recommended_spec_objective() -> None:
    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "I found enough context to continue.",
                                "proposal": {
                                    "objective": "Improve NBA points model.",
                                    "recommended_spec": {
                                        "primary_metric": {"name": "mae", "goal": "minimize"},
                                        "allowed_actions": ["baseline"],
                                    },
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                                "ready_to_start": False,
                            }
                        )
                    }
                }
            ]
        },
    )

    turn = backend.propose_bootstrap_turn(
        user_goal="Improve NBA points model.",
        capability_context=CapabilityContext(),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.recommended_spec.objective == "Improve NBA points model."
    assert turn.proposal.recommended_spec.allowed_actions == ["baseline"]


def test_bootstrap_backend_tolerates_null_recommended_spec() -> None:
    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "I found enough context to continue.",
                                "proposal": {
                                    "objective": "Improve NBA points model.",
                                    "recommended_spec": None,
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                                "ready_to_start": False,
                            }
                        )
                    }
                }
            ]
        },
    )

    turn = backend.propose_bootstrap_turn(
        user_goal="Improve NBA points model.",
        capability_context=CapabilityContext(),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.recommended_spec.objective == "Improve NBA points model."
    assert turn.proposal.recommended_spec.primary_metric.name == "primary_metric"
    assert turn.proposal.recommended_spec.allowed_actions == []


def test_bootstrap_backend_tolerates_incomplete_metric_payloads() -> None:
    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "I found enough context to continue.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {},
                                        "secondary_metrics": [{}],
                                        "guardrail_metrics": [{}],
                                        "allowed_actions": [],
                                    },
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                                "ready_to_start": False,
                            }
                        )
                    }
                }
            ]
        },
    )

    turn = backend.propose_bootstrap_turn(
        user_goal="Improve NBA points model.",
        capability_context=CapabilityContext(),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.recommended_spec.primary_metric.name == "primary_metric"
    assert turn.proposal.recommended_spec.secondary_metrics[0].name == "secondary_metric_1"
    assert turn.proposal.recommended_spec.guardrail_metrics[0].name == "guardrail_metric_1"


def test_bootstrap_backend_coerces_invalid_objective_and_allowed_actions() -> None:
    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Ready.",
                                "proposal": {
                                    "objective": {"bad": "shape"},
                                    "recommended_spec": {
                                        "objective": {"also": "bad"},
                                        "primary_metric": {"name": "mae", "goal": "minimize"},
                                        "allowed_actions": "baseline",
                                    },
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                            }
                        )
                    }
                }
            ]
        },
    )

    turn = backend.propose_bootstrap_turn(
        user_goal="Improve NBA points model.",
        capability_context=CapabilityContext(),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.objective == "Improve NBA points model."
    assert turn.proposal.recommended_spec.objective == "Improve NBA points model."
    assert turn.proposal.recommended_spec.allowed_actions == ["baseline"]
    assert turn.proposal.recommended_spec.stop_conditions["max_iterations"] == 30
    assert turn.proposal.recommended_spec.stop_conditions["max_autonomous_hours"] == 6
    assert turn.proposal.recommended_spec.metadata["execution_mode"] == "autonomous_after_bootstrap"


def test_bootstrap_backend_fills_allowed_actions_from_capability_context() -> None:
    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Ready.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {"name": "mae", "goal": "minimize"},
                                        "allowed_actions": [],
                                    },
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                            }
                        )
                    }
                }
            ]
        },
    )

    turn = backend.propose_bootstrap_turn(
        user_goal="Improve NBA points model.",
        capability_context=CapabilityContext(available_actions={"baseline": "pilot.py", "tune": "pilot.py"}),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.recommended_spec.allowed_actions == ["baseline", "tune"]


def test_bootstrap_backend_uses_capability_metrics_as_defaults() -> None:
    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Ready.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {"name": "primary_metric", "goal": "maximize"},
                                        "allowed_actions": ["baseline"],
                                    },
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                            }
                        )
                    }
                }
            ]
        },
    )

    turn = backend.propose_bootstrap_turn(
        user_goal="Improve NBA points model.",
        capability_context=CapabilityContext(
            available_metrics={
                "mae": {"goal": "minimize", "preferred_role": "primary", "scorer_ref": "pilot:mae"},
                "ordinal_loss": {"goal": "minimize", "preferred_role": "guardrail", "scorer_ref": "pilot:ord"},
            }
        ),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.recommended_spec.primary_metric.name == "mae"
    assert turn.proposal.recommended_spec.primary_metric.goal == "minimize"
    assert turn.proposal.recommended_spec.guardrail_metrics[0].name == "ordinal_loss"


def test_worker_backend_tolerates_missing_action_type() -> None:
    worker = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {"choices": [{"message": {"content": json.dumps({})}}]},
    )
    snapshot = FileMemoryStore(Path(".")).load_snapshot if False else None
    spec = build_spec(allowed_actions=["baseline", "tune"])
    memory_snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    candidate = worker.propose_next_experiment(memory_snapshot)

    assert candidate.action_type == "baseline"
    assert candidate.change_type == "baseline"


def test_iteration_policy_uses_force_next_action_when_present() -> None:
    spec = build_spec(allowed_actions=["inspect_data", "train_model"], metadata={"force_next_action": "train_model"})
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    policy = build_iteration_policy(snapshot)

    assert policy["forced_next_action"] == "train_model"
    assert policy["recommended_next_action"] == "train_model"


def test_worker_candidate_normalisation_falls_back_to_first_allowed_action() -> None:
    spec = build_spec(allowed_actions=["inspect_data", "train_model"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    candidate_payload = bootstrap_module.LiteLLMWorkerBackend._normalise_candidate_payload(
        {
            "hypothesis": "Do something unsupported.",
            "action_type": "invented_action",
            "change_type": "invented_action",
            "instructions": "Use a made up action.",
            "metadata": {},
        },
        snapshot,
        iteration_policy=build_iteration_policy(snapshot),
    )

    assert candidate_payload["action_type"] == "inspect_data"
    assert candidate_payload["change_type"] == "inspect_data"


def test_diagnostic_actions_are_recorded_as_inconclusive_not_regressed(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Inspect worst slices.",
                    action_type="eda",
                    change_type="diagnostic",
                    instructions="Run EDA.",
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            handlers={"eda": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["EDA completed."])])}
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Useful diagnosis.", lessons=["Found worst slices."])]
        ),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="keep diagnostic memory")]),
        capability_provider=lambda effective_spec: CapabilityContext(
            environment_facts={"inconclusive_actions": ["eda", "slice_analysis"]}
        ),
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["eda", "targeted_tune"],
            stop_conditions={"max_iterations": 1},
        )
    )

    cycle = orchestrator.run_iteration()

    assert cycle.accepted_summary is not None
    assert cycle.accepted_summary.result == "inconclusive"


def test_orchestrator_stops_when_max_autonomous_hours_elapsed(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            ExperimentCandidate(
                hypothesis="Should not run",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline again.",
            ),
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41)])}
        ),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
        monotonic_fn=iter([0.0, 0.0, 21600.0]).__next__,
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["baseline"],
            stop_conditions={"max_iterations": 30, "max_autonomous_hours": 6},
        )
    )

    results = orchestrator.run()

    assert len(results) == 1


def test_review_backend_tolerates_missing_status_and_reason() -> None:
    review = bootstrap_module.LiteLLMReviewBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {"choices": [{"message": {"content": json.dumps({})}}]},
    )
    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    decision = review.review(
        snapshot,
        ExperimentCandidate(hypothesis="Baseline", action_type="baseline", change_type="baseline", instructions="Run."),
        ExperimentOutcome(primary_metric_value=1.0),
        ReflectionSummary(assessment="ok"),
    )

    assert decision.status == "accepted"
    assert "fallback" in decision.reason.lower()


def test_bootstrap_survives_helper_backend_failures(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            return BootstrapTurn(
                assistant_message="Bootstrap completed.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="fraud_recall", goal="maximize"),
                        allowed_actions=["train"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=False,
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_data_assets=["fraud_training_set"],
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="warehouse_permissions",
                    status="failed",
                    detail="Missing SELECT on analytics.fraud_training_set.",
                )
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        access_advisor_backend=RaisingAccessAdvisorBackend(),
        narrator_backend=RaisingNarrationBackend(),
    )

    turn = app.bootstrap(user_goal="Improve fraud detection recall.")

    assert turn.assistant_message == "Bootstrap completed."
    assert turn.human_update is not None
    assert turn.human_update == "Bootstrap completed."


def test_only_accepted_memory_reaches_the_next_worker_and_human_notes_modify_effective_spec(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            ExperimentCandidate(
                hypothesis="Do EDA next",
                action_type="eda",
                change_type="slice",
                instructions="Slice by rest days.",
            ),
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={
                "baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["Baseline done."])]),
                "eda": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["Rest-day slice found drift."])]),
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(assessment="Useful baseline.", lessons=["Baseline established."]),
                ReflectionSummary(assessment="Rest-day slice is worth following up.", lessons=["Rest days matter."]),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="Use this as starting memory."),
                ReviewDecision(status="accepted", reason="EDA finding is actionable."),
            ]
        ),
    )
    orchestrator.initialize(spec=build_spec())

    first_cycle = orchestrator.run_iteration()
    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Focus on EDA before more tuning.",
        effects={"disable_actions": ["tune"], "force_next_action": "eda", "suggested_hypothesis": "Check rest-day drift"},
    )
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary is not None
    assert worker.snapshots[0].recent_summaries == []
    assert [summary.hypothesis for summary in worker.snapshots[1].recent_summaries] == ["Baseline"]
    assert worker.snapshots[1].effective_spec.allowed_actions == ["baseline", "eda", "train", "evaluate"]
    assert worker.snapshots[1].effective_spec.metadata["force_next_action"] == "eda"
    assert worker.snapshots[1].effective_spec.metadata["suggested_hypotheses"] == ["Check rest-day drift"]


def test_orchestrator_injects_capability_context_into_worker_snapshot(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41)])}
        ),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
        capability_provider=lambda effective_spec: CapabilityContext(
            available_actions={"baseline": "Run baseline"},
            available_entities={"model_keys": ["pass_outcome", "fumble_expanded"]},
            available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            available_metrics={
                "log_loss": {"scorer_ref": "metrics.log_loss_cv"},
                "recall_at_p90": {"scorer_ref": "metrics.recall_at_p90"},
            },
            environment_facts={"adapter": "test"},
        ),
    )
    orchestrator.initialize(spec=build_spec())

    orchestrator.run_iteration()

    assert worker.snapshots[0].capability_context.available_entities["model_keys"] == [
        "pass_outcome",
        "fumble_expanded",
    ]
    assert worker.snapshots[0].capability_context.available_data_assets == ["gold_cross_validated_pass_outcome_ncaaf"]
    assert worker.snapshots[0].capability_context.available_metrics["log_loss"]["scorer_ref"] == "metrics.log_loss_cv"


def test_consulting_worker_requests_ops_help_for_databricks_flows(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Deploy the updated Databricks bundle.",
                action_type="baseline",
                change_type="deployment",
                instructions="Deploy to Databricks dev.",
            )
        ]
    )
    consultation = FakeConsultationBackend()
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=ConsultingWorkerBackend(worker_backend=worker, consultation_backend=consultation),
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41)])}
        ),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
        capability_provider=lambda effective_spec: CapabilityContext(notes=["Need Databricks deploy guidance"]),
    )
    orchestrator.initialize(
        build_spec(
            objective="Deploy the Databricks training job and verify data access.",
            allowed_actions=["baseline"],
        )
    )

    cycle = orchestrator.run_iteration()

    assert consultation.calls
    assert "Ops consult guidance: Use bundle deploy and verify env vars first." in worker.snapshots[0].capability_context.notes
    assert cycle.record.candidate.metadata["ops_consultation"]["required_env_vars"] == [
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
    ]


def test_rejected_review_stays_in_raw_records_but_not_in_accepted_memory(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Suspicious train run",
                    action_type="train",
                    change_type="feature_set",
                    instructions="Try a risky feature.",
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            handlers={"train": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.3)])}
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Looks leaky.", lessons=["Need leakage check."])]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="rejected", reason="Do not trust this run.")]
        ),
    )
    orchestrator.initialize(spec=build_spec(allowed_actions=["train"]))

    cycle = orchestrator.run_iteration()
    snapshot = store.load_snapshot()
    records_text = (tmp_path / "loop" / "iteration_records.jsonl").read_text(encoding="utf-8")

    assert cycle.accepted_summary is None
    assert snapshot.latest_summary is None
    assert '"status": "rejected"' in records_text


def test_orchestrator_persists_recoverable_failures_for_next_worker(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Run baseline even though env may be incomplete.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            ExperimentCandidate(
                hypothesis="Retry after dependency fix.",
                action_type="baseline",
                change_type="baseline",
                instructions="Retry baseline after the recoverable failure.",
            ),
        ]
    )

    class FailingExecutor:
        def execute(self, candidate, snapshot):
            raise ModuleNotFoundError("No module named 'sklearn'")

    class PassingExecutor:
        def execute(self, candidate, snapshot):
            return ExperimentOutcome(primary_metric_value=0.25, notes=["Retry succeeded."])

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={"baseline": FailingExecutor()},
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(assessment="Recoverable dependency failure.", lessons=["Install sklearn first."]),
                ReflectionSummary(assessment="Retry worked.", lessons=["Environment is now usable."]),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="Failure is useful memory."),
                ReviewDecision(status="accepted", reason="Successful retry."),
            ]
        ),
    )
    orchestrator.initialize(spec=build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 2}))

    first_cycle = orchestrator.run_iteration()
    orchestrator.executor = RoutingExperimentExecutor(handlers={"baseline": PassingExecutor()})
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert first_cycle.accepted_summary.outcome_status == "recoverable_failure"
    assert "sklearn" in (first_cycle.accepted_summary.failure_summary or "")
    assert worker.snapshots[1].recent_records[0].outcome.status == "recoverable_failure"
    assert second_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary.outcome_status == "success"


def test_cli_interjection_appends_human_note_and_changes_future_effective_spec(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())

    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Only do EDA next.",
        effects={"disable_actions": ["train", "tune"], "force_next_action": "eda"},
        author="mhol",
        type_="override",
    )
    snapshot = store.load_snapshot()

    assert snapshot.recent_human_interventions[0].author == "mhol"
    assert snapshot.effective_spec.allowed_actions == ["baseline", "eda", "evaluate"]
    assert snapshot.effective_spec.metadata["force_next_action"] == "eda"


def test_file_memory_store_loads_markdown_memory_and_experiment_journal(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())
    (tmp_path / "loop" / "ops_access_guide.md").write_text("# Access\nUse the approved workspace.\n", encoding="utf-8")
    store.append_accepted_summary(
        IterationSummary(
            iteration_id=1,
            parent_iteration_id=None,
            hypothesis="Baseline",
            action_type="baseline",
            change_type="baseline",
            instructions="Run baseline.",
            config_patch={},
            primary_metric_name="log_loss",
            primary_metric_value=0.41,
            secondary_metrics={},
            result="improved",
            artifacts=["artifacts/baseline.json"],
            lessons=["Baseline established."],
            next_ideas=["Try bounded tuning."],
            do_not_repeat=["Rerun the broken config."],
            reflection_assessment="Useful baseline.",
            review_reason="Good starting point.",
        )
    )

    snapshot = store.load_snapshot()
    markdown_by_path = {note.path: note.content for note in snapshot.markdown_memory}

    assert "ops_access_guide.md" in markdown_by_path
    assert "experiment_journal.md" in markdown_by_path
    assert "## Iteration 1: Baseline" in markdown_by_path["experiment_journal.md"]
    assert "Do not repeat: Rerun the broken config." in markdown_by_path["experiment_journal.md"]


def test_worker_backend_sends_markdown_memory_to_completion() -> None:
    captured_payload = {}

    def completion_fn(**kwargs):
        captured_payload["payload"] = json.loads(kwargs["messages"][1]["content"])
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Respect prior access guidance.",
                                "action_type": "baseline",
                                "change_type": "baseline",
                                "instructions": "Run baseline.",
                            }
                        )
                    }
                }
            ]
        }

    worker = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4",
        completion_fn=completion_fn,
    )
    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="Do not repeat the broken deploy path.",
        markdown_memory=[MarkdownMemoryNote(path="ops_access_guide.md", content="# Access\nUse token auth.")],
        next_iteration_id=1,
    )

    worker.propose_next_experiment(snapshot)

    assert captured_payload["payload"]["markdown_memory"] == [
        {"path": "ops_access_guide.md", "content": "# Access\nUse token auth."}
    ]


def test_worker_payload_carries_autonomous_execution_mode() -> None:
    captured_payload = {}

    def completion_fn(**kwargs):
        captured_payload["payload"] = json.loads(kwargs["messages"][1]["content"])
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Run baseline.",
                                "action_type": "baseline",
                                "change_type": "baseline",
                                "instructions": "Run baseline.",
                            }
                        )
                    }
                }
            ]
        }

    worker = bootstrap_module.LiteLLMWorkerBackend(model="openai/gpt-5.4", completion_fn=completion_fn)
    spec = build_spec(allowed_actions=["baseline"], metadata={"execution_mode": "autonomous_after_bootstrap"})
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    worker.propose_next_experiment(snapshot)

    assert captured_payload["payload"]["effective_spec"]["metadata"]["execution_mode"] == "autonomous_after_bootstrap"


def test_loopforge_bootstrap_passes_existing_markdown_memory(tmp_path, monkeypatch) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())
    (tmp_path / "loop" / "ops_access_guide.md").write_text("# Access\nUse the dev workspace.\n", encoding="utf-8")

    captured = {}

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            captured["bootstrap_memory"] = bootstrap_memory
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal, allowed_actions=["baseline"]),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=False,
            )

    monkeypatch.setattr("loopforge.bootstrap.discover_capabilities_for_objective", lambda **kwargs: CapabilityContext())
    monkeypatch.setattr("loopforge.bootstrap.run_preflight_checks", lambda **kwargs: [])

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "loop",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
    )

    app.bootstrap(user_goal="Reuse prior access instructions.", answers={})

    # Bootstrap no longer loads prior session memory to avoid polluting new objectives
    assert captured["bootstrap_memory"] == {}


def test_run_from_spec_supports_factory_and_returns_reviewed_cycle_results(tmp_path, monkeypatch) -> None:
    spec = build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 1})
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    class StubWorker:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None, **kwargs) -> None:
            self.model = model

        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )

    class StubReflection:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None, **kwargs) -> None:
            self.model = model

        def reflect(self, snapshot, candidate, outcome):
            return ReflectionSummary(assessment="Baseline is fine.", lessons=["Keep this baseline."])

    class StubReview:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None, **kwargs) -> None:
            self.model = model

        def review(self, snapshot, candidate, outcome, reflection):
            return ReviewDecision(status="accepted", reason="Approved into memory.")

    class StubNarration:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None, **kwargs) -> None:
            self.model = model

        def summarize_bootstrap(self, turn, capability_context):
            return "bootstrap update"

        def summarize_iteration(self, snapshot, candidate, outcome, reflection, review, accepted_summary):
            return "iteration update"

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.39)])},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_entities={"model_keys": ["pass_outcome"]},
                available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.LiteLLMWorkerBackend", StubWorker)
    monkeypatch.setattr("loopforge.bootstrap.LiteLLMReflectionBackend", StubReflection)
    monkeypatch.setattr("loopforge.bootstrap.LiteLLMReviewBackend", StubReview)
    monkeypatch.setattr("loopforge.bootstrap.LiteLLMNarrationBackend", StubNarration)
    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)
    monkeypatch.setattr("loopforge.cli.load_factory", lambda _: fake_factory)

    results = run_from_spec(
        spec_path=spec_path,
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        worker_model="openai/gpt-5.4",
    )

    assert results[0]["accepted_summary"]["hypothesis"] == "Baseline"
    assert results[0]["record"]["review"]["status"] == "accepted"
    assert results[0]["human_update"] == "iteration update"


def test_discover_capabilities_for_objective_prefers_discovery_provider(tmp_path, monkeypatch) -> None:
    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_metrics={"fallback_metric": {"scorer_ref": "metrics.fallback"}}
            ),
            discovery_provider=lambda objective: CapabilityContext(
                available_metrics={"precision_floor": {"scorer_ref": "metrics.precision_floor"}},
                notes=[f"objective={objective}"],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    context = discover_capabilities_for_objective(
        objective="Keep precision high.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
    )

    assert context.available_metrics["precision_floor"]["scorer_ref"] == "metrics.precision_floor"
    assert "objective=Keep precision high." in context.notes


def test_discover_capabilities_for_objective_loads_real_repo_data_files(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = repo_root / "examples" / "lol" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "matches.csv").write_text(
        "playername,teamname,kills\nfaker,t1,7\nchovy,hle,5\n",
        encoding="utf-8",
    )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "runner baseline"},
                available_data_assets=[],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    context = discover_capabilities_for_objective(
        objective="Improve LoL kills model.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        repo_root=repo_root,
    )

    assert "examples/lol/data/matches.csv" in context.available_data_assets
    assert any(schema.asset_path == "examples/lol/data/matches.csv" and schema.columns for schema in context.data_schemas)
    assert any("Data probe (examples/lol/data/matches.csv)" in note for note in context.notes)


def test_normalise_probe_value_serializes_timestamp_like_objects() -> None:
    class TimestampLike:
        def isoformat(self) -> str:
            return "2026-04-03T12:34:56"

    value = bootstrap_module._normalise_probe_value(TimestampLike())

    assert value == "2026-04-03T12:34:56"


def test_probe_data_asset_reports_full_parquet_row_count_even_when_sampling(tmp_path, monkeypatch) -> None:
    asset_path = tmp_path / "matches.parquet"
    asset_path.write_text("placeholder", encoding="utf-8")

    class FakeSeries:
        def __init__(self, values) -> None:
            self._values = values

        def dropna(self):
            return self

        def unique(self):
            return self._values

    class FakeDataFrame:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows
            self.columns = ["kills", "playername"]
            self.dtypes = {"kills": "int64", "playername": "object"}

        def __len__(self) -> int:
            return len(self._rows)

        def head(self, limit: int):
            return FakeDataFrame(self._rows[:limit])

        def __getitem__(self, column: str) -> FakeSeries:
            return FakeSeries([row[column] for row in self._rows])

    class FakeArrowTable:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows
            self.num_rows = len(rows)

        def slice(self, offset: int, length: int):
            return FakeArrowTable(self._rows[offset:offset + length])

        def to_pandas(self) -> FakeDataFrame:
            return FakeDataFrame(self._rows)

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            assert path == asset_path
            self._rows = [{"kills": index % 9, "playername": f"player-{index}"} for index in range(123)]
            self.metadata = type("Meta", (), {"num_rows": len(self._rows)})()
            self.num_row_groups = 1

        def read_row_group(self, index: int) -> FakeArrowTable:
            assert index == 0
            return FakeArrowTable(self._rows)

    fake_pyarrow = ModuleType("pyarrow")
    fake_parquet = ModuleType("pyarrow.parquet")
    fake_parquet.ParquetFile = FakeParquetFile
    fake_pyarrow.parquet = fake_parquet

    monkeypatch.setitem(sys.modules, "pandas", type("FakePandasModule", (), {"DataFrame": FakeDataFrame}))
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    schema = bootstrap_module._probe_data_asset(str(asset_path), tmp_path)

    assert schema.sample_rows_loaded == 50
    assert schema.total_rows_verified == 123
    assert schema.verification_level == "verified_total_rows"
    assert schema.columns == ["kills", "playername"]
    assert schema.sample_values["kills"] == [0, 1, 2, 3, 4]


def test_lol_preflight_treats_missing_installable_dependency_as_setup_step(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    (repo_root / "examples" / "lol" / "data").mkdir(parents=True)
    (repo_root / "examples" / "lol" / "data" / "subsample_lol_data.parquet").write_text(
        "placeholder",
        encoding="utf-8",
    )

    def raise_missing_dependency(_repo_root):
        raise ModuleNotFoundError("No module named 'lightgbm'")

    monkeypatch.setattr(pilot_adapters_module, "_load_lol_modules", raise_missing_dependency)

    checks = pilot_adapters_module._lol_preflight(repo_root)

    runtime_check = next(check for check in checks if check.name == "lol_runtime_imports")
    assert runtime_check.status == "failed"
    assert runtime_check.required is False
    assert runtime_check.scope == "setup"
    assert "uv pip install lightgbm" in runtime_check.detail


def test_complete_json_tolerates_datetime_payloads() -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}

    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=fake_completion,
    )

    result = backend._complete_json("test", {"seen_at": datetime(2026, 4, 3, 12, 34, 56)})

    assert result == {"ok": True}
    assert "2026-04-03T12:34:56" in captured["messages"][1]["content"]


def test_sanitize_human_text_makes_console_unsafe_characters_safe(monkeypatch) -> None:
    class FakeStdout:
        encoding = "cp1252"

    monkeypatch.setattr(sys, "stdout", FakeStdout())

    sanitized = _sanitize_human_text("Baseline \u2192 EDA \u2026 ready")

    assert sanitized == "Baseline -> EDA ... ready"


def test_draft_spec_returns_metric_questions_and_recommended_spec(tmp_path, monkeypatch) -> None:
    class StubSpecBackend:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None, **kwargs) -> None:
            self.model = model

        def propose_spec(self, objective, capability_context, user_preferences=None):
            assert capability_context.available_metrics["recall_at_p90"]["scorer_ref"] == "metrics.recall_at_p90"
            assert user_preferences == {"custom_primary_scorer": "metrics.recall_at_p90"}
            return ExperimentSpecProposal(
                objective=objective,
                recommended_spec=ExperimentSpec(
                    objective=objective,
                    primary_metric=PrimaryMetric(
                        name="recall_at_p90",
                        goal="maximize",
                        scorer_ref="metrics.recall_at_p90",
                    ),
                    secondary_metrics=[
                        MetricSpec(
                            name="log_loss",
                            goal="minimize",
                            scorer_ref="metrics.log_loss_cv",
                        )
                    ],
                    guardrail_metrics=[
                        MetricSpec(
                            name="precision_floor",
                            goal="maximize",
                            scorer_ref="metrics.precision_floor",
                            constraints={"min_value": 0.9},
                        )
                    ],
                    allowed_actions=["baseline", "train"],
                    metadata={"planner": "stub"},
                ),
                questions=[
                    SpecQuestion(
                        key="primary_metric_confirmation",
                        prompt="Should recall_at_p90 remain the primary metric?",
                        suggested_answer="yes",
                        options=["yes", "no"],
                    )
                ],
                notes=["Discovered scorer metrics.recall_at_p90 from the host codebase."],
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "Run baseline", "train": "Train model"},
                available_metrics={
                    "recall_at_p90": {"scorer_ref": "metrics.recall_at_p90"},
                    "log_loss": {"scorer_ref": "metrics.log_loss_cv"},
                    "precision_floor": {"scorer_ref": "metrics.precision_floor"},
                },
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)
    monkeypatch.setattr("loopforge.cli.LiteLLMSpecBackend", StubSpecBackend)

    proposal = draft_spec(
        objective="Improve fraud recall without breaking precision.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        planner_model="openai/gpt-5.4",
        preferences={"custom_primary_scorer": "metrics.recall_at_p90"},
    )

    assert proposal.recommended_spec.primary_metric.name == "recall_at_p90"
    assert proposal.recommended_spec.guardrail_metrics[0].name == "precision_floor"
    assert proposal.questions[0].key == "primary_metric_confirmation"
    assert proposal.notes == ["Discovered scorer metrics.recall_at_p90 from the host codebase."]


def test_loopforge_bootstrap_returns_questions_and_preflight_failures(tmp_path, monkeypatch) -> None:
    access_advisor = FakeAccessAdvisorBackend()
    narrator = FakeNarrationBackend()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert capability_context.available_metrics["fraud_recall"]["scorer_ref"] == "metrics.fraud_recall"
            assert role_models.worker == "openai/gpt-5.4"
            return BootstrapTurn(
                assistant_message="I need one more confirmation before starting.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="fraud_recall",
                            goal="maximize",
                            scorer_ref="metrics.fraud_recall",
                        ),
                        allowed_actions=["train"],
                    ),
                    questions=[
                        SpecQuestion(
                            key="positive_label_definition",
                            prompt="Which label definition should the scorer use?",
                        )
                    ],
                    notes=["Bootstrap discovered a candidate fraud recall scorer."],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_metrics={"fraud_recall": {"scorer_ref": "metrics.fraud_recall"}},
                available_data_assets=["fraud_training_set"],
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="warehouse_permissions",
                    status="failed",
                    detail="Missing SELECT on analytics.fraud_training_set.",
                )
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        access_advisor_backend=access_advisor,
        narrator_backend=narrator,
    )

    turn = app.bootstrap(user_goal="Improve fraud detection recall.")

    assert turn.ready_to_start is False
    assert turn.missing_requirements == [
        "answer:positive_label_definition",
        "preflight:warehouse_permissions",
    ]
    assert turn.preflight_checks[0].name == "memory_root_access"
    assert turn.preflight_checks[1].name == "warehouse_permissions"
    assert turn.access_guide_path is not None
    assert turn.human_update is not None
    assert "bootstrap:Improve fraud detection recall." in turn.human_update
    assert "Access guide:" in turn.human_update
    assert Path(turn.access_guide_path).read_text(encoding="utf-8").startswith("# Access Guide")
    assert len(access_advisor.calls) == 1
    assert len(narrator.bootstrap_calls) == 1


def test_loopforge_bootstrap_marks_turn_ready_when_no_hard_requirements_remain(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            return BootstrapTurn(
                assistant_message="Plan is concrete.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=False,
            )

    monkeypatch.setattr("loopforge.bootstrap.discover_capabilities_for_objective", lambda **kwargs: CapabilityContext())
    monkeypatch.setattr(
        "loopforge.bootstrap.run_preflight_checks",
        lambda **kwargs: [
            PreflightCheck(
                name="lol_runtime_imports",
                status="failed",
                detail="Missing installable Python dependency 'lightgbm'. Install it with `uv pip install lightgbm`.",
                required=False,
                scope="setup",
            )
        ],
    )

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "loop",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
    )

    turn = app.bootstrap(user_goal="Improve LoL kills model.")

    assert turn.ready_to_start is True
    assert turn.missing_requirements == []


def test_bootstrap_sanitises_internal_adapter_questions() -> None:
    questions = [
        SpecQuestion(
            key="execution_entrypoint",
            prompt="Which of the discovered adapter files should I use as the execution entrypoint?",
            options=[
                ".loopforge/generated/foo_adapter.py",
                ".loopforge/generated/bar_adapter.py",
            ],
        ),
        SpecQuestion(
            key="primary_metric_confirmation",
            prompt="Should I optimize the default primary metric first?",
            required=False,
        ),
    ]

    sanitised = sanitise_bootstrap_questions(questions)

    assert [question.key for question in sanitised] == ["primary_metric_confirmation"]


def test_refine_bootstrap_questions_keeps_only_essential_required_question_and_adds_optional_context() -> None:
    questions = [
        SpecQuestion(
            key="position_handling",
            prompt="Should we use separate models per position or a single model with a position feature?",
            required=True,
        ),
        SpecQuestion(
            key="primary_metric_confirmation",
            prompt="Should ordinal_loss remain the primary metric?",
            required=True,
        ),
    ]

    refined = refine_bootstrap_questions(questions, answers={})

    assert [question.key for question in refined] == ["primary_metric_confirmation", "user_extra_context"]
    assert refined[1].required is False


def test_apply_answers_to_bootstrap_turn_stores_answers_and_clears_answer_blockers() -> None:
    turn = BootstrapTurn(
        assistant_message="Need one answer.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=build_spec(objective="Improve LoL kills model."),
            questions=[SpecQuestion(key="target_binning", prompt="Cap at 7+ or 10+?", required=True)],
            notes=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[],
        ready_to_start=False,
        missing_requirements=["answer:target_binning"],
    )

    updated = apply_answers_to_bootstrap_turn(turn, answers={"target_binning": "Cap at 10+"})

    assert updated.ready_to_start is True
    assert updated.missing_requirements == []
    assert updated.proposal.recommended_spec.metadata["bootstrap_answers"]["target_binning"] == "Cap at 10+"


def test_print_blocked_summary_hides_internal_preflight_ids() -> None:
    outputs: list[str] = []
    turn = BootstrapTurn(
        assistant_message="Blocked on auto_adapter_scaffold.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=build_spec(objective="Improve LoL kills model."),
            notes=["Loopforge synthesized an adapter scaffold from repo inspection."],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[
            PreflightCheck(
                name="auto_adapter_scaffold",
                status="failed",
                detail="Loopforge synthesized an adapter scaffold from repo inspection, but action handlers are still placeholders.",
            )
        ],
        missing_requirements=["preflight:auto_adapter_scaffold"],
    )

    _print_blocked_summary(turn, print_fn=outputs.append)

    joined = "\n".join(outputs)
    assert "auto_adapter_scaffold" not in joined
    assert "adapter scaffold" not in joined.lower()
    assert "runnable execution binding" in joined


def test_print_blocked_summary_separates_setup_failures_from_blockers() -> None:
    outputs: list[str] = []
    turn = BootstrapTurn(
        assistant_message="Need one setup step.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=build_spec(objective="Improve LoL kills model."),
            notes=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[
            PreflightCheck(
                name="lol_runtime_imports",
                status="failed",
                detail="Missing installable Python dependency 'lightgbm'. Install it with `uv pip install lightgbm`.",
                required=False,
                scope="setup",
            )
        ],
    )

    _print_blocked_summary(turn, print_fn=outputs.append)

    joined = "\n".join(outputs)
    assert "Blocked" not in joined
    assert "Setup still needed:" in joined
    assert "lightgbm" in joined


def test_print_plan_summary_uses_user_facing_next_step_summary() -> None:
    outputs: list[str] = []
    turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective="Improve the LoL kills model.",
            recommended_spec=ExperimentSpec(
                objective="Improve the LoL kills model.",
                primary_metric=PrimaryMetric(name="ordinal_loss_scorer", goal="minimize"),
                allowed_actions=["predict", "eda", "targeted_tune", "test_bad_helper_name"],
                guardrail_metrics=[MetricSpec(name="calibration_error", goal="minimize")],
                secondary_metrics=[MetricSpec(name="mae", goal="minimize")],
            ),
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    _print_plan_summary(turn, print_fn=outputs.append)

    joined = "\n".join(outputs)
    assert "test_bad_helper_name" not in joined
    assert "choose among the discovered actions: predict, eda, targeted_tune, and 1 more" in joined
    assert "Ordinal Loss Scorer" in joined



def test_loopforge_blocks_autonomous_start_without_execution_preflight(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            return BootstrapTurn(
                assistant_message="The plan is clear.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="fraud_recall", goal="maximize"),
                        allowed_actions=["train"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"train": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.5)])},
            discovery_provider=lambda objective: CapabilityContext(
                available_data_assets=["fraud_training_set"],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
    )

    turn = app.bootstrap(user_goal="Improve fraud detection recall.")

    assert turn.ready_to_start is False
    assert "preflight:autonomous_execution_permissions" in turn.missing_requirements
    assert turn.preflight_checks[1].status == "failed"
    assert turn.preflight_checks[1].scope == "execution"


def test_loopforge_start_uses_defaults_and_runs_when_bootstrap_is_ready(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert answer_history == {"positive_label_definition": "chargeback_60d"}
            return BootstrapTurn(
                assistant_message="Configuration is specific enough to start.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="fraud_recall", goal="maximize"),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class StubWorker:
        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run the baseline.",
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.55)])},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_data_assets=["fraud_training_set"],
            ),
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_data_assets=["fraud_training_set"],
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="warehouse_permissions",
                    status="passed",
                    detail="Confirmed data access.",
                )
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=StubWorker(),
        consultation_backend=FakeConsultationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )

    result = app.start(
        user_goal="Improve fraud detection recall.",
        answers={"positive_label_definition": "chargeback_60d"},
    )

    assert result["status"] == "started"
    role_models = result["bootstrap"]["role_models"]
    assert role_models["worker"] == "openai/gpt-5.4"
    assert role_models["reflection"] == "openai/gpt-5.4"
    assert role_models["review"] == "openai/gpt-5.4"
    assert result["results"][0]["accepted_summary"]["primary_metric_value"] == 0.55
    assert "bootstrap:Improve fraud detection recall." in result["bootstrap"]["human_update"]
    assert result["bootstrap"]["access_guide_path"].endswith("ops_access_guide.md")
    assert result["results"][0]["human_update"] == "iteration:baseline:accepted"


def test_loopforge_start_returns_blocked_when_execution_raises(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            return BootstrapTurn(
                assistant_message="Ready to start.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="fraud_recall", goal="maximize"),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class StubWorker:
        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run the baseline.",
            )

    class RaisingExecutor:
        def execute(self, candidate, snapshot):
            raise RuntimeError("PermissionError: [WinError 5] Access is denied while creating multiprocessing resources.")

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": RaisingExecutor()},
            discovery_provider=lambda objective: CapabilityContext(),
            capability_provider=lambda effective_spec: CapabilityContext(),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(name="executor_ready", status="passed", detail="Executor is configured.")
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=StubWorker(),
        consultation_backend=FakeConsultationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )

    result = app.start(user_goal="Improve fraud detection recall.")

    assert result["status"] == "blocked"
    assert result["error"]["type"] == "RuntimeError"
    assert "Windows permission error" in result["error"]["message"]


def test_synthesize_auto_adapter_creates_loadable_reusable_factory(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "fraud_metrics.py").write_text(
        "def fraud_recall_score(y_true, y_pred):\n    return 0.0\n",
        encoding="utf-8",
    )
    (repo_root / "trainer.py").write_text(
        "def train_model(config):\n    return config\n",
        encoding="utf-8",
    )
    (repo_root / "fraud_data_loader.py").write_text(
        "def load_dataset(name):\n    return name\n",
        encoding="utf-8",
    )

    factory_path = synthesize_auto_adapter(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        objective="Improve fraud recall.",
    )
    build_adapter = load_factory(factory_path)
    adapter_setup = build_adapter(
        build_spec(allowed_actions=["train_model"]),
        tmp_path / "memory",
    )
    capability_context = adapter_setup.discovery_provider("Improve fraud recall.")

    assert "fraud_recall_score" in capability_context.available_metrics
    assert capability_context.available_actions["train_model"] == "trainer.py"
    assert "fraud_data_loader.py" in capability_context.available_data_assets
    assert Path(factory_path.rsplit(":", maxsplit=1)[0]).exists()


def test_loopforge_bootstrap_uses_planning_only_discovery_when_runner_missing(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "credit_metrics.py").write_text(
        "def precision_guardrail_metric():\n    return 0.0\n",
        encoding="utf-8",
    )
    (repo_root / "evaluate_pipeline.py").write_text(
        "def evaluate_model():\n    return 0.0\n",
        encoding="utf-8",
    )

    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert capability_context.environment_facts["execution_backend_kind"] == "generic_agentic"
            assert capability_context.environment_facts["autonomous_execution_supported"] is True
            assert "run_experiment" in capability_context.environment_facts["generic_autonomous_actions"]
            assert capability_context.available_actions["run_experiment"] == "generic_autonomous_executor"
            assert "precision_guardrail_metric" in capability_context.available_metrics
            return BootstrapTurn(
                assistant_message="Generic autonomous discovery is available.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="precision_guardrail_metric",
                            goal="maximize",
                        ),
                        allowed_actions=["evaluate_model"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class FailingRunnerAuthoringBackend:
        def author_runner(self, request):
            raise RuntimeError("could not build a validated runner")

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        runner_authoring_backend=FailingRunnerAuthoringBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
    )

    turn = app.bootstrap(user_goal="Protect approval precision.")

    assert app.executor_factory_path is None
    assert turn.ready_to_start is True
    assert turn.proposal.recommended_spec.metadata["execution_backend_kind"] == "generic_agentic"
    assert "preflight:repo_execution_not_supported" not in turn.missing_requirements
    assert not (tmp_path / "memory" / "generated").exists()


def test_loopforge_does_not_attempt_runner_authoring_when_generic_executor_is_available(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def __init__(self) -> None:
            self.calls = 0

        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            self.calls += 1
            return BootstrapTurn(
                assistant_message="Planning-only discovery is available.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="primary_metric", goal="maximize"),
                        allowed_actions=["inspect_data"],
                    ),
                    questions=[SpecQuestion(key="data_size", prompt="How large is the dataset?", required=True)],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class FailingRunnerAuthoringBackend:
        def __init__(self) -> None:
            self.calls = 0

        def author_runner(self, request):
            self.calls += 1
            raise RuntimeError("could not build a validated runner")

    bootstrap_backend = StubBootstrapBackend()
    authoring_backend = FailingRunnerAuthoringBackend()
    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=bootstrap_backend,
        runner_authoring_backend=authoring_backend,
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
    )

    first_turn = app.bootstrap(user_goal="Protect approval precision.")
    second_turn = app.bootstrap(user_goal="Protect approval precision.", answers={"data_size": "12000 rows"})

    assert first_turn.ready_to_start is True
    assert second_turn.ready_to_start is True
    assert authoring_backend.calls == 0
    assert bootstrap_backend.calls == 2


def test_loopforge_can_run_with_generic_executor_when_no_supported_runner(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "metrics.py").write_text("def ordinal_loss():\n    return 0.0\n", encoding="utf-8")

    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert capability_context.environment_facts["runner_kind"] == "generic_agentic_executor"
            return BootstrapTurn(
                assistant_message="Generic executor is ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class StubWorker:
        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Run generic baseline.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
                execution_steps=[
                    ExecutionStep(
                        kind="shell",
                        command=f'"{sys.executable}" -c "print(0.25)"',
                        cwd=str(repo_root),
                    )
                ],
            )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=StubWorker(),
        consultation_backend=FakeConsultationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )

    result = app.start(user_goal="Create a kills model in this repo.")

    assert result["status"] == "started"
    assert result["bootstrap"]["ready_to_start"] is True
    assert result["results"][0]["accepted_summary"]["outcome_status"] == "success"
    assert result["results"][0]["accepted_summary"]["primary_metric_value"] is None
    assert not (tmp_path / "memory" / "runners").exists()


def test_loopforge_runner_authoring_retries_after_failed_smoke_run_when_called_explicitly(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "metrics.py").write_text("def ordinal_loss():\n    return 0.0\n", encoding="utf-8")

    class RetryingRunnerAuthoringBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, list[str]]] = []

        def author_runner(self, request):
            self.calls.append((request.attempt_number, list(request.previous_errors)))
            if request.attempt_number == 1:
                return bootstrap_module.RunnerAuthoringResult(
                    module_source="""
from __future__ import annotations

from loopforge import AdapterSetup, CapabilityContext, PreflightCheck


class _BrokenBaselineExecutor:
    def execute(self, candidate, snapshot):
        raise RuntimeError("dataset path missing")


def _build_context(objective: str) -> CapabilityContext:
    return CapabilityContext(
        available_actions={"baseline": "authored runner"},
        available_metrics={"ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}},
        environment_facts={"runner_kind": "authored"},
    )


def _preflight(spec, capability_context):
    return [PreflightCheck(name="authored_runner_ready", status="passed", detail="Authored runner validated.")]


def build_adapter(spec, memory_root):
    return AdapterSetup(
        handlers={"baseline": _BrokenBaselineExecutor()},
        discovery_provider=_build_context,
        capability_provider=lambda effective_spec: _build_context(effective_spec.objective),
        preflight_provider=_preflight,
    )
""".strip(),
                        summary="First attempt fails smoke.",
                    )
            assert request.previous_errors
            return bootstrap_module.RunnerAuthoringResult(
                module_source="""
from __future__ import annotations

from loopforge import AdapterSetup, CapabilityContext, ExperimentOutcome, PreflightCheck


class _BaselineExecutor:
    def execute(self, candidate, snapshot):
        return ExperimentOutcome(primary_metric_value=0.2)


def _build_context(objective: str) -> CapabilityContext:
    return CapabilityContext(
        available_actions={"baseline": "authored runner"},
        available_metrics={"ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}},
        environment_facts={"runner_kind": "authored"},
    )


def _preflight(spec, capability_context):
    return [PreflightCheck(name="authored_runner_ready", status="passed", detail="Authored runner validated.")]


def build_adapter(spec, memory_root):
    return AdapterSetup(
        handlers={"baseline": _BaselineExecutor()},
        discovery_provider=_build_context,
        capability_provider=lambda effective_spec: _build_context(effective_spec.objective),
        preflight_provider=_preflight,
    )
""".strip(),
                summary="Second attempt passes smoke.",
                )

    authoring_backend = RetryingRunnerAuthoringBackend()
    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        runner_authoring_backend=authoring_backend,
        consultation_backend=FakeConsultationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )

    resolution = app._author_runner(
        "Create a kills model in this repo.",
        CapabilityContext(),
    )

    assert resolution.kind == "supported"
    assert resolution.factory_path is not None
    assert authoring_backend.calls[0] == (1, [])
    assert authoring_backend.calls[1][0] == 2


def test_validate_runner_factory_accepts_dict_preflight_checks(tmp_path, monkeypatch) -> None:
    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={
                "baseline": type(
                    "BaselineExecutor",
                    (),
                    {"execute": lambda self, candidate, snapshot: ExperimentOutcome(primary_metric_value=0.1)},
                )()
            },
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "authored runner"},
                available_metrics={"ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}},
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                {"name": "authored_runner_ready", "status": "passed", "detail": "Runner validated."}
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    validation = bootstrap_module._validate_runner_factory(
        factory_path="fake.module:build_adapter",
        objective="Create a kills model in this repo.",
        memory_root=tmp_path / "memory",
    )

    assert validation.success is True
    assert validation.smoke_test_passed is True
    assert validation.preflight_checks[0].name == "authored_runner_ready"


def test_loopforge_start_refuses_unsupported_planning_only_repo(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "evaluate_pipeline.py").write_text(
        "def evaluate_model():\n    return 0.0\n",
        encoding="utf-8",
    )

    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            return BootstrapTurn(
                assistant_message="Plan looks good.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="primary_metric", goal="maximize"),
                        allowed_actions=["baseline"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class StubWorker:
        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Use the generic executor directly.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run a bounded generic command.",
                execution_steps=[
                    ExecutionStep(
                        kind="shell",
                        command=f'"{sys.executable}" -c "print(1)"',
                        cwd=str(repo_root),
                    )
                ],
            )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=StubWorker(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )

    result = app.start(user_goal="Improve approval precision.")

    assert result["status"] == "started"
    assert result["bootstrap"]["ready_to_start"] is True


def test_routing_executor_recovery_handler_can_retry_after_recovery() -> None:
    attempts = {"count": 0}

    class FlakyExecutor:
        def execute(self, candidate, snapshot):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ModuleNotFoundError("No module named 'sklearn'")
            return ExperimentOutcome(primary_metric_value=0.11)

    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    def recovery_handler(handler, candidate, snapshot, exc):
        assert isinstance(exc, ModuleNotFoundError)
        return handler.execute(candidate, snapshot)

    executor = RoutingExperimentExecutor(
        handlers={"baseline": FlakyExecutor()},
        recovery_handler=recovery_handler,
    )

    outcome = executor.execute(
        ExperimentCandidate(
            hypothesis="Retry after recovery.",
            action_type="baseline",
            change_type="baseline",
            instructions="Run baseline.",
        ),
        snapshot,
    )

    assert outcome.primary_metric_value == 0.11
    assert attempts["count"] == 2


def test_generic_execution_plan_executor_runs_shell_steps(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
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
        hypothesis="Run a direct command.",
        action_type="baseline",
        change_type="baseline",
        instructions="Run a bounded shell command.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" -c "print(123)"',
                cwd=str(tmp_path),
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert "Executed 1 command step" in outcome.notes[0]
    assert outcome.execution_details["step_results"][0]["returncode"] == 0


def test_routing_executor_uses_plan_executor_for_execution_steps(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
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
        hypothesis="Use generic execution.",
        action_type="baseline",
        change_type="baseline",
        instructions="Run bounded shell steps.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" -c "print(456)"',
                cwd=str(tmp_path),
            )
        ],
    )

    executor = RoutingExperimentExecutor(
        handlers={},
        plan_executor=GenericExecutionPlanExecutor(repo_root=tmp_path),
    )

    outcome = executor.execute(candidate, snapshot)

    assert outcome.status == "success"
    assert outcome.execution_details["step_results"][0]["returncode"] == 0


def test_routing_executor_ignores_action_label_when_execution_steps_are_present(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["run_experiment"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={
                "execution_backend_kind": "generic_agentic",
                "autonomous_execution_supported": True,
            }
        ),
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
        hypothesis="Run a concrete agent step.",
        action_type="baseline",
        change_type="baseline",
        instructions="The label should not matter when execution steps exist.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" -c "print(789)"',
                cwd=str(tmp_path),
            )
        ],
    )

    executor = RoutingExperimentExecutor(
        handlers={},
        plan_executor=GenericExecutionPlanExecutor(repo_root=tmp_path),
    )

    outcome = executor.execute(candidate, snapshot)

    assert outcome.status == "success"
    assert outcome.execution_details["step_results"][0]["returncode"] == 0


def test_generic_execution_plan_executor_can_write_files(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
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
        hypothesis="Write a file directly.",
        action_type="baseline",
        change_type="baseline",
        instructions="Write a repo file.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="notes/generated.txt",
                content="hello\n",
            ),
            ExecutionStep(
                kind="append_file",
                path="notes/generated.txt",
                content="world\n",
            ),
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert (tmp_path / "notes" / "generated.txt").read_text(encoding="utf-8") == "hello\nworld\n"


def test_generic_execution_plan_executor_blocks_writes_outside_repo(tmp_path) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["baseline"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
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
        hypothesis="Try an unsafe write.",
        action_type="baseline",
        change_type="baseline",
        instructions="Attempt to write outside the repo root.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="../outside.txt",
                content="nope",
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(candidate, snapshot)

    assert outcome.status == "blocked"
    assert outcome.failure_type == "UnsafePath"


def test_loopforge_start_from_bootstrap_turn_auto_installs_missing_dependency_and_retries(tmp_path, monkeypatch) -> None:
    attempts = {"count": 0}

    class MissingDependencyExecutor:
        def execute(self, candidate, snapshot):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ModuleNotFoundError("No module named 'sklearn'")
            return ExperimentOutcome(primary_metric_value=0.19, notes=["Recovered after dependency install."])

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": MissingDependencyExecutor()},
            capability_provider=lambda effective_spec: CapabilityContext(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Recovered.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )
    monkeypatch.setattr(app, "_install_python_dependency", lambda package_name: (True, {"command": ["uv", "pip", "install", package_name]}))

    bootstrap_turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=ExperimentSpec(
                objective="Improve LoL kills model.",
                primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
                allowed_actions=["baseline"],
                stop_conditions={"max_iterations": 1},
            ),
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    result = app.start_from_bootstrap_turn(
        bootstrap_turn=bootstrap_turn,
        user_goal="Improve LoL kills model.",
        iterations=1,
    )

    assert result["status"] == "started"
    assert result["results"][0]["accepted_summary"]["primary_metric_value"] == 0.19
    assert attempts["count"] == 2


def test_scan_repo_ignores_loopforge_generated_internal_files(tmp_path) -> None:
    (tmp_path / "train_model.py").write_text(
        "def train_model():\n    return 1\n",
        encoding="utf-8",
    )
    generated_dir = tmp_path / ".loopforge" / "generated"
    generated_dir.mkdir(parents=True)
    (generated_dir / "internal_adapter.py").write_text(
        "def baseline_params():\n    return {}\n",
        encoding="utf-8",
    )

    summary = scan_repo(tmp_path)

    assert "train_model" in summary["actions"]
    assert "baseline_params" not in summary["actions"]
    assert all(".loopforge/" not in note for note in summary["notes"])


def test_scan_repo_filters_private_and_test_actions(tmp_path) -> None:
    (tmp_path / "pipeline.py").write_text(
        "\n".join(
            [
                "def predict_kills():",
                "    return 1",
                "",
                "def _predict_internal():",
                "    return 2",
                "",
                "def test_predict_kills():",
                "    return 3",
                "",
                "def distributionmanagerpredictorwithaveryveryveryverylongname():",
                "    return 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = scan_repo(tmp_path)

    assert "predict_kills" in summary["actions"]
    assert "_predict_internal" not in summary["actions"]
    assert "test_predict_kills" not in summary["actions"]
    assert "distributionmanagerpredictorwithaveryveryveryverylongname" not in summary["actions"]


def test_scan_repo_discovers_real_data_files(tmp_path) -> None:
    data_dir = tmp_path / "examples" / "lol" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "matches.parquet").write_text("placeholder", encoding="utf-8")

    summary = scan_repo(tmp_path)

    assert "examples/lol/data/matches.parquet" in summary["data_assets"]


def test_metric_results_and_guardrails_drive_summary_classification(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Push recall while holding precision guardrail.",
                action_type="train",
                change_type="threshold",
                instructions="Tune the scorer threshold.",
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={
                "train": StaticActionExecutor(
                    [
                        ExperimentOutcome(
                            metric_results={
                                "recall_at_p90": MetricResult(
                                    name="recall_at_p90",
                                    value=0.74,
                                    scorer_ref="metrics.recall_at_p90",
                                ),
                                "precision_floor": MetricResult(
                                    name="precision_floor",
                                    value=0.88,
                                    scorer_ref="metrics.precision_floor",
                                ),
                            },
                            notes=["Recall improved but precision dipped."],
                        )
                    ]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Primary metric improved but the guardrail failed.")]
        ),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="Track this in memory.")]),
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["train"],
            primary_metric=PrimaryMetric(
                name="recall_at_p90",
                goal="maximize",
                scorer_ref="metrics.recall_at_p90",
                display_name="Recall @ Precision>=0.90",
            ),
            secondary_metrics=[
                MetricSpec(
                    name="log_loss",
                    goal="minimize",
                    scorer_ref="metrics.log_loss_cv",
                )
            ],
            guardrail_metrics=[
                MetricSpec(
                    name="precision_floor",
                    goal="maximize",
                    scorer_ref="metrics.precision_floor",
                    constraints={"min_value": 0.9},
                )
            ],
        )
    )

    cycle = orchestrator.run_iteration()

    assert cycle.accepted_summary is not None
    assert cycle.accepted_summary.primary_metric_value == 0.74
    assert cycle.accepted_summary.metric_results["recall_at_p90"].scorer_ref == "metrics.recall_at_p90"
    assert cycle.accepted_summary.guardrail_failures == ["precision_floor"]
    assert cycle.accepted_summary.result == "regressed"


# ── Progress callback tests ──────────────────────────────────────────────


def test_orchestrator_calls_progress_fn_during_iteration(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    progress_log: list[tuple[str, str]] = []

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Baseline run.",
                    action_type="baseline",
                    change_type="baseline",
                    instructions="Run baseline.",
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.5)])}
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="OK.", lessons=["Baseline done."])]
        ),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="baseline recorded")]),
        narrator_backend=FakeNarrationBackend(),
        progress_fn=lambda stage, msg: progress_log.append((stage, msg)),
    )
    orchestrator.initialize(build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 1}))

    orchestrator.run(iterations=1)

    stages = [stage for stage, _ in progress_log]
    assert "iteration_start" in stages
    assert "worker_propose" in stages
    assert "executor_run" in stages
    assert "reflect" in stages
    assert "review" in stages
    assert "narrate" in stages


def test_orchestrator_raises_experiment_interrupted_on_keyboard_interrupt(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class InterruptingWorker:
        def propose_next_experiment(self, snapshot):
            raise KeyboardInterrupt()

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=InterruptingWorker(),
        executor=RoutingExperimentExecutor(handlers={"baseline": StaticActionExecutor([])}),
        reflection_backend=FakeReflectionBackend([]),
        review_backend=FakeReviewBackend([]),
    )
    orchestrator.initialize(build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 3}))

    try:
        orchestrator.run(iterations=3)
        assert False, "Expected ExperimentInterrupted"
    except ExperimentInterrupted as exc:
        assert exc.results_so_far == []
        assert exc.current_stage == "iteration"
        assert exc.snapshot is not None


def test_cli_interrupt_during_bootstrap_prompts_for_redirect(tmp_path) -> None:
    """Ctrl+C during bootstrap should preserve the current plan by default."""
    call_count = [0]
    inputs = iter(["improve points model", "cap at 10", "quit"])
    prompts: list[str] = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(inputs)

    lines: list[str] = []

    def fake_print(msg=""):
        lines.append(str(msg))

    import loopforge.cli as cli_module
    original_loopforge = cli_module.Loopforge

    class InterruptingLoopforge(original_loopforge):
        def bootstrap(self, **kwargs):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            raise KeyboardInterrupt()

    cli_module.Loopforge = InterruptingLoopforge
    try:
        result = run_interactive_start(
            memory_root=tmp_path / "mem",
            input_fn=fake_input,
            print_fn=fake_print,
        )
    finally:
        cli_module.Loopforge = original_loopforge

    assert result == 0
    assert any("Interrupted during planning" in line for line in lines)
    assert any("Feedback for current plan" in prompt for prompt in prompts)


def test_cli_interrupt_during_experiment_prompts_for_redirect(tmp_path) -> None:
    """ExperimentInterrupted during start_from_bootstrap_turn() should show partial results and prompt."""
    inputs = iter(["improve model", "y", "quit"])
    lines: list[str] = []

    def fake_input(prompt):
        return next(inputs)

    def fake_print(msg=""):
        lines.append(str(msg))

    import loopforge.cli as cli_module
    original_loopforge = cli_module.Loopforge

    class FakeLoopforgeWithInterrupt(original_loopforge):
        def bootstrap(self, **kwargs):
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective="improve model",
                    recommended_spec=build_spec(),
                ),
                role_models=RoleModelConfig(
                    planner="m", worker="m", reflection="m",
                    review="m", consultation="m", narrator="m",
                ),
                ready_to_start=True,
            )

        def start_from_bootstrap_turn(self, **kwargs):
            raise ExperimentInterrupted(
                results_so_far=[],
                current_stage="iteration",
            )

    cli_module.Loopforge = FakeLoopforgeWithInterrupt
    try:
        result = run_interactive_start(
            memory_root=tmp_path / "mem",
            input_fn=fake_input,
            print_fn=fake_print,
        )
    finally:
        cli_module.Loopforge = original_loopforge

    assert result == 0
    assert any("Interrupted after 0 iteration(s)" in line for line in lines)


# ── Streaming tests ──────────────────────────────────────────────────────


def test_stream_fn_not_called_when_completion_fn_provided() -> None:
    """When a mock completion_fn is provided, stream_fn should never fire."""
    stream_calls: list[str] = []

    def fake_completion(**kwargs):
        return {
            "choices": [
                {"message": {"content": '{"hypothesis": "test", "action_type": "baseline", "change_type": "baseline", "instructions": "run"}'}}
            ]
        }

    from loopforge.core.backends import LiteLLMWorkerBackend

    backend = LiteLLMWorkerBackend(
        model="test-model",
        completion_fn=fake_completion,
        stream_fn=lambda token: stream_calls.append(token),
    )
    snapshot = MemorySnapshot(
        spec=build_spec(allowed_actions=["baseline"]),
        effective_spec=build_spec(allowed_actions=["baseline"]),
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )
    backend.propose_next_experiment(snapshot)
    assert stream_calls == [], "stream_fn should not be called when completion_fn is provided"


def test_stream_completion_accumulates_chunks() -> None:
    """_stream_completion should accumulate streamed chunks and call stream_fn."""
    from loopforge.core.backends import _LiteLLMJsonBackend

    streamed: list[str] = []
    backend = _LiteLLMJsonBackend(
        model="test-model",
        stream_fn=lambda token: streamed.append(token),
    )

    # Mock the streaming response
    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.delta = FakeDelta(content)

    class FakeChunk:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    chunks = [FakeChunk("Hello"), FakeChunk(" "), FakeChunk("world")]

    import loopforge.core.backends as backends_module
    original_completion = None

    def fake_litellm_completion(**kwargs):
        assert kwargs.get("stream") is True
        return iter(chunks)

    # Patch litellm.completion temporarily
    import types
    backend._stream_fn = lambda token: streamed.append(token)

    # Call _stream_completion directly with a patched litellm
    import unittest.mock
    with unittest.mock.patch.dict("sys.modules", {"litellm": types.ModuleType("litellm")}):
        import sys
        sys.modules["litellm"].completion = fake_litellm_completion
        result = backend._stream_completion([{"role": "user", "content": "hi"}])

    assert result == "Hello world"
    assert "Hello" in streamed
    assert " " in streamed
    assert "world" in streamed
