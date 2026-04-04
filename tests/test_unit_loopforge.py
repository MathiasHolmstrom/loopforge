from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest

import loopforge.bootstrap as bootstrap_module
import loopforge.core.agentic_execution as agentic_execution_module
import loopforge.pilot_adapters as pilot_adapters_module
from loopforge import (
    AdapterSetup,
    BootstrapTurn,
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentInterrupted,
    ExperimentOrchestrator,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentSpecProposal,
    FileMemoryStore,
    HumanIntervention,
    IterationRecord,
    IterationSummary,
    Loopforge,
    MarkdownMemoryNote,
    MemorySnapshot,
    MetricResult,
    MetricSpec,
    PreflightCheck,
    PrimaryMetric,
    ReflectionSummary,
    ReviewDecision,
    RoleModelConfig,
    RoutingExperimentExecutor,
    SpecQuestion,
)
from loopforge.auto_adapter import scan_repo, synthesize_auto_adapter
from loopforge.core.types import RunnerAuthoringRequest, RunnerAuthoringResult
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
from loopforge.core.backends import (
    build_execution_handoff,
    build_iteration_policy,
)
from tests.support import (
    FakeAccessAdvisorBackend,
    FakeNarrationBackend,
    FakeReflectionBackend,
    FakeReviewBackend,
    FakeWorkerBackend,
    RaisingAccessAdvisorBackend,
    RaisingNarrationBackend,
    StaticActionExecutor,
    build_adapter_setup,
    build_candidate,
    build_spec,
    passed_check,
    patch_litellm_test_backends,
)


@pytest.mark.parametrize(
    ("helpers_supported", "kwargs", "expected_helper_model", "expected_core_model"),
    [
        (
            False,
            {
                "consultation_model": "anthropic/claude-sonnet-4-5",
                "narrator_model": "anthropic/claude-sonnet-4-5",
            },
            bootstrap_module.DEFAULT_OPENAI_MODEL,
            bootstrap_module.DEFAULT_OPENAI_MODEL,
        ),
        (True, {}, bootstrap_module.DEFAULT_CLAUDE_MODEL, None),
    ],
)
def test_default_role_models_helper_routing(
    monkeypatch,
    helpers_supported: bool,
    kwargs: dict[str, str],
    expected_helper_model: str,
    expected_core_model: str | None,
) -> None:
    monkeypatch.setattr(
        "loopforge.bootstrap._can_use_anthropic_helpers", lambda: helpers_supported
    )
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)

    role_models = bootstrap_module.default_role_models(**kwargs)

    if expected_core_model is not None:
        assert role_models.planner == expected_core_model
        assert role_models.worker == expected_core_model
        assert role_models.reflection == expected_core_model
        assert role_models.review == expected_core_model
    assert role_models.consultation == expected_helper_model
    assert role_models.narrator == expected_helper_model


def test_detect_builtin_executor_factory_for_lol_kills_repo(tmp_path) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    (repo_root / "examples" / "lol").mkdir(parents=True)
    (repo_root / "examples" / "lol" / "pipeline_transformer_example.py").write_text(
        "print('lol')\n", encoding="utf-8"
    )

    factory = pilot_adapters_module.detect_builtin_executor_factory(
        repo_root,
        "create player kills model in player-performance-ratings repo for lol",
    )

    assert factory == "loopforge.pilot_adapters:build_lol_kills_adapter"


def test_loopforge_does_not_prefer_builtin_lol_adapter_by_default(tmp_path) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    (repo_root / "examples" / "lol").mkdir(parents=True)
    (repo_root / "examples" / "lol" / "pipeline_transformer_example.py").write_text(
        "print('lol')\n", encoding="utf-8"
    )

    app = Loopforge(repo_root=repo_root, memory_root=tmp_path / "memory")

    resolution = app.resolve_execution_backend(
        "create player kills model in player-performance-ratings repo for lol"
    )

    assert resolution.kind == "planning_only"
    assert resolution.factory_path is None


def test_loopforge_init_wires_default_models_to_expected_backends(tmp_path) -> None:
    app = Loopforge(memory_root=tmp_path / "memory")

    assert app.bootstrap_backend.model == app.role_models.planner
    assert app.worker_backend.model == app.role_models.worker
    assert app.reflection_backend.model == app.role_models.reflection
    assert app.review_backend.model == app.role_models.review
    expected_helper = app.role_models.consultation
    assert app.access_advisor_backend.model == expected_helper
    assert app.execution_fix_backend.model == expected_helper
    expected_narrator = app.role_models.narrator
    assert app.narrator_backend.model == expected_narrator
    assert (
        app.bootstrap_backend.max_completion_tokens
        == bootstrap_module.DEFAULT_ROLE_MAX_COMPLETION_TOKENS["planner"]
    )
    assert (
        app.worker_backend.max_completion_tokens
        == bootstrap_module.DEFAULT_ROLE_MAX_COMPLETION_TOKENS["worker"]
    )


def test_baseline_first_worker_backend_skips_llm_for_first_constrained_iteration() -> None:
    class ExplodingWorker:
        model = "openai/gpt-5.4"

        def propose_next_experiment(self, snapshot):
            raise AssertionError("LLM worker should not be called for baseline-first plan.")

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            raise AssertionError("not used")

    backend = bootstrap_module.BaselineFirstWorkerBackend(ExplodingWorker())
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
                "enforcement_scope": "until_first_successful_iteration",
            }
        },
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={
                "execution_backend_kind": "generic_agentic",
                "repo_root": "C:/repo",
                "python_executable": "C:/repo/.venv/Scripts/python.exe",
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

    candidate = backend.propose_next_experiment(snapshot)

    assert candidate.metadata["baseline_first_contract"] is True
    assert candidate.execution_steps[0].command.endswith('"src/train.py"')
    assert "existing baseline script" in candidate.instructions.lower()


def test_litellm_backend_passes_max_completion_tokens_to_completion_fn() -> None:
    captured_kwargs = {}

    def completion_fn(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Run baseline.",
                                "action_type": "run_experiment",
                                "change_type": "baseline",
                                "instructions": "Run the existing baseline.",
                                "execution_steps": [],
                            }
                        )
                    }
                }
            ]
        }

    worker = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4",
        completion_fn=completion_fn,
        max_completion_tokens=321,
    )
    spec = build_spec(allowed_actions=["run_experiment"])
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

    assert captured_kwargs["max_completion_tokens"] == 321
    assert captured_kwargs["response_format"] == {"type": "json_object"}


def test_loopforge_prefers_explicit_executor_factory(tmp_path) -> None:
    app = Loopforge(
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        executor_factory_path="custom.module:build_adapter",
    )

    resolved = app.resolve_executor_factory_path("Improve the model.")

    assert resolved == "custom.module:build_adapter"


def test_loopforge_cached_authored_runner_requires_matching_objective(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    memory_root = tmp_path / "memory"
    runner_path, manifest_path = bootstrap_module._authored_runner_paths(
        memory_root, repo_root
    )
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(
        "def build_adapter(spec, memory_root):\n    return None\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "repo_root": str(repo_root.resolve()),
                "user_goal": "Improve churn model.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bootstrap_module,
        "_validate_runner_factory",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache validation should not run for another objective")
        ),
    )
    app = Loopforge(repo_root=repo_root, memory_root=memory_root)

    resolution = app._load_cached_authored_runner("Improve fraud model.")

    assert resolution is None


def test_loopforge_uses_planning_only_backend_when_no_supported_runner(
    tmp_path,
) -> None:
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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: "y" if "Ready to start?" in prompt else "Improve model",
        print_fn=lambda *_args, **_kwargs: None,
    )

    assert exit_code == 0
    assert "stream_fn" not in captured_kwargs or captured_kwargs["stream_fn"] is None


def test_run_interactive_start_can_enable_raw_llm_output_streaming(monkeypatch) -> None:
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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **kwargs
        ):
            return {
                "status": "completed",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: "y" if "Ready to start?" in prompt else "Improve model",
        print_fn=lambda *_args, **_kwargs: None,
        stream_llm_output=True,
    )

    assert exit_code == 0
    assert callable(captured_kwargs["stream_fn"])


def test_run_interactive_start_asks_for_goal_then_follow_up_and_starts(
    monkeypatch,
) -> None:
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
                        questions=[
                            SpecQuestion(
                                key="workspace",
                                prompt="Which workspace should I target?",
                            )
                        ],
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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            self.start_calls.append(
                (
                    user_goal,
                    bootstrap_turn.proposal.recommended_spec.metadata,
                    iterations,
                )
            )
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

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
    assert any(
        "Ready to start?" in p or "What would you like to do?" in p for p in prompts
    )
    assert any("Scanning repo" in o for o in outputs)
    assert any("finished" in o.lower() or "started" in o.lower() for o in outputs)
    assert len(created_apps) == 1
    assert created_apps[0].bootstrap_calls == [("Fix Databricks deploy", {})]
    assert (
        created_apps[0].start_calls[0][1]["bootstrap_answers"]["workspace"]
        == "dev workspace"
    )


def test_run_interactive_start_accepts_natural_language_start_intent(
    monkeypatch,
) -> None:
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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            self.start_calls += 1
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: (
            "Go ahead start man" if "Ready to start?" in prompt else "Improve model"
        ),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any("Starting experiment loop" in output for output in outputs)


def test_run_interactive_start_does_not_treat_pasted_objective_text_as_feedback(
    monkeypatch,
) -> None:
    prompts = []
    outputs = []
    answers = iter(
        [
            "improve lol player kills model through a new script",
            "through a new script",
            "n",
            "quit",
        ]
    )
    created_apps = []

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = []
            self.apply_feedback_calls = []
            created_apps.append(self)

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls.append((user_goal, dict(answers or {})))
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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

        def apply_feedback(self, turn, response):
            self.apply_feedback_calls.append(response)
            return None

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any(
        "Treat that text as feedback for the current plan?" in prompt
        for prompt in prompts
    )
    assert created_apps[0].bootstrap_calls == [
        ("improve lol player kills model through a new script", {})
    ]
    assert created_apps[0].apply_feedback_calls == []


def test_run_interactive_start_confirms_buffered_multiline_paste_before_feedback(
    monkeypatch,
) -> None:
    prompts = []
    outputs = []
    answers = iter(
        [
            "improve lol player kills model that exists in here. use or",
            "dinalosscorer as primary metric. just create a new script.",
            "y",
            "quit",
        ]
    )
    created_apps = []

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = []
            self.apply_feedback_calls = []
            created_apps.append(self)

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls.append((user_goal, dict(answers or {})))
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

        def apply_feedback(self, turn, response):
            self.apply_feedback_calls.append(response)
            return None

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any(
        "Treat that text as feedback for the current plan?" in prompt
        for prompt in prompts
    )
    assert created_apps[0].bootstrap_calls == [
        ("improve lol player kills model that exists in here. use or", {}),
        (
            "improve lol player kills model that exists in here. use or dinalosscorer as primary metric. just create a new script.",
            {},
        ),
    ]
    assert created_apps[0].apply_feedback_calls == []


def test_run_interactive_start_does_not_reprint_same_ready_plan_after_blank_input(
    monkeypatch,
) -> None:
    outputs = []
    answers = iter(["Improve model", "", "quit"])

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            pass

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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    joined = "\n".join(str(item) for item in outputs)
    assert joined.count("--- Experiment plan ---") == 1


def test_run_interactive_start_reprints_ready_plan_after_replanning_same_signature(
    monkeypatch,
) -> None:
    outputs = []
    answers = iter(["Improve model", "need another look", "quit"])

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = 0

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls += 1
            return BootstrapTurn(
                assistant_message=(
                    "Ready." if self.bootstrap_calls == 1 else "Still ready."
                ),
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=True,
                human_update="Ready to start.",
            )

        def apply_feedback(self, turn, response):
            return None

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    joined = "\n".join(str(item) for item in outputs)
    assert joined.count("--- Experiment plan ---") == 2


def test_run_interactive_start_does_not_print_plan_update_when_replanning_after_required_answer(
    monkeypatch,
) -> None:
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
                                options=[
                                    "Use the larger dataset",
                                    "Use the subsample first",
                                ],
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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            return {
                "status": "started",
                "bootstrap": bootstrap_turn.to_dict(),
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any(
        "Analysing repository and planning experiment" in output for output in outputs
    )
    assert not any("Updating plan..." in output for output in outputs)
    assert len(captured_bootstraps) == 1


def test_run_interactive_start_patches_plan_with_feedback_before_replanning(
    monkeypatch,
) -> None:
    """Feedback tries to patch the plan first; falls through to replan if patch returns None."""
    outputs = []
    answers = iter(
        ["Improve the LoL kills model", "Use Bayesian smoothing", "y", "quit"]
    )

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

        def apply_feedback(self, turn, feedback):
            # Return None to signal full replan needed
            return None

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert any("Updating plan..." in output for output in outputs)
    assert any("Plan revised around your feedback." in output for output in outputs)


def test_run_interactive_start_does_not_duplicate_plan_after_inline_feedback_patch(
    monkeypatch,
) -> None:
    outputs = []
    answers = iter(
        ["Improve the LoL kills model", "Use Bayesian smoothing", "y", "quit"]
    )

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            pass

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

        def apply_feedback(self, turn, feedback):
            return BootstrapTurn(
                assistant_message="Updated.",
                proposal=ExperimentSpecProposal(
                    objective=turn.proposal.objective,
                    recommended_spec=replace(
                        turn.proposal.recommended_spec,
                        secondary_metrics=[
                            MetricSpec(name="mae", goal="minimize")
                        ],
                    ),
                    questions=[],
                ),
                role_models=turn.role_models,
                ready_to_start=True,
                human_update="Plan updated with your guidance.",
            )

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    joined = "\n".join(str(item) for item in outputs)
    assert joined.count("--- Experiment plan ---") == 2


def test_run_interactive_start_offers_optional_context_once_then_replans(
    monkeypatch,
) -> None:
    outputs = []
    prompts = []
    answers = iter(
        ["Improve the LoL kills model", "Try position-aware features if useful", "y"]
    )

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

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            self.start_calls.append(
                (
                    user_goal,
                    bootstrap_turn.proposal.recommended_spec.metadata,
                    iterations,
                )
            )
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert prompts[0] == "What problem are we solving? "
    assert prompts.count("> ") >= 1
    assert not any("Updating plan..." in output for output in outputs)


def test_run_interactive_start_accepts_cont_as_start_intent_after_optional_question(
    monkeypatch,
) -> None:
    prompts = []
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
                    questions=[
                        SpecQuestion(
                            key="baseline_choice",
                            prompt="Use existing baseline or build from scratch?",
                            required=True,
                            options=["Use existing baseline", "Start from scratch"],
                        ),
                        SpecQuestion(
                            key="user_extra_context",
                            prompt="Any extra modelling context, constraints, or ideas the agent should consider before it proceeds autonomously? Leave blank to skip.",
                            required=False,
                        ),
                    ],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=True,
                human_update="Ready to start.",
            )

        def start_from_bootstrap_turn(
            self, *, bootstrap_turn, user_goal, iterations=None, **_kwargs
        ):
            self.start_calls += 1
            return {
                "status": "started",
                "bootstrap": {"objective": user_goal},
                "results": [],
            }

    monkeypatch.setattr("loopforge.cli.Loopforge", StubLoopforge)

    answers = iter(["Build LoL kills model", "2", "cont"])
    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert prompts[0] == "What problem are we solving? "
    assert "Your choice: " in prompts
    assert any("Starting experiment loop" in output for output in outputs)


def test_main_without_args_enters_interactive_mode(monkeypatch) -> None:
    captured_kwargs = {}

    def fake_run_interactive_start(**kwargs):
        captured_kwargs.update(kwargs)
        return 7

    monkeypatch.setattr("loopforge.cli.run_interactive_start", fake_run_interactive_start)

    assert main([]) == 7
    assert captured_kwargs == {"stream_llm_output": False}


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
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
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
    # No domain defaults injected — agent returned null spec so fields are absent/empty
    assert turn.proposal.recommended_spec.allowed_actions == []


def test_bootstrap_backend_tolerates_incomplete_metric_payloads() -> None:
    """Incomplete metric payloads pass through — no domain defaults injected."""
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
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
                                        "secondary_metrics": [
                                            {"name": "r2", "goal": "maximize"}
                                        ],
                                        "guardrail_metrics": [],
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

    # Agent's explicit choices are preserved
    assert turn.proposal.recommended_spec.primary_metric.name == "mae"
    assert turn.proposal.recommended_spec.primary_metric.goal == "minimize"
    assert turn.proposal.recommended_spec.secondary_metrics[0].name == "r2"
    assert turn.proposal.recommended_spec.guardrail_metrics == []


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
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
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


def test_bootstrap_backend_respects_agent_empty_allowed_actions() -> None:
    """If agent returns empty allowed_actions, code doesn't auto-fill from capability context."""
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
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
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
        capability_context=CapabilityContext(
            available_actions={"baseline": "pilot.py", "tune": "pilot.py"}
        ),
        role_models=bootstrap_module.default_role_models(),
    )

    # Agent chose empty actions — respected, not auto-filled
    assert turn.proposal.recommended_spec.allowed_actions == []


def test_bootstrap_backend_does_not_override_agent_metric_decisions() -> None:
    """Agent decisions on metrics are respected — no auto-substitution or auto-injection."""
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
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
                                        "guardrail_metrics": [],
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
                "mae": {
                    "goal": "minimize",
                    "preferred_role": "primary",
                    "scorer_ref": "pilot:mae",
                },
                "ordinal_loss": {
                    "goal": "minimize",
                    "preferred_role": "guardrail",
                    "scorer_ref": "pilot:ord",
                },
            }
        ),
        role_models=bootstrap_module.default_role_models(),
    )

    assert turn.proposal.recommended_spec.primary_metric.name == "mae"
    assert turn.proposal.recommended_spec.primary_metric.goal == "minimize"
    # Agent chose no guardrails — respected, not auto-injected
    assert turn.proposal.recommended_spec.guardrail_metrics == []


def test_bootstrap_prompt_requires_grounding_in_existing_implementation() -> None:
    captured: dict[str, str] = {}

    def fake_completion(**kwargs):
        captured["system_prompt"] = kwargs["messages"][0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Grounded on the current baseline.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {
                                            "name": "ordinal_loss",
                                            "goal": "minimize",
                                        },
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
        }

    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=fake_completion,
    )

    backend.propose_bootstrap_turn(
        user_goal="Improve LoL kills model.",
        capability_context=CapabilityContext(
            notes=[
                "Inspect experiments/lol_kills_autonomous_baseline.py before proposing changes."
            ]
        ),
        role_models=bootstrap_module.default_role_models(),
    )

    system_prompt = captured["system_prompt"]
    assert "GROUNDING IN THE CURRENT IMPLEMENTATION" in system_prompt
    assert "Treat capability_context notes, environment facts, file paths, and discovered actions as the current source of truth." in system_prompt
    assert (
        "Do not ask the user to tell you what model currently exists when the repo context already answers it."
        in system_prompt
    )
    assert "Make a qualified guess when the objective and repo context make one path much more plausible than the others." in system_prompt


def test_runner_authoring_prompt_requires_reusing_existing_pipeline() -> None:
    captured: dict[str, str] = {}

    def fake_completion(**kwargs):
        captured["system_prompt"] = kwargs["messages"][0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "module_source": "from pathlib import Path\n",
                                "summary": "stub",
                                "notes": [],
                            }
                        )
                    }
                }
            ]
        }

    backend = bootstrap_module.LiteLLMRunnerAuthoringBackend(
        model="openai/gpt-5.4",
        completion_fn=fake_completion,
    )

    backend.author_runner(
        RunnerAuthoringRequest(
            user_goal="Improve LoL kills model.",
            repo_root="C:/repo",
            capability_context=CapabilityContext(
                notes=[
                    "Inspect examples/lol/pipeline_transformer_example.py for the existing spforge pipeline."
                ]
            ),
            target_module_path="C:/repo/generated_adapter.py",
        )
    )

    system_prompt = captured["system_prompt"]
    assert "Ground the runner in the existing implementation described in request.capability_context." in system_prompt
    assert "reuse and adapt them instead of inventing a disconnected pipeline" in system_prompt
    assert "keep feature engineering on the repo's existing tooling stack" in system_prompt


def test_lol_capability_context_surfaces_existing_code_paths_and_spforge_stack(
    tmp_path,
) -> None:
    repo_root = tmp_path / "player-performance-ratings"

    context = pilot_adapters_module._lol_capability_context(repo_root)

    assert context.environment_facts["feature_engineering_stack"] == "spforge"
    assert any(
        path.endswith("experiments\\lol_kills_autonomous_baseline.py")
        or path.endswith("experiments/lol_kills_autonomous_baseline.py")
        for path in context.environment_facts["baseline_code_paths"]
    )
    assert any(
        "pipeline_transformer_example.py" in note for note in context.notes
    )
    assert any("spforge tooling" in note for note in context.notes)


def test_execution_guidance_gap_is_suppressed_when_local_repo_evidence_is_strong() -> (
    None
):
    detail = bootstrap_module._detect_execution_guidance_gap(
        user_goal=(
            "Improve lol player kills model that exists inside this repo. "
            "Check lol data and find where the player kills model is generated."
        ),
        capability_context=CapabilityContext(
            available_actions={
                "train_player_kills": "scripts/lol_train_models.py",
            },
            available_data_assets=[
                "lol_data/lol_game_player_historical.parquet",
                "lol_data/lol_feature_store.parquet",
            ],
            environment_facts={"repo_root": "C:/repo", "python_executable": "python"},
            notes=[
                "The pipeline may load data from s3 or catalog depending on environment."
            ],
        ),
        answers=None,
    )

    assert detail is None


def test_metric_repair_context_prioritises_objective_relevant_metrics() -> None:
    compact = bootstrap_module._compact_capability_context_for_metric_repair(
        objective="Improve lol player kills model using ordinal loss scorer.",
        assistant_message="I scanned the repo and prepared an initial bootstrap proposal.",
        capability_context=CapabilityContext(
            available_metrics={
                "mae": {"path": "scripts/analyze_spread_robust_gate.py"},
                "ordinal_loss": {
                    "path": "scripts/experiment_pbp_rebound_variance_features.py"
                },
                "NamedOrdinalLossScorer": {
                    "path": "sports_predictions_orchestrator/nba/sim/backtest_reporting.py"
                },
                "BalancedLineScorer": {
                    "path": "sports_predictions_orchestrator/examples/lol/model_experimentation/lol_player_kills.py"
                },
            },
            notes=["LoL player kills model exists in repo."],
        ),
    )

    assert "ordinal_loss" in compact["available_metrics"]
    assert "NamedOrdinalLossScorer" in compact["available_metrics"]
    assert len(compact["available_metrics"]) < 4


def test_worker_backend_tolerates_missing_action_type() -> None:
    worker = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [{"message": {"content": json.dumps({})}}]
        },
    )
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

    # Agent returned empty — visible sentinel, not silent replacement
    assert candidate.action_type == "unspecified"
    assert candidate.hypothesis == "[agent did not specify]"


def test_iteration_policy_uses_force_next_action_when_present() -> None:
    spec = build_spec(
        allowed_actions=["inspect_data", "train_model"],
        metadata={"force_next_action": "train_model"},
    )
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


def test_worker_candidate_normalisation_preserves_agent_action_type() -> None:
    """Agent's action_type is preserved even if not in allowed_actions — executor validates."""
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

    candidate_payload = (
        bootstrap_module.LiteLLMWorkerBackend._normalise_candidate_payload(
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
    )

    # Agent's choice is preserved — not silently replaced
    assert candidate_payload["action_type"] == "invented_action"
    assert candidate_payload["change_type"] == "invented_action"


def test_diagnostic_actions_are_recorded_as_inconclusive_not_regressed(
    tmp_path,
) -> None:
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
            handlers={
                "eda": StaticActionExecutor(
                    [
                        ExperimentOutcome(
                            primary_metric_value=0.41, notes=["EDA completed."]
                        )
                    ]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(
                    assessment="Useful diagnosis.", lessons=["Found worst slices."]
                )
            ]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="keep diagnostic memory")]
        ),
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
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.41)]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
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


def test_orchestrator_continuation_merges_follow_up_outcome_without_crashing(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class ContinuationWorker:
        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Initial setup step.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run setup first.",
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            return ExperimentCandidate(
                hypothesis="Follow up with actual metrics.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run the actual baseline.",
            )

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=ContinuationWorker(),
        executor=RoutingExperimentExecutor(
            handlers={
                "baseline": StaticActionExecutor(
                    [
                        ExperimentOutcome(status="success", notes=["Setup completed."]),
                        ExperimentOutcome(
                            status="success",
                            primary_metric_value=0.41,
                            notes=["Metrics produced."],
                        ),
                    ]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Baseline captured.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["baseline"],
            stop_conditions={"max_iterations": 1, "max_metricless_continuations": 2},
        )
    )

    cycle = orchestrator.run_iteration()

    assert cycle.record.outcome.status == "success"
    assert cycle.record.outcome.primary_metric_value == 0.41
    assert cycle.record.outcome.notes == ["Setup completed.", "Metrics produced."]


def test_orchestrator_blocks_repeated_metricless_successes_early(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class RepeatingContinuationWorker:
        def __init__(self) -> None:
            self.continue_calls = 0

        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Write a runner that should emit metrics.",
                action_type="run_experiment",
                change_type="run",
                instructions="Run the metric-emitting script.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python emit_metrics.py")
                ],
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            self.continue_calls += 1
            return ExperimentCandidate(
                hypothesis="Retry the same runner path.",
                action_type="run_experiment",
                change_type="run",
                instructions="Retry the same script.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python emit_metrics.py")
                ],
            )

    worker = RepeatingContinuationWorker()
    repeated_outcome = ExperimentOutcome(
        status="success",
        notes=["Executed runner, but no metrics were captured."],
        execution_details={
            "step_results": [
                {
                    "index": 1,
                    "kind": "shell",
                    "command": "python emit_metrics.py",
                    "stdout": '{"ranked_probability_score": 0.12}',
                    "stderr": "",
                    "returncode": 0,
                }
            ],
            "latest_stdout_preview": '{"ranked_probability_score": 0.12}',
        },
    )
    class RepeatingExecutor:
        def execute(self, candidate, snapshot):
            return repeated_outcome

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RepeatingExecutor(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Blocked on metrics.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="capture blocked state")]
        ),
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["run_experiment"],
            stop_conditions={
                "max_iterations": 1,
                "max_metricless_continuations": 1,
            },
            metadata={"execution_mode": "autonomous_after_bootstrap"},
        )
    )

    cycle = orchestrator.run_iteration()

    assert cycle.record.outcome.status == "blocked"
    assert cycle.record.outcome.failure_type == "MetriclessIterationExhausted"
    assert worker.continue_calls == 1
    assert any(
        "Loopforge refused to advance to the next iteration" in note
        for note in cycle.record.outcome.notes
    )


def test_orchestrator_disables_metricless_continuation_during_baseline_first_phase(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class BaselineFirstWorker:
        def __init__(self) -> None:
            self.continue_calls = 0

        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Run the existing baseline script first.",
                action_type="run_experiment",
                change_type="baseline_reuse",
                instructions="Run the baseline script.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python src/train.py")
                ],
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            self.continue_calls += 1
            return previous_candidate

    worker = BaselineFirstWorker()
    repeated_outcome = ExperimentOutcome(
        status="success",
        notes=["Baseline ran but did not emit metrics."],
    )

    class MetriclessExecutor:
        def execute(self, candidate, snapshot):
            return repeated_outcome

    spec = build_spec(
        allowed_actions=["run_experiment"],
        stop_conditions={"max_iterations": 1, "max_metricless_continuations": 3},
        metadata={
            "execution_mode": "autonomous_after_bootstrap",
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
                "enforcement_scope": "until_first_successful_iteration",
            },
        },
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=MetriclessExecutor(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Blocked on baseline metrics.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="capture blocked state")]
        ),
    )
    orchestrator.initialize(spec)

    cycle = orchestrator.run_iteration()

    assert cycle.record.outcome.status == "blocked"
    assert cycle.record.outcome.failure_type == "MetriclessIterationExhausted"
    assert worker.continue_calls == 0


def test_orchestrator_skips_review_backend_after_gateway_reflection_failure(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class GatewayReflectionBackend:
        def reflect(self, snapshot, candidate, outcome):
            raise RuntimeError("litellm.BadGatewayError: 502 Bad Gateway")

    class ExplodingReviewBackend:
        def review(self, snapshot, candidate, outcome, reflection):
            raise AssertionError(
                "review backend should have been skipped after gateway reflection failure"
            )

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Run baseline",
                    action_type="baseline",
                    change_type="baseline",
                    instructions="Run baseline.",
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.41)]
                )
            }
        ),
        reflection_backend=GatewayReflectionBackend(),
        review_backend=ExplodingReviewBackend(),
    )
    orchestrator.initialize(
        build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 1})
    )

    cycle = orchestrator.run_iteration()

    assert cycle.record.review.status == "accepted"
    assert (
        "Accepted by fallback because review generation failed"
        in cycle.record.review.reason
    )


def test_review_backend_tolerates_missing_status_and_reason() -> None:
    review = bootstrap_module.LiteLLMReviewBackend(
        model="openai/gpt-5.4",
        completion_fn=lambda **kwargs: {
            "choices": [{"message": {"content": json.dumps({})}}]
        },
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
        ExperimentCandidate(
            hypothesis="Baseline",
            action_type="baseline",
            change_type="baseline",
            instructions="Run.",
        ),
        ExperimentOutcome(primary_metric_value=1.0),
        ReflectionSummary(assessment="ok"),
    )

    assert decision.status == "pending_human"
    assert "fallback" in decision.reason.lower()


def test_bootstrap_survives_helper_backend_failures(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Bootstrap completed.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="fraud_recall", goal="maximize"
                        ),
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


def test_only_accepted_memory_reaches_the_next_worker_and_human_notes_modify_effective_spec(
    tmp_path,
) -> None:
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
                "baseline": StaticActionExecutor(
                    [
                        ExperimentOutcome(
                            primary_metric_value=0.41, notes=["Baseline done."]
                        )
                    ]
                ),
                "eda": StaticActionExecutor(
                    [
                        ExperimentOutcome(
                            primary_metric_value=0.41,
                            notes=["Rest-day slice found drift."],
                        )
                    ]
                ),
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(
                    assessment="Useful baseline.", lessons=["Baseline established."]
                ),
                ReflectionSummary(
                    assessment="Rest-day slice is worth following up.",
                    lessons=["Rest days matter."],
                ),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(
                    status="accepted", reason="Use this as starting memory."
                ),
                ReviewDecision(status="accepted", reason="EDA finding is actionable."),
            ]
        ),
    )
    orchestrator.initialize(spec=build_spec())

    first_cycle = orchestrator.run_iteration()
    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Focus on EDA before more tuning.",
        effects={
            "disable_actions": ["tune"],
            "force_next_action": "eda",
            "suggested_hypothesis": "Check rest-day drift",
        },
    )
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary is not None
    assert worker.snapshots[0].recent_summaries == []
    assert [summary.hypothesis for summary in worker.snapshots[1].recent_summaries] == [
        "Baseline"
    ]
    assert worker.snapshots[1].effective_spec.allowed_actions == [
        "baseline",
        "eda",
        "train",
        "evaluate",
    ]
    assert worker.snapshots[1].effective_spec.metadata["force_next_action"] == "eda"
    assert worker.snapshots[1].effective_spec.metadata["suggested_hypotheses"] == [
        "Check rest-day drift"
    ]


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
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.41)]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
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
    assert worker.snapshots[0].capability_context.available_data_assets == [
        "gold_cross_validated_pass_outcome_ncaaf"
    ]
    assert (
        worker.snapshots[0].capability_context.available_metrics["log_loss"][
            "scorer_ref"
        ]
        == "metrics.log_loss_cv"
    )


def test_build_execution_handoff_exposes_verified_lane_from_markdown_memory() -> None:
    spec = build_spec(
        metadata={
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
            }
        }
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={
                "repo_root": "C:/repo",
                "execution_shell": "cmd.exe",
                "shell_family": "windows_cmd",
                "python_executable": "C:/repo/.venv/Scripts/python.exe",
                "execution_backend_kind": "generic_agentic",
            }
        ),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[
            MarkdownMemoryNote(
                path="bootstrap_handoff.md",
                content="# Bootstrap Handoff\nReuse src/train.py.",
            ),
            MarkdownMemoryNote(
                path="execution_runbook.md",
                content="# Execution Runbook\nUse the verified Python path.",
            ),
            MarkdownMemoryNote(
                path="experiment_guide.md",
                content="# Experiment Guide\nBuild and run one script.",
            ),
        ],
        next_iteration_id=1,
    )

    handoff = build_execution_handoff(snapshot)

    assert handoff["verified_execution_lane"] is True
    assert handoff["must_reuse_verified_lane"] is True
    assert handoff["execution_contract"]["must_reference_baseline_paths"] is True
    assert handoff["bootstrap_handoff"].startswith("# Bootstrap Handoff")
    assert handoff["execution_runbook"].startswith("# Execution Runbook")
    assert handoff["experiment_guide"].startswith("# Experiment Guide")


def test_rejected_review_stays_in_raw_records_but_not_in_accepted_memory(
    tmp_path,
) -> None:
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
            handlers={
                "train": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.3)]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(
                    assessment="Looks leaky.", lessons=["Need leakage check."]
                )
            ]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="rejected", reason="Do not trust this run.")]
        ),
    )
    orchestrator.initialize(spec=build_spec(allowed_actions=["train"]))

    cycle = orchestrator.run_iteration()
    snapshot = store.load_snapshot()
    records_text = (tmp_path / "loop" / "iteration_records.jsonl").read_text(
        encoding="utf-8"
    )

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
            return ExperimentOutcome(
                primary_metric_value=0.25, notes=["Retry succeeded."]
            )

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={"baseline": FailingExecutor()},
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(
                    assessment="Recoverable dependency failure.",
                    lessons=["Install sklearn first."],
                ),
                ReflectionSummary(
                    assessment="Retry worked.", lessons=["Environment is now usable."]
                ),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="Failure is useful memory."),
                ReviewDecision(status="accepted", reason="Successful retry."),
            ]
        ),
    )
    orchestrator.initialize(
        spec=build_spec(
            allowed_actions=["baseline"], stop_conditions={"max_iterations": 2}
        )
    )

    first_cycle = orchestrator.run_iteration()
    orchestrator.executor = RoutingExperimentExecutor(
        handlers={"baseline": PassingExecutor()}
    )
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert first_cycle.accepted_summary.outcome_status == "recoverable_failure"
    assert "sklearn" in (first_cycle.accepted_summary.failure_summary or "")
    assert worker.snapshots[1].recent_records[0].outcome.status == "recoverable_failure"
    assert second_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary.outcome_status == "success"


def test_orchestrator_repairs_generic_autonomous_failure_within_same_iteration(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Run the repo experiment directly.",
                action_type="run_experiment",
                change_type="run",
                instructions="Run the experiment.",
            ),
            ExperimentCandidate(
                hypothesis="Repair the missing execution plan.",
                action_type="fix_failure",
                change_type="repair",
                instructions="Create and run the minimal metric-producing script.",
                execution_steps=[ExecutionStep(kind="shell", command="fake", cwd=".")],
            ),
        ]
    )
    progress_log: list[tuple[str, str]] = []
    spec = build_spec(
        allowed_actions=["run_experiment", "fix_failure"],
        stop_conditions={"max_iterations": 1, "max_same_iteration_repairs": 3},
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )

    class RepairingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, candidate, snapshot):
            self.calls += 1
            if self.calls == 1:
                return ExperimentOutcome(
                    status="recoverable_failure",
                    failure_type="MissingExecutionSteps",
                    failure_summary="The worker proposed an agentic step without concrete execution steps.",
                    recoverable=True,
                )
            return ExperimentOutcome(status="success", primary_metric_value=0.21)

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RepairingExecutor(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Recovered in-place.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
        progress_fn=lambda stage, msg: progress_log.append((stage, msg)),
    )
    orchestrator.initialize(spec=spec)

    cycle = orchestrator.run_iteration()

    assert cycle.record.outcome.status == "success"
    assert cycle.record.outcome.primary_metric_value == 0.21
    assert len(worker.snapshots) == 2
    assert (
        worker.snapshots[1].recent_records[-1].outcome.status == "recoverable_failure"
    )
    assert any(
        "repair within the same iteration" in message for _, message in progress_log
    )


def test_orchestrator_disables_same_iteration_repair_during_baseline_first_phase(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class RepairWorker:
        def __init__(self) -> None:
            self.propose_calls = 0

        def propose_next_experiment(self, snapshot):
            self.propose_calls += 1
            if self.propose_calls > 1:
                raise AssertionError("baseline-first phase should not request same-iteration repair")
            return ExperimentCandidate(
                hypothesis="Run the existing baseline script first.",
                action_type="run_experiment",
                change_type="baseline_reuse",
                instructions="Run the baseline script.",
                execution_steps=[
                    ExecutionStep(kind="shell", command="python src/train.py")
                ],
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            raise AssertionError("not used")

    class FailingExecutor:
        def execute(self, candidate, snapshot):
            return ExperimentOutcome(
                status="recoverable_failure",
                failure_type="MissingExecutionSteps",
                failure_summary="simulated recoverable failure",
                recoverable=True,
            )

    spec = build_spec(
        allowed_actions=["run_experiment", "fix_failure"],
        stop_conditions={"max_iterations": 1, "max_same_iteration_repairs": 3},
        metadata={
            "execution_mode": "autonomous_after_bootstrap",
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
                "enforcement_scope": "until_first_successful_iteration",
            },
        },
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=RepairWorker(),
        executor=FailingExecutor(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Blocked on baseline failure.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="capture blocked state")]
        ),
    )
    orchestrator.initialize(spec)

    cycle = orchestrator.run_iteration()

    assert cycle.record.outcome.status == "blocked"
    assert cycle.record.outcome.failure_type == "IterationRepairBudgetExceeded"


def test_cli_interjection_appends_human_note_and_changes_future_effective_spec(
    tmp_path,
) -> None:
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


def test_file_memory_store_loads_markdown_memory_and_experiment_journal(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())
    (tmp_path / "loop" / "agent_markdown").mkdir(parents=True, exist_ok=True)
    (tmp_path / "loop" / "agent_markdown" / "ops_access_guide.md").write_text(
        "# Access\nUse the approved workspace.\n", encoding="utf-8"
    )
    (tmp_path / "loop" / "agent_markdown" / "execution_runbook.md").write_text(
        "# Runbook\nUse Python scripts.\n", encoding="utf-8"
    )
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

    assert "agent_markdown/ops_access_guide.md" in markdown_by_path
    assert "agent_markdown/execution_runbook.md" in markdown_by_path
    assert "agent_markdown/experiment_journal.md" in markdown_by_path
    assert (
        "## Iteration 1: Baseline"
        in markdown_by_path["agent_markdown/experiment_journal.md"]
    )
    assert (
        "Do not repeat: Rerun the broken config."
        in markdown_by_path["agent_markdown/experiment_journal.md"]
    )


def test_file_memory_store_still_reads_legacy_root_markdown_memory(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())
    (tmp_path / "loop" / "ops_access_guide.md").write_text(
        "# Access\nLegacy location.\n", encoding="utf-8"
    )

    snapshot = store.load_snapshot()
    markdown_by_path = {note.path: note.content for note in snapshot.markdown_memory}

    assert "ops_access_guide.md" in markdown_by_path
    assert markdown_by_path["ops_access_guide.md"].startswith("# Access")


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
        markdown_memory=[
            MarkdownMemoryNote(
                path="ops_access_guide.md", content="# Access\nUse token auth."
            )
        ],
        next_iteration_id=1,
    )

    worker.propose_next_experiment(snapshot)

    assert captured_payload["payload"]["markdown_memory"] == [
        {"path": "ops_access_guide.md", "content": "# Access\nUse token auth."}
    ]


def test_worker_backend_filters_first_iteration_handoff_for_generic_autonomous_mode() -> None:
    captured_payload = {}

    def completion_fn(**kwargs):
        captured_payload["payload"] = json.loads(kwargs["messages"][1]["content"])
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Write one script.",
                                "action_type": "run_experiment",
                                "change_type": "run",
                                "instructions": "Write and run one script.",
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
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
        ),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[
            MarkdownMemoryNote(
                path="agent_markdown/bootstrap_handoff.md",
                content="# Bootstrap Handoff",
            ),
            MarkdownMemoryNote(
                path="agent_markdown/execution_runbook.md", content="# Runbook"
            ),
            MarkdownMemoryNote(
                path="agent_markdown/ops_access_guide.md", content="# Access"
            ),
            MarkdownMemoryNote(
                path="agent_markdown/experiment_guide.md", content="# Experiment Guide"
            ),
        ],
        next_iteration_id=1,
    )

    worker.propose_next_experiment(snapshot)

    assert captured_payload["payload"]["markdown_memory"] == [
        {
            "path": "agent_markdown/bootstrap_handoff.md",
            "content": "# Bootstrap Handoff",
        },
        {"path": "agent_markdown/execution_runbook.md", "content": "# Runbook"},
        {"path": "agent_markdown/ops_access_guide.md", "content": "# Access"},
        {
            "path": "agent_markdown/experiment_guide.md",
            "content": "# Experiment Guide",
        },
    ]
    assert captured_payload["payload"]["iteration_policy"][
        "prefer_smallest_runnable_script"
    ] is True


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

    worker = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4", completion_fn=completion_fn
    )
    spec = build_spec(
        allowed_actions=["baseline"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
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

    assert (
        captured_payload["payload"]["effective_spec"]["metadata"]["execution_mode"]
        == "autonomous_after_bootstrap"
    )


def test_loopforge_bootstrap_passes_existing_markdown_memory(
    tmp_path, monkeypatch
) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())
    (tmp_path / "loop" / "agent_markdown").mkdir(parents=True, exist_ok=True)
    (tmp_path / "loop" / "agent_markdown" / "ops_access_guide.md").write_text(
        "# Access\nUse the dev workspace.\n", encoding="utf-8"
    )

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
                    recommended_spec=build_spec(
                        objective=user_goal, allowed_actions=["baseline"]
                    ),
                    questions=[],
                ),
                role_models=bootstrap_module.default_role_models(),
                ready_to_start=False,
            )

    monkeypatch.setattr(
        "loopforge.bootstrap.discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(),
    )
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


def test_run_from_spec_supports_factory_and_returns_reviewed_cycle_results(
    tmp_path, monkeypatch
) -> None:
    spec = build_spec(
        allowed_actions=["baseline"], stop_conditions={"max_iterations": 1}
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    patch_litellm_test_backends(
        monkeypatch,
        candidate=build_candidate(),
        reflection=ReflectionSummary(
            assessment="Baseline is fine.", lessons=["Keep this baseline."]
        ),
    )

    def factory(spec, memory_root):
        return build_adapter_setup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.39)]
                )
            },
            capability_context=CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_entities={"model_keys": ["pass_outcome"]},
                available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            ),
            preflight_checks=passed_check(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: factory)

    results = run_from_spec(
        spec_path=spec_path,
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        worker_model="openai/gpt-5.4",
    )

    assert results[0]["accepted_summary"]["hypothesis"] == "Baseline"
    assert results[0]["record"]["review"]["status"] == "accepted"
    assert results[0]["human_update"] == "iteration update"


def test_run_from_spec_uses_configured_repo_root_for_generic_executor(
    tmp_path, monkeypatch
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(build_spec().to_dict()), encoding="utf-8")
    captured_repo_roots: list[Path] = []
    patch_litellm_test_backends(monkeypatch)

    class CapturingPlanExecutor:
        def __init__(self, *, repo_root: Path | str, **kwargs) -> None:
            captured_repo_roots.append(Path(repo_root))

        def execute(self, candidate, snapshot):
            return ExperimentOutcome(primary_metric_value=0.5)

    def factory(spec, memory_root):
        return build_adapter_setup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.39)]
                )
            },
            capability_context=CapabilityContext(
                available_actions={"baseline": "Run baseline"}
            ),
            preflight_checks=passed_check(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: factory)
    monkeypatch.setattr(
        "loopforge.bootstrap.GenericExecutionPlanExecutor", CapturingPlanExecutor
    )

    run_from_spec(
        spec_path=spec_path,
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        worker_model="openai/gpt-5.4",
    )

    assert captured_repo_roots == [tmp_path / "repo"]


def test_run_from_spec_wires_execution_fix_backend_into_generic_executor(
    tmp_path, monkeypatch
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(build_spec().to_dict()), encoding="utf-8")
    captured_fix_backends = []
    patch_litellm_test_backends(monkeypatch)

    class CapturingPlanExecutor:
        def __init__(
            self, *, repo_root: Path | str, fix_backend=None, **kwargs
        ) -> None:
            captured_fix_backends.append(fix_backend)

        def execute(self, candidate, snapshot):
            return ExperimentOutcome(primary_metric_value=0.5)

    def factory(spec, memory_root):
        return build_adapter_setup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.39)]
                )
            },
            capability_context=CapabilityContext(
                available_actions={"baseline": "Run baseline"}
            ),
            preflight_checks=passed_check(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: factory)
    monkeypatch.setattr(
        "loopforge.bootstrap.GenericExecutionPlanExecutor", CapturingPlanExecutor
    )

    run_from_spec(
        spec_path=spec_path,
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        worker_model="openai/gpt-5.4",
    )

    assert len(captured_fix_backends) == 1
    assert captured_fix_backends[0] is not None


def test_run_from_spec_blocks_invalid_metric_goal_before_execution(
    tmp_path, monkeypatch
) -> None:
    spec = build_spec(
        allowed_actions=["baseline"],
        primary_metric=PrimaryMetric(name="mae", goal="unspecified"),
        stop_conditions={"max_iterations": 1},
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    def fake_factory(spec, memory_root: Path):
        return build_adapter_setup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.39)]
                )
            },
            capability_context=CapabilityContext(),
            preflight_checks=passed_check(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    try:
        run_from_spec(
            spec_path=spec_path,
            memory_root=tmp_path / "memory",
            executor_factory_path="fake.module:factory",
            worker_model="openai/gpt-5.4",
        )
        raise AssertionError(
            "Expected run_from_spec to reject an unspecified metric goal"
        )
    except ValueError as exc:
        assert "maximize" in str(exc)
        assert "mae" in str(exc)


def test_discover_capabilities_for_objective_prefers_discovery_provider(
    tmp_path, monkeypatch
) -> None:
    def fake_factory(spec, memory_root: Path):
        return build_adapter_setup(
            handlers={},
            capability_context=CapabilityContext(
                available_metrics={
                    "fallback_metric": {"scorer_ref": "metrics.fallback"}
                }
            ),
            discovery_context=CapabilityContext(
                available_metrics={
                    "precision_floor": {"scorer_ref": "metrics.precision_floor"}
                },
                notes=["objective=Keep precision high."],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    context = discover_capabilities_for_objective(
        objective="Keep precision high.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
    )

    assert (
        context.available_metrics["precision_floor"]["scorer_ref"]
        == "metrics.precision_floor"
    )
    assert "objective=Keep precision high." in context.notes


def test_discover_capabilities_for_objective_exposes_execution_shell_facts(
    tmp_path, monkeypatch
) -> None:
    def fake_factory(spec, memory_root: Path):
        return build_adapter_setup(
            handlers={},
            discovery_context=CapabilityContext(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    context = discover_capabilities_for_objective(
        objective="Keep precision high.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        repo_root=tmp_path,
    )

    assert context.environment_facts["execution_shell"]
    assert context.environment_facts["shell_family"]
    assert context.environment_facts["python_executable"]
    assert any("Runtime platform:" in note for note in context.notes)


def test_discover_capabilities_for_objective_loads_real_repo_data_files(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = repo_root / "examples" / "lol" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "matches.csv").write_text(
        "playername,teamname,kills\nfaker,t1,7\nchovy,hle,5\n",
        encoding="utf-8",
    )

    def fake_factory(spec, memory_root: Path):
        return build_adapter_setup(
            handlers={},
            discovery_context=CapabilityContext(
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
    assert any(
        schema.asset_path == "examples/lol/data/matches.csv" and schema.columns
        for schema in context.data_schemas
    )
    assert any(
        "Data probe (examples/lol/data/matches.csv)" in note for note in context.notes
    )


def test_normalise_probe_value_serializes_timestamp_like_objects() -> None:
    class TimestampLike:
        def isoformat(self) -> str:
            return "2026-04-03T12:34:56"

    value = bootstrap_module._normalise_probe_value(TimestampLike())

    assert value == "2026-04-03T12:34:56"


def test_probe_data_asset_reports_full_parquet_row_count_even_when_sampling(
    tmp_path, monkeypatch
) -> None:
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
            return FakeArrowTable(self._rows[offset : offset + length])

        def to_pandas(self) -> FakeDataFrame:
            return FakeDataFrame(self._rows)

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            assert path == asset_path
            self._rows = [
                {"kills": index % 9, "playername": f"player-{index}"}
                for index in range(123)
            ]
            self.metadata = type("Meta", (), {"num_rows": len(self._rows)})()
            self.num_row_groups = 1

        def read_row_group(self, index: int) -> FakeArrowTable:
            assert index == 0
            return FakeArrowTable(self._rows)

    fake_pyarrow = ModuleType("pyarrow")
    fake_parquet = ModuleType("pyarrow.parquet")
    fake_parquet.ParquetFile = FakeParquetFile
    fake_pyarrow.parquet = fake_parquet

    monkeypatch.setitem(
        sys.modules,
        "pandas",
        type("FakePandasModule", (), {"DataFrame": FakeDataFrame}),
    )
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    schema = bootstrap_module._probe_data_asset(str(asset_path), tmp_path)

    assert schema.sample_rows_loaded == 50
    assert schema.total_rows_verified == 123
    assert schema.verification_level == "verified_total_rows"
    assert schema.columns == ["kills", "playername"]
    assert schema.sample_values["kills"] == [0, 1, 2, 3, 4]


def test_probe_data_asset_timeout_reports_timeout_without_real_sleep(
    tmp_path, monkeypatch
) -> None:
    asset_path = tmp_path / "matches.csv"
    asset_path.write_text("kills\n1\n", encoding="utf-8")
    join_timeouts: list[float] = []

    class HungThread:
        def __init__(self, *, target, daemon) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(bootstrap_module, "_ACTIVE_DATA_PROBES", {})

    schema = bootstrap_module._probe_data_asset(
        str(asset_path),
        tmp_path,
        timeout_seconds=0.01,
        thread_factory=HungThread,
    )

    assert schema.load_error == "Probe timed out after 0.01s"
    assert join_timeouts == [0.01]


def test_probe_data_asset_skips_reprobe_for_same_asset_after_timeout_without_real_sleep(
    tmp_path, monkeypatch
) -> None:
    asset_path = tmp_path / "matches.csv"
    asset_path.write_text("kills\n1\n", encoding="utf-8")

    class HungThread:
        def __init__(self, *, target, daemon) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(bootstrap_module, "_ACTIVE_DATA_PROBES", {})

    first_schema = bootstrap_module._probe_data_asset(
        str(asset_path),
        tmp_path,
        timeout_seconds=0.01,
        thread_factory=HungThread,
    )
    second_schema = bootstrap_module._probe_data_asset(
        str(asset_path),
        tmp_path,
        timeout_seconds=0.01,
        thread_factory=HungThread,
    )

    assert first_schema.load_error == "Probe timed out after 0.01s"
    assert (
        second_schema.load_error
        == "Probe skipped because a previous timed-out probe is still running for this asset."
    )


def test_lol_preflight_treats_missing_installable_dependency_as_setup_step(
    monkeypatch, tmp_path
) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    (repo_root / "examples" / "lol" / "data").mkdir(parents=True)
    (repo_root / "examples" / "lol" / "data" / "subsample_lol_data.parquet").write_text(
        "placeholder",
        encoding="utf-8",
    )

    def raise_missing_dependency(_repo_root):
        raise ModuleNotFoundError("No module named 'lightgbm'")

    monkeypatch.setattr(
        pilot_adapters_module, "_load_lol_modules", raise_missing_dependency
    )

    checks = pilot_adapters_module._lol_preflight(repo_root)

    runtime_check = next(
        check for check in checks if check.name == "lol_runtime_imports"
    )
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

    result = backend._complete_json(
        "test", {"seen_at": datetime(2026, 4, 3, 12, 34, 56)}
    )

    assert result == {"ok": True}
    assert "2026-04-03T12:34:56" in captured["messages"][1]["content"]


def test_sanitize_human_text_makes_console_unsafe_characters_safe(monkeypatch) -> None:
    class FakeStdout:
        encoding = "cp1252"

    monkeypatch.setattr(sys, "stdout", FakeStdout())

    sanitized = _sanitize_human_text("Baseline \u2192 EDA \u2026 ready")

    assert sanitized == "Baseline -> EDA ... ready"


def test_draft_spec_returns_metric_questions_and_recommended_spec(
    tmp_path, monkeypatch
) -> None:
    class StubSpecBackend:
        def __init__(
            self,
            model: str,
            completion_fn=None,
            temperature: float = 0.2,
            extra_kwargs=None,
            **kwargs,
        ) -> None:
            self.model = model

        def propose_spec(self, objective, capability_context, user_preferences=None):
            assert (
                capability_context.available_metrics["recall_at_p90"]["scorer_ref"]
                == "metrics.recall_at_p90"
            )
            assert user_preferences == {
                "custom_primary_scorer": "metrics.recall_at_p90"
            }
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
                notes=[
                    "Discovered scorer metrics.recall_at_p90 from the host codebase."
                ],
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
    assert proposal.notes == [
        "Discovered scorer metrics.recall_at_p90 from the host codebase."
    ]


def test_loopforge_bootstrap_returns_questions_and_preflight_failures(
    tmp_path, monkeypatch
) -> None:
    access_advisor = FakeAccessAdvisorBackend()
    narrator = FakeNarrationBackend()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            assert (
                capability_context.available_metrics["fraud_recall"]["scorer_ref"]
                == "metrics.fraud_recall"
            )
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
                available_metrics={
                    "fraud_recall": {"scorer_ref": "metrics.fraud_recall"}
                },
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
    assert (
        Path(turn.access_guide_path)
        .read_text(encoding="utf-8")
        .startswith("# Access Guide")
    )
    runbook_path = tmp_path / "memory" / "agent_markdown" / "execution_runbook.md"
    assert runbook_path.exists()
    assert "Execution Runbook" in runbook_path.read_text(encoding="utf-8")
    assert len(access_advisor.calls) == 1
    assert len(narrator.bootstrap_calls) == 1


def test_loopforge_bootstrap_marks_turn_ready_when_no_hard_requirements_remain(
    tmp_path, monkeypatch
) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Plan is concrete.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=False,
            )

    monkeypatch.setattr(
        "loopforge.bootstrap.discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(),
    )
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


def test_bootstrap_repairs_unspecified_metric_goal_from_metric_catalog(
    tmp_path, monkeypatch
) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Use ordinal_loss for the kills model.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="unspecified"
                        ),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    monkeypatch.setattr(
        "loopforge.bootstrap.discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_metrics={
                "ordinal_loss": {
                    "goal": "minimize",
                    "preferred_role": "primary",
                    "scorer_ref": "metrics.ordinal_loss",
                }
            }
        ),
    )
    monkeypatch.setattr("loopforge.bootstrap.run_preflight_checks", lambda **kwargs: [])

    app = Loopforge(
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
    )
    monkeypatch.setattr(
        app,
        "resolve_execution_backend",
        lambda user_goal: bootstrap_module.ExecutionBackendResolution(
            kind="supported", factory_path="fake.module:factory"
        ),
    )

    turn = app.bootstrap(user_goal="Improve LoL player kills using ordinal loss.")

    assert turn.ready_to_start is True
    assert turn.proposal.recommended_spec.primary_metric.goal == "minimize"
    assert (
        turn.proposal.recommended_spec.primary_metric.scorer_ref
        == "metrics.ordinal_loss"
    )
    assert not any(
        check.name == "metric_goal_unspecified" and check.status == "failed"
        for check in turn.preflight_checks
    )


def test_bootstrap_passes_full_context_to_metric_repair_backend(
    tmp_path, monkeypatch
) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Primary metric should be ordinal_loss.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="unspecified"
                        ),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class RepairNarrator(FakeNarrationBackend):
        def __init__(self) -> None:
            super().__init__()
            self.fix_calls = []

        def fix_incomplete_metrics(
            self,
            current_spec,
            assistant_message,
            objective=None,
            capability_context=None,
        ):
            self.fix_calls.append(
                {
                    "current_spec": current_spec,
                    "assistant_message": assistant_message,
                    "objective": objective,
                    "capability_context": capability_context,
                }
            )
            return {
                "primary_metric": {"name": "ordinal_loss", "goal": "minimize"},
                "secondary_metrics": [],
                "guardrail_metrics": [],
            }

    monkeypatch.setattr(
        "loopforge.bootstrap.discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_metrics={"ordinal_loss": {"preferred_role": "primary"}},
            notes=["Ordinal loss is available in the repo."],
        ),
    )
    monkeypatch.setattr("loopforge.bootstrap.run_preflight_checks", lambda **kwargs: [])

    narrator = RepairNarrator()
    app = Loopforge(
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=narrator,
        access_advisor_backend=FakeAccessAdvisorBackend(),
    )
    monkeypatch.setattr(
        app,
        "resolve_execution_backend",
        lambda user_goal: bootstrap_module.ExecutionBackendResolution(
            kind="supported", factory_path="fake.module:factory"
        ),
    )

    turn = app.bootstrap(user_goal="Improve LoL player kills using ordinal loss.")

    assert turn.proposal.recommended_spec.primary_metric.goal == "minimize"
    assert narrator.fix_calls
    assert narrator.fix_calls[0]["objective"] == turn.proposal.objective
    assert narrator.fix_calls[0]["assistant_message"] == "Primary metric should be ordinal_loss."
    assert narrator.fix_calls[0]["capability_context"]["available_metrics"] == {
        "ordinal_loss": {"preferred_role": "primary"}
    }


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


def test_refine_bootstrap_questions_keeps_all_required_questions_and_adds_optional_context() -> (
    None
):
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

    # All agent-proposed questions are kept — the agent decides what to ask, code doesn't inject extras
    assert [q.key for q in refined] == [
        "position_handling",
        "primary_metric_confirmation",
    ]


def test_run_preflight_checks_blocks_unspecified_metric_goal(tmp_path) -> None:
    checks = bootstrap_module.run_preflight_checks(
        spec=build_spec(primary_metric=PrimaryMetric(name="mae", goal="unspecified")),
        capability_context=CapabilityContext(
            environment_facts={"autonomous_execution_supported": True}
        ),
        memory_root=tmp_path / "memory",
        executor_factory_path=None,
    )

    invalid_goal_check = next(
        check for check in checks if check.name == "metric_goal_unspecified"
    )

    assert invalid_goal_check.status == "failed"
    assert "mae" in invalid_goal_check.detail


def test_run_preflight_checks_verifies_generic_execution_lane(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    checks = bootstrap_module.run_preflight_checks(
        spec=build_spec(),
        capability_context=CapabilityContext(
            environment_facts={
                "repo_root": str(repo_root),
                "python_executable": sys.executable,
                "autonomous_execution_supported": True,
            }
        ),
        memory_root=tmp_path / "memory",
        executor_factory_path=None,
    )

    assert [check.name for check in checks] == [
        "memory_root_access",
        "generic_agentic_execution",
        "generic_agentic_execution_probe",
    ]
    assert checks[-1].status == "passed"
    assert "Verified the generic autonomous execution lane" in checks[-1].detail


def test_apply_answers_to_bootstrap_turn_stores_answers_and_clears_answer_blockers() -> (
    None
):
    turn = BootstrapTurn(
        assistant_message="Need one answer.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=build_spec(objective="Improve LoL kills model."),
            questions=[
                SpecQuestion(
                    key="target_binning", prompt="Cap at 7+ or 10+?", required=True
                )
            ],
            notes=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[],
        ready_to_start=False,
        missing_requirements=["answer:target_binning"],
    )

    updated = apply_answers_to_bootstrap_turn(
        turn, answers={"target_binning": "Cap at 10+"}
    )

    assert updated.ready_to_start is True
    assert updated.missing_requirements == []
    assert (
        updated.proposal.recommended_spec.metadata["bootstrap_answers"][
            "target_binning"
        ]
        == "Cap at 10+"
    )


def test_apply_feedback_preserves_previous_bootstrap_answers(tmp_path) -> None:
    class FeedbackNarrator:
        def interpret_feedback(self, turn, feedback, capability_context):
            return {
                "action": "patch",
                "spec_updates": {"constraints": {"max_runtime_minutes": 45}},
                "message": "Updated runtime budget.",
            }

    turn = BootstrapTurn(
        assistant_message="Need one answer.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=replace(
                build_spec(objective="Improve LoL kills model."),
                metadata={"bootstrap_answers": {"target_binning": "Cap at 10+"}},
            ),
            questions=[
                SpecQuestion(
                    key="target_binning", prompt="Cap at 7+ or 10+?", required=True
                )
            ],
            notes=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[],
        ready_to_start=True,
        missing_requirements=[],
    )

    app = Loopforge(
        memory_root=tmp_path / "memory",
        narrator_backend=FeedbackNarrator(),
    )

    updated = app.apply_feedback(turn, "Increase runtime budget.")

    assert updated is not None
    assert updated.ready_to_start is True
    assert updated.missing_requirements == []
    assert (
        updated.proposal.recommended_spec.metadata["bootstrap_answers"][
            "target_binning"
        ]
        == "Cap at 10+"
    )
    assert updated.proposal.recommended_spec.constraints["max_runtime_minutes"] == 45


def test_apply_feedback_falls_back_to_primary_metric_patch_when_interpreter_replans(
    tmp_path,
) -> None:
    class FeedbackNarrator:
        def interpret_feedback(self, turn, feedback, capability_context):
            return {
                "action": "replan",
                "message": "Needs full replan.",
            }

    turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=replace(
                build_spec(objective="Improve LoL kills model."),
                metadata={"bootstrap_answers": {"target_binning": "Cap at 10+"}},
            ),
            questions=[],
            notes=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[],
        ready_to_start=True,
        missing_requirements=[],
    )

    app = Loopforge(
        memory_root=tmp_path / "memory",
        narrator_backend=FeedbackNarrator(),
    )
    app._cached_capability_context = CapabilityContext(
        available_metrics={"DinalLossScorer": {"goal": "minimize"}}
    )

    feedback = (
        "DinalLossScorer as primary metric. Just create a new script and get the baseline first."
    )
    updated = app.apply_feedback(turn, feedback)

    assert updated is not None
    assert updated.proposal.recommended_spec.primary_metric.name == "DinalLossScorer"
    assert updated.proposal.recommended_spec.primary_metric.goal == "minimize"
    assert (
        updated.proposal.recommended_spec.metadata["bootstrap_answers"][
            "target_binning"
        ]
        == "Cap at 10+"
    )
    assert (
        updated.proposal.recommended_spec.metadata["bootstrap_answers"][
            "user_feedback"
        ]
        == feedback
    )
    assert feedback in updated.proposal.recommended_spec.metadata["operator_guidance"]


def test_apply_feedback_overrides_generic_placeholder_metric_with_feedback_metric(
    tmp_path,
) -> None:
    class FeedbackNarrator:
        def interpret_feedback(self, turn, feedback, capability_context):
            return {
                "action": "patch",
                "spec_updates": {
                    "primary_metric": {"name": "metric", "goal": "minimize"},
                    "metadata": {"operator_guidance": [feedback]},
                },
                "message": "Updated the primary metric to metric and carried your guidance into execution.",
            }

    turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model.",
            recommended_spec=replace(
                build_spec(objective="Improve LoL kills model."),
                primary_metric=PrimaryMetric(
                    name="OverUnderLogLossScorer", goal="minimize"
                ),
            ),
            questions=[],
            notes=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        preflight_checks=[],
        ready_to_start=True,
        missing_requirements=[],
    )

    app = Loopforge(
        memory_root=tmp_path / "memory",
        narrator_backend=FeedbackNarrator(),
    )
    app._cached_capability_context = CapabilityContext(
        available_metrics={
            "DinalLossScorer": {"goal": "minimize"},
            "OverUnderLogLossScorer": {"goal": "minimize"},
        }
    )

    feedback = (
        "DinalLossScorer as primary metric. Just create a new script and get the baseline first."
    )
    updated = app.apply_feedback(turn, feedback)

    assert updated is not None
    assert updated.proposal.recommended_spec.primary_metric.name == "DinalLossScorer"
    assert updated.proposal.recommended_spec.primary_metric.goal == "minimize"
    assert (
        updated.human_update
        == "Updated the primary metric to DinalLossScorer and carried your guidance into execution."
    )


def test_extract_primary_metric_from_feedback_ignores_generic_metric_catalog_entries() -> None:
    capability_context = CapabilityContext(
        available_metrics={
            "metric": {"goal": "minimize"},
            "DinalLossScorer": {"goal": "minimize"},
        }
    )
    feedback = (
        "dinalosscorer as primary metric. Just create a new script and get the baseline first."
    )

    extracted = bootstrap_module._extract_primary_metric_from_feedback(
        feedback, capability_context
    )

    assert extracted == "dinalosscorer"


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
                primary_metric=PrimaryMetric(
                    name="ordinal_loss_scorer", goal="minimize"
                ),
                allowed_actions=[
                    "predict",
                    "eda",
                    "targeted_tune",
                    "test_bad_helper_name",
                ],
                guardrail_metrics=[
                    MetricSpec(name="calibration_error", goal="minimize")
                ],
                secondary_metrics=[MetricSpec(name="mae", goal="minimize")],
            ),
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    _print_plan_summary(turn, print_fn=outputs.append)

    joined = "\n".join(outputs)
    assert "test_bad_helper_name" not in joined
    assert (
        "choose among the discovered actions: predict, eda, targeted_tune, and 1 more"
        in joined
    )
    assert "Ordinal Loss Scorer" in joined


def test_loopforge_blocks_autonomous_start_without_execution_preflight(
    tmp_path, monkeypatch
) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="The plan is clear.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="fraud_recall", goal="maximize"
                        ),
                        allowed_actions=["train"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={
                "train": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.5)]
                )
            },
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


def test_loopforge_start_uses_defaults_and_runs_when_bootstrap_is_ready(
    tmp_path, monkeypatch
) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            assert answer_history == {"positive_label_definition": "chargeback_60d"}
            return BootstrapTurn(
                assistant_message="Configuration is specific enough to start.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="fraud_recall", goal="maximize"
                        ),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_factory(spec, memory_root: Path):
        return build_adapter_setup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.55)]
                )
            },
            capability_context=CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_data_assets=["fraud_training_set"],
            ),
            discovery_context=CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_data_assets=["fraud_training_set"],
            ),
            preflight_checks=passed_check(
                name="warehouse_permissions", detail="Confirmed data access."
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=FakeWorkerBackend(
            [build_candidate(instructions="Run the baseline.")]
        ),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
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
    assert (
        "bootstrap:Improve fraud detection recall."
        in result["bootstrap"]["human_update"]
    )
    assert result["bootstrap"]["access_guide_path"].endswith("ops_access_guide.md")
    assert result["results"][0]["human_update"] == "iteration:baseline:accepted"


def test_orchestrator_classifies_multiprocessing_permission_error_as_recoverable() -> (
    None
):
    outcome = ExperimentOrchestrator._build_failure_outcome(
        RuntimeError(
            "PermissionError: [WinError 5] Access is denied while creating multiprocessing resources. "
            "Traceback...\nself._ssock, self._csock = socket.socketpair()"
        )
    )

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "MultiprocessingPermissionError"
    assert "worker count reduced to 1" in outcome.recovery_actions[0]


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


def test_loopforge_bootstrap_uses_planning_only_discovery_when_runner_missing(
    tmp_path,
) -> None:
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
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            assert (
                capability_context.environment_facts["execution_backend_kind"]
                == "generic_agentic"
            )
            assert (
                capability_context.environment_facts["autonomous_execution_supported"]
                is True
            )
            assert (
                "run_experiment"
                in capability_context.environment_facts["generic_autonomous_actions"]
            )
            assert (
                capability_context.available_actions["run_experiment"]
                == "generic_autonomous_executor"
            )
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
    assert (
        turn.proposal.recommended_spec.metadata["execution_backend_kind"]
        == "generic_agentic"
    )
    assert "preflight:repo_execution_not_supported" not in turn.missing_requirements
    assert not (tmp_path / "memory" / "generated").exists()


def test_loopforge_does_not_attempt_runner_authoring_when_generic_executor_is_available(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def __init__(self) -> None:
            self.calls = 0

        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            self.calls += 1
            return BootstrapTurn(
                assistant_message="Planning-only discovery is available.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="primary_metric", goal="maximize"
                        ),
                        allowed_actions=["inspect_data"],
                    ),
                    questions=[
                        SpecQuestion(
                            key="data_size",
                            prompt="How large is the dataset?",
                            required=False,
                        )
                    ],
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
    second_turn = app.bootstrap(
        user_goal="Protect approval precision.", answers={"data_size": "12000 rows"}
    )

    assert first_turn.ready_to_start is True
    assert second_turn.ready_to_start is True
    # Generic autonomous execution should not bounce through runner authoring.
    assert authoring_backend.calls == 0
    assert bootstrap_backend.calls == 2


def test_loopforge_can_run_with_generic_executor_when_no_supported_runner(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "metrics.py").write_text(
        "def ordinal_loss():\n    return 0.0\n", encoding="utf-8"
    )

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            assert (
                capability_context.environment_facts["runner_kind"]
                == "generic_agentic_executor"
            )
            return BootstrapTurn(
                assistant_message="Generic executor is ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_execute(self, candidate, snapshot):
        return ExperimentOutcome(
            status="success",
            primary_metric_value=0.25,
            metric_results={
                "ordinal_loss": MetricResult(name="ordinal_loss", value=0.25)
            },
            execution_details={
                "step_results": [{"returncode": 0, "stdout": "ordinal_loss=0.25"}]
            },
        )

    monkeypatch.setattr(
        agentic_execution_module.GenericExecutionPlanExecutor, "execute", fake_execute
    )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=FakeWorkerBackend(
            [build_candidate(hypothesis="Run generic baseline.", execution_steps=[ExecutionStep(kind="shell", command="echo ok")])]
        ),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    result = app.start(user_goal="Create a kills model in this repo.")

    assert result["status"] == "started"
    assert result["bootstrap"]["ready_to_start"] is True
    assert result["results"][0]["accepted_summary"]["outcome_status"] == "success"
    assert result["results"][0]["accepted_summary"]["primary_metric_value"] == 0.25


def test_loopforge_tolerates_narration_payloads_without_message_key(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["inspect_repo"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Inspection completed and the loop can continue autonomously."
                            }
                        )
                    }
                }
            ]
        }

    narrator = bootstrap_module.LiteLLMNarrationBackend(
        model="openai/gpt-5.4",
        completion_fn=fake_completion,
    )

    def fake_execute(self, candidate, snapshot):
        return ExperimentOutcome(
            status="success",
            primary_metric_value=0.5,
            metric_results={
                "ordinal_loss": MetricResult(name="ordinal_loss", value=0.5)
            },
            execution_details={
                "step_results": [
                    {
                        "returncode": 0,
                        "stdout": '{"metric_results":{"ordinal_loss":0.5}}',
                    }
                ]
            },
        )

    monkeypatch.setattr(
        agentic_execution_module.GenericExecutionPlanExecutor, "execute", fake_execute
    )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=FakeWorkerBackend(
            [
                build_candidate(
                    hypothesis="Inspect the repo.",
                    action_type="inspect_repo",
                    change_type="inspect",
                    instructions="Run a minimal step that also reports the primary metric.",
                    execution_steps=[ExecutionStep(kind="shell", command="echo ok")],
                )
            ]
        ),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=narrator,
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    result = app.start(user_goal="Inspect the repo.")

    assert result["status"] == "started"
    assert (
        result["bootstrap"]["human_update"]
        == "Inspection completed and the loop can continue autonomously."
    )
    assert (
        result["results"][0]["human_update"]
        == "Inspection completed and the loop can continue autonomously."
    )


def test_loopforge_runner_authoring_retries_after_failed_smoke_run_when_called_explicitly(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "metrics.py").write_text(
        "def ordinal_loss():\n    return 0.0\n", encoding="utf-8"
    )

    class RetryingRunnerAuthoringBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, list[str]]] = []

        def author_runner(self, request):
            self.calls.append((request.attempt_number, list(request.previous_errors)))
            if request.attempt_number == 1:
                return RunnerAuthoringResult(
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
            return RunnerAuthoringResult(
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
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    resolution = app._author_runner(
        "Create a kills model in this repo.",
        CapabilityContext(),
    )

    assert resolution.kind == "supported"
    assert resolution.factory_path is not None
    assert authoring_backend.calls[0] == (1, [])
    assert authoring_backend.calls[1][0] == 2


def test_validate_runner_factory_accepts_dict_preflight_checks(
    tmp_path, monkeypatch
) -> None:
    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={
                "baseline": type(
                    "BaselineExecutor",
                    (),
                    {
                        "execute": lambda self, candidate, snapshot: ExperimentOutcome(
                            primary_metric_value=0.1
                        )
                    },
                )()
            },
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "authored runner"},
                available_metrics={
                    "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
                },
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                {
                    "name": "authored_runner_ready",
                    "status": "passed",
                    "detail": "Runner validated.",
                }
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


def test_bootstrap_injects_high_level_execution_guidance_question_for_unresolved_remote_data(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="I found the likely repo entrypoints.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_data_assets=["s3://team-bucket/training/latest.parquet"],
            available_metrics={
                "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
            },
            environment_facts={
                "repo_root": str(repo_root.resolve()),
                "python_executable": sys.executable,
            },
            notes=[
                "The pipeline may load data from S3 or Databricks depending on the environment."
            ],
        ),
    )

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )
    monkeypatch.setattr(
        app,
        "_author_runner",
        lambda user_goal, capability_context, answers=None: (
            bootstrap_module.ExecutionBackendResolution(
                kind="planning_only",
                reason="Runner authoring still needs one execution hint.",
            )
        ),
    )

    turn = app.bootstrap(user_goal="Improve ranking quality.")

    assert turn.ready_to_start is False
    assert any(
        question.key == bootstrap_module.EXECUTION_GUIDANCE_QUESTION_KEY
        for question in turn.proposal.questions
    )
    assert "answer:execution_strategy_hint" in turn.missing_requirements
    assert "preflight:execution_strategy_unresolved" in turn.missing_requirements
    assert any(
        check.name == "execution_strategy_unresolved" for check in turn.preflight_checks
    )


def test_bootstrap_does_not_author_runner_before_execution_direction_is_known(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="I found the likely repo entrypoints.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_data_assets=["s3://team-bucket/training/latest.parquet"],
            available_metrics={
                "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
            },
            environment_facts={
                "repo_root": str(repo_root.resolve()),
                "python_executable": sys.executable,
            },
            notes=["The training data is stored remotely."],
        ),
    )

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Runner authoring should not run before execution direction is known."
        )

    monkeypatch.setattr(app, "_author_runner", fail_if_called)

    turn = app.bootstrap(user_goal="Improve ranking quality.")

    assert turn.ready_to_start is False
    assert any(
        question.key == bootstrap_module.EXECUTION_GUIDANCE_QUESTION_KEY
        for question in turn.proposal.questions
    )


def test_bootstrap_uses_high_level_execution_hint_to_unlock_authored_runner(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="Ready to run.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="ordinal_loss", goal="minimize"
                        ),
                        allowed_actions=["baseline"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_discover(*, executor_factory_path, **kwargs):
        if executor_factory_path:
            return CapabilityContext(
                available_actions={"baseline": "authored baseline"},
                available_metrics={
                    "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
                },
                environment_facts={
                    "repo_root": str(repo_root.resolve()),
                    "python_executable": sys.executable,
                    "runner_kind": "authored",
                },
                notes=["Authored runner resolved the execution path."],
            )
        return CapabilityContext(
            available_data_assets=["s3://team-bucket/training/latest.parquet"],
            available_metrics={
                "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
            },
            environment_facts={
                "repo_root": str(repo_root.resolve()),
                "python_executable": sys.executable,
            },
            notes=["The training data is stored remotely."],
        )

    def fake_adapter_setup(*, factory_path, spec, memory_root):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "authored baseline"},
                available_metrics={
                    "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
                },
                environment_facts={"runner_kind": "authored"},
            ),
            capability_provider=lambda effective_spec: CapabilityContext(
                available_actions={"baseline": "authored baseline"},
                available_metrics={
                    "ordinal_loss": {"goal": "minimize", "preferred_role": "primary"}
                },
                environment_facts={"runner_kind": "authored"},
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="authored_runner_ready",
                    status="passed",
                    detail="Runner validated.",
                )
            ],
        )

    monkeypatch.setattr(
        bootstrap_module, "discover_capabilities_for_objective", fake_discover
    )
    monkeypatch.setattr(bootstrap_module, "load_adapter_setup", fake_adapter_setup)

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    def fake_author_runner(user_goal, capability_context, answers=None):
        assert answers == {
            bootstrap_module.EXECUTION_GUIDANCE_QUESTION_KEY: "Use the repo's local Python loader against S3."
        }
        return bootstrap_module.ExecutionBackendResolution(
            kind="supported",
            factory_path="generated.runner:build_adapter",
        )

    monkeypatch.setattr(app, "_author_runner", fake_author_runner)

    turn = app.bootstrap(
        user_goal="Improve ranking quality.",
        answers={
            bootstrap_module.EXECUTION_GUIDANCE_QUESTION_KEY: "Use the repo's local Python loader against S3."
        },
    )

    assert turn.ready_to_start is True
    assert not any(
        question.key == bootstrap_module.EXECUTION_GUIDANCE_QUESTION_KEY
        for question in turn.proposal.questions
    )
    assert not any(check.status == "failed" for check in turn.preflight_checks)
    assert any(check.name == "authored_runner_ready" for check in turn.preflight_checks)


def test_loopforge_start_refuses_unsupported_planning_only_repo(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "evaluate_pipeline.py").write_text(
        "def evaluate_model():\n    return 0.0\n",
        encoding="utf-8",
    )

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self, user_goal, capability_context, answer_history=None, role_models=None
        ):
            return BootstrapTurn(
                assistant_message="Plan looks good.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="primary_metric", goal="maximize"
                        ),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=FakeWorkerBackend(
            [
                build_candidate(
                    hypothesis="Use the generic executor directly.",
                    instructions="Run a bounded generic command that reports the primary metric.",
                    execution_steps=[ExecutionStep(kind="shell", command="echo ok")],
                )
            ]
        ),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    monkeypatch.setattr(
        agentic_execution_module.GenericExecutionPlanExecutor,
        "execute",
        lambda self, candidate, snapshot: ExperimentOutcome(
            status="success",
            primary_metric_value=1.0,
            metric_results={
                "primary_metric": MetricResult(name="primary_metric", value=1.0)
            },
        ),
    )

    result = app.start(user_goal="Improve approval precision.")

    assert result["status"] == "started", f"Expected started but got: {result.get('error')}"
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


def test_inline_python_command_args_parses_multiline_python_c_commands() -> None:
    args = agentic_execution_module._inline_python_command_args(
        (
            f'"{sys.executable}" -c "from pathlib import Path; files=[\'pyproject.toml\'];\n'
            "for fp in files:\n"
            "    print(fp)\n"
            "print('ordinal_loss=0.27')"
            '"'
        )
    )

    assert args is not None
    assert args[0] == sys.executable
    assert args[1] == "-c"
    assert "print('ordinal_loss=0.27')" in args[2]
    assert "\nfor fp in files:\n" in args[2]


def test_classify_shell_failure_treats_permission_denied_as_recoverable() -> None:
    failure_type, recoverable, recovery_actions = (
        agentic_execution_module._classify_shell_failure(
            "",
            "PermissionError: [WinError 5] Access is denied",
        )
    )

    assert failure_type == "ShellPermissionDenied"
    assert recoverable is True
    assert "writable repo path" in recovery_actions[0]


def test_generic_execution_fix_uses_latest_revised_plan_on_subsequent_repairs(
    monkeypatch,
) -> None:
    repo_root = Path(".").resolve()
    spec = build_spec(allowed_actions=["run_experiment", "fix_failure"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={
                "execution_shell": "cmd.exe",
                "shell_family": "windows_cmd",
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
        hypothesis="Repair the failing command incrementally.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run the experiment.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command="python missing_script.py",
                cwd=str(repo_root),
            )
        ],
    )

    class RecordingFixBackend:
        def __init__(self) -> None:
            self.model = "anthropic/claude-opus-4-6-v1"

        def fix_execution_plan(
            self, candidate, failed_step_index, failure_summary, step_results
        ):
            current_command = candidate.execution_steps[0].command
            if current_command == "python missing_script.py":
                return [
                    ExecutionStep(
                        kind="shell",
                        command="head missing_script.py",
                        cwd=str(repo_root),
                    )
                ]
            return [
                ExecutionStep(
                    kind="shell", command='python -c "print(123)"', cwd=str(repo_root)
                )
            ]

    attempts: list[str] = []

    def fake_run_steps(
        steps, repo_root, default_timeout_seconds, max_captured_chars, progress_fn=None
    ):
        attempts.append(steps[0].command)
        if len(attempts) < 3:
            return (
                ExperimentOutcome(
                    status="recoverable_failure",
                    failure_type="ShellCommandFailed",
                    failure_summary="simulated command failure",
                    recoverable=True,
                    execution_details={"step_results": [{"command": steps[0].command}]},
                ),
                [{"command": steps[0].command}],
            )
        return (
            None,
            [
                {
                    "command": steps[0].command,
                    "returncode": 0,
                    "stdout": "123",
                    "stderr": "",
                }
            ],
        )

    monkeypatch.setattr(
        agentic_execution_module,
        "_validate_steps_pre_execution",
        lambda steps, repo_root: (True, 0, ""),
    )
    monkeypatch.setattr(agentic_execution_module, "_run_steps", fake_run_steps)

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root,
        fix_backend=RecordingFixBackend(),
        max_retries=2,
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert attempts == [
        "python missing_script.py",
        "head missing_script.py",
        'python -c "print(123)"',
    ]


def test_validate_steps_pre_execution_rejects_raw_file_dump_shell_command(
    tmp_path,
) -> None:
    valid, failed_idx, reason = agentic_execution_module._validate_steps_pre_execution(
        [
            ExecutionStep(
                kind="shell",
                command="type scripts\\experiment.py",
                cwd=str(tmp_path),
            )
        ],
        tmp_path,
    )

    assert valid is False
    assert failed_idx == 1
    assert "raw file-dump shell command" in reason


def test_generic_execution_plan_executor_blocks_invalid_prevalidated_plan(
    tmp_path,
) -> None:
    spec = build_spec(allowed_actions=["run_experiment"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_shell": "cmd.exe"}
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
        hypothesis="Inspect the script with a shell dump.",
        action_type="run_experiment",
        change_type="inspect",
        instructions="Dump the file before doing anything else.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command="type scripts\\experiment.py",
                cwd=str(tmp_path),
            )
        ],
    )

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=tmp_path,
        max_retries=0,
    ).execute(candidate, snapshot)

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "InvalidExecutionPlan"
    assert "raw file-dump shell command" in (outcome.failure_summary or "")


def test_generic_execution_plan_executor_captures_metrics_from_json_stdout(
    monkeypatch,
) -> None:
    repo_root = Path(".").resolve()
    spec = build_spec(
        allowed_actions=["run_experiment"],
        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
        secondary_metrics=[MetricSpec(name="accuracy", goal="maximize")],
        guardrail_metrics=[MetricSpec(name="latency_ms", goal="minimize")],
    )
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
        hypothesis="Emit metrics in machine-readable form.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run the experiment and print metrics as JSON.",
        execution_steps=[
            ExecutionStep(kind="shell", command="fake", cwd=str(repo_root))
        ],
    )

    monkeypatch.setattr(
        agentic_execution_module,
        "_validate_steps_pre_execution",
        lambda steps, repo_root: (True, 0, ""),
    )
    monkeypatch.setattr(
        agentic_execution_module,
        "_run_steps",
        lambda steps,
        repo_root,
        default_timeout_seconds,
        max_captured_chars,
        progress_fn=None: (
            None,
            [
                {
                    "index": 1,
                    "kind": "shell",
                    "command": "fake",
                    "cwd": str(repo_root),
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "metric_results": {"ordinal_loss": 0.19, "accuracy": 0.81},
                            "guardrail_metrics": {"latency_ms": 123.0},
                        }
                    ),
                    "stderr": "",
                }
            ],
        ),
    )

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert outcome.primary_metric_value == 0.19
    assert outcome.metric_results["ordinal_loss"].value == 0.19
    assert outcome.secondary_metrics["accuracy"] == 0.81
    assert outcome.guardrail_metrics["latency_ms"] == 123.0
    assert "Captured metric output" in outcome.notes[-1]


def test_generic_execution_plan_executor_matches_metric_names_case_insensitively(
    monkeypatch,
) -> None:
    repo_root = Path(".").resolve()
    spec = build_spec(
        allowed_actions=["run_experiment"],
        primary_metric=PrimaryMetric(name="Ordinallossscorer", goal="minimize"),
    )
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
        hypothesis="Emit the primary metric with normal Python casing.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run the experiment and print the metric.",
        execution_steps=[
            ExecutionStep(kind="shell", command="fake", cwd=str(repo_root))
        ],
    )

    monkeypatch.setattr(
        agentic_execution_module,
        "_validate_steps_pre_execution",
        lambda steps, repo_root: (True, 0, ""),
    )
    monkeypatch.setattr(
        agentic_execution_module,
        "_run_steps",
        lambda steps,
        repo_root,
        default_timeout_seconds,
        max_captured_chars,
        progress_fn=None: (
            None,
            [
                {
                    "index": 1,
                    "kind": "shell",
                    "command": "fake",
                    "cwd": str(repo_root),
                    "returncode": 0,
                    "stdout": "OrdinalLossScorer=0.19",
                    "stderr": "",
                }
            ],
        ),
    )

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert outcome.primary_metric_value == 0.19
    assert outcome.metric_results["Ordinallossscorer"].value == 0.19


def test_generic_execution_plan_executor_repairs_metricless_success_inline(
    monkeypatch,
) -> None:
    repo_root = Path(".").resolve()
    spec = build_spec(
        allowed_actions=["run_experiment", "fix_failure"],
        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
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
        hypothesis="Run a script that should report metrics.",
        action_type="run_experiment",
        change_type="run",
        instructions="Write and run the experiment script.",
        execution_steps=[
            ExecutionStep(kind="shell", command="first", cwd=str(repo_root))
        ],
    )

    class RecordingFixBackend:
        def __init__(self) -> None:
            self.failure_summaries: list[str] = []

        def fix_execution_plan(
            self,
            candidate,
            failed_step_index,
            failure_summary,
            step_results,
        ):
            self.failure_summaries.append(failure_summary)
            return [ExecutionStep(kind="shell", command="fixed", cwd=str(repo_root))]

    attempts: list[str] = []

    def fake_run_steps(
        steps, repo_root, default_timeout_seconds, max_captured_chars, progress_fn=None
    ):
        attempts.append(steps[0].command)
        if steps[0].command == "first":
            return (
                None,
                [
                    {
                        "index": 1,
                        "kind": "shell",
                        "command": "first",
                        "cwd": str(repo_root),
                        "returncode": 0,
                        "stdout": "setup completed",
                        "stderr": "",
                    }
                ],
            )
        return (
            None,
            [
                {
                    "index": 1,
                    "kind": "shell",
                    "command": "fixed",
                    "cwd": str(repo_root),
                    "returncode": 0,
                    "stdout": '{"metric_results":{"primary_metric":0.23}}',
                    "stderr": "",
                }
            ],
        )

    monkeypatch.setattr(
        agentic_execution_module,
        "_validate_steps_pre_execution",
        lambda steps, repo_root: (True, 0, ""),
    )
    monkeypatch.setattr(agentic_execution_module, "_run_steps", fake_run_steps)

    fix_backend = RecordingFixBackend()
    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root,
        fix_backend=fix_backend,
        max_retries=1,
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert outcome.primary_metric_value == 0.23
    assert attempts == ["first", "fixed"]
    assert len(fix_backend.failure_summaries) == 1
    assert "did not print a parseable value" in fix_backend.failure_summaries[0]


def test_generic_execution_plan_executor_treats_metricless_success_as_failure_in_autonomous_mode(
    monkeypatch,
) -> None:
    repo_root = Path(".").resolve()
    spec = build_spec(
        allowed_actions=["run_experiment"],
        primary_metric=PrimaryMetric(name="ordinal_loss", goal="minimize"),
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
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
        hypothesis="Run a script that forgets to print metrics.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run the script.",
        execution_steps=[
            ExecutionStep(kind="shell", command="fake", cwd=str(repo_root))
        ],
    )

    monkeypatch.setattr(
        agentic_execution_module,
        "_validate_steps_pre_execution",
        lambda steps, repo_root: (True, 0, ""),
    )
    monkeypatch.setattr(
        agentic_execution_module,
        "_run_steps",
        lambda steps,
        repo_root,
        default_timeout_seconds,
        max_captured_chars,
        progress_fn=None: (
            None,
            [
                {
                    "index": 1,
                    "kind": "shell",
                    "command": "fake",
                    "cwd": str(repo_root),
                    "returncode": 0,
                    "stdout": "finished without metrics",
                    "stderr": "",
                }
            ],
        ),
    )

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root,
        max_retries=0,
    ).execute(candidate, snapshot)

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "MetricsNotReported"


def test_generic_execution_plan_executor_reports_step_level_progress(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    spec = build_spec(allowed_actions=["run_experiment"])
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
        hypothesis="Write and run a script.",
        action_type="run_experiment",
        change_type="run",
        instructions="Write then execute the script.",
        execution_steps=[
            ExecutionStep(kind="write_file", path="scripts/run_it.py", content="print(1)\n"),
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" scripts\\run_it.py',
                cwd=str(repo_root),
            ),
        ],
    )

    monkeypatch.setattr(
        agentic_execution_module,
        "_validate_steps_pre_execution",
        lambda steps, repo_root: (True, 0, ""),
    )
    monkeypatch.setattr(
        agentic_execution_module,
        "_run_steps",
        agentic_execution_module._run_steps,
    )

    progress_log: list[tuple[str, str]] = []
    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root,
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert any("Running step 1/2: write scripts\\run_it.py" in msg for _, msg in progress_log)
    assert any("Running step 2/2:" in msg and "run_it.py" in msg for _, msg in progress_log)


def test_describe_step_hides_shell_boilerplate() -> None:
    step = ExecutionStep(
        kind="shell",
        command=(
            'cd /d C:\\repo && "C:\\repo\\.venv\\Scripts\\python.exe" '
            'scripts\\experiment_lol_player_kills_features_v2.py'
        ),
    )

    description = agentic_execution_module._describe_step(step)

    assert description == "run scripts\\experiment_lol_player_kills_features_v2.py"


def test_run_steps_reports_long_running_process_progress(tmp_path, monkeypatch) -> None:
    script_path = tmp_path / "sleepy.py"
    script_path.write_text(
        "import time\n"
        "time.sleep(0.15)\n"
        "print('done')\n",
        encoding="utf-8",
    )
    progress_log: list[tuple[str, str]] = []
    monkeypatch.setattr(
        agentic_execution_module,
        "STEP_PROGRESS_CHECKPOINTS",
        (0.01, 0.03),
    )

    failure_outcome, step_results = agentic_execution_module._run_steps(
        [
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" sleepy.py',
                cwd=str(tmp_path),
                rationale="Wait for the model run to finish so we can capture its metrics.",
            )
        ],
        tmp_path,
        default_timeout_seconds=5,
        max_captured_chars=4000,
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
    )

    assert failure_outcome is None
    assert step_results[0]["returncode"] == 0
    assert any(
        "Still running step 1/1: run sleepy.py - Wait for the model run to finish so we can capture its metrics." in message
        for _, message in progress_log
    )


def test_run_steps_handles_large_stdout_without_deadlocking(tmp_path) -> None:
    failure_outcome, step_results = agentic_execution_module._run_steps(
        [
            ExecutionStep(
                kind="shell",
                command=(
                    f'"{sys.executable}" -c "import sys; sys.stdout.write(\'x\' * 200000)"'
                ),
                cwd=str(tmp_path),
            )
        ],
        tmp_path,
        default_timeout_seconds=5,
        max_captured_chars=4000,
        progress_fn=None,
    )

    assert failure_outcome is None
    assert step_results[0]["returncode"] == 0
    assert step_results[0]["stdout"]


def test_worker_backend_preserves_operator_facing_reasoning_metadata() -> None:
    def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Try leakage-safe rolling features.",
                                "action_type": "run_experiment",
                                "change_type": "feature_engineering",
                                "instructions": "Build a bounded script and run grouped CV.",
                                "execution_steps": [
                                    {
                                        "kind": "write_file",
                                        "path": "scripts/run.py",
                                        "content": "print('ok')\n",
                                    }
                                ],
                                "metadata": {
                                    "observation_summary": "The latest accepted run used a simpler baseline feature set.",
                                    "reasoning_summary": "The repo already contains grouped CV and scorer plumbing, so the next move is to extend features rather than redesign execution.",
                                    "next_step_summary": "Write one script that trains and prints the primary metric.",
                                },
                            }
                        )
                    }
                }
            ]
        }

    backend = bootstrap_module.LiteLLMWorkerBackend(
        model="test-model",
        completion_fn=fake_completion,
    )
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
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

    candidate = backend.propose_next_experiment(snapshot)

    assert (
        candidate.metadata["observation_summary"]
        == "The latest accepted run used a simpler baseline feature set."
    )
    assert "grouped CV and scorer plumbing" in candidate.metadata["reasoning_summary"]
    assert "prints the primary metric" in candidate.metadata["next_step_summary"]


def test_orchestrator_progress_includes_operator_reasoning(tmp_path) -> None:
    progress_log: list[tuple[str, str]] = []
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
    memory_store = FileMemoryStore(tmp_path / "memory")
    memory_store.initialize(spec=spec, reset_state=True)
    candidate = ExperimentCandidate(
        hypothesis="Extend the ordinal baseline with leakage-safe rolling context.",
        action_type="run_experiment",
        change_type="feature_engineering",
        instructions="Build a new script that reuses the existing scorer path and prints the metric payload.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="scripts/experiment.py",
                content="print('ok')\n",
                rationale="Write a single runnable experiment entrypoint instead of scattering probes across iterations.",
            )
        ],
        metadata={
            "observation_summary": "The repo already has an ordinal scorer path and local LoL parquet assets.",
            "reasoning_summary": "The fastest next move is to extend the existing pipeline in one self-contained run, not rediscover APIs.",
            "next_step_summary": "Create the experiment script and evaluate the primary metric in the same iteration.",
        },
    )
    outcome = ExperimentOutcome(
        status="success",
        metric_results={"log_loss": MetricResult(name="log_loss", value=0.42)},
        primary_metric_value=0.42,
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=memory_store,
        worker_backend=FakeWorkerBackend([candidate]),
        executor=RoutingExperimentExecutor(
            handlers={},
            plan_executor=StaticActionExecutor([outcome]),
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Useful run.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="Keep iterating.")]
        ),
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
    )

    orchestrator.run_iteration()

    assert any("Observed   : The repo already has an ordinal scorer path" in message for _, message in progress_log)
    assert any("Why now    : The fastest next move is to extend the existing pipeline" in message for _, message in progress_log)
    assert any("Next check : Create the experiment script and evaluate the primary metric" in message for _, message in progress_log)


def test_generic_execution_plan_executor_surfaces_repair_reasoning(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
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
        hypothesis="Fix the missing script path and rerun the metric-emitting script.",
        action_type="run_experiment",
        change_type="repair",
        instructions="Write the runner and execute it.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" scripts\\missing.py',
                cwd=str(repo_root),
            )
        ],
    )
    progress_log: list[tuple[str, str]] = []
    call_count = {"value": 0}

    def fake_run_steps(
        steps, repo_root, default_timeout_seconds, max_captured_chars, progress_fn=None
    ):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return (
                ExperimentOutcome(
                    status="recoverable_failure",
                    failure_type="MissingScriptFile",
                    failure_summary="Script missing.py does not exist.",
                    recoverable=True,
                    execution_details={
                        "step_results": [
                            {
                                "index": 1,
                                "kind": "shell",
                                "command": steps[0].command,
                                "cwd": str(repo_root),
                                "stderr": "can't open file",
                            }
                        ]
                    },
                ),
                [
                    {
                        "index": 1,
                        "kind": "shell",
                        "command": steps[0].command,
                        "cwd": str(repo_root),
                        "stderr": "can't open file",
                    }
                ],
            )
        return (
            None,
            [
                {
                    "index": 1,
                    "kind": "shell",
                    "command": steps[0].command,
                    "cwd": str(repo_root),
                    "stdout": "log_loss=0.4",
                    "stderr": "",
                    "returncode": 0,
                }
            ],
        )

    class RepairingFixBackend:
        model = "test-fixer"

        def __init__(self) -> None:
            self.last_repair_summary = None

        def fix_execution_plan(self, *args, **kwargs):
            self.last_repair_summary = (
                "The failure is just a missing script path, so replace the shell step with a generated inline metric emitter."
            )
            return [
                ExecutionStep(
                    kind="shell",
                    command=(
                        f'"{sys.executable}" -c "print(\'metric_results={{\\\"log_loss\\\": 0.4}}\')"'
                    ),
                    cwd=str(repo_root),
                    rationale="Emit the metric payload directly after fixing the missing script path.",
                )
            ]

    monkeypatch.setattr(agentic_execution_module, "_run_steps", fake_run_steps)

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root,
        fix_backend=RepairingFixBackend(),
        max_retries=1,
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
    ).execute(candidate, snapshot)

    assert outcome.status in {"success", "recoverable_failure"}
    assert any("Revised plan with 1 step(s)" in message for _, message in progress_log)
    assert any("Repair reasoning: The failure is just a missing script path" in message for _, message in progress_log)


def test_generic_execution_plan_executor_locally_repairs_missing_script_before_llm(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path
    spec = build_spec(allowed_actions=["run_experiment"])
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
        hypothesis="Write and run the script.",
        action_type="run_experiment",
        change_type="run",
        instructions="Write a script, then run it.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="scripts/generated_runner.py",
                content="print('ok')\n",
            ),
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" scripts\\missing_runner.py',
                cwd=str(repo_root),
            ),
        ],
    )

    attempts: list[str] = []

    def fake_run_steps(
        steps, repo_root, default_timeout_seconds, max_captured_chars, progress_fn=None
    ):
        attempts.append(steps[-1].command)
        return (
            None,
            [
                {
                    "index": 2,
                    "kind": "shell",
                    "command": steps[-1].command,
                    "cwd": str(repo_root),
                    "returncode": 0,
                    "stdout": "ok",
                    "stderr": "",
                }
            ],
        )

    class ExplodingFixBackend:
        def fix_execution_plan(self, *args, **kwargs):
            raise AssertionError("LLM repair should not be called for this obvious missing script fix")

    monkeypatch.setattr(agentic_execution_module, "_run_steps", fake_run_steps)

    outcome = agentic_execution_module.GenericExecutionPlanExecutor(
        repo_root=repo_root,
        fix_backend=ExplodingFixBackend(),
        max_retries=1,
    ).execute(candidate, snapshot)

    assert outcome.status == "success"
    assert len(attempts) == 1
    assert "generated_runner.py" in attempts[0]


def test_format_outcome_surfaces_failed_step_command() -> None:
    spec = build_spec(allowed_actions=["run_experiment"])
    outcome = ExperimentOutcome(
        status="recoverable_failure",
        failure_summary="Traceback (most recent call last): boom",
        recovery_actions=["Fix the script and retry."],
        execution_details={
            "step_results": [
                {
                    "index": 1,
                    "command": f'"{sys.executable}" scripts\\experiment.py',
                    "returncode": 1,
                }
            ]
        },
    )

    message = ExperimentOrchestrator._format_outcome(outcome, spec)

    assert "Failed step:" in message
    assert "scripts\\experiment.py" in message


def test_loopforge_build_orchestrator_uses_fast_generic_executor_defaults(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    app = Loopforge(repo_root=repo_root, memory_root=tmp_path / "memory")
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )

    monkeypatch.setattr(
        app,
        "resolve_runtime_binding",
        lambda **kwargs: bootstrap_module.RuntimeBinding(
            executor_factory_path=None,
            handlers={},
            capability_provider=None,
            capability_context=CapabilityContext(
                environment_facts={"execution_backend_kind": "generic_agentic"}
            ),
        ),
    )

    orchestrator, _ = app.build_orchestrator(
        spec=spec,
        objective="Improve ordinal loss.",
    )

    plan_executor = orchestrator.executor.plan_executor
    assert plan_executor is not None
    assert plan_executor.max_retries == 2
    assert plan_executor.default_timeout_seconds == 30


def test_bootstrap_backend_emits_heartbeat_updates_for_slow_planning() -> None:
    progress_log: list[tuple[str, str]] = []

    def slow_completion(**kwargs):
        time.sleep(0.03)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Planned.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
                                        "allowed_actions": ["baseline"],
                                    }
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                                "ready_to_start": False,
                            }
                        )
                    }
                }
            ]
        }

    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=slow_completion,
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
        heartbeat_seconds=0.01,
    )

    backend.propose_bootstrap_turn(
        user_goal="Improve the existing model.",
        capability_context=CapabilityContext(),
        role_models=bootstrap_module.default_role_models(),
    )

    assert any(stage.startswith("bootstrap_plan_llm_wait_") for stage, _ in progress_log)
    assert any("Still drafting the experiment plan" in message for _, message in progress_log)


def test_worker_backend_emits_heartbeat_updates_for_slow_iteration_planning() -> None:
    progress_log: list[tuple[str, str]] = []

    def slow_completion(**kwargs):
        time.sleep(0.03)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Try a bounded experiment.",
                                "action_type": "run_experiment",
                                "change_type": "run",
                                "instructions": "Write and run a script.",
                                "execution_steps": [
                                    {"kind": "write_file", "path": "scripts/run.py", "content": "print(1)\n"},
                                    {"kind": "shell", "command": "python scripts/run.py"},
                                ],
                            }
                        )
                    }
                }
            ]
        }

    backend = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4",
        completion_fn=slow_completion,
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
        heartbeat_seconds=0.01,
    )
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
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

    candidate = backend.propose_next_experiment(snapshot)

    assert candidate.action_type == "run_experiment"
    assert any(stage.startswith("worker_propose_llm_wait_") for stage, _ in progress_log)
    assert any("Still planning the next experiment" in message for _, message in progress_log)


def test_loopforge_bootstrap_surfaces_live_planning_updates(tmp_path) -> None:
    progress_log: list[tuple[str, str]] = []

    def slow_completion(**kwargs):
        time.sleep(0.03)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Planned.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
                                        "allowed_actions": ["baseline"],
                                    }
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                                "ready_to_start": False,
                            }
                        )
                    }
                }
            ]
        }

    app = Loopforge(
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        bootstrap_backend=bootstrap_module.LiteLLMBootstrapBackend(
            model="openai/gpt-5.4",
            completion_fn=slow_completion,
            progress_fn=lambda stage, message: progress_log.append((stage, message)),
            heartbeat_seconds=0.01,
        ),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
    )
    app.repo_root.mkdir(parents=True, exist_ok=True)

    turn = app.bootstrap(user_goal="Improve the model.")

    assert turn.proposal.recommended_spec.primary_metric.name == "mae"
    assert any(stage == "propose_plan" for stage, _ in progress_log)
    assert any(stage.startswith("bootstrap_plan_llm_wait_") for stage, _ in progress_log)
    assert any("Still drafting the experiment plan" in message for _, message in progress_log)


def test_bootstrap_skips_experiment_guide_for_generic_autonomous_execution(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="mae", goal="minimize"),
                        allowed_actions=["run_experiment"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

        def build_experiment_guide(self, *args, **kwargs):
            raise AssertionError(
                "Generic autonomous bootstrap should not build an experiment guide."
            )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_actions={"run_experiment": "generic_autonomous_executor"},
            environment_facts={
                "repo_root": str(repo_root.resolve()),
                "python_executable": sys.executable,
                "execution_backend_kind": "generic_agentic",
                "runner_kind": "generic_agentic_executor",
            },
            notes=["Generic autonomous execution is available."],
        ),
    )

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    turn = app.bootstrap(user_goal="Improve the model.")

    assert turn.ready_to_start is True
    assert not (tmp_path / "memory" / "agent_markdown" / "experiment_guide.md").exists()


def test_bootstrap_writes_bootstrap_handoff_for_generic_autonomous_execution(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="Reuse the existing trainer script under src/train.py.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="mae", goal="minimize"),
                        allowed_actions=["run_experiment"],
                        metadata={
                            "operator_guidance": [
                                "Copy the existing trainer script and work from there."
                            ]
                        },
                    ),
                    questions=[],
                    notes=["Baseline path is src/train.py."],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_actions={"run_experiment": "generic_autonomous_executor"},
            environment_facts={
                "repo_root": str(repo_root.resolve()),
                "python_executable": sys.executable,
                "execution_backend_kind": "generic_agentic",
                "runner_kind": "generic_agentic_executor",
                "baseline_code_paths": ["src/train.py"],
            },
            notes=["Current baseline script: src/train.py"],
        ),
    )

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    turn = app.bootstrap(
        user_goal="Copy the existing trainer script and improve the model from there."
    )

    handoff_path = tmp_path / "memory" / "agent_markdown" / "bootstrap_handoff.md"
    assert handoff_path.exists()
    handoff_text = handoff_path.read_text(encoding="utf-8")
    contract = turn.proposal.recommended_spec.metadata["execution_contract"]
    assert contract["must_reference_baseline_paths"] is True
    assert contract["baseline_paths"] == ["src/train.py"]
    assert "Reuse the existing trainer script under src/train.py." in handoff_text
    assert "src/train.py" in handoff_text
    assert "Copy the existing trainer script and work from there." in handoff_text


def test_bootstrap_skips_narrator_summary_for_generic_autonomous_execution(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="Ready from bootstrap.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="mae", goal="minimize"),
                        allowed_actions=["run_experiment"],
                    ),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class ExplodingNarrator:
        def summarize_bootstrap(self, turn, capability_context):
            raise AssertionError(
                "Generic autonomous bootstrap should not call narrator summary."
            )

    monkeypatch.setattr(
        bootstrap_module,
        "discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            available_actions={"run_experiment": "generic_autonomous_executor"},
            environment_facts={
                "repo_root": str(repo_root.resolve()),
                "python_executable": sys.executable,
                "execution_backend_kind": "generic_agentic",
                "runner_kind": "generic_agentic_executor",
            },
            notes=["Generic autonomous execution is available."],
        ),
    )

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=ExplodingNarrator(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fine.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    turn = app.bootstrap(user_goal="Improve the model.")

    assert turn.human_update == "Ready from bootstrap."


def test_bootstrap_backend_uses_sparse_backoff_heartbeat_schedule() -> None:
    progress_log: list[tuple[str, str]] = []

    def slow_completion(**kwargs):
        time.sleep(0.05)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assistant_message": "Planned.",
                                "proposal": {
                                    "recommended_spec": {
                                        "primary_metric": {
                                            "name": "mae",
                                            "goal": "minimize",
                                        },
                                        "allowed_actions": ["baseline"],
                                    }
                                },
                                "role_models": bootstrap_module.default_role_models().to_dict(),
                                "ready_to_start": False,
                            }
                        )
                    }
                }
            ]
        }

    backend = bootstrap_module.LiteLLMBootstrapBackend(
        model="openai/gpt-5.4",
        completion_fn=slow_completion,
        progress_fn=lambda stage, message: progress_log.append((stage, message)),
        heartbeat_schedule_seconds=(0.01, 0.03),
    )

    backend.propose_bootstrap_turn(
        user_goal="Improve the existing model.",
        capability_context=CapabilityContext(),
        role_models=bootstrap_module.default_role_models(),
    )

    wait_stages = [stage for stage, _ in progress_log if stage.startswith("bootstrap_plan_llm_wait_")]
    assert wait_stages == ["bootstrap_plan_llm_wait_1", "bootstrap_plan_llm_wait_2"]


def test_routing_executor_uses_plan_executor_for_execution_steps_without_running_shell() -> (
    None
):
    class StubPlanExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, candidate, snapshot):
            self.calls.append(candidate.action_type)
            return ExperimentOutcome(
                status="success",
                execution_details={"step_results": [{"returncode": 0}]},
            )

    plan_executor = StubPlanExecutor()
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
        execution_steps=[ExecutionStep(kind="shell", command="fake", cwd=".")],
    )

    executor = RoutingExperimentExecutor(handlers={}, plan_executor=plan_executor)

    outcome = executor.execute(candidate, snapshot)

    assert outcome.status == "success"
    assert plan_executor.calls == ["baseline"]


@pytest.mark.parametrize(
    ("spec_kwargs", "capability_context", "action_type"),
    [
        (
            {},
            CapabilityContext(
                environment_facts={
                    "execution_backend_kind": "generic_agentic",
                    "autonomous_execution_supported": True,
                }
            ),
            "make_df",
        ),
        (
            {"metadata": {"execution_mode": "autonomous_after_bootstrap"}},
            CapabilityContext(),
            "unknown_generic_action",
        ),
    ],
)
def test_routing_executor_treats_unbound_generic_action_as_recoverable_failure(
    spec_kwargs: dict[str, object],
    capability_context: CapabilityContext,
    action_type: str,
) -> None:
    spec = build_spec(
        allowed_actions=[
            "inspect_repo",
            "inspect_data",
            "edit_code",
            "run_experiment",
            "fix_failure",
        ],
        **spec_kwargs,
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
        hypothesis="Call a repo helper directly.",
        action_type=action_type,
        change_type="inspect",
        instructions="Inspect the repo helper.",
    )

    executor = RoutingExperimentExecutor(handlers={})

    outcome = executor.execute(candidate, snapshot)

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "UnboundActionProposal"
    assert "execution_steps" in outcome.failure_summary


def test_generic_execution_plan_executor_detects_missing_script_before_running(
    tmp_path,
) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(allowed_actions=["run_experiment"])
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
        hypothesis="Run a temporary script.",
        action_type="run_experiment",
        change_type="run",
        instructions="Run a missing script.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command="python tmp_missing_script.py",
                cwd=str(tmp_path),
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "MissingScriptFile"
    assert "does not exist" in outcome.failure_summary


def test_generic_execution_plan_executor_rejects_plan_that_ignores_baseline_contract(
    tmp_path,
) -> None:
    from loopforge.core.agentic_execution import GenericExecutionPlanExecutor

    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
                "enforcement_scope": "until_first_successful_iteration",
            }
        },
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
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
        hypothesis="Write a new scratch pipeline.",
        action_type="run_experiment",
        change_type="run",
        instructions="Create a new standalone script.",
        execution_steps=[
            ExecutionStep(
                kind="write_file",
                path="scratch/new_model.py",
                content="print('new pipeline')\n",
            ),
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" scratch/new_model.py',
                cwd=str(tmp_path),
            ),
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "recoverable_failure"
    assert outcome.failure_type == "InvalidExecutionPlan"
    assert "src/train.py" in outcome.failure_summary
    assert "existing framework" in outcome.failure_summary.lower()


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

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "success"
    assert (tmp_path / "notes" / "generated.txt").read_text(
        encoding="utf-8"
    ) == "hello\nworld\n"


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

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "blocked"
    assert outcome.failure_type == "UnsafePath"


def test_generic_execution_plan_executor_blocks_shell_cwd_outside_repo(
    tmp_path,
) -> None:
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
    outside_dir = tmp_path.parent
    candidate = ExperimentCandidate(
        hypothesis="Try an unsafe shell cwd.",
        action_type="baseline",
        change_type="baseline",
        instructions="Attempt to run outside the repo root.",
        execution_steps=[
            ExecutionStep(
                kind="shell",
                command=f'"{sys.executable}" -c "print(123)"',
                cwd=str(outside_dir),
            )
        ],
    )

    outcome = GenericExecutionPlanExecutor(repo_root=tmp_path).execute(
        candidate, snapshot
    )

    assert outcome.status == "blocked"
    assert outcome.failure_type == "UnsafePath"
    assert "cwd" in outcome.failure_summary


def test_bootstrap_recomputes_capabilities_when_objective_changes(
    tmp_path, monkeypatch
) -> None:
    discovered_objectives = []

    def fake_discover_capabilities_for_objective(
        *, objective, memory_root, executor_factory_path, repo_root="."
    ):
        discovered_objectives.append(objective)
        return CapabilityContext(notes=[f"objective={objective}"])

    class StubBootstrapBackend:
        def __init__(self) -> None:
            self.notes_seen = []

        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            self.notes_seen.append(list(capability_context.notes))
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    backend = StubBootstrapBackend()
    monkeypatch.setattr(
        bootstrap_module,
        "discover_capabilities_for_objective",
        fake_discover_capabilities_for_objective,
    )

    app = Loopforge(
        repo_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
        bootstrap_backend=backend,
        narrator_backend=FakeNarrationBackend(),
    )

    app.bootstrap(user_goal="Improve recall.")
    app.bootstrap(user_goal="Reduce latency.")

    assert discovered_objectives == ["Improve recall.", "Reduce latency."]
    assert "objective=Improve recall." in backend.notes_seen[0]
    assert "objective=Reduce latency." in backend.notes_seen[1]


def test_build_iteration_policy_prefers_fix_failure_for_generic_recoverable_failures() -> (
    None
):
    spec = build_spec(
        allowed_actions=[
            "inspect_repo",
            "inspect_data",
            "edit_code",
            "run_experiment",
            "fix_failure",
        ]
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"execution_backend_kind": "generic_agentic"}
        ),
        best_summary=None,
        latest_summary=None,
        recent_records=[
            IterationRecord(
                iteration_id=1,
                parent_iteration_id=None,
                candidate=ExperimentCandidate(
                    hypothesis="Run a temp script.",
                    action_type="inspect_repo",
                    change_type="inspect",
                    instructions="Inspect.",
                ),
                outcome=ExperimentOutcome(
                    status="recoverable_failure",
                    failure_type="MissingScriptFile",
                    failure_summary="tmp script missing",
                    recoverable=True,
                ),
                reflection=ReflectionSummary(assessment="Need a fix."),
                review=ReviewDecision(status="accepted", reason="ok"),
            )
        ],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=2,
    )

    policy = build_iteration_policy(snapshot)

    assert policy["recommended_next_action"] == "fix_failure"


def test_build_iteration_policy_treats_metadata_only_autonomous_mode_as_generic() -> (
    None
):
    spec = build_spec(
        allowed_actions=[
            "inspect_repo",
            "inspect_data",
            "edit_code",
            "run_experiment",
            "fix_failure",
        ],
        metadata={"execution_mode": "autonomous_after_bootstrap"},
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[
            IterationRecord(
                iteration_id=1,
                parent_iteration_id=None,
                candidate=ExperimentCandidate(
                    hypothesis="Run a temp script.",
                    action_type="inspect_repo",
                    change_type="inspect",
                    instructions="Inspect.",
                ),
                outcome=ExperimentOutcome(
                    status="recoverable_failure",
                    failure_type="MissingScriptFile",
                    failure_summary="tmp script missing",
                    recoverable=True,
                ),
                reflection=ReflectionSummary(assessment="Need a fix."),
                review=ReviewDecision(status="accepted", reason="ok"),
            )
        ],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=2,
    )

    policy = build_iteration_policy(snapshot)

    assert policy["generic_autonomous"] is True
    assert policy["recommended_next_action"] == "fix_failure"


def test_memory_store_initialize_resets_previous_run_state(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    spec = build_spec(allowed_actions=["baseline"])
    store.initialize(spec, reset_state=True)
    record = IterationRecord(
        iteration_id=1,
        parent_iteration_id=None,
        candidate=ExperimentCandidate(
            hypothesis="Initial baseline.",
            action_type="baseline",
            change_type="baseline",
            instructions="Run baseline.",
        ),
        outcome=ExperimentOutcome(
            primary_metric_value=0.5, notes=["Baseline recorded."]
        ),
        reflection=ReflectionSummary(assessment="Looks fine."),
        review=ReviewDecision(status="accepted", reason="ok"),
    )
    summary = IterationSummary(
        iteration_id=1,
        parent_iteration_id=None,
        hypothesis="Initial baseline.",
        action_type="baseline",
        change_type="baseline",
        instructions="Run baseline.",
        config_patch={},
        primary_metric_name="log_loss",
        primary_metric_value=0.5,
        secondary_metrics={},
        result="improved",
        artifacts=["artifact.json"],
        lessons=["Keep the baseline."],
        next_ideas=["Tune next."],
        do_not_repeat=[],
        reflection_assessment="Looks fine.",
        review_reason="ok",
    )
    store.append_iteration_record(record)
    store.append_accepted_summary(summary)
    store.write_best_summary(summary)
    store.append_human_intervention(
        HumanIntervention(
            author="human",
            type="note",
            message="carry this forward",
            timestamp="2026-04-03T00:00:00Z",
        )
    )

    store.initialize(
        build_spec(objective="Fresh objective", allowed_actions=["baseline"]),
        reset_state=True,
    )
    snapshot = store.load_snapshot()

    assert snapshot.spec.objective == "Fresh objective"
    assert snapshot.next_iteration_id == 1
    assert snapshot.best_summary is None
    assert snapshot.recent_records == []
    assert snapshot.recent_summaries == []
    assert snapshot.recent_human_interventions == []
    assert snapshot.lessons_learned == ""


def test_start_from_bootstrap_turn_preserves_existing_memory_when_resuming(
    tmp_path, monkeypatch
) -> None:
    memory_root = tmp_path / "memory"
    store = FileMemoryStore(memory_root)
    original_spec = build_spec(
        allowed_actions=["baseline"], stop_conditions={"max_iterations": 1}
    )
    store.initialize(original_spec, reset_state=True)
    summary = IterationSummary(
        iteration_id=1,
        parent_iteration_id=None,
        hypothesis="Existing baseline.",
        action_type="baseline",
        change_type="baseline",
        instructions="Run baseline.",
        config_patch={},
        primary_metric_name="log_loss",
        primary_metric_value=0.5,
        secondary_metrics={},
        result="improved",
        artifacts=[],
        lessons=["Keep the baseline."],
        next_ideas=["Inspect error slices."],
        do_not_repeat=[],
        reflection_assessment="Looks fine.",
        review_reason="ok",
    )
    store.append_iteration_record(
        IterationRecord(
            iteration_id=1,
            parent_iteration_id=None,
            candidate=ExperimentCandidate(
                hypothesis="Existing baseline.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            outcome=ExperimentOutcome(primary_metric_value=0.5),
            reflection=ReflectionSummary(assessment="Looks fine."),
            review=ReviewDecision(status="accepted", reason="ok"),
        )
    )
    store.append_accepted_summary(summary)
    store.write_best_summary(summary)
    store.append_human_intervention(
        HumanIntervention(
            author="human",
            type="note",
            message="Continue from the existing memory.",
            timestamp="2026-04-03T00:00:00Z",
        )
    )

    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Resume with a follow-up.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline again.",
            )
        ]
    )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.4)]
                )
            },
            capability_provider=lambda effective_spec: CapabilityContext(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=memory_root,
        worker_backend=worker,
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Resumed.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )
    bootstrap_turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective=original_spec.objective,
            recommended_spec=original_spec,
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    result = app.start_from_bootstrap_turn(
        bootstrap_turn=bootstrap_turn,
        user_goal=original_spec.objective,
        iterations=1,
        reset_state=False,
    )

    assert result["status"] == "started"
    assert worker.snapshots[0].recent_summaries[0].hypothesis == "Existing baseline."
    assert (
        worker.snapshots[0].recent_human_interventions[0].message
        == "Continue from the existing memory."
    )
    assert result["results"][0]["record"]["iteration_id"] == 2


def test_start_from_bootstrap_turn_resets_state_when_resumed_spec_changes(
    tmp_path, monkeypatch
) -> None:
    memory_root = tmp_path / "memory"
    store = FileMemoryStore(memory_root)
    original_spec = build_spec(
        allowed_actions=["baseline"], stop_conditions={"max_iterations": 1}
    )
    store.initialize(original_spec, reset_state=True)
    store.append_iteration_record(
        IterationRecord(
            iteration_id=1,
            parent_iteration_id=None,
            candidate=ExperimentCandidate(
                hypothesis="Existing baseline.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            outcome=ExperimentOutcome(primary_metric_value=0.5),
            reflection=ReflectionSummary(assessment="Looks fine."),
            review=ReviewDecision(status="accepted", reason="ok"),
        )
    )
    store.append_human_intervention(
        HumanIntervention(
            author="human",
            type="note",
            message="Continue from the existing memory.",
            timestamp="2026-04-03T00:00:00Z",
        )
    )

    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Fresh plan after spec drift.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )
        ]
    )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.4)]
                )
            },
            capability_provider=lambda effective_spec: CapabilityContext(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=memory_root,
        worker_backend=worker,
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Reset.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )
    updated_spec = build_spec(
        objective=original_spec.objective,
        primary_metric=PrimaryMetric(name="new_metric", goal="maximize"),
        allowed_actions=["baseline"],
        stop_conditions={"max_iterations": 1},
    )
    bootstrap_turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective=updated_spec.objective,
            recommended_spec=updated_spec,
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    result = app.start_from_bootstrap_turn(
        bootstrap_turn=bootstrap_turn,
        user_goal=updated_spec.objective,
        iterations=1,
        reset_state=False,
    )

    assert result["status"] == "started"
    assert worker.snapshots[0].recent_records == []
    assert worker.snapshots[0].recent_human_interventions == []
    assert result["results"][0]["record"]["iteration_id"] == 1


def test_start_resets_previous_memory_for_fresh_run(tmp_path, monkeypatch) -> None:
    memory_root = tmp_path / "memory"
    store = FileMemoryStore(memory_root)
    existing_spec = build_spec(
        allowed_actions=["baseline"], stop_conditions={"max_iterations": 1}
    )
    store.initialize(existing_spec, reset_state=True)
    store.append_iteration_record(
        IterationRecord(
            iteration_id=1,
            parent_iteration_id=None,
            candidate=ExperimentCandidate(
                hypothesis="Stale baseline.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            outcome=ExperimentOutcome(primary_metric_value=0.5),
            reflection=ReflectionSummary(assessment="Looks fine."),
            review=ReviewDecision(status="accepted", reason="ok"),
        )
    )

    class StubBootstrapBackend:
        def propose_bootstrap_turn(
            self,
            user_goal,
            capability_context,
            answer_history=None,
            role_models=None,
            bootstrap_memory=None,
        ):
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(
                        objective=user_goal,
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Fresh baseline.",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )
        ]
    )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.4)]
                )
            },
            capability_provider=lambda effective_spec: CapabilityContext(),
            discovery_provider=lambda objective: CapabilityContext(),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="executor_ready",
                    status="passed",
                    detail="Executor is configured.",
                )
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=memory_root,
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=worker,
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Fresh.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )

    result = app.start(user_goal="New objective")

    assert result["status"] == "started"
    assert worker.snapshots[0].recent_records == []
    assert result["results"][0]["record"]["iteration_id"] == 1


def test_loopforge_start_from_bootstrap_turn_auto_installs_missing_dependency_and_retries(
    tmp_path, monkeypatch
) -> None:
    attempts = {"count": 0}

    class MissingDependencyExecutor:
        def execute(self, candidate, snapshot):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ModuleNotFoundError("No module named 'sklearn'")
            return ExperimentOutcome(
                primary_metric_value=0.19, notes=["Recovered after dependency install."]
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": MissingDependencyExecutor()},
            capability_provider=lambda effective_spec: CapabilityContext(),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Run baseline.",
                    action_type="baseline",
                    change_type="baseline",
                    instructions="Train baseline model.",
                ),
            ]
        ),
        narrator_backend=FakeNarrationBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Recovered.")]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="ok")]
        ),
    )
    monkeypatch.setattr(
        app,
        "_install_python_dependency",
        lambda package_name: (
            True,
            {"command": ["uv", "pip", "install", package_name]},
        ),
    )

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
    assert (
        "distributionmanagerpredictorwithaveryveryveryverylongname"
        not in summary["actions"]
    )


def test_scan_repo_discovers_real_data_files(tmp_path) -> None:
    data_dir = tmp_path / "examples" / "lol" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "matches.parquet").write_text("placeholder", encoding="utf-8")

    summary = scan_repo(tmp_path)

    assert "examples/lol/data/matches.parquet" in summary["data_assets"]


def test_scan_repo_discovers_imported_metric_symbol_usage(tmp_path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "lol_train_models.py").write_text(
        "\n".join(
            [
                "from spforge.scorer import OrdinalLossScorer",
                "",
                "def train_player_kills(df):",
                "    scorer = OrdinalLossScorer(pred_column='player_kills_probabilities', target='kills')",
                "    return scorer",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = scan_repo(tmp_path)

    assert "OrdinalLossScorer" in summary["metrics"]
    assert summary["metrics"]["OrdinalLossScorer"]["path"] == "scripts/lol_train_models.py"


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
            [
                ReflectionSummary(
                    assessment="Primary metric improved but the guardrail failed."
                )
            ]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="Track this in memory.")]
        ),
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
    assert (
        cycle.accepted_summary.metric_results["recall_at_p90"].scorer_ref
        == "metrics.recall_at_p90"
    )
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
            handlers={
                "baseline": StaticActionExecutor(
                    [ExperimentOutcome(primary_metric_value=0.5)]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="OK.", lessons=["Baseline done."])]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="accepted", reason="baseline recorded")]
        ),
        narrator_backend=FakeNarrationBackend(),
        progress_fn=lambda stage, msg: progress_log.append((stage, msg)),
    )
    orchestrator.initialize(
        build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 1})
    )

    orchestrator.run(iterations=1)

    stages = [stage for stage, _ in progress_log]
    assert "iteration_start" in stages
    assert "worker_propose" in stages
    assert "executor_run" in stages
    assert "reflect" in stages
    assert "review" in stages
    assert "narrate" in stages


def test_orchestrator_raises_experiment_interrupted_on_keyboard_interrupt(
    tmp_path,
) -> None:
    store = FileMemoryStore(tmp_path / "loop")

    class InterruptingWorker:
        def propose_next_experiment(self, snapshot):
            raise KeyboardInterrupt()

    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=InterruptingWorker(),
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([])}
        ),
        reflection_backend=FakeReflectionBackend([]),
        review_backend=FakeReviewBackend([]),
    )
    orchestrator.initialize(
        build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 3})
    )

    try:
        orchestrator.run(iterations=3)
        raise AssertionError("Expected ExperimentInterrupted")
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
                    planner="m",
                    worker="m",
                    reflection="m",
                    review="m",
                    consultation="m",
                    narrator="m",
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
                {
                    "message": {
                        "content": '{"hypothesis": "test", "action_type": "baseline", "change_type": "baseline", "instructions": "run"}'
                    }
                }
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
    assert stream_calls == [], (
        "stream_fn should not be called when completion_fn is provided"
    )


@pytest.mark.parametrize(
    ("payload", "expected_assessment", "expected_lessons", "expected_risks"),
    [
        (
            {
                "reflection": {
                    "assessment": "Useful diagnosis.",
                    "lessons": ["Inspect the loader first."],
                    "risks": ["Feature export names may differ."],
                    "recommended_next_action": "inspect_repo",
                }
            },
            "Useful diagnosis.",
            ["Inspect the loader first."],
            ["Feature export names may differ."],
        ),
        (
            {
                "summary": "Execution succeeded and produced useful inspection evidence.",
                "lessons": "Avoid bulk multi-file stdout reads.",
                "recommended_next_action": {
                    "candidate": {"action_type": "inspect_repo"},
                    "reason": "Need one more narrow inspection pass.",
                },
            },
            "Execution succeeded and produced useful inspection evidence.",
            ["Avoid bulk multi-file stdout reads."],
            [],
        ),
    ],
)
def test_reflection_backend_normalizes_supported_payload_shapes(
    payload: dict[str, object],
    expected_assessment: str,
    expected_lessons: list[str],
    expected_risks: list[str],
) -> None:
    from loopforge.core.backends import LiteLLMReflectionBackend

    backend = LiteLLMReflectionBackend(
        model="test-model",
        completion_fn=lambda **kwargs: {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        },
    )
    snapshot = MemorySnapshot(
        spec=build_spec(),
        effective_spec=build_spec(),
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
        hypothesis="Inspect the repo.",
        action_type="inspect_repo",
        change_type="inspect",
        instructions="Inspect bounded files.",
    )
    outcome = ExperimentOutcome(status="success")

    reflection = backend.reflect(snapshot, candidate, outcome)

    assert reflection.assessment == expected_assessment
    assert reflection.lessons == expected_lessons
    assert reflection.risks == expected_risks
    assert reflection.recommended_next_action == "inspect_repo"


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

    def fake_litellm_completion(**kwargs):
        assert kwargs.get("stream") is True
        return iter(chunks)

    # Patch litellm.completion temporarily
    import types

    backend._stream_fn = lambda token: streamed.append(token)

    # Call _stream_completion directly with a patched litellm
    import unittest.mock

    with unittest.mock.patch.dict(
        "sys.modules", {"litellm": types.ModuleType("litellm")}
    ):
        import sys

        sys.modules["litellm"].completion = fake_litellm_completion
        result = backend._stream_completion([{"role": "user", "content": "hi"}])

    assert result == "Hello world"
    assert "Hello" in streamed
    assert " " in streamed
    assert "world" in streamed


def test_complete_json_falls_back_when_streaming_json_request_fails() -> None:
    from loopforge.core.backends import _LiteLLMJsonBackend

    backend = _LiteLLMJsonBackend(
        model="test-model",
        stream_fn=lambda _token: None,
        progress_fn=lambda *_args: None,
    )
    backend._stream_completion = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("streaming rejected")
    )

    response_payload = {"ok": True}

    def fake_completion(**kwargs):
        assert kwargs.get("response_format") == {"type": "json_object"}
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(response_payload),
                    }
                }
            ]
        }

    import types
    import unittest.mock

    with unittest.mock.patch.dict(
        "sys.modules", {"litellm": types.ModuleType("litellm")}
    ):
        import sys

        sys.modules["litellm"].completion = fake_completion
        result = backend._complete_json(
            "Return JSON.",
            {"x": 1},
            progress_stage="worker_propose_llm",
            progress_label="planning the next experiment",
        )

    assert result == response_payload
