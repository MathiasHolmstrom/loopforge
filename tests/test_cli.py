from __future__ import annotations

import loopforge.bootstrap as bootstrap_module
from loopforge import (
    BootstrapTurn,
    ExperimentCandidate,
    ExperimentInterrupted,
    ExperimentOutcome,
    ExperimentSpecProposal,
    FileMemoryStore,
    IterationCycleResult,
    IterationRecord,
    ReflectionSummary,
    ReviewDecision,
)
from loopforge.cli import build_argument_parser, run_interactive_start
from tests.support import build_spec


def test_ready_prompt_feedback_patches_current_plan_without_rebootstrap(
    monkeypatch,
) -> None:
    prompts: list[str] = []
    outputs: list[str] = []
    answers = iter(
        [
            "Improve the LoL kills model",
            "Use dinalosscorer as primary metric. Get the baseline first.",
            "quit",
        ]
    )

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = 0
            self.apply_feedback_calls: list[str] = []

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls += 1
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
            self.apply_feedback_calls.append(feedback)
            return BootstrapTurn(
                assistant_message="Updated.",
                proposal=ExperimentSpecProposal(
                    objective=turn.proposal.objective,
                    recommended_spec=build_spec(objective=turn.proposal.objective),
                    questions=[],
                ),
                role_models=turn.role_models,
                ready_to_start=True,
                human_update="Patched current plan.",
            )

    created_apps: list[StubLoopforge] = []

    def fake_loopforge(**kwargs):
        app = StubLoopforge(**kwargs)
        created_apps.append(app)
        return app

    monkeypatch.setattr("loopforge.cli.Loopforge", fake_loopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert len(created_apps) == 1
    assert created_apps[0].bootstrap_calls == 1
    assert created_apps[0].apply_feedback_calls == [
        "Use dinalosscorer as primary metric. Get the baseline first."
    ]
    assert any("Patched current plan." in str(output) for output in outputs)


def test_question_failure_does_not_silently_replan(
    monkeypatch,
) -> None:
    prompts: list[str] = []
    outputs: list[str] = []
    answers = iter(
        [
            "Improve the LoL kills model",
            "Why this metric?",
            "quit",
        ]
    )

    class StubNarrator:
        def answer_question(self, question, turn, capability_context):
            raise RuntimeError("narrator unavailable")

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.bootstrap_calls = 0
            self.apply_feedback_calls: list[str] = []
            self._cached_capability_context = None
            self.narrator_backend = StubNarrator()

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls += 1
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
            self.apply_feedback_calls.append(feedback)
            return turn

    created_apps: list[StubLoopforge] = []

    def fake_loopforge(**kwargs):
        app = StubLoopforge(**kwargs)
        created_apps.append(app)
        return app

    monkeypatch.setattr("loopforge.cli.Loopforge", fake_loopforge)

    exit_code = run_interactive_start(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    assert exit_code == 0
    assert len(created_apps) == 1
    assert created_apps[0].bootstrap_calls == 1
    assert created_apps[0].apply_feedback_calls == []
    assert any(
        "narrator failed: narrator unavailable" in str(output) for output in outputs
    )


def test_cli_parses_max_autonomous_hours_override() -> None:
    args = build_argument_parser().parse_args(
        [
            "run",
            "--spec",
            "spec.json",
            "--executor-factory",
            "pkg.module:build",
            "--max-autonomous-hours",
            "2.5",
            "--max-iterations",
            "12",
        ]
    )

    assert args.max_autonomous_hours == 2.5
    assert args.iterations == 12


def test_interrupt_feedback_restarts_same_iteration_without_rebootstrap(
    monkeypatch, tmp_path
) -> None:
    prompts: list[str] = []
    outputs: list[str] = []
    answers = iter(
        [
            "Improve the LoL kills model",
            "y",
            "champion isnt know at the time of prediction so cant be used as a feature",
        ]
    )

    memory_root = tmp_path / "memory"
    spec = build_spec(objective="Improve the LoL kills model")
    store = FileMemoryStore(memory_root)
    store.initialize(spec)
    record = IterationRecord(
        iteration_id=1,
        parent_iteration_id=None,
        candidate=ExperimentCandidate(
            hypothesis="Use champion feature.",
            action_type="train",
            change_type="train",
            instructions="Train with champion as a feature.",
        ),
        outcome=ExperimentOutcome(primary_metric_value=0.4),
        reflection=ReflectionSummary(assessment="Looks strong."),
        review=ReviewDecision(status="accepted", reason="ok"),
    )
    store.append_iteration_record(record)

    class StubLoopforge:
        def __init__(self, **kwargs) -> None:
            self.memory_root = kwargs["memory_root"]
            self.bootstrap_calls = 0
            self.start_calls = 0
            self.role_models = bootstrap_module.default_role_models()
            self._cached_capability_context = None
            self.narrator_backend = None

        def bootstrap(self, *, user_goal, answers=None):
            self.bootstrap_calls += 1
            return BootstrapTurn(
                assistant_message="Ready.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=build_spec(objective=user_goal),
                    questions=[],
                ),
                role_models=self.role_models,
                ready_to_start=True,
                human_update="Ready to start.",
            )

        def start_from_bootstrap_turn(self, **kwargs):
            self.start_calls += 1
            if self.start_calls == 1:
                cycle = IterationCycleResult(
                    record=record,
                    accepted_summary=None,
                    human_update=(
                        "Iteration 1 completed with review status accepted. Outcome: success."
                    ),
                )
                raise ExperimentInterrupted(
                    results_so_far=[cycle],
                    current_stage="iteration",
                )
            snapshot = FileMemoryStore(self.memory_root).load_snapshot()
            assert snapshot.next_iteration_id == 1
            return {
                "status": "started",
                "bootstrap": kwargs["bootstrap_turn"].to_dict(),
                "results": [],
            }

    created_apps: list[StubLoopforge] = []

    def fake_loopforge(**kwargs):
        app = StubLoopforge(**kwargs)
        created_apps.append(app)
        return app

    monkeypatch.setattr("loopforge.cli.Loopforge", fake_loopforge)

    exit_code = run_interactive_start(
        memory_root=memory_root,
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        print_fn=outputs.append,
    )

    snapshot = FileMemoryStore(memory_root).load_snapshot()

    assert exit_code == 0
    assert len(created_apps) == 1
    assert created_apps[0].bootstrap_calls == 1
    assert created_apps[0].start_calls == 2
    assert snapshot.next_iteration_id == 1
    assert any("Reopened iteration 1" in str(output) for output in outputs)
