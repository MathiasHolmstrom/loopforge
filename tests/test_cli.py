from __future__ import annotations

import loopforge.bootstrap as bootstrap_module
from loopforge import BootstrapTurn, ExperimentSpecProposal
from loopforge.cli import run_interactive_start
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
