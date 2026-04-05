from __future__ import annotations

import pytest

import loopforge.bootstrap as bootstrap_module
from loopforge import (
    BootstrapTurn,
    CapabilityContext,
    ExperimentSpecProposal,
    Loopforge,
)
from loopforge.auto_adapter import build_repo_scan_context
from tests.support import build_spec


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


def test_loopforge_init_wires_default_models_to_expected_backends(tmp_path) -> None:
    app = Loopforge(memory_root=tmp_path / "memory")

    assert app.bootstrap_backend.model == app.role_models.planner
    assert app.reflection_backend.model == app.role_models.reflection
    assert app.review_backend.model == app.role_models.review
    expected_helper = app.role_models.consultation
    assert app.access_advisor_backend.model == expected_helper
    assert app.narrator_backend.model == app.role_models.narrator
    assert (
        app.bootstrap_backend.max_completion_tokens
        == bootstrap_module.DEFAULT_ROLE_MAX_COMPLETION_TOKENS["planner"]
    )


def test_repo_scan_metric_catalog_does_not_auto_enrich_planning_metrics() -> None:
    spec_dict = build_spec().to_dict()
    spec_dict["primary_metric"] = {"name": "OrdinalLossScorer", "goal": "unspecified"}

    patched = bootstrap_module._apply_metric_catalog_defaults(
        spec_dict,
        CapabilityContext(
            available_metrics={
                "OrdinalLossScorer": {
                    "goal": "minimize",
                    "scorer_ref": "repo_scan:experiments/lol.py:OrdinalLossScorer",
                }
            },
            environment_facts={"observation_mode": "repo_scan_only"},
        ),
    )

    assert patched["primary_metric"]["goal"] == "minimize"
    assert (
        patched["primary_metric"]["scorer_ref"]
        == "repo_scan:experiments/lol.py:OrdinalLossScorer"
    )


def test_extract_primary_metric_from_feedback_fuzzy_matches_repo_metric_names() -> None:
    metric = bootstrap_module._extract_primary_metric_from_feedback(
        "use dinalosscorer as primary metric",
        CapabilityContext(
            available_metrics={
                "OrdinalLossScorer": {},
                "RankedProbabilityScorer": {},
            }
        ),
    )

    assert metric == "OrdinalLossScorer"


def test_repo_scan_context_surfaces_likely_baseline_files_and_metric_symbols(
    tmp_path,
) -> None:
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    baseline_path = experiments_dir / "lol_kills_autonomous_baseline.py"
    baseline_path.write_text(
        "\n".join(
            [
                "from scorers import OrdinalLossScorer, RankedProbabilityScorer",
                "",
                "def train_baseline():",
                "    return OrdinalLossScorer(), RankedProbabilityScorer()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "helpers.py").write_text(
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )

    context = build_repo_scan_context(
        tmp_path,
        objective="improve existing lol kills baseline and keep the current framework",
    )

    assert (
        "experiments/lol_kills_autonomous_baseline.py"
        in context.environment_facts["baseline_code_paths"]
    )
    assert any(
        "OrdinalLossScorer" in note and "lol_kills_autonomous_baseline.py" in note
        for note in context.notes
    )
    assert set(context.available_metrics) == {
        "OrdinalLossScorer",
        "RankedProbabilityScorer",
    }


def test_apply_feedback_resolves_goal_for_explicit_primary_metric(tmp_path) -> None:
    app = Loopforge(memory_root=tmp_path / "memory")
    app._cached_capability_context = CapabilityContext(
        available_metrics={"OrdinalLossScorer": {"path": "experiments/lol.py"}},
        environment_facts={"observation_mode": "repo_scan_only"},
    )

    class StubNarrator:
        def interpret_feedback(self, turn, feedback, capability_context):
            return None

        def fix_incomplete_metrics(
            self,
            current_spec,
            assistant_message,
            objective=None,
            capability_context=None,
        ):
            return {
                "primary_metric": {
                    "name": "OrdinalLossScorer",
                    "goal": "minimize",
                }
            }

    app.narrator_backend = StubNarrator()
    turn = BootstrapTurn(
        assistant_message="Blocked on missing metric goal.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model",
            recommended_spec=build_spec(
                objective="Improve LoL kills model",
                primary_metric=bootstrap_module.PrimaryMetric(
                    name="primary_metric",
                    goal="unspecified",
                ),
            ),
            questions=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=False,
    )

    updated = app.apply_feedback(turn, "use dinalosscorer as primary metric")

    assert updated is not None
    assert updated.proposal.recommended_spec.primary_metric.name == "OrdinalLossScorer"
    assert updated.proposal.recommended_spec.primary_metric.goal == "minimize"


def test_apply_feedback_does_not_persist_free_text_as_bootstrap_answer(
    tmp_path,
) -> None:
    app = Loopforge(memory_root=tmp_path / "memory")
    app._cached_capability_context = CapabilityContext(
        available_metrics={"OrdinalLossScorer": {"path": "experiments/lol.py"}}
    )

    class StubNarrator:
        def interpret_feedback(self, turn, feedback, capability_context):
            return None

        def fix_incomplete_metrics(
            self,
            current_spec,
            assistant_message,
            objective=None,
            capability_context=None,
        ):
            return {
                "primary_metric": {
                    "name": "OrdinalLossScorer",
                    "goal": "minimize",
                }
            }

    app.narrator_backend = StubNarrator()
    turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective="Improve LoL kills model",
            recommended_spec=build_spec(objective="Improve LoL kills model"),
            questions=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    updated = app.apply_feedback(
        turn,
        "use dinalosscorer as primary metric and load cross validated predictions",
    )

    assert updated is not None
    bootstrap_answers = updated.proposal.recommended_spec.metadata.get(
        "bootstrap_answers", {}
    )
    assert "user_feedback" not in bootstrap_answers
