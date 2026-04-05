from __future__ import annotations

import json

import pytest

import loopforge.bootstrap as bootstrap_module
from loopforge import (
    BootstrapTurn,
    CapabilityContext,
    ExperimentSpecProposal,
    FileMemoryStore,
    Loopforge,
)
from loopforge.auto_adapter import build_repo_scan_context
from tests.support import build_spec


@pytest.mark.parametrize(
    ("openai_available", "anthropic_available", "expected_models"),
    [
        (
            True,
            False,
            {
                "planner": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "worker": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "reflection": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "review": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "consultation": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "narrator": bootstrap_module.DEFAULT_OPENAI_MODEL,
            },
        ),
        (
            False,
            True,
            {
                "planner": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "worker": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "reflection": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "review": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "consultation": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "narrator": bootstrap_module.DEFAULT_CLAUDE_MODEL,
            },
        ),
        (
            True,
            True,
            {
                "planner": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "worker": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "reflection": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "review": bootstrap_module.DEFAULT_OPENAI_MODEL,
                "consultation": bootstrap_module.DEFAULT_CLAUDE_MODEL,
                "narrator": bootstrap_module.DEFAULT_CLAUDE_MODEL,
            },
        ),
    ],
)
def test_default_role_models_provider_routing(
    monkeypatch,
    openai_available: bool,
    anthropic_available: bool,
    expected_models: dict[str, str],
) -> None:
    monkeypatch.setattr(
        "loopforge.bootstrap._can_use_openai_models", lambda: openai_available
    )
    monkeypatch.setattr(
        "loopforge.bootstrap._can_use_anthropic_helpers",
        lambda: anthropic_available,
    )
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)

    role_models = bootstrap_module.default_role_models()

    assert role_models.to_dict() == expected_models


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


def test_loopforge_loads_repo_role_model_settings_and_preserves_explicit_overrides(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "loopforge.models.json").write_text(
        json.dumps(
            {
                "model_profile": "all_codex",
                "roles": {
                    "planner": "anthropic/claude-sonnet-4-5",
                    "worker": "anthropic/claude-opus-4-6-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("loopforge.bootstrap._can_use_openai_models", lambda: True)
    monkeypatch.setattr("loopforge.bootstrap._can_use_anthropic_helpers", lambda: True)

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        worker_model="openai/gpt-5.4-mini",
    )

    assert app.role_models.planner == "anthropic/claude-sonnet-4-5"
    assert app.role_models.worker == "openai/gpt-5.4-mini"


def test_loopforge_warns_when_repo_role_model_provider_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "loopforge.models.json").write_text(
        json.dumps(
            {
                "roles": {
                    "worker": "openai/gpt-5.4",
                    "review": "openai/gpt-5.4",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("loopforge.bootstrap._can_use_openai_models", lambda: False)
    monkeypatch.setattr("loopforge.bootstrap._can_use_anthropic_helpers", lambda: True)
    progress: list[tuple[str, str]] = []

    app = Loopforge(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        progress_fn=lambda stage, message: progress.append((stage, message)),
    )

    assert app.role_models.worker == bootstrap_module.DEFAULT_CLAUDE_MODEL
    assert app.role_models.review == bootstrap_module.DEFAULT_CLAUDE_MODEL
    assert any("Loaded role model settings from" in message for _, message in progress)
    assert any(
        "OpenAI credentials were not detected; routing all agent roles through Claude models."
        in message
        for _, message in progress
    )
    assert any(
        "worker requested `openai/gpt-5.4`" in message for _, message in progress
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


def test_bootstrap_surfaces_runbook_generation_warning_without_failing(
    tmp_path, monkeypatch
) -> None:
    progress: list[tuple[str, str]] = []

    class StubBootstrapBackend:
        def build_experiment_guide(self, turn, capability_context, answers):
            return "# Guide"

    class StubNarrator:
        def summarize_bootstrap(self, turn, capability_context):
            return "Ready to start."

    app = Loopforge(
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        narrator_backend=StubNarrator(),
        progress_fn=lambda stage, message: progress.append((stage, message)),
    )

    monkeypatch.setattr(
        app,
        "resolve_execution_backend",
        lambda objective: bootstrap_module.ExecutionBackendResolution(
            kind="supported", factory_path="fake.module:build_adapter"
        ),
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.discover_capabilities_for_objective",
        lambda **kwargs: CapabilityContext(
            environment_facts={
                "execution_backend_kind": "supported",
                "python_executable": "python",
                "repo_root": str(tmp_path),
            }
        ),
    )
    monkeypatch.setattr(
        "loopforge.bootstrap._trace_data_source",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.run_preflight_checks",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "loopforge.bootstrap._verify_execution_environment",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.should_prepare_access_guide",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.build_execution_runbook",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("runbook failed")),
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.build_bootstrap_handoff",
        lambda **kwargs: "# Handoff",
    )

    class StubPlanner:
        def __init__(self, **kwargs) -> None:
            pass

        def plan(self, **kwargs):
            return {
                "contract": {
                    "source_script": "src/train.py",
                    "baseline_function": "train_model",
                    "data_loading": "load_data()",
                    "target_column": "target",
                    "primary_metric": "rmse",
                    "primary_metric_goal": "minimize",
                },
                "questions": [],
            }

    monkeypatch.setattr("loopforge.bootstrap.ToolUsePlanner", StubPlanner)

    turn = app.bootstrap(user_goal="Improve the baseline model")

    assert "Could not write execution runbook: runbook failed" in turn.human_update
    assert any(
        "Could not write execution runbook: runbook failed" == note
        for note in turn.proposal.notes
    )
    assert any(stage == "execution_runbook_warning" for stage, _ in progress)


def test_metric_repair_surfaces_warning_when_inference_fails(tmp_path) -> None:
    progress: list[tuple[str, str]] = []

    class StubNarrator:
        def fix_incomplete_metrics(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    app = Loopforge(
        memory_root=tmp_path / "memory",
        narrator_backend=StubNarrator(),
        progress_fn=lambda stage, message: progress.append((stage, message)),
    )
    spec = build_spec(
        primary_metric=bootstrap_module.PrimaryMetric(
            name="primary_metric",
            goal="unspecified",
        )
    )

    repaired = app._repair_incomplete_metrics(
        spec=spec,
        capability_context=CapabilityContext(),
        assistant_message="Need to infer the metric goal.",
    )

    assert repaired == spec
    assert any(
        stage == "fix_metrics_warning_llm" and "LLM unavailable" in message
        for stage, message in progress
    )


def test_start_from_bootstrap_turn_blocks_on_unreadable_persisted_state(
    tmp_path,
) -> None:
    memory_root = tmp_path / "memory"
    spec = build_spec(objective="Improve the baseline model")
    store = FileMemoryStore(memory_root)
    store.initialize(spec)
    (memory_root / "experiment_spec.json").write_text("{bad json", encoding="utf-8")

    app = Loopforge(memory_root=memory_root)
    turn = BootstrapTurn(
        assistant_message="Ready.",
        proposal=ExperimentSpecProposal(
            objective=spec.objective,
            recommended_spec=spec,
            questions=[],
        ),
        role_models=bootstrap_module.default_role_models(),
        ready_to_start=True,
    )

    result = app.start_from_bootstrap_turn(
        bootstrap_turn=turn,
        user_goal=spec.objective,
    )

    assert result["status"] == "blocked"
    assert result["error"]["type"] == "PersistedStateLoadFailed"
    assert (
        "Refusing to reset persisted memory automatically" in result["error"]["message"]
    )
