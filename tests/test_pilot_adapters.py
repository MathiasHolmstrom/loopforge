from __future__ import annotations

import importlib.util

import loopforge.pilot_adapters as pilot_adapters_module
from loopforge import (
    CapabilityContext,
    ExperimentCandidate,
    ExperimentSpec,
    Loopforge,
    MemorySnapshot,
    PrimaryMetric,
)
from loopforge.auto_adapter import synthesize_auto_adapter


def test_installable_python_dependency_suggests_pip_command() -> None:
    package_name, install_command = (
        pilot_adapters_module._installable_python_dependency(
            ModuleNotFoundError("No module named 'sklearn'")
        )
    )

    assert package_name == "sklearn"
    assert install_command.endswith(" -m pip install scikit-learn")


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


def test_synthesize_auto_adapter_generates_importable_adapter(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "train.py").write_text("def train_model():\n    return 1\n")

    factory_path = synthesize_auto_adapter(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        objective="improve training",
    )

    module_path, symbol = factory_path.rsplit(":", maxsplit=1)
    spec = importlib.util.spec_from_file_location("generated_auto_adapter", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    experiment_spec = ExperimentSpec(
        objective="improve training",
        primary_metric=PrimaryMetric(name="rmse", goal="minimize"),
        allowed_actions=["train_model"],
    )
    adapter_setup = getattr(module, symbol)(
        spec=experiment_spec,
        memory_root=tmp_path / "memory",
        repo_root=repo_root,
    )

    assert "train_model" in adapter_setup.handlers
    checks = adapter_setup.preflight_provider(
        experiment_spec,
        adapter_setup.discovery_provider("improve training"),
    )
    assert any(check.name == "auto_adapter_scaffold" for check in checks)


def test_lol_adapter_execute_uses_bound_repo_root(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "player-performance-ratings"
    repo_root.mkdir()
    captured: dict[str, object] = {}

    def fake_evaluate_lol_kills_configuration(
        *, repo_root, params, include_diagnostics=False
    ):
        captured["repo_root"] = repo_root
        return {
            "params": dict(params),
            "dataset_rows": 10,
            "validation_rows": 5,
            "metrics": {
                "kills_mae": 1.0,
                "kills_rmse": 1.2,
                "kills_mean_bias_abs": 0.1,
                "kills_mean_bias": 0.0,
            },
        }

    monkeypatch.setattr(
        pilot_adapters_module,
        "_evaluate_lol_kills_configuration",
        fake_evaluate_lol_kills_configuration,
    )
    monkeypatch.setattr(
        pilot_adapters_module,
        "_resolve_player_performance_repo",
        lambda: (_ for _ in ()).throw(
            AssertionError("should not rediscover repo root")
        ),
    )

    experiment_spec = ExperimentSpec(
        objective="improve lol kills",
        primary_metric=PrimaryMetric(name="kills_mae", goal="minimize"),
        allowed_actions=["baseline"],
    )
    adapter = pilot_adapters_module.build_lol_kills_adapter(
        spec=experiment_spec,
        memory_root=tmp_path / "memory",
        repo_root=repo_root,
    )

    outcome = adapter.handlers["baseline"].execute(
        ExperimentCandidate(
            hypothesis="Run baseline.",
            action_type="baseline",
            change_type="baseline",
            instructions="Run baseline.",
        ),
        MemorySnapshot(
            spec=experiment_spec,
            effective_spec=experiment_spec,
            capability_context=CapabilityContext(),
            best_summary=None,
            latest_summary=None,
            recent_records=[],
            recent_summaries=[],
            recent_human_interventions=[],
            lessons_learned="",
            markdown_memory=[],
            next_iteration_id=1,
        ),
    )

    assert captured["repo_root"] == repo_root.resolve()
    assert outcome.primary_metric_value == 1.0
