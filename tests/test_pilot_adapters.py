from __future__ import annotations

import loopforge.pilot_adapters as pilot_adapters_module
from loopforge import Loopforge


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
