from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loopforge.core import (
    AdapterSetup,
    CapabilityContext,
    ExperimentOutcome,
    ExperimentSpec,
    MemorySnapshot,
    PreflightCheck,
)


def _installable_python_dependency(exc: Exception) -> tuple[str, str] | None:
    text = str(exc).strip()
    match = None
    if isinstance(exc, ModuleNotFoundError):
        match = exc.name
    if match is None:
        import re

        missing = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
        if missing:
            match = missing.group(1)
    if not match:
        return None
    package_name = match.split(".", maxsplit=1)[0]
    install_name = {
        "sklearn": "scikit-learn",
    }.get(package_name, package_name)
    return package_name, f"uv pip install {install_name}"


def detect_builtin_executor_factory(
    repo_root: Path | str, objective: str
) -> str | None:
    root = Path(repo_root).resolve()
    objective_lower = objective.lower()
    if (
        "lol" in objective_lower
        and "kill" in objective_lower
        and (root / "examples" / "lol" / "pipeline_transformer_example.py").exists()
    ):
        return "loopforge.pilot_adapters:build_lol_kills_adapter"
    loopforge_root = Path(__file__).resolve().parents[2]
    if (
        "nba" in objective_lower
        and "point" in objective_lower
        and (loopforge_root / "experiments" / "nba_points_real_pilot.py").exists()
    ):
        return "loopforge.pilot_adapters:build_nba_points_adapter"
    return None


def build_lol_kills_adapter(
    spec: ExperimentSpec, memory_root: Path | str
) -> AdapterSetup:
    handler = _LoLKillsHandler(spec=spec, memory_root=Path(memory_root))
    return AdapterSetup(
        handlers={
            "baseline": handler,
            "eda": handler,
            "slice_analysis": handler,
            "targeted_tune": handler,
        },
        capability_provider=lambda effective_spec: _lol_capability_context(
            _resolve_player_performance_repo()
        ),
        discovery_provider=lambda objective: _lol_capability_context(
            _resolve_player_performance_repo()
        ),
        preflight_provider=lambda effective_spec, capability_context: _lol_preflight(
            _resolve_player_performance_repo()
        ),
    )


def build_nba_points_adapter(
    spec: ExperimentSpec, memory_root: Path | str
) -> AdapterSetup:
    handler = _NBAPointsPilotHandler(memory_root=Path(memory_root))
    return AdapterSetup(
        handlers={
            "baseline": handler,
            "eda": handler,
            "slice_analysis": handler,
            "targeted_tune": handler,
        },
        capability_provider=lambda effective_spec: _nba_capability_context(),
        discovery_provider=lambda objective: _nba_capability_context(),
        preflight_provider=lambda effective_spec, capability_context: _nba_preflight(),
    )


def _resolve_player_performance_repo() -> Path:
    here = Path.cwd().resolve()
    candidates = [
        here,
        here / "player-performance-ratings",
        here.parent / "player-performance-ratings",
        Path(__file__).resolve().parents[2].parent / "player-performance-ratings",
    ]
    for candidate in candidates:
        if (
            candidate / "examples" / "lol" / "pipeline_transformer_example.py"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find player-performance-ratings repo with the LoL example files."
    )


def _add_repo_to_path(repo_root: Path) -> None:
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _lol_preflight(repo_root: Path) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            name="lol_repo_detected",
            status="passed" if repo_root.exists() else "failed",
            detail=f"Resolved player-performance-ratings repo: {repo_root}",
        ),
        PreflightCheck(
            name="lol_dataset_available",
            status="passed"
            if (
                repo_root / "examples" / "lol" / "data" / "subsample_lol_data.parquet"
            ).exists()
            else "failed",
            detail="LoL sample dataset is available.",
        ),
    ]
    try:
        _load_lol_modules(repo_root)
    except Exception as exc:
        recoverable_dependency = _installable_python_dependency(exc)
        if recoverable_dependency is not None:
            package_name, install_command = recoverable_dependency
            detail = (
                f"Missing installable Python dependency '{package_name}'. "
                f"Install it with `{install_command}` before running the LoL modeling stack."
            )
            checks.append(
                PreflightCheck(
                    name="lol_runtime_imports",
                    status="failed",
                    detail=detail,
                    required=False,
                    scope="setup",
                )
            )
            return checks
        checks.append(
            PreflightCheck(
                name="lol_runtime_imports",
                status="failed",
                detail=f"Could not import the LoL modeling stack: {exc}",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="lol_runtime_imports",
                status="passed",
                detail="LoL modeling dependencies import successfully.",
            )
        )
    return checks


def _lol_capability_context(repo_root: Path) -> CapabilityContext:
    data_asset = repo_root / "examples" / "lol" / "data" / "subsample_lol_data.parquet"
    pipeline_example = repo_root / "examples" / "lol" / "pipeline_transformer_example.py"
    ordinal_baseline = repo_root / "experiments" / "lol_kills_autonomous_baseline.py"
    empirical_bayes = repo_root / "experiments" / "lol_kills_empirical_bayes.py"
    end_to_end_test = repo_root / "tests" / "end_to_end" / "test_lol_player_kills.py"
    return CapabilityContext(
        available_actions={
            "baseline": "Run the current LoL kills baseline on the sample data.",
            "eda": "Diagnose model errors by position, kills bucket, and player slices.",
            "slice_analysis": "Inspect worst-performing slices before changing features.",
            "targeted_tune": "Try a bounded parameter change after diagnostics.",
        },
        available_data_assets=[str(data_asset)],
        available_metrics={
            "kills_mae": {
                "goal": "minimize",
                "preferred_role": "primary",
                "target": "kills",
            },
            "kills_rmse": {
                "goal": "minimize",
                "preferred_role": "secondary",
                "target": "kills",
            },
            "kills_mean_bias_abs": {
                "goal": "minimize",
                "preferred_role": "guardrail",
                "target": "kills",
            },
        },
        environment_facts={
            "autonomous_execution_supported": True,
            "runner_kind": "built_in_lol_kills",
            "iteration_policy": "diagnostic_first",
            "baseline_action": "baseline",
            "diagnostic_actions": ["eda", "slice_analysis"],
            "change_actions": ["targeted_tune"],
            "default_model_type": "point_prediction",
            "feature_engineering_stack": "spforge",
            "baseline_code_paths": [
                str(ordinal_baseline),
                str(empirical_bayes),
                str(pipeline_example),
                str(end_to_end_test),
            ],
        },
        notes=[
            "Built-in LoL kills runner uses the player-performance-ratings sample dataset and feature pipeline.",
            "The current autonomous runner is point-prediction first: LightGBM regressor with player/team/rating lag features.",
            f"Inspect {ordinal_baseline} for an existing ordinal/distribution baseline with capped kills classes, probability outputs, RankedProbabilityScorer, and OrdinalLossScorer.",
            f"Inspect {empirical_bayes} for a stronger follow-up baseline that still predicts kills as ordered classes.",
            f"Inspect {pipeline_example} for the existing spforge pipeline example: LagTransformer, RollingWindowTransformer, EstimatorTransformer, and NegativeBinomialEstimator.",
            f"Inspect {end_to_end_test} for the full player kills feature pipeline using spforge FeatureGeneratorPipeline, PlayerRatingGenerator, and MatchKFoldCrossValidator.",
            "Feature engineering changes should reuse spforge tooling rather than introducing a separate feature stack.",
            "Diagnostics report worst positions and worst kills buckets before any targeted tuning.",
        ],
    )


class _LoLKillsHandler:
    def __init__(self, *, spec: ExperimentSpec, memory_root: Path) -> None:
        self.spec = spec
        self.memory_root = memory_root

    def execute(self, candidate, snapshot: MemorySnapshot) -> ExperimentOutcome:
        repo_root = _resolve_player_performance_repo()
        action = candidate.action_type
        if action == "baseline":
            result = _evaluate_lol_kills_configuration(
                repo_root=repo_root, params=_baseline_lol_params()
            )
            notes = [
                f"Baseline MAE on validation rows: {result['metrics']['kills_mae']:.4f}",
                f"Baseline RMSE on validation rows: {result['metrics']['kills_rmse']:.4f}",
            ]
            return _build_lol_outcome(
                result=result,
                notes=notes,
                artifact_path=_write_artifact(
                    self.memory_root, "lol_kills_baseline", result
                ),
            )
        if action in {"eda", "slice_analysis"}:
            result = _evaluate_lol_kills_configuration(
                repo_root=repo_root,
                params=_baseline_lol_params(),
                include_diagnostics=True,
            )
            notes = [
                f"Worst position slice: {result['diagnostics']['worst_position']['position']} ({result['diagnostics']['worst_position']['mae']:.4f} MAE)",
                f"Worst kills bucket: {result['diagnostics']['worst_bucket']['bucket']} ({result['diagnostics']['worst_bucket']['mae']:.4f} MAE)",
            ]
            next_ideas = [
                f"Target feature changes at {result['diagnostics']['worst_position']['position']} players.",
                "Review whether recent-form windows are too short for volatile kill counts.",
            ]
            return _build_lol_outcome(
                result=result,
                notes=notes,
                next_ideas=next_ideas,
                artifact_path=_write_artifact(
                    self.memory_root, "lol_kills_eda", result
                ),
            )
        if action == "targeted_tune":
            search_space = [
                _baseline_lol_params(),
                {
                    **_baseline_lol_params(),
                    "kills_learning_rate": 0.05,
                    "kills_num_leaves": 31,
                },
                {
                    **_baseline_lol_params(),
                    "kills_learning_rate": 0.03,
                    "kills_num_leaves": 63,
                },
                {**_baseline_lol_params(), "lag_length": 5, "rolling_window": 30},
            ]
            if candidate.config_patch:
                search_space.append(
                    {**_baseline_lol_params(), **candidate.config_patch}
                )
            trials = [
                _evaluate_lol_kills_configuration(repo_root=repo_root, params=params)
                for params in search_space
            ]
            best = min(trials, key=lambda item: item["metrics"]["kills_mae"])
            notes = [
                f"Tried {len(trials)} bounded configurations.",
                f"Best MAE: {best['metrics']['kills_mae']:.4f}",
            ]
            return _build_lol_outcome(
                result=best,
                notes=notes,
                artifact_path=_write_artifact(
                    self.memory_root,
                    "lol_kills_tuned",
                    {"trials": trials, "best": best},
                ),
                code_or_config_changes=[f"Best params: {best['params']}"],
            )
        raise ValueError(f"Unsupported LoL action: {action}")


def _baseline_lol_params() -> dict[str, Any]:
    return {
        "lag_length": 3,
        "rolling_window": 20,
        "kills_learning_rate": 0.05,
        "kills_num_leaves": 31,
        "winner_c": 1.0,
        "n_splits": 3,
    }


def _load_lol_modules(repo_root: Path) -> dict[str, Any]:
    _add_repo_to_path(repo_root)
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import LogisticRegression
    from spforge import AutoPipeline, ColumnNames, FeatureGeneratorPipeline
    from spforge.cross_validator import MatchKFoldCrossValidator
    from spforge.feature_generator import LagTransformer, RollingWindowTransformer
    from spforge.performance_transformers._performance_manager import ColumnWeight
    from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures

    return {
        "AutoPipeline": AutoPipeline,
        "ColumnNames": ColumnNames,
        "ColumnWeight": ColumnWeight,
        "FeatureGeneratorPipeline": FeatureGeneratorPipeline,
        "LGBMRegressor": LGBMRegressor,
        "LagTransformer": LagTransformer,
        "LogisticRegression": LogisticRegression,
        "MatchKFoldCrossValidator": MatchKFoldCrossValidator,
        "PlayerRatingGenerator": PlayerRatingGenerator,
        "RatingKnownFeatures": RatingKnownFeatures,
        "RollingWindowTransformer": RollingWindowTransformer,
    }


def _prepare_lol_frames(
    repo_root: Path, params: dict[str, Any]
) -> tuple[Any, Any, Any, Any, Any]:
    modules = _load_lol_modules(repo_root)
    import pandas as pd

    column_names = modules["ColumnNames"](
        team_id="teamname",
        match_id="gameid",
        start_date="date",
        player_id="playername",
        league="league",
        position="position",
    )
    df = pd.read_parquet(
        repo_root / "examples" / "lol" / "data" / "subsample_lol_data.parquet"
    )
    df = (
        df.loc[lambda x: x.position != "team"]
        .assign(team_count=df.groupby("gameid")["teamname"].transform("nunique"))
        .loc[lambda x: x.team_count == 2]
        .assign(
            player_count=df.groupby(["gameid", "teamname"])["playername"].transform(
                "nunique"
            )
        )
        .loc[lambda x: x.player_count == 5]
        .drop_duplicates(subset=["gameid", "playername", "teamname"])
        .sort_values(["date", "gameid", "teamname", "playername"])
        .copy()
    )
    df["date"] = pd.to_datetime(df["date"])
    rating_generator_player_kills = modules["PlayerRatingGenerator"](
        features_out=[modules["RatingKnownFeatures"].PLAYER_RATING],
        performance_column="kills",
        auto_scale_performance=True,
        performance_weights=[modules["ColumnWeight"](name="kills", weight=1)],
        column_names=column_names,
    )
    rating_generator_result = modules["PlayerRatingGenerator"](
        features_out=[modules["RatingKnownFeatures"].TEAM_RATING_DIFFERENCE_PROJECTED],
        performance_column="result",
        non_predictor_features_out=[modules["RatingKnownFeatures"].PLAYER_RATING],
        column_names=column_names,
    )
    lag_generators = [
        modules["LagTransformer"](
            features=["kills", "deaths", "result"],
            lag_length=params["lag_length"],
            granularity=["playername"],
        ),
        modules["RollingWindowTransformer"](
            features=["kills", "deaths", "result"],
            window=params["rolling_window"],
            min_periods=1,
            granularity=["playername"],
        ),
    ]
    feature_generator = modules["FeatureGeneratorPipeline"](
        feature_generators=[
            rating_generator_player_kills,
            rating_generator_result,
            *lag_generators,
        ],
        column_names=column_names,
    )
    historical_df = feature_generator.fit_transform(df)
    game_winner_model = modules["AutoPipeline"](
        estimator=modules["LogisticRegression"](max_iter=1000, C=params["winner_c"]),
        impute_missing_values=True,
        scale_features=False,
        estimator_features=rating_generator_result.features_out,
    )
    winner_cv = modules["MatchKFoldCrossValidator"](
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=game_winner_model,
        prediction_column_name="result_oof",
        n_splits=params["n_splits"],
        target_column="result",
    )
    historical_df = winner_cv.generate_validation_df(historical_df)
    historical_df["result_prob1"] = historical_df["result_oof"].apply(_prob1)
    kill_features = ["result_prob1", *feature_generator.features_out]
    kills_model = modules["AutoPipeline"](
        estimator=modules["LGBMRegressor"](
            verbose=-100,
            random_state=42,
            learning_rate=params["kills_learning_rate"],
            num_leaves=params["kills_num_leaves"],
            n_estimators=150,
        ),
        impute_missing_values=True,
        scale_features=False,
        estimator_features=kill_features,
    )
    kills_cv = modules["MatchKFoldCrossValidator"](
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=kills_model,
        prediction_column_name="kills_oof",
        n_splits=params["n_splits"],
        target_column="kills",
    )
    kills_cv.features = kill_features
    kills_cv.target = "kills"
    historical_df = kills_cv.generate_validation_df(historical_df)
    validation_df = historical_df[historical_df["is_validation"]].copy()
    return modules, column_names, feature_generator, historical_df, validation_df


def _evaluate_lol_kills_configuration(
    *,
    repo_root: Path,
    params: dict[str, Any],
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    _, column_names, _, historical_df, validation_df = _prepare_lol_frames(
        repo_root, params
    )
    mae = float(mean_absolute_error(validation_df["kills"], validation_df["kills_oof"]))
    rmse = float(
        math.sqrt(
            mean_squared_error(validation_df["kills"], validation_df["kills_oof"])
        )
    )
    mean_bias = float((validation_df["kills_oof"] - validation_df["kills"]).mean())
    result = {
        "params": dict(params),
        "dataset_rows": int(len(historical_df)),
        "validation_rows": int(len(validation_df)),
        "metrics": {
            "kills_mae": mae,
            "kills_rmse": rmse,
            "kills_mean_bias_abs": abs(mean_bias),
            "kills_mean_bias": mean_bias,
        },
    }
    if include_diagnostics:
        diag_df = validation_df.copy()
        diag_df["kills_bucket"] = diag_df["kills"].clip(upper=7).astype(int).astype(str)
        diag_df.loc[diag_df["kills"] >= 7, "kills_bucket"] = "7+"
        position_slice = (
            diag_df.groupby("position")
            .apply(lambda x: float(mean_absolute_error(x["kills"], x["kills_oof"])))
            .sort_values(ascending=False)
        )
        bucket_slice = (
            diag_df.groupby("kills_bucket")
            .apply(lambda x: float(mean_absolute_error(x["kills"], x["kills_oof"])))
            .sort_values(ascending=False)
        )
        worst_rows = diag_df.assign(
            abs_error=(diag_df["kills_oof"] - diag_df["kills"]).abs()
        ).nlargest(5, "abs_error")
        result["diagnostics"] = {
            "worst_position": {
                "position": str(position_slice.index[0]),
                "mae": float(position_slice.iloc[0]),
            },
            "worst_bucket": {
                "bucket": str(bucket_slice.index[0]),
                "mae": float(bucket_slice.iloc[0]),
            },
            "position_mae": {
                str(idx): float(val) for idx, val in position_slice.items()
            },
            "bucket_mae": {str(idx): float(val) for idx, val in bucket_slice.items()},
            "worst_rows": worst_rows[
                [
                    column_names.match_id,
                    column_names.player_id,
                    "position",
                    "kills",
                    "kills_oof",
                    "abs_error",
                ]
            ].to_dict(orient="records"),
        }
    return result


def _build_lol_outcome(
    *,
    result: dict[str, Any],
    notes: list[str],
    artifact_path: str,
    next_ideas: list[str] | None = None,
    code_or_config_changes: list[str] | None = None,
) -> ExperimentOutcome:
    metrics = result["metrics"]
    guardrails = {"kills_mean_bias_abs": float(metrics["kills_mean_bias_abs"])}
    secondary = {"kills_rmse": float(metrics["kills_rmse"])}
    if "diagnostics" in result:
        diagnostics = result["diagnostics"]
        secondary["worst_position_mae"] = float(diagnostics["worst_position"]["mae"])
        secondary["worst_bucket_mae"] = float(diagnostics["worst_bucket"]["mae"])
    return ExperimentOutcome(
        primary_metric_value=float(metrics["kills_mae"]),
        secondary_metrics=secondary,
        guardrail_metrics=guardrails,
        artifacts=[artifact_path],
        notes=notes,
        next_ideas=next_ideas or [],
        code_or_config_changes=code_or_config_changes or [],
        dataset_version=f"lol-subsample-{result['dataset_rows']}",
    )


def _write_artifact(memory_root: Path, stem: str, payload: dict[str, Any]) -> str:
    artifact_dir = memory_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = artifact_dir / f"{stem}_{timestamp}.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return str(artifact_path)


def _prob1(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[1] if len(value) > 1 else value[0])
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return float(value[1] if len(value) > 1 else value[0])
    except Exception:
        pass
    return float(value)


def _nba_capability_context() -> CapabilityContext:
    return CapabilityContext(
        available_actions={
            "baseline": "Run the current NBA points baseline.",
            "eda": "Inspect worst prediction slices for the NBA points pilot.",
            "slice_analysis": "Inspect worst prediction slices for the NBA points pilot.",
            "targeted_tune": "Run a bounded NBA points parameter search.",
        },
        available_metrics={
            "mae": {
                "goal": "minimize",
                "preferred_role": "primary",
                "target": "points",
            },
            "ordinal_loss": {
                "goal": "minimize",
                "preferred_role": "guardrail",
                "target": "points",
            },
        },
        environment_facts={
            "autonomous_execution_supported": True,
            "runner_kind": "built_in_nba_points",
            "iteration_policy": "diagnostic_first",
            "baseline_action": "baseline",
            "diagnostic_actions": ["eda", "slice_analysis"],
            "change_actions": ["targeted_tune"],
        },
    )


def _nba_preflight() -> list[PreflightCheck]:
    pilot_path = (
        Path(__file__).resolve().parents[2] / "experiments" / "nba_points_real_pilot.py"
    )
    return [
        PreflightCheck(
            name="nba_pilot_available",
            status="passed" if pilot_path.exists() else "failed",
            detail=f"NBA points pilot script resolved at {pilot_path}",
        )
    ]


class _NBAPointsPilotHandler:
    def __init__(self, *, memory_root: Path) -> None:
        self.memory_root = memory_root

    def execute(self, candidate, snapshot: MemorySnapshot) -> ExperimentOutcome:
        from experiments.nba_points_real_pilot import (
            baseline_params,
            evaluate_configuration,
            load_modules,
            load_nba_dataframe,
            run_diagnostics,
        )

        repo_root = _resolve_player_performance_repo()
        if candidate.action_type in {"eda", "slice_analysis"}:
            diagnostics = run_diagnostics(repo_root=repo_root, n_splits=3)
            return ExperimentOutcome(
                primary_metric_value=float(diagnostics["diagnostic_metrics"]["mae"]),
                secondary_metrics={
                    key: float(value)
                    for key, value in diagnostics["diagnostic_metrics"].items()
                    if key != "mae"
                },
                artifacts=[
                    _write_artifact(self.memory_root, "nba_points_eda", diagnostics)
                ],
                notes=[f"Primary issue: {diagnostics['primary_issue']}"],
            )
        modules = load_modules(repo_root)
        df, column_names = load_nba_dataframe(repo_root, modules)
        params = baseline_params()
        if candidate.action_type == "targeted_tune" and candidate.config_patch:
            params.update(candidate.config_patch)
        metrics = evaluate_configuration(
            df=df,
            column_names=column_names,
            modules=modules,
            params=params,
            n_splits=3,
        )
        return ExperimentOutcome(
            primary_metric_value=float(metrics["mae"]),
            secondary_metrics={"ordinal_loss": float(metrics["ordinal_loss"])},
            artifacts=[
                _write_artifact(
                    self.memory_root,
                    "nba_points_run",
                    {"params": params, "metrics": metrics},
                )
            ],
            notes=[
                f"Ran NBA points {candidate.action_type} with MAE {metrics['mae']:.4f}."
            ],
        )
