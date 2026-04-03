

# Experiment Guide: LoL Player Kills Prediction (Ordinal Multiclass)

## 1. Objective

Build a model predicting player kills (capped at 10) as ordinal classes 0–10, outputting 11-class probabilities. Minimize `OrdinalLossScorer` on time-based cross-validation. Monitor `MeanBiasScorer` as guardrail.

## 2. Data Loading

```python
# Primary data loader
from examples.lol.data.utils import get_sub_sample_lol_data
```

**Data file:** `examples/lol/data/subsample_lol_data.parquet` — 12,432 rows, 16 columns.

Read the loader function first to understand what preprocessing it does. If it just returns raw data, load directly:

```python
import pandas as pd
df = pd.read_parquet("examples/lol/data/subsample_lol_data.parquet")
```

## 3. Schema — Key Columns

| Column | Type | Description |
|---|---|---|
| `gameid` | str | Unique game identifier (use for CV grouping) |
| `date` | str | Datetime string, e.g. `"2023-01-10 19:20:51"` — parse to datetime for time-based splits |
| `teamname` | str | Team name — use as `team_id` equivalent |
| `playername` | str | Player name — use as `player_id` equivalent |
| `league` | str | League/competition |
| `position` | str | Player position: top/jng/mid/bot/sup |
| `result` | int64 | Win (1) / Loss (0) |
| `gamelength` | int64 | Game length in seconds |
| `totalgold` | int64 | Player's total gold |
| `teamkills` | int64 | Team's total kills |
| `teamdeaths` | int64 | Team's total deaths |
| `damagetochampions` | int64 | Player damage to champions |
| `champion` | str | Champion played |
| `kills` | int64 | **TARGET** — player kills |
| `assists` | int64 | Player assists |
| `deaths` | int64 | Player deaths |

## 4. Target Variable

```python
# Cap kills at 10, creating ordinal classes 0-10
df["kills_capped"] = df["kills"].clip(upper=10).astype(int)
```

This gives 11 classes (0, 1, 2, ..., 10). The model must output a list/array of 11 probabilities per row.

## 5. Existing Baseline to Study

**Read first:** `experiments/lol_naive_ordinal_baseline.py` — this contains the `main()` function that likely demonstrates the full pipeline pattern including:
- How `ColumnNames` / data structures are configured
- How the `AutoPipeline` is set up for ordinal prediction
- How ratings, features, and estimators are composed

Also read:
- `spforge/autopipeline.py` — the `AutoPipeline` class (or similar) that orchestrates ratings → features → estimator
- `spforge/data_structures.py` — `ColumnNames` or similar config objects
- `examples/lol/data/utils.py` — the `get_sub_sample_lol_data()` function

## 6. Feature Engineering

Use the spforge framework's built-in feature generators. Key features to create:

### 6a. Player Ratings
Use `PlayerRatingGenerator` from `spforge/ratings/` to generate Elo-style player ratings. The framework tracks player skill over time. Key config:
- `player_id` column: `playername`
- `team_id` column: `teamname`  
- Performance column: likely `kills_capped` or a normalized version
- The rating system needs a `match_id` (`gameid`) and `start_date` (parsed `date`)

### 6b. Team Ratings
Use `TeamRatingGenerator` — aggregates player ratings to team level. Produces columns like `team_rating_projected`, `opponent_rating_projected`, `rating_difference_projected`.

### 6c. Rolling Features
Use feature generators from `spforge/feature_generator/`:
- `RollingMeanWindow` — rolling averages of kills, assists, deaths, etc.
- `Lag` — lagged values of performance stats
- `RollingMeanDays` — time-weighted rolling means

Example pattern (inspect the NBA examples for exact API):
```python
from spforge.feature_generator import RollingMeanWindow, Lag
```

### 6d. Derived Columns to Create Before Pipeline

```python
df["date"] = pd.to_datetime(df["date"])
df["start_date"] = df["date"].dt.strftime("%Y-%m-%d")  # or keep as datetime, check what framework expects
df["kills_capped"] = df["kills"].clip(upper=10).astype(int)
# May need opponent team — each gameid has 2 teams with 5 players each
# Construct team_id_opponent by mapping gameid → both teams → pick the other one
```

**Important:** The framework likely expects an opponent team column. Build it:
```python
# Each game has exactly 2 teams
game_teams = df.groupby("gameid")["teamname"].apply(lambda x: list(x.unique())).to_dict()
df["team_id_opponent"] = df.apply(
    lambda row: [t for t in game_teams[row["gameid"]] if t != row["teamname"]][0], axis=1
)
```

## 7. Cross-Validation Strategy

**Time-based CV** — split by date to prevent future leakage. Use `spforge/cross_validator/cross_validator.py`:

```python
from spforge.cross_validator import CrossValidator  # check exact import
```

The cross-validator likely uses a `date_column` and `match_id_column` to create expanding-window or rolling-window time splits. It marks rows with `is_validation` column.

Key parameters to set:
- `date_column`: the parsed date column name
- `match_id_column`: `"gameid"`
- Number of folds / validation window size — inspect the CV class

**Grouping:** Ensure all rows from the same `gameid` are in the same fold (no leakage of same-game info).

## 8. Model / Estimator

The framework wraps LightGBM for multiclass. For ordinal multiclass with 11 classes:

```python
import lightgbm as lgb

lgbm_params = {
    "objective": "multiclass",
    "num_class": 11,
    "metric": "multi_logloss",
    "verbosity": -1,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
}
```

The framework likely has its own estimator wrapper — check `spforge/autopipeline.py` and `spforge/estimator/` for how to configure multiclass prediction that outputs probability arrays.

## 9. Scoring / Metrics

### Primary: OrdinalLossScorer
Located in `spforge/scorer/_score.py`. Read this class carefully.

```python
from spforge.scorer import OrdinalLossScorer

scorer = OrdinalLossScorer(
    target_column="kills_capped",
    prediction_column="kills_capped_pred",  # column containing list of 11 probabilities
    # Check constructor for other required params
)
```

**How it works:** Ordinal loss penalizes predicted probability mass proportional to its distance from the true class. For true class `k`, it computes a weighted sum of probabilities assigned to each class `j`, weighted by `|j - k|`. Lower is better.

### Guardrail: MeanBiasScorer
```python
from spforge.scorer import MeanBiasScorer

bias_scorer = MeanBiasScorer(
    target_column="kills_capped",
    prediction_column="kills_capped_pred",  # accepts probability lists; computes expected value
)
```

**How it works:** Computes `mean(predicted - actual)`. For probability predictions, it first converts to expected value (sum of class_label × probability). Target: close to 0.

## 10. Step-by-Step Execution Plan

1. **Read existing code:**
   - `cat experiments/lol_naive_ordinal_baseline.py`
   - `cat examples/lol/data/utils.py`
   - `cat spforge/data_structures.py` (for ColumnNames)
   - `cat spforge/autopipeline.py` (for pipeline API)
   - `cat spforge/scorer/_score.py` (for OrdinalLossScorer signature)
   - `cat spforge/cross_validator/cross_validator.py`

2. **Create experiment script** at `experiments/lol_kills_ordinal.py` based on the baseline, modifying:
   - Target: `kills_capped` (kills clipped to 10)
   - Num classes: 11
   - Features: player ratings + team ratings + rolling features on kills/assists/deaths/damage
   - CV: time-based
   - Scorer: OrdinalLossScorer + MeanBiasScorer

3. **Run baseline** — get initial OrdinalLoss score.

4. **Iterate** — tune features, LightGBM hyperparameters, rating parameters, rolling window sizes.

## 11. Key Things to Watch

- **Prediction format:** OrdinalLossScorer expects a column containing lists/arrays of length 11 (one probability per class 0–10). Ensure the pipeline outputs this format.
- **Data sorting:** The framework requires data sorted by date. Call `validate_sorting` or sort explicitly: `df = df.sort_values("date")`.
- **10 players per game:** Each `gameid` has 10 rows (5 per team). Features like `teamkills` are shared within a team — don't accidentally leak the target through team-level stats. Use only lagged/rolling versions.
- **Feature leakage:** `totalgold`, `damagetochampions`, `teamkills`, `assists`, `deaths` are all same-game outcomes. They CANNOT be used as raw features. Only use rolling/lagged historical versions, or exclude them and rely on ratings + position + champion.
- **Position encoding:** `position` (top/jng/mid/bot/sup) is a strong predictor of kill distribution — include it as a categorical feature.
