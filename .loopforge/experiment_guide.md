# Experiment Guide: LoL Player Kills Prediction (RPS-Optimized)

## Objective
Build a multiclass model predicting player kills (capped at 10, so 11 ordinal classes 0–10) using spforge's player rating system. Evaluate primarily on Ranked Probability Score (minimize). Secondary guardrails: OrdinalLossScorer, MeanBiasScorer.

---

## 1. Data Loading

**File:** `examples/lol/data/subsample_lol_data.parquet` (12,432 rows, 16 columns)

**Loader utility:** `examples/lol/data/utils.py` → `get_sub_sample_lol_data()` — inspect this first to see what it returns (likely a pandas or polars DataFrame).

```python
# Fallback direct load:
import pandas as pd
df = pd.read_parquet("examples/lol/data/subsample_lol_data.parquet")
```

## 2. Schema — Key Columns

| Column | Type | Meaning |
|---|---|---|
| `gameid` | str | Unique match identifier |
| `date` | str | Match datetime string (e.g. `"2023-01-10 19:20:51"`) |
| `teamname` | str | Team name — use as `team_id` |
| `playername` | str | Player name — use as `player_id` |
| `position` | str | One of `top`, `jng`, `mid`, `bot`, `sup` |
| `kills` | int64 | **TARGET** — raw kill count |
| `deaths` | int64 | Available feature |
| `assists` | int64 | Available feature |
| `result` | int64 | Win (1) / Loss (0) |
| `gamelength` | int64 | Game length in seconds |
| `teamkills` | int64 | Total team kills |
| `teamdeaths` | int64 | Total team deaths |
| `totalgold` | int64 | Player gold |
| `damagetochampions` | int64 | Player damage |
| `champion` | str | Champion played |
| `league` | str | League/competition |

## 3. Target Variable

```python
MAX_KILLS = 10
df["kills_capped"] = df["kills"].clip(upper=MAX_KILLS)
# This gives 11 ordinal classes: 0, 1, 2, ..., 10
```

The model must output a probability vector of length 11 for each row.

## 4. Existing Experiment to Study First

**Read `experiments/lol_kills_rps_experiment.py` carefully.** It already contains:
- `normalize_probs(probs)` — normalizes probability arrays to sum to 1
- `counts_to_probs(counts)` — converts count arrays to probability distributions
- `_ranked_probability_score(y_true, probs)` — standalone RPS computation
- `_ordinal_loss(y_true, probs)` — standalone ordinal loss computation
- `main()` — the full pipeline entry point

**This is your starting template.** Understand its pipeline, then improve it.

## 5. spforge Rating System

The repo implements player/team rating systems. Key modules:

- **`spforge/data_structures.py`** — `ColumnNames` dataclass that maps semantic roles to column names. Inspect to find fields like `match_id`, `team_id`, `player_id`, `start_date`, `performance`, etc.
- **`spforge/ratings/`** — Player rating generator. Look at `tests/ratings/test_player_rating_generator.py` for usage patterns.
- **`spforge/autopipeline.py`** — `AutoPipeline` or similar high-level API. Inspect for convenience wrappers.
- **`spforge/cross_validator/cross_validator.py`** — Time-based cross-validation. Uses `__match_num` and `is_validation` columns.
- **`spforge/feature_generator/`** — Rolling window, lag, rolling mean features.

### Column mapping for spforge:
```python
# You'll need to create a ColumnNames-like mapping:
# match_id -> "gameid"
# start_date -> "date" (parse to datetime first)
# team_id -> "teamname"
# player_id -> "playername"
# performance -> "kills_capped" (or a normalized version)
```

## 6. Feature Engineering

### Must-create derived columns:
```python
# Parse date
df["start_date"] = pd.to_datetime(df["date"])

# Sort chronologically (required by spforge)
df = df.sort_values("start_date").reset_index(drop=True)

# Position is critical — one-hot or use as categorical feature
# Kill share within team
df["kill_share"] = df["kills"] / df["teamkills"].clip(lower=1)
```

### Features to use with spforge feature generators:
- **Rolling window** on `kills_capped` per player (e.g., last 5/10/20 games)
- **Rolling window** on `kill_share` per player
- **Player ratings** from `PlayerRatingGenerator`
- **Position** as categorical (strongly influences kill distribution)
- **`gamelength`** — longer games → more kills
- **`result`** — winners tend to have more kills

### spforge feature generators to reuse:
- `spforge.feature_generator.RollingMeanWindow` — rolling averages
- `spforge.feature_generator.Lag` — lagged values
- Look at `examples/nba/feature_engineering_example.py` for patterns

## 7. Metrics — How They Work

### Primary: RankedProbabilityScorer
**Location:** `spforge/scorer/_score.py` → class `RankedProbabilityScorer`

RPS measures calibration of cumulative probability distributions. For each row with true class `k` and predicted probabilities `[p0, p1, ..., p10]`:
```
RPS = (1/K) * sum_{i=0}^{K-1} (CDF_pred(i) - CDF_true(i))^2
```
where `CDF_true(i) = 1 if i >= k, else 0`. Lower is better.

The scorer expects a DataFrame with:
- A target column (integer class label)
- A prediction column containing list/array of 11 probabilities

### Guardrail: OrdinalLossScorer
**Location:** `spforge/scorer/_score.py` → class `OrdinalLossScorer`

Cross-entropy loss adapted for ordinal outcomes. Also expects list-column of probabilities.

### Guardrail: MeanBiasScorer
**Location:** `spforge/scorer/_score.py` → class `MeanBiasScorer`

When given probability predictions, computes expected value = `sum(i * p_i)` and measures `mean(expected - actual)`. Checks systematic over/under-prediction.

### Scorer instantiation pattern (inspect test files for exact API):
```python
from spforge.scorer import RankedProbabilityScorer, OrdinalLossScorer, MeanBiasScorer

rps_scorer = RankedProbabilityScorer(
    target_column="kills_capped",
    prediction_column="kills_probs",  # column containing list of 11 floats
)
score = rps_scorer.score(result_df)
```

Check `tests/scorer/test_score.py` for exact constructor signatures (especially `pred_column`, `target_column`, `validation_column`, `labels` params).

## 8. Cross-Validation / Splits

**Use time-based splitting** to prevent leakage (games are chronological):

```python
from spforge.cross_validator import CrossValidator  # inspect actual class name

# The CV likely splits on match number or date
# Key: group by gameid so all players from same game are in same fold
```

Alternatively, do a simple temporal split:
```python
# Use last ~20-30% of data chronologically as validation
df = df.sort_values("start_date")
n = len(df)
df["is_validation"] = 0
df.loc[df.index[int(n * 0.7):], "is_validation"] = 1
```

**Critical:** Never leak future game data into training features. All rolling/lag features must be computed on past data only. spforge handles this if you use its pipeline correctly.

## 9. Model Approach

### Approach A: LightGBM multiclass (baseline)
```python
import lightgbm as lgb

# objective="multiclass", num_class=11
# Features: player rating, rolling kills, position, gamelength, etc.
# Output: 11-class probabilities directly
```

### Approach B: spforge AutoPipeline (preferred)
Inspect `spforge/autopipeline.py` for the high-level API. It likely chains:
1. Rating generation
2. Feature engineering
3. LightGBM estimation
4. Cross-validation

Look at `tests/end_to_end/test_nba_player_points.py` for a complete end-to-end pattern with player-level predictions.

### Approach C: Negative Binomial distribution
`spforge/distributions/_negative_binomial_estimator.py` — kills are count data, so a negative binomial could work. Convert continuous distribution to discrete probabilities for classes 0–10.

## 10. Output Format

The final prediction column must contain a list/array of 11 floats (probabilities for classes 0–10) per row, summing to 1.0:

```python
# Example row: [0.05, 0.15, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.01, 0.005, 0.005]
```

## 11. Execution Steps

1. **Read** `experiments/lol_kills_rps_experiment.py` end-to-end
2. **Read** `spforge/scorer/_score.py` to understand RankedProbabilityScorer API
3. **Read** `spforge/data_structures.py` for ColumnNames
4. **Read** one end-to-end test (e.g., `tests/end_to_end/test_nba_player_points.py`)
5. **Run** the existing experiment: `python experiments/lol_kills_rps_experiment.py` to get baseline scores
6. **Iterate** on features, rating params, model hyperparameters to minimize RPS
7. **Check** OrdinalLossScorer and MeanBiasScorer as guardrails after each change

## 12. Key Pitfalls

- **Probability normalization:** Always ensure probabilities sum to 1.0 per row. Use `normalize_probs()` from the experiment file.
- **Class count mismatch:** Scorer expects exactly 11 probabilities (classes 0–10). If LightGBM sees fewer classes in training data, pad missing classes with 0.
- **Sorting:** spforge requires data sorted by date. Call `validate_sorting()` or sort manually.
- **Position conditioning:** Position is the single strongest predictor of kill distribution shape. Consider separate models per position or use it as a key feature.
- **Small data:** Only 12,432 rows. Avoid overly complex models. Regularize LightGBM (low `num_leaves`, high `min_child_samples`, `lambda_l1`/`lambda_l2`).
