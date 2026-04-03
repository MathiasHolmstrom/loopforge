# LoL Player Kills Ordinal Prediction — Access & Permissions Playbook

## Summary

Build a LoL player kills prediction model (kills capped at 10 → 11 ordinal classes, 0–10) using spforge's player rating system from the `player-performance-ratings` repo. The model outputs multiclass probabilities via the AutoPipeline, evaluated with `OrdinalLossScorer` and/or `RankedProbabilityScorer`.

---

## Required Environment Variables

| Variable | Value | Reason |
|---|---|---|
| `PYTHONPATH` | `C:\Users\m.holmstrom\PycharmProjects\player-performance-ratings` | Ensures `spforge` and `examples` packages are importable |
| `LOOPFORGE_MEMORY_ROOT` | `.loopforge` | Writable memory root for execution state (pre-verified) |

---

## Required Permissions

| Resource | Permission | Reason |
|---|---|---|
| Repo root (`C:\Users\m.holmstrom\PycharmProjects\player-performance-ratings`) | Read/Write | Read sources, write experiment scripts and artifacts |
| `.loopforge/` | Read/Write | Loopforge checkpointing |
| `examples/lol/data/subsample_lol_data.parquet` | Read | Training data: 12,432 rows, 16 columns (kills, deaths, assists, champion, position, etc.) |
| `rush_yards_cv.parquet` | Read | Reference for pipeline output format with probability columns |
| `examples/ncaaf/rush_yards_cv.parquet` | Read | Reference for ordinal probability column structure |

---

## Pre-flight Commands

Run these **in order** before the experiment:

### 1. Install dependencies
```bash
cd C:\Users\m.holmstrom\PycharmProjects\player-performance-ratings
pip install -e .[all]
```

### 2. Verify spforge import
```bash
python -c "import spforge; print(spforge.__file__)"
```

### 3. Verify LightGBM
```bash
python -c "import lightgbm; print(lightgbm.__version__)"
```
> LightGBM is the default estimator for multiclass probability output. If missing: `pip install lightgbm`

### 4. Verify data loads
```bash
python -c "from examples.lol.data.utils import get_sub_sample_lol_data; df = get_sub_sample_lol_data(); print(df.shape, list(df.columns) if hasattr(df, 'columns') else 'check type')"
```

### 5. Run existing tests (health check)
```bash
python -m pytest tests/ -x -q --tb=short
```

---

## Step-by-Step Experiment Plan

### Step 1 — Inspect Repo

Read these files to understand the exact API before writing the experiment:

| File | Purpose |
|---|---|
| `experiments/lol_naive_ordinal_baseline.py` | **Primary template** — existing LoL ordinal baseline, copy its structure |
| `spforge/autopipeline.py` | AutoPipeline constructor signature and `lgbm_in_root` helper |
| `spforge/data_structures.py` | `ColumnNames` dataclass — required/optional fields |
| `spforge/ratings/player_performance_predictor.py` | Player rating predictor internals |
| `spforge/scorer/_score.py` | `OrdinalLossScorer`, `RankedProbabilityScorer` constructors |
| `spforge/hyperparameter_tuning/_default_search_spaces.py` | `get_default_player_rating_search_space` |
| `examples/lol/data/utils.py` | Data loader function |

### Step 2 — Inspect Data

```python
from examples.lol.data.utils import get_sub_sample_lol_data

df = get_sub_sample_lol_data()
# Expected columns: gameid, league, date, teamname, playername, result,
#   gamelength, totalgold, teamkills, teamdeaths, position,
#   damagetochampions, champion, kills, assists, deaths
# 12,432 rows, kills range typically 0–20+
# Cap kills at 10 → 11 ordinal classes
```

### Step 3 — Create Experiment Script

Create `experiments/lol_kills_ordinal.py`:

```python
import polars as pl
import pandas as pd
from examples.lol.data.utils import get_sub_sample_lol_data
from spforge.data_structures import ColumnNames
from spforge.autopipeline import AutoPipeline
from spforge.scorer._score import OrdinalLossScorer, RankedProbabilityScorer

def main():
    # Load data
    df = get_sub_sample_lol_data()
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Derive opponent team per game
    game_teams = df.groupby('gameid')['teamname'].apply(
        lambda x: list(x.unique())
    ).to_dict()

    def get_opponent(row):
        teams = game_teams[row['gameid']]
        return [t for t in teams if t != row['teamname']][0] if len(teams) == 2 else row['teamname']

    df['team_id_opponent'] = df.apply(get_opponent, axis=1)

    # Cap kills at 10 → 11 ordinal classes (0-10)
    df['kills_capped'] = df['kills'].clip(upper=10).astype(int)

    # Sort by date (required by spforge)
    df = df.sort_values('date').reset_index(drop=True)

    # Add team/player count columns
    df['team_count'] = df.groupby('gameid')['teamname'].transform('nunique')
    df['player_count'] = df.groupby(['gameid', 'teamname'])['playername'].transform('count')

    # Define column names
    column_names = ColumnNames(
        team_id='teamname',
        match_id='gameid',
        start_date='date',
        player_id='playername',
        target='kills_capped',
        team_id_opponent='team_id_opponent',
    )

    # Build pipeline — adapt constructor based on Step 1 inspection
    pipeline = AutoPipeline(
        column_names=column_names,
        estimator_type='classifier',
        n_classes=11,
    )

    # Fit and predict
    result_df = pipeline.fit_transform(df)

    # Evaluate
    prob_cols = [c for c in result_df.columns if 'probabilities' in c.lower() or 'prob' in c.lower()]
    print(f'Probability columns: {prob_cols}')
    print(f'Result shape: {result_df.shape}')

    if prob_cols:
        scorer = OrdinalLossScorer(
            target='kills_capped',
            prediction=prob_cols[0],
            n_classes=11,
        )
        score = scorer.score(result_df)
        print(f'OrdinalLossScorer: {score}')

    return result_df

if __name__ == '__main__':
    main()
```

> **IMPORTANT**: The exact `AutoPipeline` constructor may differ. After inspecting `spforge/autopipeline.py` and `experiments/lol_naive_ordinal_baseline.py` in Step 1, adapt the constructor arguments accordingly. The baseline file is the most reliable template.

### Step 4 — Run Experiment

```bash
python experiments/lol_kills_ordinal.py
```

### Step 5 — Fix Failures (if needed)

| Failure | Fix |
|---|---|
| `AutoPipeline` constructor mismatch | Mirror `experiments/lol_naive_ordinal_baseline.py` exactly, replacing target with `kills_capped` |
| `ColumnNames` missing required fields | Add synthetic columns: `df['is_home'] = 0` |
| Sorting validation error | Ensure `df.sort_values('date')` before `fit_transform` |
| LightGBM missing | `pip install lightgbm` |
| Games with ≠ 2 teams | Filter: `df = df[df['gameid'].isin(valid_game_ids)]` |
| Probability column is separate per class | Gather into list column before scoring |
| `OrdinalLossScorer` expects list column | Ensure prediction column contains 11-element probability lists per row |

### Step 6 — Re-run and Validate

```bash
python experiments/lol_kills_ordinal.py
```

**Success criteria:**
- ✅ `result_df` contains a probability column with 11-element lists per row
- ✅ `OrdinalLossScorer` returns a finite numeric score
- ✅ No import or runtime errors

---

## Data Assets Summary

| Asset | Rows | Columns | Role |
|---|---|---|---|
| `examples/lol/data/subsample_lol_data.parquet` | 12,432 | 16 | Training data |
| `rush_yards_cv.parquet` | 38,716 | 144 | Output format reference |
| `examples/ncaaf/rush_yards_cv.parquet` | 4,554 | 147 | Ordinal probability reference |
| `examples/nba/data/game_player_subsample.parquet` | 19,872 | 15 | Alternative player-level reference |

## Key spforge Symbols

| Symbol | Location | Use |
|---|---|---|
| `AutoPipeline` | `spforge/autopipeline.py` | Main pipeline entry point |
| `ColumnNames` | `spforge/data_structures.py` | Column mapping config |
| `OrdinalLossScorer` | `spforge/scorer/_score.py` | Primary evaluation metric |
| `RankedProbabilityScorer` | `spforge/scorer/_score.py` | Secondary evaluation metric |
| `get_sub_sample_lol_data` | `examples/lol/data/utils.py` | Data loader |
| `get_default_player_rating_search_space` | `spforge/hyperparameter_tuning/_default_search_spaces.py` | Hyperparameter tuning |
| `main` | `experiments/lol_naive_ordinal_baseline.py` | Existing baseline to mirror |
