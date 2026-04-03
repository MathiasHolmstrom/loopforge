# LoL Player Kills Prediction — Access & Permissions Playbook

## Summary

Build a LoL player kills prediction model (kills capped at 10 → 11 ordinal classes) using spforge's player rating system. The repo `player-performance-ratings` provides the pipeline framework. Execution is on Windows via `cmd.exe` using a local `.venv` Python environment through the generic autonomous executor.

---

## Prerequisites

### Required Environment Variables

| Variable | Value | Reason |
|---|---|---|
| `PYTHONPATH` | `C:\Users\m.holmstrom\PycharmProjects\player-performance-ratings` | Ensures spforge and examples are importable |
| `VIRTUAL_ENV` | `C:\Users\m.holmstrom\PycharmProjects\loopforge\.venv` | Points to the active virtual environment |

### Required Permissions

| Resource | Permission | Reason |
|---|---|---|
| Repo root | Read/Write | Source code, data files, experiment outputs |
| `.loopforge/` | Read/Write | Execution state memory (preflight confirmed) |
| `examples/lol/data/subsample_lol_data.parquet` | Read | Primary LoL dataset (12,432 rows, 16 cols) |

### Required Python Packages

- `spforge` (installed from repo in editable mode)
- `polars`
- `lightgbm`
- `scikit-learn`
- `narwhals`
- `numpy`
- `pandas` (optional, for compatibility)

---

## Step-by-Step Playbook

### Step 1 — Activate Environment & Verify Imports

```cmd
C:\Users\m.holmstrom\PycharmProjects\loopforge\.venv\Scripts\activate.bat

cd C:\Users\m.holmstrom\PycharmProjects\player-performance-ratings

python -c "import spforge; print(spforge.__file__)"
```

If spforge is not importable:

```cmd
pip install -e C:\Users\m.holmstrom\PycharmProjects\player-performance-ratings
```

Verify critical dependencies:

```cmd
python -c "import polars; import lightgbm; import sklearn; import narwhals; print('OK')"
```

### Step 2 — Inspect LoL Data

```cmd
python -c "import polars as pl; df = pl.read_parquet('examples/lol/data/subsample_lol_data.parquet'); print(df.shape); print(df.columns); print(df['kills'].describe())"
```

**Expected:** 12,432 rows × 16 columns. Key columns: `gameid`, `teamname`, `playername`, `date`, `position`, `kills` (int64), `deaths`, `assists`, `damagetochampions`, `result`.

The canonical loader is `get_sub_sample_lol_data()` in `examples/lol/data/utils.py`.

### Step 3 — Inspect Reference Patterns

Read these files for pipeline patterns before writing the experiment:

| File | What to learn |
|---|---|
| `experiments/lol_kills_rps_experiment.py` | Existing LoL kills scoring with `_ranked_probability_score`, `_ordinal_loss`, `counts_to_probs` |
| `experiments/lol_kills_position_history_baseline.py` | Baseline prediction flow: `build_predictions`, `normalize_probs`, `expected_value`, `main` |
| `examples/lol/pipeline_transformer_example.py` | How `player_count`, `playername`, `team_count`, `teamname` integrate with spforge pipelines |
| `spforge/ratings/player_performance_predictor.py` | Player rating prediction internals |
| `spforge/hyperparameter_tuning/_default_search_spaces.py` | `get_default_player_rating_search_space`, `get_full_player_rating_search_space` |
| `tests/ratings/test_player_rating_generator.py` | Player rating generator construction patterns, `base_cn` fixture |
| `spforge/scorer/_score.py` | `OrdinalLossScorer`, `RankedProbabilityScorer` — both accept list-column probability predictions |

### Step 4 — Create the Experiment Script

Create `experiments/lol_kills_prediction.py` with this structure:

```python
import polars as pl
from examples.lol.data.utils import get_sub_sample_lol_data
from spforge.data_structures import ColumnNames
# Import pipeline components from spforge as discovered in Step 3

def main():
    # 1. Load data
    df = get_sub_sample_lol_data()

    # 2. Cap kills at 10 → 11 ordinal classes (0-10)
    df = df.with_columns(
        pl.col("kills").clip(0, 10).alias("kills_capped")
    )

    # 3. Convert date string to datetime if needed
    df = df.with_columns(
        pl.col("date").str.to_datetime().alias("date")
    )

    # 4. Define column names
    column_names = ColumnNames(
        team_id="teamname",
        match_id="gameid",
        start_date="date",
        player_id="playername",
        target="kills_capped",
        # ... additional fields as required by the pipeline
    )

    # 5. Build player rating generator + feature generators
    #    (rolling kills, deaths, assists, damagetochampions)
    # 6. Construct pipeline (AutoPipeline or manual)
    # 7. Fit and predict → output 11-class probabilities per row
    # 8. Score with OrdinalLossScorer or RankedProbabilityScorer

if __name__ == "__main__":
    main()
```

**Key design decisions:**
- Target: `kills_capped` (int, 0–10)
- Output: list column of 11 floats (probabilities) per row
- Scorer: `RankedProbabilityScorer` or `OrdinalLossScorer` from `spforge.scorer._score`
- Use `compare_to_naive=True` on scorer to benchmark against uniform baseline

### Step 5 — Smoke Test the Repo

```cmd
python -m pytest tests/ -x -q --tb=short
```

All tests must pass before proceeding.

### Step 6 — Run the Experiment

```cmd
python experiments/lol_kills_prediction.py
```

**Common failure modes and fixes:**

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: spforge` | Run `pip install -e .` from repo root |
| Missing `team_count` / `player_count` columns | Add them: `df = df.with_columns(pl.lit(1).alias("team_count"))` or compute from data |
| `date` column not datetime | Convert: `pl.col("date").str.to_datetime()` |
| Scorer expects list column but gets scalar | Ensure prediction output is a list of 11 probabilities per row |
| LightGBM not found | `pip install lightgbm` |

### Step 7 — Validate Outputs

```python
# In validation script or at end of experiment:
# 1. Check probability column
assert all(len(row) == 11 for row in predictions)
assert all(abs(sum(row) - 1.0) < 1e-6 for row in predictions)

# 2. Check scorer returns finite value
from spforge.scorer._score import RankedProbabilityScorer
scorer = RankedProbabilityScorer(
    target="kills_capped",
    prediction="kills_probabilities",
    compare_to_naive=True
)
score = scorer.score(result_df)
assert score is not None and not math.isnan(score)
```

---

## Data Assets Summary

| Asset | Rows | Columns | Role |
|---|---|---|---|
| `examples/lol/data/subsample_lol_data.parquet` | 12,432 | 16 | Primary training/validation data |
| `examples/nba/data/game_player_subsample.parquet` | 19,872 | 15 | Reference for player-level pipeline patterns |
| `examples/ncaaf/rush_yards_cv.parquet` | 4,554 | 147 | Reference for probability output schema |

## No External Credentials Required

This experiment uses only local data files bundled in the repo. No API keys, database connections, or cloud credentials are needed.
