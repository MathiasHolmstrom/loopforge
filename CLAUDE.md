# Loopforge development guide

## Running tests

```bash
uv run --extra test pytest tests/ -q
```

That's it. All tests use `tmp_path` fixtures and mock backends — no special setup, no temp directories, no PowerShell scripts needed.

To run a single test:
```bash
uv run --extra test pytest tests/test_experimentation_loop.py::test_name -xvs
```

## Project structure

- `src/loopforge/` — main package
- `src/loopforge/cli.py` — CLI entrypoint
- `src/loopforge/bootstrap.py` — bootstrap/planning logic, `Loopforge` class
- `src/loopforge/core/backends.py` — LLM backend implementations (worker, planner, reflection, review, narrator)
- `src/loopforge/core/types.py` — data types (ExperimentSpec, CapabilityContext, etc.)
- `src/loopforge/core/memory.py` — FileMemoryStore
- `src/loopforge/core/orchestrator.py` — ExperimentOrchestrator (iteration loop)
- `src/loopforge/auto_adapter.py` — repo scanning (AST-based)
- `src/loopforge/pilot_adapters.py` — built-in adapters (NBA points)
- `tests/` — pytest tests
- `experiments/` — example experiment scripts

## Running the CLI

```bash
uv run python src/loopforge/cli.py
```

## Dependencies

```bash
uv sync --extra test
```

## Design principles

- **Never hardcode domain logic — let the agent reason.** Loopforge delegates decisions to LLM agents. Do NOT add hardcoded heuristics for things like metric goal direction, feature importance, model selection, data quality rules, etc. If the agent gets something wrong (e.g. "maximize" for a loss metric), fix it by improving the system prompt guidance so the agent reasons correctly — not by adding token-matching code. The agents are the brains; the code is the plumbing.
- **Prompts over code for domain decisions.** When the LLM makes a wrong domain call, the fix goes in `DATA_SCIENCE_METRICS_REASONING` or the relevant system prompt in `backends.py`, not in `_normalise_*` methods.

## Environment

- Python 3.11+
- Windows (bash via Git Bash)
- `OPENAI_API_KEY` required for GPT models
- `ANTHROPIC_BEDROCK_BASE_URL` enables Claude via Bedrock proxy
