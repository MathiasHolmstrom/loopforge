# Agent instructions for loopforge

## Running tests

```bash
uv run --extra test pytest tests/ -q
```

Single test:
```bash
uv run --extra test pytest tests/test_experimentation_loop.py::test_name -xvs
```

No special setup needed. Tests use `tmp_path` fixtures and mock backends.

Do NOT create temporary directories, PowerShell scripts, or custom test harnesses. Just use pytest.

## Running the CLI

```bash
uv run python src/loopforge/cli.py
```

## Installing dependencies

```bash
uv sync --extra test
```

## Critical design rule: no hardcoded domain logic

Loopforge delegates data science decisions to LLM agents. **NEVER** add hardcoded heuristics, token matching, or rule-based logic for domain decisions such as:
- Metric goal direction (minimize vs maximize)
- Feature importance or selection
- Model architecture choices
- Data quality thresholds
- Statistical significance criteria

If an agent gets a domain decision wrong, fix the **system prompt** (usually in `DATA_SCIENCE_METRICS_REASONING` in `backends.py`) so the agent reasons better. The agents are the intelligence; the code is infrastructure.

## Environment

- Python 3.11+
- `OPENAI_API_KEY` required for GPT models
- `ANTHROPIC_BEDROCK_BASE_URL` enables Claude via Bedrock proxy
