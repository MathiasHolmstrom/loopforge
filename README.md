# Loopforge

Loopforge is an agent-driven experimentation runtime for ML and data workflows. It can inspect a repo, propose an experiment spec, run iterative worker and review loops, and keep durable memory across iterations.

## What It Does

- Starts from a plain-English objective
- Scans the repo and discovered metrics before execution
- Splits work across planner, worker, reflection, review, consultation, and narrator roles
- Keeps raw iteration records separate from accepted memory
- Supports human checkpoints and structured interjections
- Can synthesize an adapter scaffold when a repo does not already expose one

## Setup

Requirements:

- Python `3.11+`
- `uv`
- At least one model provider configured

Install dependencies:

```bash
uv sync --extra test
```

Run tests:

```bash
uv run --extra test pytest tests/ -q
```

Run the CLI:

```bash
uv run python src/loopforge/cli.py start
```

Provider credentials:

- OpenAI / Codex-style models: set `OPENAI_API_KEY`
- Claude direct: set `ANTHROPIC_API_KEY`
- Claude through Bedrock proxy: set `ANTHROPIC_BEDROCK_BASE_URL`
- Claude through AWS credentials: set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or `AWS_PROFILE`

## Quick Start

Examples:

```bash
uv run python src/loopforge/cli.py start --message "Improve fraud recall without breaking precision"
uv run python src/loopforge/cli.py start --message "Deploy the Databricks training flow" --executor-factory package.module:factory
uv run python src/loopforge/cli.py draft-spec --objective "Improve fraud recall without breaking precision" --memory-root .memory
uv run python src/loopforge/cli.py run --spec spec.json --memory-root .memory --executor-factory package.module:factory
uv run python src/loopforge/cli.py interject --memory-root .memory --message "Focus on slice analysis next" --effects-json '{"force_next_action": "eda"}'
```

Python API:

```python
from loopforge import Loopforge

app = Loopforge(executor_factory_path="package.module:factory")
result = app.start(user_goal="Improve fraud recall without breaking precision.")
```

If you do not pass `executor_factory_path`, Loopforge scans the repo, writes a generated adapter scaffold under `.loopforge/generated/`, and uses that scaffold for discovery and preflight.

## Role Models

Loopforge has six model roles:

- `planner`
- `worker`
- `reflection`
- `review`
- `consultation`
- `narrator`

The built-in default profile is `codex_with_claude_support`:

- `planner`: `anthropic/claude-opus-4-6-v1`
- `worker`: `openai/gpt-5.4`
- `reflection`: `openai/gpt-5.4`
- `review`: `openai/gpt-5.4`
- `consultation`: `anthropic/claude-opus-4-6-v1`
- `narrator`: `anthropic/claude-opus-4-6-v1`

There is also an `all_codex` profile that routes every role to `openai/gpt-5.4`.

## Repo-Local Model Config

You can commit a repo-local JSON file so Loopforge uses specific models for specific roles whenever it is run from that repo.

Supported file locations:

- `loopforge.models.json`
- `.loopforge/models.json`

Example:

```json
{
  "model_profile": "codex_with_claude_support",
  "roles": {
    "planner": "anthropic/claude-opus-4-6-v1",
    "worker": "openai/gpt-5.4",
    "reflection": "openai/gpt-5.4-mini",
    "review": "openai/gpt-5.4-mini",
    "consultation": "anthropic/claude-opus-4-6-v1",
    "narrator": "anthropic/claude-sonnet-4-5"
  }
}
```

You can also use top-level keys such as `planner_model`, `worker_model`, `reflection_model`, and `narrator_model` if you prefer that shape.

Precedence is:

1. CLI or constructor arguments
2. Repo-local JSON config
3. Built-in profile defaults

## Provider Fallback Rules

Loopforge resolves provider availability like this:

- If only OpenAI credentials are available, all roles are routed to OpenAI or Codex models
- If only Claude credentials are available, all roles are routed to Claude models
- If both are available, the configured per-role mix is respected
- If neither is available, Loopforge keeps the configured model ids and warns before model calls fail

If the repo config asks for a provider that is unavailable, Loopforge emits clear warnings and logs the replacement model it selected.

## Bootstrap Flow

If the bootstrap agent still needs clarification, `result["status"]` will be `needs_input` and the payload will contain:

- the assistant's next message
- a recommended experiment spec
- required clarifying questions
- preflight checks for memory, data access, and permissions
- an `ops_access_guide.md` under the memory root when access, auth, env vars, or permissions need direct operator steps

Once the required questions are answered and execution-scope preflight checks pass, `start()` initializes memory and runs the loop.

Autonomous execution is blocked unless execution-level preflight checks pass. Discovery alone is not enough. If an adapter can identify likely datasets, jobs, or permissions but cannot verify access through `preflight_provider`, Loopforge keeps `status="needs_input"` and reports a blocking `autonomous_execution_permissions` failure.

## Package Layout

- `loopforge.core`: schemas, memory, orchestrator, LiteLLM backends
- `loopforge.bootstrap`: bootstrap flow, model routing, repo discovery, and preflight checks
- `loopforge.cli`: `start`, `draft-spec`, `run`, and `interject`

## Auto Adapter

When no adapter factory is provided, Loopforge:

- scans Python files for candidate metric and action symbols
- writes a reusable adapter scaffold and discovery summary under the memory root
- exposes discovered metrics and actions back to the bootstrap agent
- fails preflight until placeholder action handlers are replaced with real repo bindings

That means the system can build reusable integration components during discovery, but it still refuses to pretend execution is safe until the generated adapter has been grounded in real code.

## Metric Discovery

Adapters can expose repo-specific scorer metadata through `AdapterSetup.discovery_provider(objective)` or, as a fallback, `AdapterSetup.capability_provider(spec)`. They can also expose `AdapterSetup.preflight_provider(spec, capability_context)` to verify that the runtime can actually load the required data and has the needed permissions before execution starts.

The planning backend uses `available_metrics` from that capability context to recommend:

- one concrete primary metric
- supporting secondary metrics
- guardrail metrics with explicit constraints
- follow-up questions when the user needs to confirm a calculation variant or custom scorer function
