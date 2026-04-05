# Loopforge

Loopforge is an AI-powered experimentation loop for ML projects. You describe what you want to improve in plain English, and it iteratively plans, runs, and reviews experiments — learning from each attempt.

## How it works

1. **You describe a goal** — e.g. "improve fraud recall without breaking precision"
2. **Loopforge scans your repo** — finds your scripts, metrics, and data
3. **A planner proposes an experiment** — you review and approve (or redirect)
4. **A worker runs iterations** — each one trains/adjusts a model and reports metrics
5. **A reviewer checks the results** — decides whether to accept, reject, or try something different
6. **Repeat** — the loop continues until the stopping condition is hit or you interrupt

You can steer the loop at any point with feedback, and Loopforge remembers what worked across iterations.

## Setup

**Requirements:** Python 3.11+, at least one model provider key.

```bash
# Install
uv sync --extra test          # or: pip install -e .[test]

# Run
loopforge start               # interactive mode — asks what you want to solve
loopforge start --message "Improve fraud recall without breaking precision"
```

**Model provider credentials** (set at least one):

| Provider | Environment variable |
|----------|---------------------|
| OpenAI / Codex | `OPENAI_API_KEY` |
| Claude (direct) | `ANTHROPIC_API_KEY` |
| Claude (Bedrock proxy) | `ANTHROPIC_BEDROCK_BASE_URL` |
| Claude (AWS) | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, or `AWS_PROFILE` |

If only one provider is configured, all roles automatically use that provider. If both are available, Loopforge uses its default role mix (see [Model config](#model-config) below).

## Configuring iterations and run time

By default, Loopforge runs up to **30 iterations** with a **6-hour** time limit — whichever comes first.

### From the command line

```bash
loopforge start --iterations 10 --max-autonomous-hours 2
```

### In a repo-local settings file

Create `loopforge.settings.json` (or `.loopforge/settings.json`) in your repo root:

```json
{
  "stop_conditions": {
    "max_iterations": 10,
    "max_autonomous_hours": 2
  }
}
```

This applies every time Loopforge runs from that repo, without needing CLI flags. CLI flags override these if both are set.

### From Python

```python
from loopforge import Loopforge

app = Loopforge()
result = app.start(
    user_goal="Improve fraud recall without breaking precision",
    iterations=10,
    max_autonomous_hours=2,
)
```

## Model config

Loopforge uses six internal roles: planner, worker, reflection, review, consultation, and narrator. You don't need to think about these unless you want to override which model handles which role.

Create `loopforge.models.json` (or `.loopforge/models.json`) in your repo root:

```json
{
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

You can also override individual roles from the CLI:

```bash
loopforge start --worker-model openai/gpt-5.4-mini --planner-model anthropic/claude-opus-4-6-v1
```

## Interrupting and steering

Press **Ctrl+C** at any time during a run. Loopforge pauses and asks for new instructions — you can redirect the experiment, restart with a new goal, or quit.

During planning, type feedback instead of confirming to revise the plan. You can also ask questions (e.g. "why did you pick that metric?") and Loopforge will answer without restarting the plan.

## CLI reference

```bash
loopforge start       # Interactive planning and execution
loopforge run         # Run from a saved experiment spec (JSON)
loopforge draft-spec  # Generate a spec without executing
loopforge interject   # Inject a human message into a running experiment's memory
```

Run `loopforge <command> --help` for full options.
