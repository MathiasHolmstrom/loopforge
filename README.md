# Loopforge

Loopforge is a provider-agnostic experimentation runtime for repeated agent-driven experiments with human checkpoints.

## What it does

- Runs a fresh worker for each iteration
- Separates raw records from accepted memory
- Supports reflection and review agents
- Lets humans interject with structured overrides
- Keeps domain execution behind adapters or handler factories

## Current package shape

- `loopforge.core`: schemas, memory, orchestrator, LiteLLM backends
- `loopforge.cli`: run the loop from a JSON spec or append a human intervention

## CLI

```bash
loopforge run --spec spec.json --memory-root .memory --executor-factory package.module:factory --worker-model openai/gpt-5.4
loopforge interject --memory-root .memory --message "Focus on slice analysis next" --effects-json '{"force_next_action": "eda"}'
```
