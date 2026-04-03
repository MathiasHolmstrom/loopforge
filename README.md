# Loopforge

Loopforge is a provider-agnostic experimentation runtime for repeated agent-driven experiments with human checkpoints.

## What it does

- Runs a fresh worker for each iteration
- Separates raw records from accepted memory
- Supports reflection and review agents
- Lets humans interject with structured overrides
- Keeps domain execution behind adapters or handler factories
- Can draft an initial experiment spec by proposing primary, secondary, and guardrail metrics from discovered scorer metadata
- Can start from a plain-language goal, ask clarifying questions, verify data access, and then launch with default OpenAI-backed roles
- Can synthesize a reusable adapter scaffold from repo inspection when no adapter factory is supplied

## Current package shape

- `loopforge.core`: schemas, memory, orchestrator, LiteLLM backends
- `loopforge.bootstrap`: zero-config bootstrap API with default OpenAI role assignment and adapter preflight checks
- `loopforge.cli`: start from a plain-language goal, draft a spec, run the loop from a JSON spec, or append a human intervention

## CLI

```bash
loopforge start --message "Improve fraud recall without breaking precision"
loopforge start --message "Improve fraud recall without breaking precision" --executor-factory package.module:factory
loopforge draft-spec --objective "Improve fraud recall without breaking precision" --memory-root .memory --executor-factory package.module:factory --planner-model openai/gpt-5.4
loopforge run --spec spec.json --memory-root .memory --executor-factory package.module:factory --worker-model openai/gpt-5.4
loopforge interject --memory-root .memory --message "Focus on slice analysis next" --effects-json '{"force_next_action": "eda"}'
```

## Package Bootstrap

```python
from loopforge import Loopforge

app = Loopforge(executor_factory_path="package.module:factory")
result = app.start(user_goal="Improve fraud recall without breaking precision.")
```

If you do not pass `executor_factory_path`, Loopforge scans the repo, writes a generated adapter scaffold under `.loopforge/generated/`, and uses that scaffold for discovery and preflight.

If the bootstrap agent still needs clarification, `result["status"]` will be `needs_input` and the payload will contain:

- the assistant's next message
- a recommended experiment spec
- required clarifying questions
- preflight checks for memory, data access, and permissions

Once the required questions are answered and preflight checks pass, `start()` initializes memory and runs the loop.

## Auto Adapter

When no adapter factory is provided, Loopforge now:

- scans Python files for candidate metric and action symbols
- writes a reusable adapter scaffold and discovery summary under the memory root
- exposes discovered metrics and actions back to the bootstrap agent
- fails preflight until placeholder action handlers are replaced with real repo bindings

That means the system can build reusable integration components during discovery, but it still refuses to pretend execution is safe until the generated adapter has been grounded in real code.

## Metric Discovery

Adapters can expose repo-specific scorer metadata through `AdapterSetup.discovery_provider(objective)` or, as a fallback, `AdapterSetup.capability_provider(spec)`. They can also expose `AdapterSetup.preflight_provider(spec, capability_context)` to verify that the runtime can actually load the required data and has the needed permissions before execution starts. The planning backend uses `available_metrics` from that capability context to recommend:

- one concrete primary metric
- supporting secondary metrics
- guardrail metrics with explicit constraints
- follow-up questions when the user needs to confirm a calculation variant or custom scorer function
