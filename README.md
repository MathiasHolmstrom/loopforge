# Loopforge

Loopforge is a provider-agnostic experimentation runtime for repeated agent-driven experiments with human checkpoints.

## What it does

- Runs a fresh worker for each iteration
- Separates raw records from accepted memory
- Supports reflection and review agents
- Uses Codex-style primary roles by default, with Claude-style support for ops guidance and human-facing narration
- Lets humans interject with structured overrides
- Keeps domain execution behind adapters or handler factories
- Can draft an initial experiment spec by proposing primary, secondary, and guardrail metrics from discovered scorer metadata
- Can start from a plain-language goal, ask clarifying questions, verify data access, and then launch with default OpenAI-backed roles
- Can synthesize a reusable adapter scaffold from repo inspection when no adapter factory is supplied

## Current package shape

- `loopforge.core`: schemas, memory, orchestrator, LiteLLM backends
- `loopforge.bootstrap`: zero-config bootstrap API with default OpenAI role assignment and adapter preflight checks
- `loopforge.bootstrap`: zero-config bootstrap API with Codex-primary defaults, Claude ops consultation, and generated access guides
- `loopforge.cli`: start from a plain-language goal, draft a spec, run the loop from a JSON spec, or append a human intervention

## CLI

```bash
loopforge start --message "Improve fraud recall without breaking precision"
loopforge start --message "Improve fraud recall without breaking precision" --executor-factory package.module:factory
loopforge start --message "Deploy the Databricks training flow" --model-profile codex_with_claude_support
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
- a Claude-written `ops_access_guide.md` under the memory root when access, auth, env vars, or permissions need direct operator steps

Once the required questions are answered and preflight checks pass, `start()` initializes memory and runs the loop.

## Default Role Behavior

The default profile is `codex_with_claude_support`:

- `planner`, `worker`, `reflection`, `review`: `openai/gpt-5.4`
- `consultation`: `anthropic/claude-sonnet-4-5`
- `narrator`: `anthropic/claude-sonnet-4-5`

That means Codex-style models own spec formation, experiment proposals, reflection, and review. Claude-style models are reserved for:

- external-service and CLI consultation
- permissions and environment-variable guidance
- direct human-readable updates and summaries

You can override the profile or any individual role from the CLI with `--model-profile`, `--consultation-model`, and `--narrator-model`.

Autonomous execution is blocked unless execution-level preflight checks pass. Discovery alone is not enough. If an adapter can identify likely datasets, jobs, or permissions but cannot verify access through `preflight_provider`, Loopforge keeps `status="needs_input"` and reports a blocking `autonomous_execution_permissions` failure.

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

`PreflightCheck` entries can be scoped to bootstrap or execution. In practice, `start()` only proceeds when required execution-scope checks pass, so unattended loops do not begin in environments that still depend on human approval or missing credentials.
