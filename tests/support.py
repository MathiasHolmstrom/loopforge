from __future__ import annotations

from loopforge import (
    AccessGuide,
    AdapterSetup,
    CapabilityContext,
    ExecutionStep,
    ExperimentCandidate,
    ExperimentOutcome,
    ExperimentSpec,
    PreflightCheck,
    PrimaryMetric,
    ReflectionSummary,
    ReviewDecision,
)


class StubLoopforgeLiteLLMBackend:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_completion_tokens: int | None = None,
        stream_fn=None,
        progress_fn=None,
        extra_kwargs=None,
        **kwargs,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.stream_fn = stream_fn
        self.progress_fn = progress_fn
        self.extra_kwargs = extra_kwargs or {}


def patch_loopforge_backend_constructors(monkeypatch) -> None:
    monkeypatch.setattr(
        "loopforge.bootstrap.LiteLLMBootstrapBackend", StubLoopforgeLiteLLMBackend
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.LiteLLMRunnerAuthoringBackend",
        StubLoopforgeLiteLLMBackend,
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.LiteLLMAccessAdvisorBackend",
        StubLoopforgeLiteLLMBackend,
    )
    monkeypatch.setattr(
        "loopforge.bootstrap.LiteLLMNarrationBackend", StubLoopforgeLiteLLMBackend
    )


class FakeWorkerBackend:
    def __init__(self, candidates: list[ExperimentCandidate]) -> None:
        self._candidates = list(candidates)
        self.snapshots = []

    def propose_next_experiment(self, snapshot):
        self.snapshots.append(snapshot)
        return self._candidates.pop(0)


class StaticActionExecutor:
    def __init__(self, outcomes: list[ExperimentOutcome]) -> None:
        self._outcomes = list(outcomes)

    def execute(self, candidate: ExperimentCandidate, snapshot):
        return self._outcomes.pop(0)


class FakeReviewer:
    def __init__(
        self,
        reflections: list[ReflectionSummary],
        decisions: list[ReviewDecision],
    ) -> None:
        self._reflections = list(reflections)
        self._decisions = list(decisions)

    def review(self, snapshot, candidate, outcome):
        return self._reflections.pop(0), self._decisions.pop(0)


class FakeNarrationBackend:
    def __init__(self) -> None:
        self.bootstrap_calls = []
        self.iteration_calls = []

    def summarize_bootstrap(self, turn, capability_context):
        self.bootstrap_calls.append((turn, capability_context))
        return f"bootstrap:{turn.proposal.objective}"

    def summarize_iteration(
        self, snapshot, candidate, outcome, reflection, review, accepted_summary
    ):
        self.iteration_calls.append(
            (snapshot, candidate, outcome, reflection, review, accepted_summary)
        )
        return f"iteration:{candidate.action_type}:{review.status}"

    def fix_incomplete_metrics(
        self,
        current_spec,
        assistant_message,
        objective=None,
        capability_context=None,
    ):
        return {}


class FakeAccessAdvisorBackend:
    def __init__(self) -> None:
        self.calls = []

    def build_access_guide(self, user_goal, capability_context, preflight_checks):
        self.calls.append((user_goal, capability_context, preflight_checks))
        return AccessGuide(
            summary="Check Databricks credentials before running.",
            required_env_vars=["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
            required_permissions=["Workspace access", "Job run permission"],
            commands=["databricks auth profiles", "databricks bundle validate"],
            steps=["Export env vars", "Validate access", "Run the bundle"],
            markdown=(
                "# Access Guide\n\n## Environment Variables\n- DATABRICKS_HOST\n- DATABRICKS_TOKEN\n"
            ),
        )


class RaisingNarrationBackend:
    def summarize_bootstrap(self, turn, capability_context):
        raise RuntimeError("invalid bearer token")

    def summarize_iteration(
        self, snapshot, candidate, outcome, reflection, review, accepted_summary
    ):
        raise RuntimeError("invalid bearer token")


class RaisingAccessAdvisorBackend:
    def build_access_guide(self, user_goal, capability_context, preflight_checks):
        raise RuntimeError("invalid bearer token")


def build_spec(**overrides) -> ExperimentSpec:
    payload = {
        "objective": "Improve pass outcome validation loss.",
        "primary_metric": PrimaryMetric(name="log_loss", goal="minimize"),
        "allowed_actions": ["baseline", "eda", "train", "tune", "evaluate"],
        "constraints": {"max_runtime_minutes": 30},
        "search_space": {"learning_rate": [0.01, 0.03, 0.05]},
        "stop_conditions": {"max_iterations": 4, "patience": 2},
        "metadata": {"model_key": "pass_outcome"},
    }
    payload.update(overrides)
    return ExperimentSpec(**payload)


def build_candidate(
    *,
    hypothesis: str = "Baseline",
    action_type: str = "baseline",
    change_type: str | None = None,
    instructions: str = "Run baseline.",
    execution_steps: list[ExecutionStep] | None = None,
) -> ExperimentCandidate:
    return ExperimentCandidate(
        hypothesis=hypothesis,
        action_type=action_type,
        change_type=change_type or action_type,
        instructions=instructions,
        execution_steps=execution_steps or [],
    )


def passed_check(
    *,
    name: str = "executor_ready",
    detail: str = "Executor is configured.",
) -> list[PreflightCheck]:
    return [PreflightCheck(name=name, status="passed", detail=detail)]


def build_adapter_setup(
    *,
    handlers: dict[str, object] | None = None,
    capability_context: CapabilityContext | None = None,
    discovery_context: CapabilityContext | None = None,
    preflight_checks: list[PreflightCheck] | None = None,
) -> AdapterSetup:
    return AdapterSetup(
        handlers=handlers or {},
        capability_provider=(
            (lambda effective_spec: capability_context)
            if capability_context is not None
            else None
        ),
        discovery_provider=(
            (lambda objective: discovery_context)
            if discovery_context is not None
            else None
        ),
        preflight_provider=(
            (lambda effective_spec, discovered_context: list(preflight_checks))
            if preflight_checks is not None
            else None
        ),
    )
