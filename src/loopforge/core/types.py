from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal


MetricGoal = Literal["maximize", "minimize", "unspecified"]
IterationResult = Literal["improved", "regressed", "inconclusive"]
OutcomeStatus = Literal["success", "recoverable_failure", "blocked"]
ReviewStatus = Literal["accepted", "rejected", "pending_human"]
HumanInterventionType = Literal["note", "override", "hypothesis"]
PreflightStatus = Literal["passed", "warning", "failed"]
PreflightScope = Literal["bootstrap", "execution", "setup"]
ExecutionStepKind = Literal["shell", "write_file", "append_file"]

ProgressFn = Callable[[str, str], None]
StreamFn = Callable[[str], None]


def _noop_progress(stage: str, message: str) -> None:
    pass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    goal: MetricGoal
    scorer_ref: str | None = None
    display_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    aggregation: str = "scalar"
    dataset_scope: str | None = None
    split: str | None = None
    slice_by: list[str] = field(default_factory=list)
    comparator: str = "value"
    min_effect_size: float | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    target_value: float | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricSpec":
        return cls(
            name=payload["name"],
            goal=payload["goal"],
            scorer_ref=payload.get("scorer_ref"),
            display_name=payload.get("display_name"),
            params=payload.get("params", {}),
            aggregation=payload.get("aggregation", "scalar"),
            dataset_scope=payload.get("dataset_scope"),
            split=payload.get("split"),
            slice_by=payload.get("slice_by", []),
            comparator=payload.get("comparator", "value"),
            min_effect_size=payload.get("min_effect_size"),
            constraints=payload.get("constraints", {}),
            target_value=payload.get("target_value"),
        )

    def is_improvement(self, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        threshold = self.min_effect_size or 0.0
        if self.goal == "maximize":
            return (candidate - incumbent) > threshold
        if self.goal == "minimize":
            return (incumbent - candidate) > threshold
        raise ValueError(f"Unsupported metric goal: {self.goal!r}")

    def resolve_passed(self, result: "MetricResult | None") -> bool | None:
        if result is None:
            return None
        if result.passed is not None:
            return result.passed
        if result.value is None:
            return None
        min_value = self.constraints.get("min_value")
        if min_value is not None and result.value < float(min_value):
            return False
        max_value = self.constraints.get("max_value")
        if max_value is not None and result.value > float(max_value):
            return False
        if self.target_value is not None:
            max_distance = self.constraints.get("max_distance")
            if max_distance is not None:
                return abs(result.value - self.target_value) <= float(max_distance)
        if self.constraints:
            return True
        return None


PrimaryMetric = MetricSpec


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    passed: bool | None = None
    scorer_ref: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricResult":
        return cls(
            name=payload["name"],
            value=payload.get("value"),
            passed=payload.get("passed"),
            scorer_ref=payload.get("scorer_ref"),
            details=payload.get("details", {}),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    objective: str
    primary_metric: PrimaryMetric
    allowed_actions: list[str]
    secondary_metrics: list[MetricSpec] = field(default_factory=list)
    guardrail_metrics: list[MetricSpec] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "primary_metric": self.primary_metric.to_dict(),
            "secondary_metrics": [metric.to_dict() for metric in self.secondary_metrics],
            "guardrail_metrics": [metric.to_dict() for metric in self.guardrail_metrics],
            "allowed_actions": self.allowed_actions,
            "constraints": self.constraints,
            "search_space": self.search_space,
            "stop_conditions": self.stop_conditions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            objective=payload["objective"],
            primary_metric=MetricSpec.from_dict(payload["primary_metric"]),
            secondary_metrics=[MetricSpec.from_dict(item) for item in payload.get("secondary_metrics", [])],
            guardrail_metrics=[MetricSpec.from_dict(item) for item in payload.get("guardrail_metrics", [])],
            allowed_actions=payload["allowed_actions"],
            constraints=payload.get("constraints", {}),
            search_space=payload.get("search_space", {}),
            stop_conditions=payload.get("stop_conditions", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class DataAssetSchema:
    """Quick schema snapshot of a data asset discovered during bootstrap."""
    asset_path: str
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    sample_rows_loaded: int | None = None
    total_rows_verified: int | None = None
    total_rows_estimate: int | None = None
    verification_level: str = "unknown"
    sample_values: dict[str, list[Any]] = field(default_factory=dict)
    load_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataAssetSchema":
        legacy_row_count = payload.get("row_count")
        return cls(
            asset_path=payload.get("asset_path", ""),
            columns=payload.get("columns", []),
            dtypes=payload.get("dtypes", {}),
            sample_rows_loaded=payload.get("sample_rows_loaded"),
            total_rows_verified=payload.get("total_rows_verified", legacy_row_count),
            total_rows_estimate=payload.get("total_rows_estimate"),
            verification_level=payload.get("verification_level", "unknown"),
            sample_values=payload.get("sample_values", {}),
            load_error=payload.get("load_error"),
        )


@dataclass(frozen=True)
class CapabilityContext:
    available_actions: dict[str, str] = field(default_factory=dict)
    available_entities: dict[str, list[str]] = field(default_factory=dict)
    available_data_assets: list[str] = field(default_factory=list)
    available_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    environment_facts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    data_schemas: list[DataAssetSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterSetup:
    handlers: dict[str, Any]
    capability_provider: Callable[[ExperimentSpec], CapabilityContext] | None = None
    discovery_provider: Callable[[str], CapabilityContext] | None = None
    preflight_provider: Callable[[ExperimentSpec, CapabilityContext], list["PreflightCheck"]] | None = None


@dataclass(frozen=True)
class ExecutionStep:
    kind: ExecutionStepKind
    command: str = ""
    path: str | None = None
    content: str | None = None
    cwd: str | None = None
    timeout_seconds: int | None = None
    allow_failure: bool = False
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionStep":
        return cls(
            kind=payload["kind"],
            command=payload.get("command", ""),
            path=payload.get("path"),
            content=payload.get("content"),
            cwd=payload.get("cwd"),
            timeout_seconds=payload.get("timeout_seconds"),
            allow_failure=payload.get("allow_failure", False),
            rationale=payload.get("rationale", ""),
        )


@dataclass(frozen=True)
class ExperimentCandidate:
    hypothesis: str
    action_type: str
    change_type: str
    instructions: str
    execution_steps: list[ExecutionStep] = field(default_factory=list)
    config_patch: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["execution_steps"] = [step.to_dict() for step in self.execution_steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentCandidate":
        return cls(
            hypothesis=payload["hypothesis"],
            action_type=payload["action_type"],
            change_type=payload["change_type"],
            instructions=payload["instructions"],
            execution_steps=[ExecutionStep.from_dict(item) for item in payload.get("execution_steps", [])],
            config_patch=payload.get("config_patch", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ExperimentOutcome:
    status: OutcomeStatus = "success"
    metric_results: dict[str, MetricResult] = field(default_factory=dict)
    primary_metric_value: float | None = None
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    guardrail_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_ideas: list[str] = field(default_factory=list)
    do_not_repeat: list[str] = field(default_factory=list)
    dataset_version: str | None = None
    code_or_config_changes: list[str] = field(default_factory=list)
    candidate_fingerprint: str | None = None
    failure_type: str | None = None
    failure_summary: str | None = None
    recoverable: bool = False
    recovery_actions: list[str] = field(default_factory=list)
    execution_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metric_results"] = {
            name: result.to_dict() for name, result in self.metric_results.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentOutcome":
        return cls(
            status=payload.get("status", "success"),
            metric_results={
                name: MetricResult.from_dict(result_payload)
                for name, result_payload in payload.get("metric_results", {}).items()
            },
            primary_metric_value=payload.get("primary_metric_value"),
            secondary_metrics=payload.get("secondary_metrics", {}),
            guardrail_metrics=payload.get("guardrail_metrics", {}),
            artifacts=payload.get("artifacts", []),
            notes=payload.get("notes", []),
            next_ideas=payload.get("next_ideas", []),
            do_not_repeat=payload.get("do_not_repeat", []),
            dataset_version=payload.get("dataset_version"),
            code_or_config_changes=payload.get("code_or_config_changes", []),
            candidate_fingerprint=payload.get("candidate_fingerprint"),
            failure_type=payload.get("failure_type"),
            failure_summary=payload.get("failure_summary"),
            recoverable=payload.get("recoverable", False),
            recovery_actions=payload.get("recovery_actions", []),
            execution_details=payload.get("execution_details", {}),
        )

    def resolved_metric_results(self, spec: ExperimentSpec) -> dict[str, MetricResult]:
        resolved = dict(self.metric_results)
        if spec.primary_metric.name not in resolved and self.primary_metric_value is not None:
            resolved[spec.primary_metric.name] = MetricResult(
                name=spec.primary_metric.name,
                value=self.primary_metric_value,
                scorer_ref=spec.primary_metric.scorer_ref,
            )
        secondary_specs = {metric.name: metric for metric in spec.secondary_metrics}
        for name, value in self.secondary_metrics.items():
            if name not in resolved:
                metric = secondary_specs.get(name)
                resolved[name] = MetricResult(
                    name=name,
                    value=value,
                    scorer_ref=metric.scorer_ref if metric is not None else None,
                )
        guardrail_specs = {metric.name: metric for metric in spec.guardrail_metrics}
        for name, value in self.guardrail_metrics.items():
            if name not in resolved:
                metric = guardrail_specs.get(name)
                resolved[name] = MetricResult(
                    name=name,
                    value=value,
                    scorer_ref=metric.scorer_ref if metric is not None else None,
                )
        for metric in spec.guardrail_metrics:
            result = resolved.get(metric.name)
            if result is not None and result.passed is None:
                resolved[metric.name] = replace(result, passed=metric.resolve_passed(result))
        return resolved


@dataclass(frozen=True)
class ReflectionSummary:
    assessment: str
    lessons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommended_next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReflectionSummary":
        return cls(**payload)


@dataclass(frozen=True)
class OpsConsultation:
    focus: str
    guidance: str
    commands: list[str] = field(default_factory=list)
    required_env_vars: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    should_consult: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpsConsultation":
        focus = payload.get("focus")
        guidance = payload.get("guidance")
        if not isinstance(focus, str) or not focus.strip():
            focus = str(payload.get("topic") or payload.get("problem") or "general execution help")
        if not isinstance(guidance, str) or not guidance.strip():
            guidance = str(
                payload.get("advice")
                or payload.get("recommendation")
                or payload.get("summary")
                or "No concrete guidance was returned."
            )
        return cls(
            focus=focus,
            guidance=guidance,
            commands=payload.get("commands", []),
            required_env_vars=payload.get("required_env_vars", []),
            risks=payload.get("risks", []),
            should_consult=payload.get("should_consult", True),
        )


@dataclass(frozen=True)
class AccessGuide:
    summary: str
    required_env_vars: list[str] = field(default_factory=list)
    required_permissions: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AccessGuide":
        return cls(
            summary=payload["summary"],
            required_env_vars=payload.get("required_env_vars", []),
            required_permissions=payload.get("required_permissions", []),
            commands=payload.get("commands", []),
            steps=payload.get("steps", []),
            markdown=payload.get("markdown", ""),
        )


@dataclass(frozen=True)
class RunnerAuthoringRequest:
    user_goal: str
    repo_root: str
    capability_context: CapabilityContext
    target_module_path: str
    build_symbol: str = "build_adapter"
    attempt_number: int = 1
    previous_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunnerAuthoringRequest":
        return cls(
            user_goal=payload["user_goal"],
            repo_root=payload["repo_root"],
            capability_context=CapabilityContext(**payload["capability_context"]),
            target_module_path=payload["target_module_path"],
            build_symbol=payload.get("build_symbol", "build_adapter"),
            attempt_number=payload.get("attempt_number", 1),
            previous_errors=payload.get("previous_errors", []),
        )


@dataclass(frozen=True)
class RunnerAuthoringResult:
    module_source: str
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunnerAuthoringResult":
        return cls(
            module_source=payload["module_source"],
            summary=payload.get("summary", ""),
            notes=payload.get("notes", []),
        )


@dataclass(frozen=True)
class ReviewDecision:
    status: ReviewStatus
    reason: str
    should_update_memory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewDecision":
        return cls(**payload)


@dataclass(frozen=True)
class HumanIntervention:
    author: str
    type: HumanInterventionType
    message: str
    timestamp: str
    effects: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HumanIntervention":
        return cls(**payload)


@dataclass(frozen=True)
class IterationSummary:
    iteration_id: int
    parent_iteration_id: int | None
    hypothesis: str
    action_type: str
    change_type: str
    instructions: str
    config_patch: dict[str, Any]
    primary_metric_name: str
    primary_metric_value: float | None
    secondary_metrics: dict[str, float]
    result: IterationResult
    artifacts: list[str]
    lessons: list[str]
    next_ideas: list[str]
    do_not_repeat: list[str]
    reflection_assessment: str
    review_reason: str
    metric_results: dict[str, MetricResult] = field(default_factory=dict)
    guardrail_failures: list[str] = field(default_factory=list)
    dataset_version: str | None = None
    code_or_config_changes: list[str] = field(default_factory=list)
    candidate_fingerprint: str | None = None
    outcome_status: OutcomeStatus = "success"
    failure_type: str | None = None
    failure_summary: str | None = None
    recovery_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metric_results"] = {
            name: result.to_dict() for name, result in self.metric_results.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IterationSummary":
        return cls(
            iteration_id=payload["iteration_id"],
            parent_iteration_id=payload.get("parent_iteration_id"),
            hypothesis=payload["hypothesis"],
            action_type=payload["action_type"],
            change_type=payload["change_type"],
            instructions=payload["instructions"],
            config_patch=payload.get("config_patch", {}),
            primary_metric_name=payload["primary_metric_name"],
            primary_metric_value=payload.get("primary_metric_value"),
            secondary_metrics=payload.get("secondary_metrics", {}),
            metric_results={
                name: MetricResult.from_dict(result_payload)
                for name, result_payload in payload.get("metric_results", {}).items()
            },
            guardrail_failures=payload.get("guardrail_failures", []),
            result=payload["result"],
            artifacts=payload.get("artifacts", []),
            lessons=payload.get("lessons", []),
            next_ideas=payload.get("next_ideas", []),
            do_not_repeat=payload.get("do_not_repeat", []),
            reflection_assessment=payload["reflection_assessment"],
            review_reason=payload["review_reason"],
            dataset_version=payload.get("dataset_version"),
            code_or_config_changes=payload.get("code_or_config_changes", []),
            candidate_fingerprint=payload.get("candidate_fingerprint"),
            outcome_status=payload.get("outcome_status", "success"),
            failure_type=payload.get("failure_type"),
            failure_summary=payload.get("failure_summary"),
            recovery_actions=payload.get("recovery_actions", []),
        )


@dataclass(frozen=True)
class IterationRecord:
    iteration_id: int
    parent_iteration_id: int | None
    candidate: ExperimentCandidate
    outcome: ExperimentOutcome
    reflection: ReflectionSummary
    review: ReviewDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_id": self.iteration_id,
            "parent_iteration_id": self.parent_iteration_id,
            "candidate": self.candidate.to_dict(),
            "outcome": self.outcome.to_dict(),
            "reflection": self.reflection.to_dict(),
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IterationRecord":
        return cls(
            iteration_id=payload["iteration_id"],
            parent_iteration_id=payload.get("parent_iteration_id"),
            candidate=ExperimentCandidate.from_dict(payload["candidate"]),
            outcome=ExperimentOutcome.from_dict(payload["outcome"]),
            reflection=ReflectionSummary.from_dict(payload["reflection"]),
            review=ReviewDecision.from_dict(payload["review"]),
        )


@dataclass(frozen=True)
class IterationCycleResult:
    record: IterationRecord
    accepted_summary: IterationSummary | None
    human_update: str | None = None


@dataclass(frozen=True)
class SpecQuestion:
    key: str
    prompt: str
    rationale: str = ""
    required: bool = True
    suggested_answer: str | None = None
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpecQuestion":
        return cls(
            key=payload["key"],
            prompt=payload["prompt"],
            rationale=payload.get("rationale", ""),
            required=payload.get("required", True),
            suggested_answer=payload.get("suggested_answer"),
            options=payload.get("options", []),
        )


@dataclass(frozen=True)
class ExperimentSpecProposal:
    objective: str
    recommended_spec: ExperimentSpec
    questions: list[SpecQuestion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "recommended_spec": self.recommended_spec.to_dict(),
            "questions": [question.to_dict() for question in self.questions],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpecProposal":
        return cls(
            objective=payload["objective"],
            recommended_spec=ExperimentSpec.from_dict(payload["recommended_spec"]),
            questions=[SpecQuestion.from_dict(item) for item in payload.get("questions", [])],
            notes=payload.get("notes", []),
        )


@dataclass(frozen=True)
class RoleModelConfig:
    planner: str
    worker: str
    reflection: str
    review: str
    consultation: str
    narrator: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoleModelConfig":
        return cls(
            planner=payload["planner"],
            worker=payload["worker"],
            reflection=payload["reflection"],
            review=payload["review"],
            consultation=payload.get("consultation", payload["worker"]),
            narrator=payload.get("narrator", payload.get("reflection", payload["worker"])),
        )


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: PreflightStatus
    detail: str
    required: bool = True
    scope: PreflightScope = "execution"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreflightCheck":
        return cls(
            name=payload["name"],
            status=payload["status"],
            detail=payload["detail"],
            required=payload.get("required", True),
            scope=payload.get("scope", "execution"),
        )


@dataclass(frozen=True)
class RunnerValidationResult:
    success: bool
    factory_path: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preflight_checks: list[PreflightCheck] = field(default_factory=list)
    smoke_test_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "factory_path": self.factory_path,
            "errors": self.errors,
            "warnings": self.warnings,
            "preflight_checks": [item.to_dict() for item in self.preflight_checks],
            "smoke_test_passed": self.smoke_test_passed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunnerValidationResult":
        return cls(
            success=payload["success"],
            factory_path=payload.get("factory_path"),
            errors=payload.get("errors", []),
            warnings=payload.get("warnings", []),
            preflight_checks=[PreflightCheck.from_dict(item) for item in payload.get("preflight_checks", [])],
            smoke_test_passed=payload.get("smoke_test_passed", False),
        )


@dataclass(frozen=True)
class BootstrapTurn:
    assistant_message: str
    proposal: ExperimentSpecProposal
    role_models: RoleModelConfig
    preflight_checks: list[PreflightCheck] = field(default_factory=list)
    ready_to_start: bool = False
    missing_requirements: list[str] = field(default_factory=list)
    human_update: str | None = None
    access_guide_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "assistant_message": self.assistant_message,
            "proposal": self.proposal.to_dict(),
            "role_models": self.role_models.to_dict(),
            "preflight_checks": [item.to_dict() for item in self.preflight_checks],
            "ready_to_start": self.ready_to_start,
            "missing_requirements": self.missing_requirements,
            "human_update": self.human_update,
            "access_guide_path": self.access_guide_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BootstrapTurn":
        return cls(
            assistant_message=payload["assistant_message"],
            proposal=ExperimentSpecProposal.from_dict(payload["proposal"]),
            role_models=RoleModelConfig.from_dict(payload["role_models"]),
            preflight_checks=[PreflightCheck.from_dict(item) for item in payload.get("preflight_checks", [])],
            ready_to_start=payload.get("ready_to_start", False),
            missing_requirements=payload.get("missing_requirements", []),
            human_update=payload.get("human_update"),
            access_guide_path=payload.get("access_guide_path"),
        )


@dataclass(frozen=True)
class AgentUpdate:
    stage: str
    message: str
    iteration_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentUpdate":
        return cls(
            stage=payload["stage"],
            message=payload["message"],
            iteration_id=payload.get("iteration_id"),
        )


@dataclass(frozen=True)
class MarkdownMemoryNote:
    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarkdownMemoryNote":
        return cls(
            path=payload["path"],
            content=payload["content"],
        )


@dataclass(frozen=True)
class MemorySnapshot:
    spec: ExperimentSpec
    effective_spec: ExperimentSpec
    capability_context: CapabilityContext
    best_summary: IterationSummary | None
    latest_summary: IterationSummary | None
    recent_records: list[IterationRecord]
    recent_summaries: list[IterationSummary]
    recent_human_interventions: list[HumanIntervention]
    lessons_learned: str
    markdown_memory: list[MarkdownMemoryNote]
    next_iteration_id: int


class ExperimentInterrupted(Exception):
    """Raised when the experiment loop is interrupted by Ctrl+C."""

    def __init__(
        self,
        results_so_far: list[IterationCycleResult],
        current_stage: str,
        snapshot: MemorySnapshot | None = None,
    ) -> None:
        self.results_so_far = results_so_far
        self.current_stage = current_stage
        self.snapshot = snapshot
        super().__init__(f"Experiment interrupted during {current_stage} after {len(results_so_far)} iteration(s)")


def apply_human_interventions(
    spec: ExperimentSpec,
    interventions: list[HumanIntervention],
) -> ExperimentSpec:
    effective_spec = replace(
        spec,
        allowed_actions=list(spec.allowed_actions),
        constraints=dict(spec.constraints),
        search_space=dict(spec.search_space),
        stop_conditions=dict(spec.stop_conditions),
        metadata=dict(spec.metadata),
    )
    suggested_hypotheses = list(effective_spec.metadata.get("suggested_hypotheses", []))
    for intervention in interventions:
        effects = intervention.effects
        disabled_actions = set(effects.get("disable_actions", []))
        if disabled_actions:
            effective_spec = replace(
                effective_spec,
                allowed_actions=[action for action in effective_spec.allowed_actions if action not in disabled_actions],
            )
        enabled_actions = list(effects.get("enable_actions", []))
        if enabled_actions:
            merged_actions = list(effective_spec.allowed_actions)
            for action in enabled_actions:
                if action not in merged_actions:
                    merged_actions.append(action)
            effective_spec = replace(effective_spec, allowed_actions=merged_actions)
        constraint_updates = effects.get("constraint_updates", {})
        if constraint_updates:
            merged_constraints = dict(effective_spec.constraints)
            merged_constraints.update(constraint_updates)
            effective_spec = replace(effective_spec, constraints=merged_constraints)
        search_space_updates = effects.get("search_space_updates", {})
        if search_space_updates:
            merged_search_space = dict(effective_spec.search_space)
            merged_search_space.update(search_space_updates)
            effective_spec = replace(effective_spec, search_space=merged_search_space)
        metadata_updates = effects.get("metadata_updates", {})
        if metadata_updates:
            merged_metadata = dict(effective_spec.metadata)
            merged_metadata.update(metadata_updates)
            effective_spec = replace(effective_spec, metadata=merged_metadata)
        if "force_next_action" in effects:
            merged_metadata = dict(effective_spec.metadata)
            merged_metadata["force_next_action"] = effects["force_next_action"]
            effective_spec = replace(effective_spec, metadata=merged_metadata)
        if "suggested_hypothesis" in effects:
            suggested_hypotheses.append(effects["suggested_hypothesis"])
    if suggested_hypotheses:
        merged_metadata = dict(effective_spec.metadata)
        merged_metadata["suggested_hypotheses"] = suggested_hypotheses
        effective_spec = replace(effective_spec, metadata=merged_metadata)
    return effective_spec
