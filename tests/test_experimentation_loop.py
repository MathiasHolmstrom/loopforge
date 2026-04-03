from __future__ import annotations

import json
from pathlib import Path

from loopforge import (
    AdapterSetup,
    BootstrapTurn,
    CapabilityContext,
    ConsultingWorkerBackend,
    ExperimentCandidate,
    ExperimentOrchestrator,
    ExperimentOutcome,
    ExperimentSpec,
    ExperimentSpecProposal,
    FileMemoryStore,
    Loopforge,
    MetricResult,
    MetricSpec,
    AccessGuide,
    OpsConsultation,
    PreflightCheck,
    PrimaryMetric,
    ReflectionSummary,
    RoleModelConfig,
    ReviewDecision,
    RoutingExperimentExecutor,
    SpecQuestion,
)
from loopforge.auto_adapter import synthesize_auto_adapter
from loopforge.bootstrap import load_factory
from loopforge.cli import append_human_intervention, discover_capabilities_for_objective, draft_spec, run_from_spec


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


class FakeReflectionBackend:
    def __init__(self, reflections: list[ReflectionSummary]) -> None:
        self._reflections = list(reflections)

    def reflect(self, snapshot, candidate, outcome):
        return self._reflections.pop(0)


class FakeReviewBackend:
    def __init__(self, decisions: list[ReviewDecision]) -> None:
        self._decisions = list(decisions)

    def review(self, snapshot, candidate, outcome, reflection):
        return self._decisions.pop(0)


class FakeNarrationBackend:
    def __init__(self) -> None:
        self.bootstrap_calls = []
        self.iteration_calls = []

    def summarize_bootstrap(self, turn, capability_context):
        self.bootstrap_calls.append((turn, capability_context))
        return f"bootstrap:{turn.proposal.objective}"

    def summarize_iteration(self, snapshot, candidate, outcome, reflection, review, accepted_summary):
        self.iteration_calls.append((snapshot, candidate, outcome, reflection, review, accepted_summary))
        return f"iteration:{candidate.action_type}:{review.status}"


class FakeConsultationBackend:
    def __init__(self, focus: str = "databricks deploy") -> None:
        self.focus = focus
        self.calls = []

    def consult(self, snapshot):
        self.calls.append(snapshot)
        return OpsConsultation(
            focus=self.focus,
            guidance="Use bundle deploy and verify env vars first.",
            commands=["databricks bundle validate", "databricks bundle deploy -t dev"],
            required_env_vars=["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
            risks=["Missing workspace permissions"],
            should_consult=True,
        )


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
                "# Access Guide\n\n"
                "## Environment Variables\n"
                "- DATABRICKS_HOST\n"
                "- DATABRICKS_TOKEN\n"
            ),
        )


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


def test_only_accepted_memory_reaches_the_next_worker_and_human_notes_modify_effective_spec(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            ),
            ExperimentCandidate(
                hypothesis="Do EDA next",
                action_type="eda",
                change_type="slice",
                instructions="Slice by rest days.",
            ),
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={
                "baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["Baseline done."])]),
                "eda": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41, notes=["Rest-day slice found drift."])]),
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [
                ReflectionSummary(assessment="Useful baseline.", lessons=["Baseline established."]),
                ReflectionSummary(assessment="Rest-day slice is worth following up.", lessons=["Rest days matter."]),
            ]
        ),
        review_backend=FakeReviewBackend(
            [
                ReviewDecision(status="accepted", reason="Use this as starting memory."),
                ReviewDecision(status="accepted", reason="EDA finding is actionable."),
            ]
        ),
    )
    orchestrator.initialize(spec=build_spec())

    first_cycle = orchestrator.run_iteration()
    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Focus on EDA before more tuning.",
        effects={"disable_actions": ["tune"], "force_next_action": "eda", "suggested_hypothesis": "Check rest-day drift"},
    )
    second_cycle = orchestrator.run_iteration()

    assert first_cycle.accepted_summary is not None
    assert second_cycle.accepted_summary is not None
    assert worker.snapshots[0].recent_summaries == []
    assert [summary.hypothesis for summary in worker.snapshots[1].recent_summaries] == ["Baseline"]
    assert worker.snapshots[1].effective_spec.allowed_actions == ["baseline", "eda", "train", "evaluate"]
    assert worker.snapshots[1].effective_spec.metadata["force_next_action"] == "eda"
    assert worker.snapshots[1].effective_spec.metadata["suggested_hypotheses"] == ["Check rest-day drift"]


def test_orchestrator_injects_capability_context_into_worker_snapshot(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41)])}
        ),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
        capability_provider=lambda effective_spec: CapabilityContext(
            available_actions={"baseline": "Run baseline"},
            available_entities={"model_keys": ["pass_outcome", "fumble_expanded"]},
            available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            available_metrics={
                "log_loss": {"scorer_ref": "metrics.log_loss_cv"},
                "recall_at_p90": {"scorer_ref": "metrics.recall_at_p90"},
            },
            environment_facts={"adapter": "test"},
        ),
    )
    orchestrator.initialize(spec=build_spec())

    orchestrator.run_iteration()

    assert worker.snapshots[0].capability_context.available_entities["model_keys"] == [
        "pass_outcome",
        "fumble_expanded",
    ]
    assert worker.snapshots[0].capability_context.available_data_assets == ["gold_cross_validated_pass_outcome_ncaaf"]
    assert worker.snapshots[0].capability_context.available_metrics["log_loss"]["scorer_ref"] == "metrics.log_loss_cv"


def test_consulting_worker_requests_ops_help_for_databricks_flows(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Deploy the updated Databricks bundle.",
                action_type="baseline",
                change_type="deployment",
                instructions="Deploy to Databricks dev.",
            )
        ]
    )
    consultation = FakeConsultationBackend()
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=ConsultingWorkerBackend(worker_backend=worker, consultation_backend=consultation),
        executor=RoutingExperimentExecutor(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.41)])}
        ),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
        capability_provider=lambda effective_spec: CapabilityContext(notes=["Need Databricks deploy guidance"]),
    )
    orchestrator.initialize(
        build_spec(
            objective="Deploy the Databricks training job and verify data access.",
            allowed_actions=["baseline"],
        )
    )

    cycle = orchestrator.run_iteration()

    assert consultation.calls
    assert "Ops consult guidance: Use bundle deploy and verify env vars first." in worker.snapshots[0].capability_context.notes
    assert cycle.record.candidate.metadata["ops_consultation"]["required_env_vars"] == [
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
    ]


def test_rejected_review_stays_in_raw_records_but_not_in_accepted_memory(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=FakeWorkerBackend(
            [
                ExperimentCandidate(
                    hypothesis="Suspicious train run",
                    action_type="train",
                    change_type="feature_set",
                    instructions="Try a risky feature.",
                )
            ]
        ),
        executor=RoutingExperimentExecutor(
            handlers={"train": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.3)])}
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Looks leaky.", lessons=["Need leakage check."])]
        ),
        review_backend=FakeReviewBackend(
            [ReviewDecision(status="rejected", reason="Do not trust this run.")]
        ),
    )
    orchestrator.initialize(spec=build_spec(allowed_actions=["train"]))

    cycle = orchestrator.run_iteration()
    snapshot = store.load_snapshot()
    records_text = (tmp_path / "loop" / "iteration_records.jsonl").read_text(encoding="utf-8")

    assert cycle.accepted_summary is None
    assert snapshot.latest_summary is None
    assert '"status": "rejected"' in records_text


def test_cli_interjection_appends_human_note_and_changes_future_effective_spec(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    store.initialize(build_spec())

    append_human_intervention(
        memory_root=tmp_path / "loop",
        message="Only do EDA next.",
        effects={"disable_actions": ["train", "tune"], "force_next_action": "eda"},
        author="mhol",
        type_="override",
    )
    snapshot = store.load_snapshot()

    assert snapshot.recent_human_interventions[0].author == "mhol"
    assert snapshot.effective_spec.allowed_actions == ["baseline", "eda", "evaluate"]
    assert snapshot.effective_spec.metadata["force_next_action"] == "eda"


def test_run_from_spec_supports_factory_and_returns_reviewed_cycle_results(tmp_path, monkeypatch) -> None:
    spec = build_spec(allowed_actions=["baseline"], stop_conditions={"max_iterations": 1})
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    class StubWorker:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run baseline.",
            )

    class StubReflection:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def reflect(self, snapshot, candidate, outcome):
            return ReflectionSummary(assessment="Baseline is fine.", lessons=["Keep this baseline."])

    class StubReview:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def review(self, snapshot, candidate, outcome, reflection):
            return ReviewDecision(status="accepted", reason="Approved into memory.")

    class StubNarration:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def summarize_bootstrap(self, turn, capability_context):
            return "bootstrap update"

        def summarize_iteration(self, snapshot, candidate, outcome, reflection, review, accepted_summary):
            return "iteration update"

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.39)])},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_entities={"model_keys": ["pass_outcome"]},
                available_data_assets=["gold_cross_validated_pass_outcome_ncaaf"],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.LiteLLMWorkerBackend", StubWorker)
    monkeypatch.setattr("loopforge.bootstrap.LiteLLMReflectionBackend", StubReflection)
    monkeypatch.setattr("loopforge.bootstrap.LiteLLMReviewBackend", StubReview)
    monkeypatch.setattr("loopforge.bootstrap.LiteLLMNarrationBackend", StubNarration)
    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)
    monkeypatch.setattr("loopforge.cli.load_factory", lambda _: fake_factory)

    results = run_from_spec(
        spec_path=spec_path,
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        worker_model="openai/gpt-5.4",
    )

    assert results[0]["accepted_summary"]["hypothesis"] == "Baseline"
    assert results[0]["record"]["review"]["status"] == "accepted"
    assert results[0]["human_update"] == "iteration update"


def test_discover_capabilities_for_objective_prefers_discovery_provider(tmp_path, monkeypatch) -> None:
    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_metrics={"fallback_metric": {"scorer_ref": "metrics.fallback"}}
            ),
            discovery_provider=lambda objective: CapabilityContext(
                available_metrics={"precision_floor": {"scorer_ref": "metrics.precision_floor"}},
                notes=[f"objective={objective}"],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    context = discover_capabilities_for_objective(
        objective="Keep precision high.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
    )

    assert context.available_metrics["precision_floor"]["scorer_ref"] == "metrics.precision_floor"
    assert context.notes == ["objective=Keep precision high."]


def test_draft_spec_returns_metric_questions_and_recommended_spec(tmp_path, monkeypatch) -> None:
    class StubSpecBackend:
        def __init__(self, model: str, completion_fn=None, temperature: float = 0.2, extra_kwargs=None) -> None:
            self.model = model

        def propose_spec(self, objective, capability_context, user_preferences=None):
            assert capability_context.available_metrics["recall_at_p90"]["scorer_ref"] == "metrics.recall_at_p90"
            assert user_preferences == {"custom_primary_scorer": "metrics.recall_at_p90"}
            return ExperimentSpecProposal(
                objective=objective,
                recommended_spec=ExperimentSpec(
                    objective=objective,
                    primary_metric=PrimaryMetric(
                        name="recall_at_p90",
                        goal="maximize",
                        scorer_ref="metrics.recall_at_p90",
                    ),
                    secondary_metrics=[
                        MetricSpec(
                            name="log_loss",
                            goal="minimize",
                            scorer_ref="metrics.log_loss_cv",
                        )
                    ],
                    guardrail_metrics=[
                        MetricSpec(
                            name="precision_floor",
                            goal="maximize",
                            scorer_ref="metrics.precision_floor",
                            constraints={"min_value": 0.9},
                        )
                    ],
                    allowed_actions=["baseline", "train"],
                    metadata={"planner": "stub"},
                ),
                questions=[
                    SpecQuestion(
                        key="primary_metric_confirmation",
                        prompt="Should recall_at_p90 remain the primary metric?",
                        suggested_answer="yes",
                        options=["yes", "no"],
                    )
                ],
                notes=["Discovered scorer metrics.recall_at_p90 from the host codebase."],
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "Run baseline", "train": "Train model"},
                available_metrics={
                    "recall_at_p90": {"scorer_ref": "metrics.recall_at_p90"},
                    "log_loss": {"scorer_ref": "metrics.log_loss_cv"},
                    "precision_floor": {"scorer_ref": "metrics.precision_floor"},
                },
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)
    monkeypatch.setattr("loopforge.cli.LiteLLMSpecBackend", StubSpecBackend)

    proposal = draft_spec(
        objective="Improve fraud recall without breaking precision.",
        memory_root=tmp_path / "memory",
        executor_factory_path="fake.module:factory",
        planner_model="openai/gpt-5.4",
        preferences={"custom_primary_scorer": "metrics.recall_at_p90"},
    )

    assert proposal.recommended_spec.primary_metric.name == "recall_at_p90"
    assert proposal.recommended_spec.guardrail_metrics[0].name == "precision_floor"
    assert proposal.questions[0].key == "primary_metric_confirmation"
    assert proposal.notes == ["Discovered scorer metrics.recall_at_p90 from the host codebase."]


def test_loopforge_bootstrap_returns_questions_and_preflight_failures(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert capability_context.available_metrics["fraud_recall"]["scorer_ref"] == "metrics.fraud_recall"
            assert role_models.worker == "openai/gpt-5.4"
            return BootstrapTurn(
                assistant_message="I need one more confirmation before starting.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="fraud_recall",
                            goal="maximize",
                            scorer_ref="metrics.fraud_recall",
                        ),
                        allowed_actions=["train"],
                    ),
                    questions=[
                        SpecQuestion(
                            key="positive_label_definition",
                            prompt="Which label definition should the scorer use?",
                        )
                    ],
                    notes=["Bootstrap discovered a candidate fraud recall scorer."],
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={},
            discovery_provider=lambda objective: CapabilityContext(
                available_metrics={"fraud_recall": {"scorer_ref": "metrics.fraud_recall"}},
                available_data_assets=["fraud_training_set"],
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="warehouse_permissions",
                    status="failed",
                    detail="Missing SELECT on analytics.fraud_training_set.",
                )
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
    )

    turn = app.bootstrap(user_goal="Improve fraud detection recall.")

    assert turn.ready_to_start is False
    assert turn.missing_requirements == [
        "answer:positive_label_definition",
        "preflight:warehouse_permissions",
    ]
    assert turn.preflight_checks[0].name == "memory_root_access"
    assert turn.preflight_checks[1].name == "warehouse_permissions"
    assert turn.access_guide_path is not None
    assert turn.human_update is not None
    assert "bootstrap:Improve fraud detection recall." in turn.human_update
    assert "Access guide:" in turn.human_update
    assert Path(turn.access_guide_path).read_text(encoding="utf-8").startswith("# Access Guide")


def test_loopforge_blocks_autonomous_start_without_execution_preflight(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            return BootstrapTurn(
                assistant_message="The plan is clear.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="fraud_recall", goal="maximize"),
                        allowed_actions=["train"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"train": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.5)])},
            discovery_provider=lambda objective: CapabilityContext(
                available_data_assets=["fraud_training_set"],
            ),
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
    )

    turn = app.bootstrap(user_goal="Improve fraud detection recall.")

    assert turn.ready_to_start is False
    assert "preflight:autonomous_execution_permissions" in turn.missing_requirements
    assert turn.preflight_checks[1].status == "failed"
    assert turn.preflight_checks[1].scope == "execution"


def test_loopforge_start_uses_defaults_and_runs_when_bootstrap_is_ready(tmp_path, monkeypatch) -> None:
    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert answer_history == {"positive_label_definition": "chargeback_60d"}
            return BootstrapTurn(
                assistant_message="Configuration is specific enough to start.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(name="fraud_recall", goal="maximize"),
                        allowed_actions=["baseline"],
                        stop_conditions={"max_iterations": 1},
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    class StubWorker:
        def propose_next_experiment(self, snapshot):
            return ExperimentCandidate(
                hypothesis="Baseline",
                action_type="baseline",
                change_type="baseline",
                instructions="Run the baseline.",
            )

    def fake_factory(spec, memory_root: Path):
        return AdapterSetup(
            handlers={"baseline": StaticActionExecutor([ExperimentOutcome(primary_metric_value=0.55)])},
            capability_provider=lambda effective_spec: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_data_assets=["fraud_training_set"],
            ),
            discovery_provider=lambda objective: CapabilityContext(
                available_actions={"baseline": "Run baseline"},
                available_data_assets=["fraud_training_set"],
            ),
            preflight_provider=lambda effective_spec, capability_context: [
                PreflightCheck(
                    name="warehouse_permissions",
                    status="passed",
                    detail="Confirmed data access.",
                )
            ],
        )

    monkeypatch.setattr("loopforge.bootstrap.load_factory", lambda _: fake_factory)

    app = Loopforge(
        executor_factory_path="fake.module:factory",
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        worker_backend=StubWorker(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
        reflection_backend=FakeReflectionBackend([ReflectionSummary(assessment="Fine.")]),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="ok")]),
    )

    result = app.start(
        user_goal="Improve fraud detection recall.",
        answers={"positive_label_definition": "chargeback_60d"},
    )

    assert result["status"] == "started"
    assert result["bootstrap"]["role_models"] == RoleModelConfig(
        planner="openai/gpt-5.4",
        worker="openai/gpt-5.4",
        reflection="openai/gpt-5.4",
        review="openai/gpt-5.4",
        consultation="anthropic/claude-sonnet-4-5",
        narrator="anthropic/claude-sonnet-4-5",
    ).to_dict()
    assert result["results"][0]["accepted_summary"]["primary_metric_value"] == 0.55
    assert "bootstrap:Improve fraud detection recall." in result["bootstrap"]["human_update"]
    assert result["bootstrap"]["access_guide_path"].endswith("ops_access_guide.md")
    assert result["results"][0]["human_update"] == "iteration:baseline:accepted"


def test_synthesize_auto_adapter_creates_loadable_reusable_factory(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "fraud_metrics.py").write_text(
        "def fraud_recall_score(y_true, y_pred):\n    return 0.0\n",
        encoding="utf-8",
    )
    (repo_root / "trainer.py").write_text(
        "def train_model(config):\n    return config\n",
        encoding="utf-8",
    )
    (repo_root / "fraud_data_loader.py").write_text(
        "def load_dataset(name):\n    return name\n",
        encoding="utf-8",
    )

    factory_path = synthesize_auto_adapter(
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        objective="Improve fraud recall.",
    )
    build_adapter = load_factory(factory_path)
    adapter_setup = build_adapter(
        build_spec(allowed_actions=["train_model"]),
        tmp_path / "memory",
    )
    capability_context = adapter_setup.discovery_provider("Improve fraud recall.")

    assert "fraud_recall_score" in capability_context.available_metrics
    assert capability_context.available_actions["train_model"] == "trainer.py"
    assert "fraud_data_loader.py" in capability_context.available_data_assets
    assert Path(factory_path.rsplit(":", maxsplit=1)[0]).exists()


def test_loopforge_bootstrap_auto_synthesizes_adapter_when_missing(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "credit_metrics.py").write_text(
        "def precision_guardrail_metric():\n    return 0.0\n",
        encoding="utf-8",
    )
    (repo_root / "evaluate_pipeline.py").write_text(
        "def evaluate_model():\n    return 0.0\n",
        encoding="utf-8",
    )

    class StubBootstrapBackend:
        def propose_bootstrap_turn(self, user_goal, capability_context, answer_history=None, role_models=None):
            assert capability_context.environment_facts["adapter_kind"] == "auto_generated_scaffold"
            assert "precision_guardrail_metric" in capability_context.available_metrics
            return BootstrapTurn(
                assistant_message="Scaffold generated from repo inspection.",
                proposal=ExperimentSpecProposal(
                    objective=user_goal,
                    recommended_spec=ExperimentSpec(
                        objective=user_goal,
                        primary_metric=PrimaryMetric(
                            name="precision_guardrail_metric",
                            goal="maximize",
                        ),
                        allowed_actions=["evaluate_model"],
                    ),
                ),
                role_models=role_models,
                ready_to_start=True,
            )

    app = Loopforge(
        executor_factory_path=None,
        repo_root=repo_root,
        memory_root=tmp_path / "memory",
        bootstrap_backend=StubBootstrapBackend(),
        access_advisor_backend=FakeAccessAdvisorBackend(),
        narrator_backend=FakeNarrationBackend(),
    )

    turn = app.bootstrap(user_goal="Protect approval precision.")

    assert app.executor_factory_path is not None
    assert turn.ready_to_start is False
    assert "preflight:auto_adapter_scaffold" in turn.missing_requirements


def test_metric_results_and_guardrails_drive_summary_classification(tmp_path) -> None:
    store = FileMemoryStore(tmp_path / "loop")
    worker = FakeWorkerBackend(
        [
            ExperimentCandidate(
                hypothesis="Push recall while holding precision guardrail.",
                action_type="train",
                change_type="threshold",
                instructions="Tune the scorer threshold.",
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        memory_store=store,
        worker_backend=worker,
        executor=RoutingExperimentExecutor(
            handlers={
                "train": StaticActionExecutor(
                    [
                        ExperimentOutcome(
                            metric_results={
                                "recall_at_p90": MetricResult(
                                    name="recall_at_p90",
                                    value=0.74,
                                    scorer_ref="metrics.recall_at_p90",
                                ),
                                "precision_floor": MetricResult(
                                    name="precision_floor",
                                    value=0.88,
                                    scorer_ref="metrics.precision_floor",
                                ),
                            },
                            notes=["Recall improved but precision dipped."],
                        )
                    ]
                )
            }
        ),
        reflection_backend=FakeReflectionBackend(
            [ReflectionSummary(assessment="Primary metric improved but the guardrail failed.")]
        ),
        review_backend=FakeReviewBackend([ReviewDecision(status="accepted", reason="Track this in memory.")]),
    )
    orchestrator.initialize(
        build_spec(
            allowed_actions=["train"],
            primary_metric=PrimaryMetric(
                name="recall_at_p90",
                goal="maximize",
                scorer_ref="metrics.recall_at_p90",
                display_name="Recall @ Precision>=0.90",
            ),
            secondary_metrics=[
                MetricSpec(
                    name="log_loss",
                    goal="minimize",
                    scorer_ref="metrics.log_loss_cv",
                )
            ],
            guardrail_metrics=[
                MetricSpec(
                    name="precision_floor",
                    goal="maximize",
                    scorer_ref="metrics.precision_floor",
                    constraints={"min_value": 0.9},
                )
            ],
        )
    )

    cycle = orchestrator.run_iteration()

    assert cycle.accepted_summary is not None
    assert cycle.accepted_summary.primary_metric_value == 0.74
    assert cycle.accepted_summary.metric_results["recall_at_p90"].scorer_ref == "metrics.recall_at_p90"
    assert cycle.accepted_summary.guardrail_failures == ["precision_floor"]
    assert cycle.accepted_summary.result == "regressed"

