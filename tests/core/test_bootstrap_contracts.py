from __future__ import annotations

from loopforge import CapabilityContext, PreflightCheck
from loopforge.core.bootstrap_contracts import (
    apply_bootstrap_execution_contract,
    resolve_repo_root_from_objective,
    should_prepare_access_guide,
)
from tests.support import build_spec


def test_should_prepare_access_guide_when_access_risk_is_present() -> None:
    capability_context = CapabilityContext(
        notes=["Need Databricks authentication before data access."],
        environment_facts={"warehouse": "analytics"},
    )

    result = should_prepare_access_guide(
        capability_context=capability_context,
        preflight_checks=[PreflightCheck(name="ready", status="passed", detail="ok")],
    )

    assert result is True


def test_apply_bootstrap_execution_contract_enforces_baseline_reuse_from_operator_intent() -> (
    None
):
    spec = build_spec(
        metadata={"operator_guidance": ["keep the existing framework and copy the script"]}
    )
    capability_context = CapabilityContext(
        environment_facts={"baseline_code_paths": ["src/train.py"]}
    )

    updated = apply_bootstrap_execution_contract(
        spec=spec,
        capability_context=capability_context,
        user_goal="Use the existing script and work from there.",
        assistant_message="Reuse the current framework.",
        answers={"approach": "copy my script"},
        answer_summary="copy my script",
    )

    contract = updated.metadata["execution_contract"]
    assert contract["baseline_paths"] == ["src/train.py"]
    assert contract["must_reference_baseline_paths"] is True


def test_resolve_repo_root_from_objective_prefers_named_child_repo(tmp_path) -> None:
    current_root = tmp_path / "workspace"
    current_root.mkdir()
    target = current_root / "player-performance-ratings"
    target.mkdir()

    resolved = resolve_repo_root_from_objective(
        "Work inside player-performance-ratings repo and reuse the existing script.",
        current_root,
    )

    assert resolved == target

