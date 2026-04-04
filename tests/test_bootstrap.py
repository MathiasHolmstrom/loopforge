from __future__ import annotations

import pytest

import loopforge.bootstrap as bootstrap_module
from loopforge import Loopforge


@pytest.mark.parametrize(
    ("helpers_supported", "kwargs", "expected_helper_model", "expected_core_model"),
    [
        (
            False,
            {
                "consultation_model": "anthropic/claude-sonnet-4-5",
                "narrator_model": "anthropic/claude-sonnet-4-5",
            },
            bootstrap_module.DEFAULT_OPENAI_MODEL,
            bootstrap_module.DEFAULT_OPENAI_MODEL,
        ),
        (True, {}, bootstrap_module.DEFAULT_CLAUDE_MODEL, None),
    ],
)
def test_default_role_models_helper_routing(
    monkeypatch,
    helpers_supported: bool,
    kwargs: dict[str, str],
    expected_helper_model: str,
    expected_core_model: str | None,
) -> None:
    monkeypatch.setattr(
        "loopforge.bootstrap._can_use_anthropic_helpers", lambda: helpers_supported
    )
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)

    role_models = bootstrap_module.default_role_models(**kwargs)

    if expected_core_model is not None:
        assert role_models.planner == expected_core_model
        assert role_models.worker == expected_core_model
        assert role_models.reflection == expected_core_model
        assert role_models.review == expected_core_model
    assert role_models.consultation == expected_helper_model
    assert role_models.narrator == expected_helper_model


def test_loopforge_init_wires_default_models_to_expected_backends(tmp_path) -> None:
    app = Loopforge(memory_root=tmp_path / "memory")

    assert app.bootstrap_backend.model == app.role_models.planner
    assert app.worker_backend.model == app.role_models.worker
    assert app.reflection_backend.model == app.role_models.reflection
    assert app.review_backend.model == app.role_models.review
    expected_helper = app.role_models.consultation
    assert app.access_advisor_backend.model == expected_helper
    assert app.execution_fix_backend.model == expected_helper
    assert app.narrator_backend.model == app.role_models.narrator
    assert (
        app.bootstrap_backend.max_completion_tokens
        == bootstrap_module.DEFAULT_ROLE_MAX_COMPLETION_TOKENS["planner"]
    )
    assert (
        app.worker_backend.max_completion_tokens
        == bootstrap_module.DEFAULT_ROLE_MAX_COMPLETION_TOKENS["worker"]
    )
