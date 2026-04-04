from __future__ import annotations

import json

import loopforge.bootstrap as bootstrap_module
from loopforge import CapabilityContext, MemorySnapshot
from tests.support import build_spec


def test_baseline_first_worker_backend_skips_llm_for_first_constrained_iteration() -> (
    None
):
    class ExplodingWorker:
        model = "openai/gpt-5.4"

        def propose_next_experiment(self, snapshot):
            raise AssertionError(
                "LLM worker should not be called for baseline-first plan."
            )

        def continue_experiment(self, snapshot, previous_candidate, previous_outcome):
            raise AssertionError("not used")

    backend = bootstrap_module.BaselineFirstWorkerBackend(ExplodingWorker())
    spec = build_spec(
        allowed_actions=["run_experiment"],
        metadata={
            "execution_contract": {
                "baseline_paths": ["src/train.py"],
                "must_reference_baseline_paths": True,
                "enforcement_scope": "until_first_successful_iteration",
            }
        },
    )
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={
                "execution_backend_kind": "generic_agentic",
                "repo_root": "C:/repo",
                "python_executable": "C:/repo/.venv/Scripts/python.exe",
            }
        ),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    candidate = backend.propose_next_experiment(snapshot)

    assert candidate.metadata["baseline_first_contract"] is True
    assert candidate.execution_steps[0].command.endswith('"src/train.py"')
    assert "existing baseline script" in candidate.instructions.lower()


def test_litellm_backend_passes_max_completion_tokens_to_completion_fn() -> None:
    captured_kwargs = {}

    def completion_fn(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hypothesis": "Run baseline.",
                                "action_type": "run_experiment",
                                "change_type": "baseline",
                                "instructions": "Run the existing baseline.",
                                "execution_steps": [],
                            }
                        )
                    }
                }
            ]
        }

    worker = bootstrap_module.LiteLLMWorkerBackend(
        model="openai/gpt-5.4",
        completion_fn=completion_fn,
        max_completion_tokens=321,
    )
    spec = build_spec(allowed_actions=["run_experiment"])
    snapshot = MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(),
        best_summary=None,
        latest_summary=None,
        recent_records=[],
        recent_summaries=[],
        recent_human_interventions=[],
        lessons_learned="",
        markdown_memory=[],
        next_iteration_id=1,
    )

    worker.propose_next_experiment(snapshot)

    assert captured_kwargs["max_completion_tokens"] == 321

