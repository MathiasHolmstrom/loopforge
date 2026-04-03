from __future__ import annotations

from loopforge.core.types import CapabilityContext, ExperimentSpec, MemorySnapshot


def is_generic_autonomous(
    *,
    snapshot: MemorySnapshot | None = None,
    capability_context: CapabilityContext | None = None,
    effective_spec: ExperimentSpec | None = None,
) -> bool:
    if snapshot is not None:
        capability_context = snapshot.capability_context
        effective_spec = snapshot.effective_spec
    environment_facts = capability_context.environment_facts if capability_context is not None else {}
    metadata = effective_spec.metadata if effective_spec is not None else {}
    return (
        environment_facts.get("execution_backend_kind") == "generic_agentic"
        or metadata.get("execution_backend_kind") == "generic_agentic"
        or metadata.get("execution_mode") == "autonomous_after_bootstrap"
    )
