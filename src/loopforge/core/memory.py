from __future__ import annotations

import json
from pathlib import Path

from loopforge.core.types import (
    AgentUpdate,
    CapabilityContext,
    ExperimentSpec,
    HumanIntervention,
    IterationRecord,
    IterationSummary,
    MemorySnapshot,
    apply_human_interventions,
)


class FileMemoryStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._artifacts_dir = self.root / "artifacts"
        self._objective_path = self.root / "objective.md"
        self._spec_path = self.root / "experiment_spec.json"
        self._best_result_path = self.root / "best_result.json"
        self._records_path = self.root / "iteration_records.jsonl"
        self._summaries_path = self.root / "iteration_summaries.jsonl"
        self._updates_path = self.root / "agent_updates.jsonl"
        self._human_notes_path = self.root / "human_notes.jsonl"
        self._lessons_path = self.root / "lessons_learned.md"
        self._artifact_index_path = self._artifacts_dir / "index.json"

    def initialize(self, spec: ExperimentSpec) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._objective_path.write_text(spec.objective.strip() + "\n", encoding="utf-8")
        self._spec_path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path, default_contents in [
            (self._records_path, ""),
            (self._summaries_path, ""),
            (self._updates_path, ""),
            (self._human_notes_path, ""),
            (self._lessons_path, ""),
            (self._artifact_index_path, "[]\n"),
        ]:
            if not path.exists():
                path.write_text(default_contents, encoding="utf-8")

    def load_spec(self) -> ExperimentSpec:
        return ExperimentSpec.from_dict(json.loads(self._spec_path.read_text(encoding="utf-8")))

    def load_snapshot(
        self,
        summary_window: int = 5,
        human_window: int = 10,
        capability_context: CapabilityContext | None = None,
    ) -> MemorySnapshot:
        spec = self.load_spec()
        summaries = self._read_summaries()
        human_interventions = self._read_human_interventions()
        effective_spec = apply_human_interventions(spec, human_interventions)
        lessons = self._lessons_path.read_text(encoding="utf-8") if self._lessons_path.exists() else ""
        return MemorySnapshot(
            spec=spec,
            effective_spec=effective_spec,
            capability_context=capability_context or CapabilityContext(),
            best_summary=self._read_best_summary(),
            latest_summary=summaries[-1] if summaries else None,
            recent_summaries=summaries[-summary_window:],
            recent_human_interventions=human_interventions[-human_window:],
            lessons_learned=lessons.strip(),
            next_iteration_id=len(self._read_records()) + 1,
        )

    def append_iteration_record(self, record: IterationRecord) -> None:
        with self._records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def append_accepted_summary(self, summary: IterationSummary) -> None:
        with self._summaries_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")
        self._append_lessons(summary.lessons)
        self._append_artifacts(summary)

    def write_best_summary(self, summary: IterationSummary) -> None:
        self._best_result_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def append_human_intervention(self, intervention: HumanIntervention) -> None:
        with self._human_notes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(intervention.to_dict(), sort_keys=True) + "\n")

    def append_agent_update(self, update: AgentUpdate) -> None:
        with self._updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(update.to_dict(), sort_keys=True) + "\n")

    def _read_records(self) -> list[IterationRecord]:
        if not self._records_path.exists():
            return []
        return [
            IterationRecord.from_dict(json.loads(line))
            for line in self._records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _read_summaries(self) -> list[IterationSummary]:
        if not self._summaries_path.exists():
            return []
        return [
            IterationSummary.from_dict(json.loads(line))
            for line in self._summaries_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _read_human_interventions(self) -> list[HumanIntervention]:
        if not self._human_notes_path.exists():
            return []
        return [
            HumanIntervention.from_dict(json.loads(line))
            for line in self._human_notes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def read_agent_updates(self) -> list[AgentUpdate]:
        if not self._updates_path.exists():
            return []
        return [
            AgentUpdate.from_dict(json.loads(line))
            for line in self._updates_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _read_best_summary(self) -> IterationSummary | None:
        if not self._best_result_path.exists():
            return None
        return IterationSummary.from_dict(json.loads(self._best_result_path.read_text(encoding="utf-8")))

    def _append_lessons(self, lessons: list[str]) -> None:
        if not lessons:
            return
        existing_lessons = [
            line[2:].strip()
            for line in self._lessons_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
        merged_lessons = list(existing_lessons)
        for lesson in lessons:
            if lesson not in merged_lessons:
                merged_lessons.append(lesson)
        self._lessons_path.write_text("".join(f"- {lesson}\n" for lesson in merged_lessons), encoding="utf-8")

    def _append_artifacts(self, summary: IterationSummary) -> None:
        existing_payload = json.loads(self._artifact_index_path.read_text(encoding="utf-8"))
        existing_payload.append(
            {
                "iteration_id": summary.iteration_id,
                "action_type": summary.action_type,
                "artifacts": summary.artifacts,
            }
        )
        self._artifact_index_path.write_text(
            json.dumps(existing_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

