from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loopforge.core.types import (
    AgentUpdate,
    CapabilityContext,
    ExperimentSpec,
    HumanIntervention,
    IterationRecord,
    IterationSummary,
    MarkdownMemoryNote,
    MemorySnapshot,
    apply_human_interventions,
)


class FileMemoryStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._artifacts_dir = self.root / "artifacts"
        self._agent_markdown_dir = self.root / "agent_markdown"
        self._objective_path = self._agent_markdown_dir / "objective.md"
        self._spec_path = self.root / "experiment_spec.json"
        self._best_result_path = self.root / "best_result.json"
        self._records_path = self.root / "iteration_records.jsonl"
        self._summaries_path = self.root / "iteration_summaries.jsonl"
        self._updates_path = self.root / "agent_updates.jsonl"
        self._human_notes_path = self.root / "human_notes.jsonl"
        self._lessons_path = self._agent_markdown_dir / "lessons_learned.md"
        self._experiment_journal_path = self._agent_markdown_dir / "experiment_journal.md"
        self._artifact_index_path = self._artifacts_dir / "index.json"
        self._legacy_markdown_paths = {
            "objective": self.root / "objective.md",
            "lessons": self.root / "lessons_learned.md",
            "journal": self.root / "experiment_journal.md",
            "ops_access_guide": self.root / "ops_access_guide.md",
            "execution_runbook": self.root / "execution_runbook.md",
            "experiment_guide": self.root / "experiment_guide.md",
        }

    def initialize(self, spec: ExperimentSpec, *, reset_state: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._agent_markdown_dir.mkdir(parents=True, exist_ok=True)
        self._objective_path.write_text(spec.objective.strip() + "\n", encoding="utf-8")
        self._spec_path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path, default_contents in [
            (self._records_path, ""),
            (self._summaries_path, ""),
            (self._updates_path, ""),
            (self._human_notes_path, ""),
            (self._lessons_path, ""),
            (self._experiment_journal_path, ""),
            (self._artifact_index_path, "[]\n"),
        ]:
            if reset_state or not path.exists():
                path.write_text(default_contents, encoding="utf-8")
        if reset_state:
            self._best_result_path.unlink(missing_ok=True)

    def load_spec(self) -> ExperimentSpec:
        return ExperimentSpec.from_dict(json.loads(self._spec_path.read_text(encoding="utf-8")))

    def load_snapshot(
        self,
        summary_window: int = 5,
        human_window: int = 10,
        capability_context: CapabilityContext | None = None,
    ) -> MemorySnapshot:
        spec = self.load_spec()
        records = self._read_records()
        summaries = self._read_summaries()
        human_interventions = self._read_human_interventions()
        effective_spec = apply_human_interventions(spec, human_interventions)
        lessons = self._read_markdown_text(self._lessons_path, self._legacy_markdown_paths["lessons"])
        return MemorySnapshot(
            spec=spec,
            effective_spec=effective_spec,
            capability_context=capability_context or CapabilityContext(),
            best_summary=self._read_best_summary(),
            latest_summary=summaries[-1] if summaries else None,
            recent_records=records[-summary_window:],
            recent_summaries=summaries[-summary_window:],
            recent_human_interventions=human_interventions[-human_window:],
            lessons_learned=lessons.strip(),
            markdown_memory=self._read_markdown_memory(),
            next_iteration_id=len(records) + 1,
        )

    def load_bootstrap_context(self, summary_window: int = 5, human_window: int = 10) -> dict[str, Any]:
        summaries = self._read_summaries()
        best_summary = self._read_best_summary()
        human_interventions = self._read_human_interventions()
        lessons = self._read_markdown_text(self._lessons_path, self._legacy_markdown_paths["lessons"])
        objective_text = self._read_markdown_text(self._objective_path, self._legacy_markdown_paths["objective"])
        objective = objective_text.strip() if objective_text else None
        return {
            "previous_objective": objective,
            "best_summary": best_summary.to_dict() if best_summary is not None else None,
            "recent_summaries": [summary.to_dict() for summary in summaries[-summary_window:]],
            "recent_human_interventions": [item.to_dict() for item in human_interventions[-human_window:]],
            "lessons_learned": lessons.strip(),
            "markdown_memory": [item.to_dict() for item in self._read_markdown_memory()],
        }

    def has_persisted_state(self) -> bool:
        return any(
            path.exists() and path.read_text(encoding="utf-8").strip()
            for path in (
                self._records_path,
                self._summaries_path,
                self._updates_path,
                self._human_notes_path,
                self._lessons_path,
                self._experiment_journal_path,
                self._artifact_index_path,
                self._best_result_path,
                self._legacy_markdown_paths["objective"],
                self._legacy_markdown_paths["lessons"],
                self._legacy_markdown_paths["journal"],
            )
        )

    def append_iteration_record(self, record: IterationRecord) -> None:
        with self._records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def append_accepted_summary(self, summary: IterationSummary) -> None:
        with self._summaries_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")
        self._append_lessons(summary.lessons)
        self._append_experiment_journal(summary)
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

    @staticmethod
    def _read_markdown_text(primary_path: Path, legacy_path: Path | None = None) -> str:
        if primary_path.exists():
            return primary_path.read_text(encoding="utf-8")
        if legacy_path is not None and legacy_path.exists():
            return legacy_path.read_text(encoding="utf-8")
        return ""

    def _append_lessons(self, lessons: list[str]) -> None:
        if not lessons:
            return
        existing_lessons = [
            line[2:].strip()
            for line in self._read_markdown_text(self._lessons_path, self._legacy_markdown_paths["lessons"]).splitlines()
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

    def _read_markdown_memory(self) -> list[MarkdownMemoryNote]:
        if not self.root.exists():
            return []
        notes: list[MarkdownMemoryNote] = []
        seen_paths: set[str] = set()
        for path in sorted(self._agent_markdown_dir.rglob("*.md")) if self._agent_markdown_dir.exists() else []:
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            rel_path = str(path.relative_to(self.root)).replace("\\", "/")
            seen_paths.add(path.name)
            notes.append(
                MarkdownMemoryNote(
                    path=rel_path,
                    content=content,
                )
            )
        for path in sorted(self.root.glob("*.md")):
            if not path.is_file() or path.name in seen_paths:
                continue
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            notes.append(
                MarkdownMemoryNote(
                    path=str(path.relative_to(self.root)).replace("\\", "/"),
                    content=content,
                )
            )
        return notes

    def _append_experiment_journal(self, summary: IterationSummary) -> None:
        lines = [
            f"## Iteration {summary.iteration_id}: {summary.hypothesis}",
            f"- Action: {summary.action_type}",
            f"- Result: {summary.result}",
            f"- Outcome status: {summary.outcome_status}",
            f"- Primary metric: {summary.primary_metric_name}={summary.primary_metric_value}",
            f"- Review reason: {summary.review_reason}",
        ]
        if summary.failure_summary:
            lines.append(f"- Failure: {summary.failure_summary}")
        if summary.recovery_actions:
            lines.append("- Recovery actions: " + "; ".join(summary.recovery_actions))
        if summary.guardrail_failures:
            lines.append("- Guardrail failures: " + ", ".join(summary.guardrail_failures))
        if summary.lessons:
            lines.append("- Lessons: " + "; ".join(summary.lessons))
        if summary.do_not_repeat:
            lines.append("- Do not repeat: " + "; ".join(summary.do_not_repeat))
        if summary.next_ideas:
            lines.append("- Next ideas: " + "; ".join(summary.next_ideas))
        with self._experiment_journal_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n\n")

