"""Tests for the interactive tool-use agents (executor, planner, reviewer)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock


from loopforge.core.tool_use_execution import (
    ToolUseExecutor,
    ToolUsePlanner,
    ToolUseReviewer,
    _execute_read_file,
    _execute_list_files,
    _execute_search_files,
    _execute_write_file,
    _brief_args,
)
from loopforge.core.types import (
    CapabilityContext,
    ExperimentCandidate,
    ExperimentOutcome,
    ExperimentSpec,
    MemorySnapshot,
    MetricResult,
    MetricSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    objective: str = "Improve RMSE",
    primary_name: str = "rmse",
    primary_goal: str = "minimize",
) -> ExperimentSpec:
    return ExperimentSpec(
        objective=objective,
        primary_metric=MetricSpec(name=primary_name, goal=primary_goal),
        allowed_actions=["run_experiment"],
    )


def _make_snapshot(
    spec: ExperimentSpec | None = None,
    repo_root: str = ".",
) -> MemorySnapshot:
    spec = spec or _make_spec()
    return MemorySnapshot(
        spec=spec,
        effective_spec=spec,
        capability_context=CapabilityContext(
            environment_facts={"repo_root": repo_root, "python_executable": "python"},
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


def _make_candidate(
    hypothesis: str = "Test baseline model",
    instructions: str = "Train a simple model and report RMSE.",
) -> ExperimentCandidate:
    return ExperimentCandidate(
        hypothesis=hypothesis,
        action_type="run_experiment",
        change_type="baseline",
        instructions=instructions,
        execution_steps=[],  # Tool-use executor ignores these
    )


def _mock_tool_call(tool_id: str, name: str, arguments: dict[str, Any]):
    """Create a mock tool_call object matching litellm response format."""
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _mock_response(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
):
    """Create a mock litellm completion response."""
    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tool function tests
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("Hello world\nLine 2")
        result = _execute_read_file({"path": "hello.txt"}, tmp_path, 8000)
        assert "Hello world" in result
        assert "Line 2" in result

    def test_missing_file(self, tmp_path):
        result = _execute_read_file({"path": "nope.txt"}, tmp_path, 8000)
        assert "Error" in result
        assert "not found" in result

    def test_truncates_long_files(self, tmp_path):
        (tmp_path / "big.txt").write_text("\n".join(f"line {i}" for i in range(1000)))
        result = _execute_read_file(
            {"path": "big.txt", "max_lines": 10}, tmp_path, 8000
        )
        assert "more lines" in result

    def test_rejects_path_outside_repo(self, tmp_path):
        result = _execute_read_file({"path": "../../etc/passwd"}, tmp_path, 8000)
        assert "Error" in result or result == ""


class TestListFiles:
    def test_lists_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        result = _execute_list_files({"pattern": "*.py"}, tmp_path)
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_no_matches(self, tmp_path):
        result = _execute_list_files({"pattern": "*.rs"}, tmp_path)
        assert "No files" in result


class TestSearchFiles:
    def test_finds_pattern(self, tmp_path):
        (tmp_path / "code.py").write_text("def train_model():\n    pass\n")
        result = _execute_search_files({"pattern": "train_model"}, tmp_path, 8000)
        assert "code.py" in result
        assert "train_model" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "code.py").write_text("hello\n")
        result = _execute_search_files({"pattern": "nonexistent_xyz"}, tmp_path, 8000)
        assert "No matches" in result


class TestWriteFile:
    def test_writes_file(self, tmp_path):
        result = _execute_write_file(
            {"path": "output.py", "content": "print('hi')"},
            tmp_path,
        )
        assert "OK" in result
        assert (tmp_path / "output.py").read_text() == "print('hi')"

    def test_creates_directories(self, tmp_path):
        result = _execute_write_file(
            {"path": "sub/dir/file.py", "content": "x = 1"},
            tmp_path,
        )
        assert "OK" in result
        assert (tmp_path / "sub" / "dir" / "file.py").exists()

    def test_rejects_path_outside_repo(self, tmp_path):
        result = _execute_write_file(
            {"path": "../../evil.py", "content": "bad"},
            tmp_path,
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# Executor integration tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestToolUseExecutor:
    def test_report_metrics_produces_success(self, tmp_path, monkeypatch):
        """Model calls report_metrics → outcome is success with metrics."""
        executor = ToolUseExecutor(
            model="test/mock",
            repo_root=tmp_path,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()

        # Simulate: model calls report_metrics on first turn
        responses = [
            _mock_response(
                content=None,
                tool_calls=[
                    _mock_tool_call(
                        "tc1",
                        "report_metrics",
                        {
                            "metrics": {"rmse": 0.42},
                            "summary": "Trained a baseline model.",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
        call_count = 0

        def mock_completion(**kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        monkeypatch.setattr(
            "loopforge.core.tool_use_execution.litellm_completion",
            mock_completion,
            raising=False,
        )
        # Patch the import inside execute()
        import loopforge.core.tool_use_execution as mod

        monkeypatch.setattr(mod, "litellm_completion", mock_completion, raising=False)

        # We need to patch the dynamic import inside execute()
        import unittest.mock

        with unittest.mock.patch("litellm.completion", mock_completion):
            outcome = executor.execute(candidate, snapshot)

        assert outcome.status == "success"
        assert "rmse" in outcome.metric_results
        assert outcome.metric_results["rmse"].value == 0.42

    def test_no_metrics_produces_failure(self, tmp_path, monkeypatch):
        """Model finishes without calling report_metrics → recoverable failure."""
        executor = ToolUseExecutor(
            model="test/mock",
            repo_root=tmp_path,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()

        # Model just sends text, no tool calls
        import unittest.mock

        with unittest.mock.patch(
            "litellm.completion",
            return_value=_mock_response(
                content="I couldn't figure it out.", tool_calls=None
            ),
        ):
            outcome = executor.execute(candidate, snapshot)

        assert outcome.status == "recoverable_failure"
        assert outcome.failure_type == "MetricsNotReported"

    def test_read_then_report(self, tmp_path, monkeypatch):
        """Model reads a file, then reports metrics."""
        (tmp_path / "data.csv").write_text("col1,col2\n1,2\n3,4\n")

        executor = ToolUseExecutor(
            model="test/mock",
            repo_root=tmp_path,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()

        responses = [
            # Turn 1: read a file
            _mock_response(
                tool_calls=[_mock_tool_call("tc1", "read_file", {"path": "data.csv"})],
                finish_reason="tool_calls",
            ),
            # Turn 2: report metrics
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc2",
                        "report_metrics",
                        {
                            "metrics": {"rmse": 0.35},
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
        call_idx = 0

        def mock_completion(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        import unittest.mock

        with unittest.mock.patch("litellm.completion", mock_completion):
            outcome = executor.execute(candidate, snapshot)

        assert outcome.status == "success"
        assert outcome.metric_results["rmse"].value == 0.35

    def test_max_errors_limit(self, tmp_path, monkeypatch):
        """Executor stops after max_errors failed run_command calls."""
        executor = ToolUseExecutor(
            model="test/mock",
            repo_root=tmp_path,
            max_errors=2,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()

        # Model always runs a failing command
        def mock_completion(**kwargs):
            return _mock_response(
                tool_calls=[
                    _mock_tool_call("tc", "run_command", {"command": "exit 1"})
                ],
                finish_reason="tool_calls",
            )

        import unittest.mock

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "error"

        with (
            unittest.mock.patch("litellm.completion", mock_completion),
            unittest.mock.patch(
                "loopforge.core.tool_use_execution._run_subprocess_with_progress",
                return_value=mock_proc,
            ),
        ):
            outcome = executor.execute(candidate, snapshot)

        assert outcome.status == "recoverable_failure"

    def test_stdout_fallback_metric_parsing(self, tmp_path, monkeypatch):
        """If model prints metrics to stdout instead of calling report_metrics."""
        executor = ToolUseExecutor(
            model="test/mock",
            repo_root=tmp_path,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()

        # Write a script that prints metrics
        (tmp_path / "train.py").write_text(
            'print(\'{"metric_results": {"rmse": 0.5}}\')'
        )

        responses = [
            # Turn 1: run command that prints metrics
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc1", "run_command", {"command": "python train.py"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            # Turn 2: model finishes
            _mock_response(content="Done.", tool_calls=None, finish_reason="stop"),
        ]
        call_idx = 0

        def mock_completion(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        import unittest.mock

        # Mock subprocess to return metric output
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"metric_results": {"rmse": 0.5}}'
        mock_proc.stderr = ""

        with (
            unittest.mock.patch("litellm.completion", mock_completion),
            unittest.mock.patch(
                "loopforge.core.tool_use_execution._run_subprocess_with_progress",
                return_value=mock_proc,
            ),
        ):
            outcome = executor.execute(candidate, snapshot)

        assert outcome.status == "success"
        assert "rmse" in outcome.metric_results
        assert outcome.metric_results["rmse"].value == 0.5


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestBriefArgs:
    def test_path_arg(self):
        assert _brief_args({"path": "src/model.py"}) == "src/model.py"

    def test_command_arg(self):
        assert _brief_args({"command": "python train.py"}) == "python train.py"

    def test_metrics_arg(self):
        result = _brief_args({"metrics": {"rmse": 0.5, "r2": 0.9}})
        assert "rmse=0.5" in result

    def test_truncates_long_command(self):
        long_cmd = "x" * 200
        result = _brief_args({"command": long_cmd}, limit=80)
        assert len(result) <= 83  # 80 + "..."


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


class TestToolUsePlanner:
    def test_planner_reads_file_and_fills_contract(self, tmp_path):
        """Planner reads the source file and calls fill_contract."""
        # Create a fake training script
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "train.py").write_text(
            "def train_model(df):\n"
            "    from sklearn.ensemble import GradientBoostingRegressor\n"
            "    model = GradientBoostingRegressor()\n"
            "    model.fit(df[['feature1']], df['target_kills'])\n"
            "    return model\n"
        )

        planner = ToolUsePlanner(
            model="test/mock",
            repo_root=tmp_path,
        )

        # Mock: turn 1 = read_file, turn 2 = think, turn 3 = fill_contract
        responses = [
            _mock_response(
                tool_calls=[
                    _mock_tool_call("tc1", "read_file", {"path": "scripts/train.py"})
                ],
                finish_reason="tool_calls",
            ),
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc2", "think", {"reasoning": "I see train_model function"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc3",
                        "fill_contract",
                        {
                            "source_script": "scripts/train.py",
                            "baseline_function": "train_model",
                            "data_loading": "DataFrame passed as argument",
                            "target_column": "target_kills",
                            "primary_metric": "rmse",
                            "primary_metric_goal": "minimize",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
        call_idx = 0

        def mock_completion(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        import unittest.mock

        with unittest.mock.patch("litellm.completion", mock_completion):
            result = planner.plan(
                user_goal="improve model in train.py",
                source_file_hint="scripts/train.py",
            )

        contract = result["contract"]
        assert contract["source_script"] == "scripts/train.py"
        assert contract["baseline_function"] == "train_model"
        assert contract["target_column"] == "target_kills"
        assert contract["primary_metric"] == "rmse"

    def test_planner_asks_user_when_uncertain(self, tmp_path):
        """Planner calls ask_user when it can't determine something."""
        planner = ToolUsePlanner(
            model="test/mock",
            repo_root=tmp_path,
        )

        responses = [
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc1",
                        "ask_user",
                        {
                            "question": "Which function trains the model?",
                            "reason": "Multiple candidates found",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc2",
                        "fill_contract",
                        {
                            "source_script": "train.py",
                            "baseline_function": "unknown",
                            "data_loading": "unknown",
                            "target_column": "unknown",
                            "primary_metric": "rmse",
                            "primary_metric_goal": "minimize",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
        call_idx = 0

        def mock_completion(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        import unittest.mock

        with unittest.mock.patch("litellm.completion", mock_completion):
            result = planner.plan(user_goal="improve model")

        assert len(result["questions"]) == 1
        assert "function" in result["questions"][0]["question"].lower()


# ---------------------------------------------------------------------------
# Reviewer tests
# ---------------------------------------------------------------------------


class TestToolUseReviewer:
    def test_reviewer_accepts_successful_iteration(self, tmp_path):
        """Reviewer accepts an iteration with good metrics."""
        reviewer = ToolUseReviewer(
            model="test/mock",
            repo_root=tmp_path,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()
        outcome = ExperimentOutcome(
            status="success",
            metric_results={"rmse": MetricResult(name="rmse", value=0.42)},
            primary_metric_value=0.42,
        )

        responses = [
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc1", "think", {"reasoning": "RMSE 0.42 is a good baseline"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc2",
                        "report_review",
                        {
                            "status": "accepted",
                            "reason": "Good baseline established",
                            "lessons": ["Baseline RMSE is 0.42"],
                            "next_experiment": "Try adding rolling averages as features",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
        call_idx = 0

        def mock_completion(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        import unittest.mock

        with unittest.mock.patch("litellm.completion", mock_completion):
            reflection, review = reviewer.review(snapshot, candidate, outcome)

        assert review.status == "accepted"
        assert "baseline" in review.reason.lower()
        assert len(reflection.lessons) > 0
        assert reflection.recommended_next_action is not None

    def test_reviewer_rejects_failed_iteration(self, tmp_path):
        """Reviewer rejects a failed iteration but proposes next steps."""
        reviewer = ToolUseReviewer(
            model="test/mock",
            repo_root=tmp_path,
        )
        spec = _make_spec()
        snapshot = _make_snapshot(spec, str(tmp_path))
        candidate = _make_candidate()
        outcome = ExperimentOutcome(
            status="recoverable_failure",
            failure_type="ScriptFailed",
            failure_summary="ImportError: No module named 'lightgbm'",
        )

        import unittest.mock

        with unittest.mock.patch(
            "litellm.completion",
            return_value=_mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "tc1",
                        "report_review",
                        {
                            "status": "rejected",
                            "reason": "Script failed due to missing dependency",
                            "lessons": ["Need to install lightgbm"],
                            "next_experiment": "Install lightgbm then rerun baseline",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ):
            reflection, review = reviewer.review(snapshot, candidate, outcome)

        assert review.status == "rejected"
        assert reflection.recommended_next_action is not None
