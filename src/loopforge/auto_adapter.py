from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


METRIC_TOKENS = ("metric", "score", "loss", "auc", "precision", "recall", "accuracy", "f1", "rmse", "mae")
ACTION_TOKENS = ("train", "baseline", "evaluate", "eval", "eda", "tune", "validate", "predict", "infer")
DATA_TOKENS = ("data", "dataset", "table", "feature", "artifact", "model")
EXCLUDED_DIRS = {".git", ".hg", ".venv", ".pytest_cache", "__pycache__", "node_modules"}


def _iter_python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _record_symbol(name: str, path: Path, repo_root: Path, bucket: dict[str, dict[str, Any]]) -> None:
    if name in bucket:
        return
    bucket[name] = {
        "symbol": name,
        "path": str(path.relative_to(repo_root)).replace("\\", "/"),
    }


def scan_repo(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    metrics: dict[str, dict[str, Any]] = {}
    actions: dict[str, dict[str, Any]] = {}
    data_assets: list[str] = []
    notes: list[str] = []

    for path in _iter_python_files(root):
        relative_path = str(path.relative_to(root)).replace("\\", "/")
        lower_path = relative_path.lower()
        if any(token in lower_path for token in DATA_TOKENS) and relative_path not in data_assets:
            data_assets.append(relative_path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            name = getattr(node, "name", None)
            if not isinstance(name, str):
                continue
            lowered = name.lower()
            if any(token in lowered for token in METRIC_TOKENS):
                _record_symbol(name, path, root, metrics)
            if any(token in lowered for token in ACTION_TOKENS):
                normalized = name.lower()
                if normalized == "eval":
                    normalized = "evaluate"
                _record_symbol(normalized, path, root, actions)

    if metrics:
        notes.append(f"Discovered {len(metrics)} candidate metric symbols from Python sources.")
    if actions:
        notes.append(f"Discovered {len(actions)} candidate action symbols from Python sources.")
    if data_assets:
        notes.append(f"Discovered {len(data_assets)} candidate data-related files from repo paths.")

    return {
        "repo_root": str(root),
        "metrics": metrics,
        "actions": actions,
        "data_assets": data_assets,
        "notes": notes,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "loopforge_auto_adapter"


def synthesize_auto_adapter(
    *,
    repo_root: Path | str,
    memory_root: Path | str,
    objective: str,
) -> str:
    summary = scan_repo(repo_root)
    memory_root_path = Path(memory_root)
    generated_dir = memory_root_path / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    module_stem = _slugify(f"{Path(repo_root).name}_{objective}")[:80]
    adapter_path = generated_dir / f"{module_stem}_adapter.py"
    summary_path = generated_dir / f"{module_stem}_summary.json"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    module_source = f"""from __future__ import annotations

from pathlib import Path

from loopforge import AdapterSetup, CapabilityContext, ExperimentOutcome, PreflightCheck

DISCOVERED_METRICS = {json.dumps(summary["metrics"], indent=4, sort_keys=True)}
DISCOVERED_ACTIONS = {json.dumps(summary["actions"], indent=4, sort_keys=True)}
DISCOVERED_DATA_ASSETS = {json.dumps(summary["data_assets"], indent=4)}
DISCOVERY_NOTES = {json.dumps(summary["notes"], indent=4)}
REPO_ROOT = r\"{summary["repo_root"]}\"
SUMMARY_PATH = r\"{summary_path}\"


class _PlaceholderExecutor:
    def __init__(self, action_name: str) -> None:
        self.action_name = action_name

    def execute(self, candidate, snapshot):
        raise NotImplementedError(
            f\"Auto-generated adapter placeholder for action '{{self.action_name}}' is not implemented yet. \"
            f\"Edit {{__file__}} to bind this action to repo-specific code.\"
        )


def _build_context(objective: str) -> CapabilityContext:
    return CapabilityContext(
        available_actions={{name: item["path"] for name, item in DISCOVERED_ACTIONS.items()}},
        available_data_assets=list(DISCOVERED_DATA_ASSETS),
        available_metrics={{
            name: {{"scorer_ref": f\"auto_adapter:{{item['path']}}:{{item['symbol']}}\", "path": item["path"]}}
            for name, item in DISCOVERED_METRICS.items()
        }},
        environment_facts={{
            "adapter_kind": "auto_generated_scaffold",
            "repo_root": REPO_ROOT,
            "summary_path": SUMMARY_PATH,
        }},
        notes=list(DISCOVERY_NOTES),
    )


def _preflight_provider(spec, capability_context):
    checks = [
        PreflightCheck(
            name="auto_adapter_scaffold",
            status="failed",
            detail=(
                "Loopforge synthesized an adapter scaffold from repo inspection, but action handlers are still "
                "placeholders. Fill in the generated adapter module before execution."
            ),
            scope="execution",
        )
    ]
    for asset in capability_context.available_data_assets:
        asset_path = Path(REPO_ROOT) / asset
        if asset_path.exists():
                checks.append(
                    PreflightCheck(
                        name=f"asset:{{asset}}",
                    status="passed",
                    detail=f"Discovered repo path exists: {{asset_path}}",
                    required=False,
                    scope="bootstrap",
                )
            )
    return checks


def build_adapter(spec, memory_root):
    context = _build_context(spec.objective)
    action_names = spec.allowed_actions or list(DISCOVERED_ACTIONS.keys())
    handlers = {{name: _PlaceholderExecutor(name) for name in action_names}}
    return AdapterSetup(
        handlers=handlers,
        capability_provider=lambda effective_spec: _build_context(effective_spec.objective),
        discovery_provider=_build_context,
        preflight_provider=_preflight_provider,
    )
"""
    adapter_path.write_text(module_source, encoding="utf-8")
    return f"{adapter_path}:build_adapter"
