from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from loopforge.core.types import CapabilityContext


METRIC_TOKENS = (
    "metric",
    "score",
    "loss",
    "auc",
    "precision",
    "recall",
    "accuracy",
    "f1",
    "rmse",
    "mae",
)
DATA_TOKENS = (
    "data",
    "dataset",
    "table",
    "feature",
    "artifact",
    "model",
    "schema",
    "ddl",
    "migration",
)
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".loopforge",
}
DATA_FILE_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}
# Patterns that indicate a string is a DataFrame column reference
COLUMN_ACCESS_FUNCS = {
    "col",
    "alias",
    "over",
    "sort",
    "select",
    "with_columns",
    "group_by",
    "filter",
}
COLUMN_NOISE = {
    "utf-8",
    "utf8",
    "r",
    "w",
    "rb",
    "wb",
    "a",
    "",
    ".",
    ",",
    "/",
    "\\",
    "ok",
    "true",
    "false",
}


def _iter_python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _iter_data_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in DATA_FILE_SUFFIXES:
            continue
        files.append(path)
    return files


def _record_symbol(
    name: str, path: Path, repo_root: Path, bucket: dict[str, dict[str, Any]]
) -> None:
    if name in bucket:
        return
    bucket[name] = {
        "symbol": name,
        "path": str(path.relative_to(repo_root)).replace("\\", "/"),
    }


def _is_candidate_action_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("_") or lowered.startswith("test_"):
        return False
    if len(lowered) > 48:
        return False
    if "__" in lowered:
        return False
    return True


def _file_uses_dataframes(tree: ast.AST) -> bool:
    """Check if a file imports pandas or polars (i.e. actually works with DataFrames)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [alias.name for alias in getattr(node, "names", [])]
            all_names = [module] + names
            if any(n in ("pandas", "polars", "pl", "pd") for n in all_names):
                return True
            if any("pandas" in n or "polars" in n for n in all_names):
                return True
    return False


def _extract_column_refs(tree: ast.AST) -> set[str]:
    """Extract likely DataFrame column names from AST — only from files that use DataFrames."""
    if not _file_uses_dataframes(tree):
        return set()
    columns: set[str] = set()
    for node in ast.walk(tree):
        # pl.col("name"), .alias("name"), .over(["col1", "col2"]), etc.
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            if isinstance(func, ast.Attribute):
                func_name = func.attr
            elif isinstance(func, ast.Name):
                func_name = func.id
            if func_name and func_name.lower() in COLUMN_ACCESS_FUNCS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        columns.add(arg.value)
                    elif isinstance(arg, ast.List):
                        for elt in arg.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                columns.add(elt.value)

    # Filter noise — keep only plausible column names (lowercase/snake_case, no class names)
    def _looks_like_column(name: str) -> bool:
        if name.lower() in COLUMN_NOISE:
            return False
        if len(name) <= 1:
            return False
        if name[0].isupper() and any(c.isupper() for c in name[1:]):
            return False  # CamelCase = likely a class name
        return True

    return {c for c in columns if _looks_like_column(c)}


def scan_repo(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    metrics: dict[str, dict[str, Any]] = {}
    actions: dict[str, dict[str, Any]] = {}
    data_assets: list[str] = []
    notes: list[str] = []
    column_refs: dict[str, list[str]] = {}  # file -> columns found

    for path in _iter_python_files(root):
        relative_path = str(path.relative_to(root)).replace("\\", "/")
        lower_path = relative_path.lower()
        if (
            any(token in lower_path for token in DATA_TOKENS)
            and relative_path not in data_assets
        ):
            data_assets.append(relative_path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.iter_child_nodes(tree):
            name = getattr(node, "name", None)
            if not isinstance(name, str):
                continue
            lowered = name.lower()
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and any(token in lowered for token in METRIC_TOKENS):
                _record_symbol(name, path, root, metrics)
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and _is_candidate_action_name(name):
                _record_symbol(name, path, root, actions)
        # Extract column references from data manipulation code
        cols = _extract_column_refs(tree)
        if cols:
            column_refs[relative_path] = sorted(cols)

    for path in _iter_data_files(root):
        relative_path = str(path.relative_to(root)).replace("\\", "/")
        if relative_path not in data_assets:
            data_assets.append(relative_path)

    if metrics:
        notes.append(
            f"Discovered {len(metrics)} candidate metric symbols from Python sources."
        )
    if actions:
        notes.append(
            f"Discovered {len(actions)} candidate action symbols from Python sources."
        )
    if data_assets:
        notes.append(
            f"Discovered {len(data_assets)} candidate data-related files from repo paths."
        )

    # Summarize discovered columns for the agent (cap at 50 to avoid noise)
    all_columns = sorted({col for cols in column_refs.values() for col in cols})
    if all_columns:
        if len(all_columns) <= 50:
            notes.append(
                f"DataFrame columns referenced in code: {', '.join(all_columns)}"
            )
        else:
            notes.append(
                f"DataFrame columns referenced in code ({len(all_columns)} total, showing first 50): "
                f"{', '.join(all_columns[:50])} ..."
            )
        for file_path, cols in column_refs.items():
            notes.append(f"  {file_path}: {', '.join(cols)}")

    return {
        "repo_root": str(root),
        "metrics": metrics,
        "actions": actions,
        "data_assets": data_assets,
        "notes": notes,
        "column_refs": column_refs,
    }


def build_repo_scan_context(repo_root: Path | str) -> CapabilityContext:
    summary = scan_repo(repo_root)
    discovered_actions = summary["actions"]
    discovered_metrics = summary["metrics"]
    notes = [
        "Loopforge can inspect this repo and plan experiments from observed repo structure, but it does not yet have a real runnable runner for this objective.",
        "Discovered symbols are observations only. They are not executable experiment steps until the repo has a supported runner.",
    ]
    notes.extend(summary["notes"])
    if discovered_actions:
        previews = [
            f"{name} ({item['path']})"
            for name, item in list(discovered_actions.items())[:10]
        ]
        notes.append("Public callables discovered in code: " + ", ".join(previews))
    return CapabilityContext(
        available_actions={
            name: item["path"] for name, item in discovered_actions.items()
        },
        available_data_assets=list(summary["data_assets"]),
        available_metrics={
            name: {
                "scorer_ref": f"repo_scan:{item['path']}:{item['symbol']}",
                "path": item["path"],
            }
            for name, item in discovered_metrics.items()
        },
        environment_facts={
            "execution_backend_kind": "planning_only_repo_scan",
            "autonomous_execution_supported": False,
            "observation_mode": "repo_scan_only",
            "repo_root": summary["repo_root"],
            "discovered_action_count": len(discovered_actions),
            "discovered_metric_count": len(discovered_metrics),
        },
        notes=notes,
    )


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

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
                "Loopforge found a likely experiment path from repo inspection, but it does not yet have a "
                "real repo-specific runner for it. Execution cannot start until that runner is wired."
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
