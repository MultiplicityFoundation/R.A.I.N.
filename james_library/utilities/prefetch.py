"""Dependency-aware prompt prefetching for orchestrator first turns.

This module keeps the Python side lightweight. It finds likely file paths in a
prompt, resolves a small set of adjacent dependencies, and asks the Rust LSP
bridge for symbol summaries when available.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


LspQuery = Callable[..., dict[str, Any]]

_BACKTICK_PATH_RE = re.compile(r"`([^`\n]+(?:/|\\)[^`\n]+\.[A-Za-z0-9_]+)`")
_BARE_PATH_RE = re.compile(r"(?<![\w.-])((?:[A-Za-z]:)?[^\s`'\"<>|]+(?:/|\\)[^\s`'\"<>|]+\.[A-Za-z0-9_]+)")
_FILENAME_RE = re.compile(r"(?<![\w/\\.-])([A-Za-z0-9_-]+\.[A-Za-z0-9_]+)(?![\w.-])")
_PY_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)", re.MULTILINE)
_PY_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+", re.MULTILINE)
_RUST_MOD_RE = re.compile(r"^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.MULTILINE)
_RUST_USE_RE = re.compile(r"^\s*use\s+(?:crate::)?([A-Za-z_][A-Za-z0-9_:]*)", re.MULTILINE)


def extract_file_paths(prompt: str, workspace_root: str | Path) -> list[Path]:
    """Find existing workspace file paths mentioned in a user prompt."""

    workspace = Path(workspace_root).resolve()
    candidates: list[str] = []
    for pattern in (_BACKTICK_PATH_RE, _BARE_PATH_RE):
        candidates.extend(match.group(1).strip() for match in pattern.finditer(prompt))
    candidates.extend(match.group(1).strip() for match in _FILENAME_RE.finditer(prompt))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = _resolve_prompt_path(candidate, workspace)
        if path is None or path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def build_prefetch_context(
    prompt: str,
    workspace_root: str | Path,
    lsp_query: LspQuery | None = None,
) -> str:
    """Build a compact first-turn context block for referenced files."""

    workspace = Path(workspace_root).resolve()
    targets = extract_file_paths(prompt, workspace)
    if not targets:
        return ""

    query = lsp_query or query_lsp_bridge
    sections = ["[IDE VISION]", "Prefetched code context for referenced files and direct dependencies."]

    for target in targets:
        target_symbols = _document_symbols_for(target, query)
        sections.append(f"- {target.relative_to(workspace).as_posix()}: {_format_symbol_summary(target_symbols)}")

        dependency_paths = _discover_dependencies(target, workspace)
        if dependency_paths:
            sections.append("  imports:")
        for dependency in dependency_paths[:6]:
            dep_symbols = _document_symbols_for(dependency, query)
            dep_label = dependency.relative_to(workspace).as_posix()
            sections.append(f"  - {dep_label}: {_format_symbol_summary(dep_symbols)}")

    return "\n".join(sections)


def query_lsp_bridge(
    action: str,
    file_path: str,
    *,
    line: int | None = None,
    character: int | None = None,
) -> dict[str, Any]:
    """Query the Rust LSP bridge.

    This is intentionally soft-failing so orchestration can continue when the
    bridge or language server is unavailable.
    """

    payload: dict[str, Any] = {"action": action, "file_path": str(Path(file_path).resolve())}
    if line is not None:
        payload["line"] = int(line)
    if character is not None:
        payload["character"] = int(character)

    command = os.environ.get("JAMES_LSP_BRIDGE_CMD")
    if command:
        raw_command = [part for part in command.split(" ") if part]
    else:
        raw_command = ["cargo", "run", "--quiet", "--", "lsp-query"]

    try:
        completed = subprocess.run(
            raw_command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"results": [], "error": f"LSP bridge unavailable: {exc}"}

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return {"results": [], "error": f"LSP bridge failed: {stderr or 'unknown error'}"}

    stdout = completed.stdout.strip()
    if not stdout:
        return {"results": [], "error": "LSP bridge returned empty output"}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {"results": [], "error": f"Invalid LSP bridge JSON: {exc}"}


def _resolve_prompt_path(candidate: str, workspace: Path) -> Path | None:
    normalized = candidate.strip().strip(".,:;()[]{}")
    if not normalized:
        return None

    path = Path(normalized).expanduser()
    if not path.is_absolute():
        direct = (workspace / path)
        if direct.exists():
            path = direct
        else:
            matches = list(workspace.rglob(normalized))
            if len(matches) == 1:
                path = matches[0]
            else:
                path = direct

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return None

    if not resolved.is_file() or not _is_within_workspace(resolved, workspace):
        return None

    return resolved


def _is_within_workspace(path: Path, workspace: Path) -> bool:
    try:
        path.relative_to(workspace)
        return True
    except ValueError:
        return False


def _discover_dependencies(file_path: Path, workspace: Path) -> list[Path]:
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return []

    if file_path.suffix == ".py":
        return _discover_python_dependencies(file_path, content, workspace)
    if file_path.suffix == ".rs":
        return _discover_rust_dependencies(file_path, content, workspace)
    return []


def _discover_python_dependencies(file_path: Path, content: str, workspace: Path) -> list[Path]:
    modules = _PY_IMPORT_RE.findall(content) + _PY_FROM_IMPORT_RE.findall(content)
    discovered: list[Path] = []
    seen: set[Path] = set()
    for module in modules:
        dependency = _resolve_python_module(file_path, module, workspace)
        if dependency is None or dependency == file_path or dependency in seen:
            continue
        seen.add(dependency)
        discovered.append(dependency)
    return discovered


def _resolve_python_module(file_path: Path, module: str, workspace: Path) -> Path | None:
    parts = [part for part in module.split(".") if part]
    if not parts:
        return None

    same_dir = file_path.parent
    candidates = [
        same_dir.joinpath(*parts).with_suffix(".py"),
        workspace.joinpath(*parts).with_suffix(".py"),
        same_dir.joinpath(*parts, "__init__.py"),
        workspace.joinpath(*parts, "__init__.py"),
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_file() and _is_within_workspace(resolved, workspace):
            return resolved
    return None


def _discover_rust_dependencies(file_path: Path, content: str, workspace: Path) -> list[Path]:
    modules = _RUST_MOD_RE.findall(content)
    modules.extend(path.split("::")[0] for path in _RUST_USE_RE.findall(content))
    discovered: list[Path] = []
    seen: set[Path] = set()
    for module in modules:
        dependency = _resolve_rust_module(file_path, module, workspace)
        if dependency is None or dependency == file_path or dependency in seen:
            continue
        seen.add(dependency)
        discovered.append(dependency)
    return discovered


def _resolve_rust_module(file_path: Path, module: str, workspace: Path) -> Path | None:
    module = module.strip()
    if not module:
        return None

    candidates = [
        file_path.parent / f"{module}.rs",
        file_path.parent / module / "mod.rs",
        workspace / "src" / f"{module}.rs",
        workspace / "src" / module / "mod.rs",
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_file() and _is_within_workspace(resolved, workspace):
            return resolved
    return None


def _document_symbols_for(file_path: Path, query: LspQuery) -> list[dict[str, Any]]:
    try:
        response = query("document_symbols", str(file_path))
    except Exception as exc:  # pragma: no cover - defensive guard around external bridge
        return [{"name": f"LSP unavailable ({exc})", "kind": "error"}]

    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list):
        return []

    flattened: list[dict[str, Any]] = []
    for symbol in results:
        if not isinstance(symbol, dict):
            continue
        name = str(symbol.get("name", "")).strip()
        if not name:
            continue
        kind = str(symbol.get("kind", "symbol")).strip().lower() or "symbol"
        flattened.append({"name": name, "kind": kind})
        children = symbol.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict) and child.get("name"):
                    flattened.append(
                        {
                            "name": str(child["name"]).strip(),
                            "kind": str(child.get("kind", "symbol")).strip().lower() or "symbol",
                        }
                    )
    return flattened


def _format_symbol_summary(symbols: list[dict[str, Any]]) -> str:
    if not symbols:
        return "no symbols found"

    labels: list[str] = []
    for symbol in symbols[:6]:
        name = str(symbol.get("name", "")).strip()
        kind = str(symbol.get("kind", "symbol")).strip().lower() or "symbol"
        if not name:
            continue
        labels.append(f"{name} ({kind})")
    return ", ".join(labels) if labels else "no symbols found"


__all__ = [
    "build_prefetch_context",
    "extract_file_paths",
    "query_lsp_bridge",
]
