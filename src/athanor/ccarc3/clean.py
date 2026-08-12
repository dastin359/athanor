"""Fail-closed single-game entrypoint for benchmark-clean ARC-AGI-3 runs.

The ordinary :mod:`athanor.ccarc3` API intentionally remains usable without
baseline ablation.  This module is the explicit boundary for a scored run: it
installs the existing allowlisting proxy and baseline strip, verifies the
solver-facing workspace before every Codex launch, and restores process-global
patches when the run ends.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from . import session as sess
from .client import HIDE_BASELINES_ENV
from .session import Ccarc3Config, Workspace

DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "athanor-ccarc3-codex" / "clean-single"
"""Outside the source checkout so repository instructions cannot reach a solver."""

_MISSING = object()
_ISOLATION_FLAGS = ("--ignore-user-config", "--ignore-rules", "--skip-git-repo-check")


def _source_checkout() -> Path:
    """Return the checkout containing this module (or its installation prefix)."""
    return Path(__file__).resolve().parents[3]


def _load_ablation() -> Any:
    """Load the audited strip from ``tools/`` without import-time side effects."""
    try:
        return importlib.import_module("ablate_baselines")
    except ModuleNotFoundError as exc:
        tools = _source_checkout() / "tools"
        script = tools / "ablate_baselines.py"
        if not script.is_file():
            raise RuntimeError(
                "benchmark-clean ARC support requires tools/ablate_baselines.py; "
                "run this command from an Athanor source checkout"
            ) from exc
        sys.path.insert(0, str(tools))
        return importlib.import_module("ablate_baselines")


def _containing_repository(path: Path) -> Path | None:
    """Find a Git worktree containing ``path`` without invoking Git."""
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        marker = candidate / ".git"
        # Managed sandboxes may mount an empty read-only ``.git`` sentinel at
        # broad writable roots such as /tmp.  It is not a repository.  A real
        # checkout has a HEAD in its directory, while a linked worktree uses a
        # ``gitdir: ...`` file.
        directory_repo = marker.is_dir() and (marker / "HEAD").is_file()
        file_repo = marker.is_file()
        if directory_repo or file_repo:
            return candidate
    return None


def assert_external_workspace(config: Ccarc3Config) -> Path:
    """Resolve the solver workspace and refuse any location inside a repository."""
    root = (Path(config.out_dir) / config.game_id).resolve()
    repository = _containing_repository(root)
    if repository is not None:
        raise RuntimeError(
            f"clean ARC workspace {root} is inside Git repository {repository}; "
            "choose --out-dir outside every repository"
        )
    return root


def _contains_secret(path: Path, secret: bytes) -> bool:
    """Search one file without loading a potentially large trace into memory."""
    overlap = max(0, len(secret) - 1)
    tail = b""
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                block = tail + chunk
                if secret in block:
                    return True
                tail = block[-overlap:] if overlap else b""
    except OSError:
        return False
    return False


def _workspace_secret_files(workspace: Workspace) -> list[str]:
    """Name files containing a parent credential, never the credential itself."""
    root = workspace.root.resolve()
    secrets = {
        value.encode()
        for name in ("ARC_API_KEY", "ARCPRIZE_API_KEY")
        if (value := os.environ.get(name))
    }
    if workspace.config.api_key:
        secrets.add(workspace.config.api_key.encode())
    if not secrets:
        return []
    leaked: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file() and any(_contains_secret(path, secret) for secret in secrets):
            leaked.append(path.relative_to(root).as_posix())
    return sorted(leaked)


def _has_disable_memories(args: list[str]) -> bool:
    return any(args[i:i + 2] == ["--disable", "memories"] for i in range(len(args) - 1))


def assert_clean_launch(
    workspace: Workspace, args: list[str], *, ablation: Any | None = None,
) -> None:
    """Refuse a solver launch unless every clean-run invariant still holds."""
    failures: list[str] = []
    root = workspace.root.resolve()
    symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        failures.append(f"workspace contains symlinks: {', '.join(symlinks)}")

    repository = _containing_repository(root)
    if repository is not None:
        failures.append(f"workspace is inside Git repository {repository}")

    env = workspace.env
    for name in ("ARC_API_KEY", "ARCPRIZE_API_KEY", "CCARC3_MAX_ACTIONS"):
        if name in env:
            failures.append(f"child environment contains {name}")
    if env.get(HIDE_BASELINES_ENV, "").lower() not in {"1", "true"}:
        failures.append(f"child environment does not set {HIDE_BASELINES_ENV}=true")
    arc_root = env.get("CCARC3_ARC_ROOT", "")
    parsed = urlparse(arc_root)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
        failures.append("child CCARC3_ARC_ROOT is not a loopback proxy URL")
    if "CCARC3_PROXY_URL" in env:
        failures.append("child environment contains the parent proxy URL")

    meta = root / "meta.json"
    if not meta.is_symlink():
        try:
            metadata = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"meta.json is unavailable or invalid ({type(exc).__name__})")
        else:
            for name in ("baseline_actions", "action_budget"):
                if name in metadata:
                    failures.append(f"meta.json contains {name}")

    session_file = root / "session.py"
    if not session_file.is_symlink():
        try:
            session_text = session_file.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"session.py is unavailable ({type(exc).__name__})")
        else:
            if "baseline_actions=()," not in session_text:
                failures.append("session.py still exposes baseline actions")

    leaked = _workspace_secret_files(workspace)
    if leaked:
        failures.append(f"ARC credential appears in workspace files: {', '.join(leaked)}")

    for flag in _ISOLATION_FLAGS:
        if flag not in args:
            failures.append(f"Codex launch is missing {flag}")
    if not _has_disable_memories(args):
        failures.append("Codex launch does not disable memories")
    if args[1:3] not in (["exec", "--json"], ["exec", "resume"]):
        failures.append("command is neither an initial nor resumed Codex exec launch")

    if ablation is not None:
        try:
            ablation.assert_installed()
            proxy = ablation.proxy_for(workspace.info.game_id)
        except Exception as exc:  # noqa: BLE001 -- any broken boundary is fatal
            failures.append(f"baseline proxy is not installed ({type(exc).__name__})")
        else:
            if arc_root != proxy.url:
                failures.append("child CCARC3_ARC_ROOT is not this game's proxy")
            expected_budget = workspace.info.suggested_budget(
                workspace.config.budget_multiple
            )
            if proxy.max_actions != expected_budget:
                failures.append("this game's proxy does not enforce the configured cap")

    if failures:
        raise RuntimeError("clean ARC launch refused: " + "; ".join(failures))


@contextmanager
def _installed_boundary() -> Iterator[Any]:
    """Install the audited strip/proxy temporarily and verify the installation."""
    import athanor.ccarc3 as package

    ablation = _load_ablation()
    old_session_builder = sess.build_workspace
    old_package_builder = package.build_workspace
    old_proxy_url = os.environ.get("CCARC3_PROXY_URL", _MISSING)
    try:
        # A pre-existing URL proves nothing about this process. Force install()
        # to bind and close its own probe before accepting the boundary.
        os.environ.pop("CCARC3_PROXY_URL", None)
        ablation.install()
        ablation.assert_installed()
        yield ablation
    finally:
        sess.build_workspace = old_session_builder
        package.build_workspace = old_package_builder
        if old_proxy_url is _MISSING:
            os.environ.pop("CCARC3_PROXY_URL", None)
        else:
            os.environ["CCARC3_PROXY_URL"] = old_proxy_url


def run_clean_game(config: Ccarc3Config, *, max_nudges: int = 3) -> dict[str, Any]:
    """Run one game through the clean boundary, restoring global state afterwards."""
    assert_external_workspace(config)
    old_out_dir = config.out_dir
    old_validator = config.launch_validator
    config.out_dir = Path(config.out_dir).resolve()
    old_nudges = os.environ.get("CCARC3_MAX_NUDGES", _MISSING)
    os.environ["CCARC3_MAX_NUDGES"] = str(max(0, max_nudges))
    try:
        with _installed_boundary() as ablation:
            config.launch_validator = lambda workspace, args: assert_clean_launch(
                workspace, args, ablation=ablation
            )
            try:
                return sess.run_game(config)
            finally:
                ablation.release_proxy(config.game_id)
    finally:
        config.out_dir = old_out_dir
        config.launch_validator = old_validator
        if old_nudges is _MISSING:
            os.environ.pop("CCARC3_MAX_NUDGES", None)
        else:
            os.environ["CCARC3_MAX_NUDGES"] = old_nudges
