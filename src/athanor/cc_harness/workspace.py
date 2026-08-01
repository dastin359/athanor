"""Run-workspace construction.

The workspace *is* the harness state. Claude Code owns the conversation, so
everything Athanor kept in orchestrator memory — puzzle data, iteration ledger,
verified invariants, distilled research state — lives on disk here instead,
where it survives context compaction and can be audited after the fact.

Ground truth never enters the workspace. Test outputs are stripped when
``task.json`` is written and are held by the harness for post-run scoring only.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import prompt as prompt_mod
from .config import CCRunConfig

ASSETS = Path(__file__).parent / "assets"


@dataclass
class Workspace:
    """A prepared puzzle workspace and the prompts that go with it."""

    root: Path
    task_id: str
    config: CCRunConfig
    system_prompt: str
    initial_prompt: str
    image_files: list[str]

    @property
    def state_path(self) -> Path:
        return self.root / ".athanor" / "state.json"

    @property
    def final_path(self) -> Path:
        return self.root / ".athanor" / "final.json"

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def read_final(self) -> dict[str, Any] | None:
        if not self.final_path.is_file():
            return None
        return json.loads(self.final_path.read_text(encoding="utf-8"))


def strip_test_outputs(puzzle_data: dict[str, Any]) -> dict[str, Any]:
    """Training pairs in full; test entries reduced to their inputs."""
    return {
        "train": [
            {"input": pair["input"], "output": pair["output"]}
            for pair in puzzle_data.get("train") or []
        ],
        "test": [{"input": pair["input"]} for pair in puzzle_data.get("test") or []],
    }


def ground_truth(puzzle_data: dict[str, Any]) -> list[list[list[int]] | None]:
    """Expected test outputs, held outside the workspace for scoring."""
    return [pair.get("output") for pair in puzzle_data.get("test") or []]


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _render_images(root: Path, puzzle_data: dict[str, Any]) -> list[str]:
    """Render every grid to a PNG the agent can open with the Read tool."""
    try:
        from athanor.solver.grid_visualizer import render_grid_to_image
    except Exception:  # noqa: BLE001 - Pillow missing, or a partial install
        return []

    images_dir = root / "task" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def emit(name: str, grid: list[list[int]]) -> None:
        data = render_grid_to_image(grid)
        if data is None:
            return
        (images_dir / name).write_bytes(data)
        written.append(f"task/images/{name}")

    for idx, pair in enumerate(puzzle_data.get("train") or []):
        emit(f"train_{idx}_input.png", pair["input"])
        emit(f"train_{idx}_output.png", pair["output"])
    for idx, pair in enumerate(puzzle_data.get("test") or []):
        emit(f"test_{idx}_input.png", pair["input"])
    return written


def athanor_src_root() -> str:
    """Directory to put on ``sys.path`` so a workspace can import athanor."""
    return str(Path(__file__).resolve().parents[2])


def build_workspace(
    *,
    task_id: str,
    puzzle_data: dict[str, Any],
    root: Path | str,
    config: CCRunConfig | None = None,
    overwrite: bool = False,
) -> Workspace:
    """Materialise a complete, self-contained puzzle workspace at ``root``."""
    config = config or CCRunConfig()
    root = Path(root).resolve()

    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Workspace {root} already exists and is not empty.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for subdir in ("task", "explore", "solution", ".athanor", ".claude/hooks"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    visible = strip_test_outputs(puzzle_data)
    _write(root / "task" / "task.json", json.dumps({"task_id": task_id, **visible}, indent=1))
    _write(root / "task" / "grids.md", prompt_mod.render_task_markdown(task_id, visible))

    image_files = _render_images(root, visible) if config.visual else []

    shutil.copyfile(ASSETS / "arc_toolkit.py", root / "arc.py")

    gate_source = (ASSETS / "gate_shim.py").read_text(encoding="utf-8")
    _write(root / "gate.py", gate_source.replace("__ATHANOR_SRC__", athanor_src_root()), executable=True)

    hook_source = (ASSETS / "on_compact.sh").read_text(encoding="utf-8")
    hook_path = root / ".claude" / "hooks" / "on_compact.sh"
    _write(
        hook_path,
        hook_source.replace("__WORKSPACE__", str(root)).replace("__PYTHON__", sys.executable),
        executable=True,
    )
    _write(root / ".claude" / "settings.json", prompt_mod.settings_json(hook_script=hook_path))

    _write(
        root / "CLAUDE.md",
        prompt_mod.build_workspace_claude_md(task_id=task_id, puzzle_data=visible, config=config),
    )
    _write(root / "NOTES.md", prompt_mod.build_notes_seed(task_id))
    _write(
        root / "explore" / "README.md",
        "Scratch scripts live here. One question per script, named for the question.\n"
        "Run them with `python explore/<name>.py`. Nothing here is budgeted or recorded;\n"
        "only `arc.verify()` results and `gate.py submit` are.\n",
    )

    state = {
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness": "claude-code",
        "max_iterations": config.max_iterations,
        "best_effort_iterations": config.best_effort_iterations,
        "min_hypothesis_chars": config.min_hypothesis_chars,
        "max_test_predictions": config.max_test_predictions,
        "solve_timeout_s": config.solve_timeout_s,
        "iterations": [],
        "accepted": None,
        "last_hypothesis_sha": "",
        "last_code_sha": "",
        "config": config.to_dict(),
    }
    _write(root / ".athanor" / "state.json", json.dumps(state, indent=2))
    _write(root / ".athanor" / "invariants.jsonl", "")

    system_prompt = prompt_mod.build_system_prompt()
    initial_prompt = prompt_mod.build_initial_prompt(
        task_id=task_id,
        puzzle_data=visible,
        config=config,
        image_files=image_files,
    )
    # Written next to the workspace so any Claude Code agent — the subprocess
    # launcher, or a sub-agent inside an existing session — can be pointed at
    # the same two prompts.
    _write(root / ".athanor" / "system_prompt.md", system_prompt)
    _write(root / ".athanor" / "initial_prompt.md", initial_prompt)

    return Workspace(
        root=root,
        task_id=task_id,
        config=config,
        system_prompt=system_prompt,
        initial_prompt=initial_prompt,
        image_files=image_files,
    )


def load_workspace(root: Path | str) -> Workspace:
    """Reopen an existing workspace (for scoring or inspection)."""
    root = Path(root).resolve()
    state = json.loads((root / ".athanor" / "state.json").read_text(encoding="utf-8"))
    config = CCRunConfig.from_dict(state.get("config") or {})

    def _maybe(relative: str) -> str:
        path = root / ".athanor" / relative
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    images_dir = root / "task" / "images"
    images = (
        sorted(f"task/images/{p.name}" for p in images_dir.glob("*.png"))
        if images_dir.is_dir()
        else []
    )
    return Workspace(
        root=root,
        task_id=str(state.get("task_id") or root.name),
        config=config,
        system_prompt=_maybe("system_prompt.md"),
        initial_prompt=_maybe("initial_prompt.md"),
        image_files=images,
    )


def relative_to_cwd(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def workspace_env() -> dict[str, str]:
    """Environment for a solver process: the dataset root is deliberately removed."""
    env = dict(os.environ)
    env.pop("ARC_DATA_ROOT", None)
    return env
