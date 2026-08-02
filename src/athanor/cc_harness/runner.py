"""Launch Claude Code as the solver agent and collect the run.

The flagship harness *is* the agent loop. Here the loop belongs to Claude Code,
and this module is only its launcher and recorder: it prepares the workspace,
starts ``claude -p`` inside it, translates the ``stream-json`` transcript into
Athanor's event vocabulary so existing consumers keep working, and scores the
result against ground truth the agent never had access to.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from athanor.data import load_task_json, resolve_task_path
from athanor.solver.events import EventCallback, EventType, OrchestratorEvent

from .config import CCRunConfig
from .scoring import contamination_scan, score_final
from .workspace import Workspace, build_workspace, ground_truth, workspace_env


class ClaudeCodeNotFound(RuntimeError):
    pass


@lru_cache(maxsize=1)
def claude_binary() -> str:
    path = os.environ.get("ATHANOR_CLAUDE_BIN") or shutil.which("claude")
    if not path:
        raise ClaudeCodeNotFound(
            "The `claude` CLI was not found on PATH. Install Claude Code, or point "
            "ATHANOR_CLAUDE_BIN at the binary."
        )
    return path


@lru_cache(maxsize=1)
def _help_text() -> str:
    try:
        completed = subprocess.run(
            [claude_binary(), "--help"], capture_output=True, text=True, timeout=60
        )
        return (completed.stdout or "") + (completed.stderr or "")
    except Exception:  # noqa: BLE001 - flag probing is best-effort
        return ""


def running_as_root() -> bool:
    getuid = getattr(os, "geteuid", None)
    return bool(getuid and getuid() == 0)


def resolve_permission_mode(mode: str) -> str:
    """Substitute a permission mode the CLI will actually accept.

    `bypassPermissions` maps to `--dangerously-skip-permissions`, which Claude
    Code refuses when running as root. A containerised benchmark harness is
    usually root, and the refusal arrives as an empty stream and a one-line
    stderr — so trade it for `acceptEdits`, which does the same job here because
    every tool the solver needs is already allow-listed.
    """
    if mode == "bypassPermissions" and running_as_root():
        return "acceptEdits"
    return mode


def supports_flag(flag: str) -> bool:
    """Whether the installed Claude Code advertises ``flag``.

    Flag availability moves between releases; probing keeps the harness working
    across versions instead of failing on an unknown option.
    """
    return flag in _help_text()


# ── argv construction ────────────────────────────────────────────────────────

def build_cli_args(workspace: Workspace, *, system_prompt_file: Path) -> list[str]:
    config = workspace.config
    args = [
        claude_binary(),
        "-p",
        workspace.initial_prompt,
        "--output-format",
        "stream-json",
        "--verbose",
    ]

    if config.bare and supports_flag("--bare"):
        args.append("--bare")
    if config.model:
        args += ["--model", config.model]
    if config.effort and supports_flag("--effort"):
        args += ["--effort", config.effort]
    if config.permission_mode:
        args += ["--permission-mode", resolve_permission_mode(config.permission_mode)]
    if config.tools and supports_flag("--tools"):
        args += ["--tools", ",".join(config.tools)]
    if config.allowed_tools:
        args += ["--allowedTools", ",".join(config.allowed_tools)]
    if config.disallowed_tools:
        args += ["--disallowed-tools", ",".join(config.disallowed_tools)]

    if supports_flag("--append-system-prompt-file"):
        args += ["--append-system-prompt-file", str(system_prompt_file)]
    else:
        args += ["--append-system-prompt", workspace.system_prompt]

    if config.stable_system_prompt and supports_flag("--exclude-dynamic-system-prompt-sections"):
        args.append("--exclude-dynamic-system-prompt-sections")
    if config.setting_sources and supports_flag("--setting-sources"):
        args += ["--setting-sources", config.setting_sources]
    if supports_flag("--strict-mcp-config"):
        args.append("--strict-mcp-config")
    if config.max_budget_usd and supports_flag("--max-budget-usd"):
        args += ["--max-budget-usd", str(config.max_budget_usd)]

    args += list(config.extra_cli_args)
    return args


# ── stream translation ───────────────────────────────────────────────────────

def stream_message_to_events(message: dict[str, Any]) -> Iterable[OrchestratorEvent]:
    """Translate one Claude Code stream-json message into Athanor events."""
    kind = message.get("type")

    if kind == "system":
        subtype = message.get("subtype")
        if subtype == "init":
            yield OrchestratorEvent(
                type=EventType.SYSTEM,
                content=f"session {message.get('session_id', '?')} started on {message.get('model', '?')}",
                metadata={"tools": message.get("tools"), "subtype": subtype},
            )
        elif subtype == "api_retry":
            yield OrchestratorEvent(
                type=EventType.SYSTEM,
                content=(
                    f"API retry {message.get('attempt')}/{message.get('max_retries')} "
                    f"({message.get('error')})"
                ),
                metadata={"subtype": subtype},
            )
        return

    if kind == "assistant":
        for block in (message.get("message") or {}).get("content") or []:
            block_type = block.get("type")
            if block_type == "text" and block.get("text"):
                yield OrchestratorEvent(type=EventType.TEXT, content=block["text"])
            elif block_type == "thinking" and block.get("thinking"):
                yield OrchestratorEvent(type=EventType.THINKING, content=block["thinking"])
            elif block_type == "tool_use":
                yield OrchestratorEvent(
                    type=EventType.TOOL_CALL,
                    content=str(block.get("name") or ""),
                    metadata={"tool_input": block.get("input"), "tool_use_id": block.get("id")},
                )
        return

    if kind == "user":
        for block in (message.get("message") or {}).get("content") or []:
            if block.get("type") != "tool_result":
                continue
            content = block.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            yield OrchestratorEvent(
                type=EventType.TOOL_RESULT,
                content=str(content or ""),
                metadata={"tool_use_id": block.get("tool_use_id"), "is_error": block.get("is_error")},
            )
        return

    if kind == "result":
        yield OrchestratorEvent(
            type=EventType.COMPLETE,
            content=str(message.get("result") or ""),
            metadata={
                "subtype": message.get("subtype"),
                "is_error": message.get("is_error"),
                "cost_usd": message.get("total_cost_usd"),
                "num_turns": message.get("num_turns"),
                "duration_ms": message.get("duration_ms"),
                "usage": message.get("usage"),
                "session_id": message.get("session_id"),
            },
        )


def default_event_printer(event: OrchestratorEvent) -> None:
    """Terse console tracing, in the spirit of ``default_cli_callback``."""
    if event.type is EventType.TOOL_CALL:
        detail = ""
        tool_input = event.metadata.get("tool_input") or {}
        if isinstance(tool_input, dict):
            detail = str(tool_input.get("command") or tool_input.get("file_path") or "")[:110]
        print(f"  → {event.content} {detail}".rstrip())
    elif event.type is EventType.SYSTEM:
        print(f"  · {event.content}")
    elif event.type is EventType.COMPLETE:
        cost = event.metadata.get("cost_usd")
        print(f"  ✓ finished in {event.metadata.get('num_turns')} turns"
              + (f", ${float(cost):.3f}" if cost is not None else ""))
    elif event.type is EventType.ERROR:
        print(f"  ! {event.content}")


# ── the run ──────────────────────────────────────────────────────────────────

def _launch(
    workspace: Workspace,
    *,
    stream_path: Path,
    log_path: Path,
    system_prompt_file: Path,
    event_callback: EventCallback | None,
) -> dict[str, Any]:
    """Run ``claude -p`` to completion, recording every stream message."""
    args = build_cli_args(workspace, system_prompt_file=system_prompt_file)
    timeout = float(workspace.config.wall_clock_timeout_s)

    outcome: dict[str, Any] = {
        "returncode": None,
        "timed_out": False,
        "result_message": None,
        "messages": 0,
        "malformed_lines": 0,
    }

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            args,
            cwd=str(workspace.root),
            env=workspace_env(),
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            bufsize=1,
        )

        killer = threading.Timer(timeout, _terminate, args=(process, outcome))
        killer.daemon = True
        killer.start()
        try:
            with stream_path.open("w", encoding="utf-8") as stream_handle:
                for line in process.stdout or []:
                    stream_handle.write(line)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        outcome["malformed_lines"] += 1
                        continue
                    outcome["messages"] += 1
                    if message.get("type") == "result":
                        outcome["result_message"] = message
                    if event_callback:
                        for event in stream_message_to_events(message):
                            event_callback(event)
            outcome["returncode"] = process.wait()
        finally:
            killer.cancel()
            if process.poll() is None:
                _terminate(process, outcome)

    outcome["elapsed_s"] = round(time.time() - started, 2)
    return outcome


def _terminate(process: subprocess.Popen, outcome: dict[str, Any]) -> None:
    outcome["timed_out"] = True
    try:
        process.terminate()
        process.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass


def run_task(
    task: str,
    *,
    out_dir: Path | str,
    config: CCRunConfig | None = None,
    dataset_root: str | None = None,
    dataset_split: str = "public_eval",
    event_callback: EventCallback | None = None,
    overwrite: bool = False,
    launch: bool = True,
) -> dict[str, Any]:
    """Solve one task end to end and write ``<out_dir>/<task_id>/result.json``.

    With ``launch=False`` the workspace and prompts are prepared but no solver is
    started — useful for driving the same workspace from an already-running
    Claude Code session (a sub-agent), or for inspecting what the agent will see.
    """
    config = config or CCRunConfig()
    puzzle_path = Path(resolve_task_path(task=task, split=dataset_split, dataset_root=dataset_root))
    puzzle_data = load_task_json(puzzle_path)
    task_id = puzzle_path.stem

    run_dir = Path(out_dir).resolve() / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    workspace = build_workspace(
        task_id=task_id,
        puzzle_data=puzzle_data,
        root=run_dir / "workspace",
        config=config,
        overwrite=overwrite,
    )
    system_prompt_file = run_dir / "system_prompt.md"
    system_prompt_file.write_text(workspace.system_prompt, encoding="utf-8")
    (run_dir / "initial_prompt.md").write_text(workspace.initial_prompt, encoding="utf-8")

    record: dict[str, Any] = {
        "task_id": task_id,
        "harness": "claude-code",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "puzzle_path": str(puzzle_path),
        "workspace": str(workspace.root),
        "config": config.to_dict(),
        "launched": launch,
    }

    stream_path = run_dir / "stream.jsonl"
    if launch:
        try:
            outcome = _launch(
                workspace,
                stream_path=stream_path,
                log_path=run_dir / "run.log",
                system_prompt_file=system_prompt_file,
                event_callback=event_callback,
            )
        except ClaudeCodeNotFound as exc:
            record["error"] = str(exc)
            outcome = {}
        record["run"] = outcome
        result_message = outcome.get("result_message") or {}
        record["cost_usd"] = result_message.get("total_cost_usd")
        record["num_turns"] = result_message.get("num_turns")
        record["duration_ms"] = result_message.get("duration_ms")
        record["session_id"] = result_message.get("session_id")
        record["usage"] = result_message.get("usage")
        if outcome.get("timed_out"):
            record["error"] = f"wall-clock timeout after {config.wall_clock_timeout_s:.0f}s"
            # The clock is a budget, not a destructor. If the ledger holds a
            # train-perfect submission the solver never got to accept, score it
            # rather than recording a zero for work that was done and verified.
            from .gate import salvage_unaccepted
            if salvage_unaccepted(workspace.root) is not None:
                record["salvaged"] = True
        elif result_message.get("is_error"):
            record["error"] = f"claude reported {result_message.get('subtype')}"
        elif not result_message and not record.get("error"):
            # No result message means the CLI died before producing one — a
            # rejected flag, a failed launch. Without the stderr this surfaces
            # as an empty stream and no explanation at all.
            stderr_tail = ""
            log_path = run_dir / "run.log"
            if log_path.is_file():
                stderr_tail = log_path.read_text(encoding="utf-8", errors="replace").strip()[-500:]
            record["error"] = (
                f"claude exited with code {outcome.get('returncode')} without a result message"
                + (f": {stderr_tail}" if stderr_tail else "")
            )

    record.update(collect_outcome(workspace, puzzle_data))
    record["integrity"] = contamination_scan(
        workspace_root=workspace.root,
        stream_path=stream_path if stream_path.is_file() else None,
        dataset_root=puzzle_path.parent,
        task_id=task_id,
    )
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Only a launched run has an outcome to record. Writing result.json for a
    # freshly prepared workspace would plant a zero-iteration record that any
    # later scoring pass could mistake for the finished run.
    if launch:
        (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def resume_task(
    run_dir: Path | str,
    *,
    config: CCRunConfig | None = None,
    dataset_root: str | None = None,
    dataset_split: str = "public_eval",
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    """Relaunch a solver into an existing workspace that was never accepted.

    Crash recovery works for the same reason compaction recovery does: the
    workspace is the state. The new agent inherits the iteration ledger, the
    verified invariants and NOTES.md, and spends only the budget that is left.
    """
    from . import prompt as prompt_mod
    from .workspace import load_workspace

    run_dir = Path(run_dir).resolve()
    workspace = load_workspace(run_dir / "workspace")
    state = workspace.read_state()

    if state.get("accepted"):
        return {**rescore_run(run_dir, dataset_root=dataset_root, dataset_split=dataset_split),
                "resumed": False, "resume_skipped": "already accepted"}

    used = len(state.get("iterations") or [])
    if used >= int(state.get("max_iterations") or 0):
        return {**rescore_run(run_dir, dataset_root=dataset_root, dataset_split=dataset_split),
                "resumed": False, "resume_skipped": "budget exhausted"}

    if config is not None:
        workspace.config = config
    workspace.initial_prompt = prompt_mod.build_resume_prompt(
        task_id=workspace.task_id,
        state=state,
        interpreter=state.get("interpreter"),
    )

    system_prompt_file = run_dir / "system_prompt.md"
    if not system_prompt_file.is_file():
        system_prompt_file.write_text(workspace.system_prompt, encoding="utf-8")

    attempt = len(list(run_dir.glob("stream*.jsonl")))
    outcome = _launch(
        workspace,
        stream_path=run_dir / f"stream.resume{attempt}.jsonl",
        log_path=run_dir / f"run.resume{attempt}.log",
        system_prompt_file=system_prompt_file,
        event_callback=event_callback,
    )
    record = rescore_run(run_dir, dataset_root=dataset_root, dataset_split=dataset_split)
    record["resumed"] = True
    record["resume_run"] = outcome
    # rescore_run merges the prior record, so the dead launch's `error` rides
    # along — and a resumed run that accepted and scored is a real result, not an
    # infrastructure loss. Re-derive the field from *this* launch, keeping the old
    # text under `previous_error` so the interruption is still on the record.
    previous_error = record.pop("error", None)
    if previous_error:
        record["previous_error"] = previous_error
    result_message = outcome.get("result_message") or {}
    if outcome.get("timed_out"):
        record["error"] = f"wall-clock timeout after {(config or workspace.config).wall_clock_timeout_s:.0f}s"
    elif result_message.get("is_error"):
        record["error"] = f"claude reported {result_message.get('subtype')}"
    result_message = outcome.get("result_message") or {}
    if result_message.get("total_cost_usd") is not None:
        record["cost_usd"] = float(record.get("cost_usd") or 0.0) + float(result_message["total_cost_usd"])
    (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def collect_outcome(workspace: Workspace, puzzle_data: dict[str, Any]) -> dict[str, Any]:
    """Read the gate's ledger out of a workspace and score it."""
    state = workspace.read_state()
    final = workspace.read_final()
    iterations = state.get("iterations") or []
    return {
        "accepted": bool(state.get("accepted")),
        "iterations_used": len(iterations),
        "max_iterations": state.get("max_iterations"),
        "train_perfect": bool(iterations and iterations[-1].get("all_train_correct")),
        "best_effort": bool((state.get("accepted") or {}).get("best_effort")),
        "confidence": (state.get("accepted") or {}).get("confidence"),
        "iteration_log": iterations,
        "verified_invariants": len(final.get("verified_invariants") or []) if final else 0,
        "hardcoding_findings": (final or {}).get("hardcoding_findings") or [],
        "hypothesis": (final or {}).get("hypothesis") or "",
        "code": (final or {}).get("code") or "",
        "score": score_final(final, ground_truth(puzzle_data)),
    }


def rescore_run(run_dir: Path | str, *, dataset_root: str | None = None,
                dataset_split: str = "public_eval") -> dict[str, Any]:
    """Recompute a run's outcome from its workspace on disk.

    Works whether or not the run was started by ``athanor cc run``: a workspace
    driven by hand, or by a sub-agent inside an existing Claude Code session,
    carries everything needed to score it. The workspace is the record.
    """
    from .workspace import load_workspace

    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "result.json"
    record: dict[str, Any] = {}
    if result_path.is_file():
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = {}

    workspace = load_workspace(run_dir / "workspace")
    record.setdefault("task_id", workspace.task_id)
    record.setdefault("harness", "claude-code")
    record.setdefault("workspace", str(workspace.root))
    record.setdefault("config", workspace.config.to_dict())

    puzzle_path = Path(
        record.get("puzzle_path")
        or resolve_task_path(task=workspace.task_id, split=dataset_split, dataset_root=dataset_root)
    )
    record["puzzle_path"] = str(puzzle_path)
    record.update(collect_outcome(workspace, load_task_json(puzzle_path)))

    stream_path = run_dir / "stream.jsonl"
    record["integrity"] = contamination_scan(
        workspace_root=workspace.root,
        stream_path=stream_path if stream_path.is_file() else None,
        dataset_root=puzzle_path.parent,
        task_id=workspace.task_id,
    )
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def discover_runs(out_dir: Path | str) -> list[Path]:
    """Run directories under ``out_dir``, with or without a result.json."""
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        return []
    return sorted(
        path
        for path in out_dir.iterdir()
        if path.is_dir() and ((path / "result.json").is_file() or (path / "workspace").is_dir())
    )


def run_batch(
    tasks: list[str],
    *,
    out_dir: Path | str,
    config: CCRunConfig | None = None,
    dataset_root: str | None = None,
    dataset_split: str = "public_eval",
    event_callback: EventCallback | None = None,
    overwrite: bool = False,
    resume_incomplete: bool = False,
    on_task_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Solve tasks one after another.

    Sequential on purpose: Athanor's cost figures assume a batch that shares the
    provider-side prompt cache, and interleaved runs make per-task cost
    attribution unreadable.

    With ``resume_incomplete`` the batch becomes restartable after the process
    dies under it, which on some hosts it will. A task that already has a
    ``result.json`` is left alone; a task with a workspace but no result — the
    shape a killed solver leaves behind — is handed to :func:`resume_task`, which
    inherits its iteration ledger and spends only the budget that is left. Only
    genuinely untouched tasks are run from scratch. Relaunching the same shard
    command therefore costs the remaining work rather than the whole shard.
    """
    if overwrite and resume_incomplete:
        raise ValueError("overwrite and resume_incomplete are contradictory: pick one")

    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task}")
        try:
            run_dir = None
            if resume_incomplete:
                task_id = Path(
                    resolve_task_path(task=task, split=dataset_split, dataset_root=dataset_root)
                ).stem
                run_dir = Path(out_dir).resolve() / task_id

            if run_dir is not None and (run_dir / "workspace").is_dir():
                # Ask the workspace, not the result file. A run killed by the
                # wall-clock timeout still gets a result.json written — scored 0,
                # accepted False — which is indistinguishable from an honest zero
                # if you key on the file's existence. The gate ledger is the only
                # thing that knows whether the solver was finished or interrupted,
                # and resume_task already reads it: it returns early for an
                # accepted run or an exhausted budget, and relaunches otherwise.
                record = resume_task(
                    run_dir,
                    config=config,
                    dataset_root=dataset_root,
                    dataset_split=dataset_split,
                    event_callback=event_callback,
                )
                if record.get("resume_skipped"):
                    record["batch_skipped"] = record["resume_skipped"]
                    print(f"  skipped: {record['resume_skipped']}")
                else:
                    print("  resumed: budget left and never accepted")
            elif run_dir is not None and (run_dir / "result.json").is_file():
                record = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                record["batch_skipped"] = "result.json present, no workspace to resume"
                print("  skipped: result present, workspace gone")
            else:
                record = run_task(
                    task,
                    out_dir=out_dir,
                    config=config,
                    dataset_root=dataset_root,
                    dataset_split=dataset_split,
                    event_callback=event_callback,
                    overwrite=overwrite,
                )
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the batch
            record = {"task_id": task, "error": f"{type(exc).__name__}: {exc}"}
        results.append(record)
        if on_task_done:
            on_task_done(record)
    return results
