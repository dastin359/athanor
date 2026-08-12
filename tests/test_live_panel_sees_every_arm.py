"""The live panel must show every running game, not the arms named in 2026-08.

The operator instruction this panel exists to satisfy is *show all currently
running games*. `live_games()` filtered solver cwds with a literal allowlist --
`"clean_rollouts" not in cwd and "scratchpad/runs" not in cwd` -- so a solver
working in `ablate_nobaseline/`, or in any arm named later, was invisible. The
panel then rendered "nothing running" rather than "I cannot see it", which is
the failure mode that looks exactly like the healthy one.

Same shape as the batch list in snapshot_results.py that silently omitted 25
banked games: a list of the arms of the day, going stale without a symptom.

These spawn real processes with real cwds and call the real function.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    not pathlib.Path("/proc").is_dir(),
    reason="live-panel process discovery requires Linux procfs",
)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_trace_audit as bta  # noqa: E402


@pytest.fixture
def fake_solver(tmp_path: pathlib.Path, monkeypatch):
    """A `claude` process sitting in a workspace under a fixture scratchpad."""
    scratch = tmp_path / "scratchpad"
    binv = tmp_path / "bin"
    binv.mkdir(parents=True)
    exe = binv / "codex"
    exe.write_text("#!/bin/bash\nsleep 120\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("CCARC3_SCRATCH", str(scratch))

    started: list[subprocess.Popen] = []

    def spawn(arm: str, *, with_trace: bool = True) -> pathlib.Path:
        ws = scratch / arm / "ar25-fixture" / "ws"
        ws.mkdir(parents=True)
        if with_trace:
            (ws / "trace.jsonl").write_text(
                json.dumps({"level": 3, "action": "ACTION1"}) + "\n", encoding="utf-8")
        proc = subprocess.Popen([str(exe)], cwd=str(ws),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(proc)
        time.sleep(0.4)
        return ws

    yield spawn, scratch

    for p in started:
        try:
            os.kill(p.pid, signal.SIGKILL)
        except OSError:
            pass


def _names(games: list[dict]) -> list[str]:
    return [str(g) for g in games]


def test_a_solver_in_the_sweep_arm_is_seen(fake_solver) -> None:
    """The arm the old allowlist did cover -- it must not regress."""
    spawn, _ = fake_solver
    spawn("clean_rollouts")
    assert bta.live_games(), "a solver in clean_rollouts was not seen"


def test_a_solver_in_another_arm_is_seen(fake_solver) -> None:
    """The bug: `ablate_nobaseline` was not on the allowlist."""
    spawn, _ = fake_solver
    spawn("ablate_nobaseline")
    games = bta.live_games()
    assert games, (
        "a solver in ablate_nobaseline was invisible to the live panel -- the "
        "cwd filter is an allowlist of arm names"
    )


def test_a_solver_in_an_arm_named_later_is_seen(fake_solver) -> None:
    """The property that matters: no list to go stale.

    An allowlist can be extended to today's arms and be wrong again next month.
    This asserts the rule is structural, not enumerated.
    """
    spawn, _ = fake_solver
    spawn("an_arm_invented_after_this_test_was_written")
    assert bta.live_games(), (
        "an arm not named in the source was invisible -- the filter still "
        "enumerates arms instead of scoping to the scratchpad"
    )


def test_a_claude_process_outside_the_scratchpad_is_ignored(fake_solver, tmp_path) -> None:
    """The other direction, or the filter could accept everything.

    This session's own CLI is a `claude` process; the panel must not list it.

    **The outside workspace carries a `trace.jsonl` on purpose.** Without one,
    the trace requirement rejects it and the scope check is never exercised --
    the first draft of this case omitted it, and deleting the scope check
    entirely then passed the whole file. A test whose subject is filtered out by
    a different guard is not testing its subject.
    """
    spawn, _scratch = fake_solver
    outside = tmp_path / "not_the_scratchpad"
    outside.mkdir()
    (outside / "trace.jsonl").write_text(
        json.dumps({"level": 1, "action": "ACTION1"}) + "\n", encoding="utf-8")
    exe = tmp_path / "bin" / "codex"
    proc = subprocess.Popen([str(exe)], cwd=str(outside),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.4)
    try:
        assert not bta.live_games(), (
            "a claude process outside the scratchpad was listed as a live game, "
            "even though it holds a trace.jsonl -- only the scope check can "
            "reject this, and it is gone"
        )
    finally:
        os.kill(proc.pid, signal.SIGKILL)


def test_a_workspace_without_a_trace_is_not_a_live_game(fake_solver) -> None:
    """No trace means no game in progress; it is a workspace being set up."""
    spawn, _ = fake_solver
    spawn("clean_rollouts", with_trace=False)
    assert not bta.live_games(), (
        "a workspace with no trace.jsonl was reported as a live game"
    )


def test_the_source_no_longer_enumerates_arm_names() -> None:
    """Pinned by name: the literal allowlist must not come back."""
    src = (REPO / "tools" / "build_trace_audit.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    fn_start = code.index("def live_games(")
    fn = code[fn_start:fn_start + 3000]
    assert '"clean_rollouts" not in' not in fn, (
        "live_games filters by arm name again"
    )
