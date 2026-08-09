"""The orphan sweep must select solvers by argv, never by substring.

`stop_politely` waited on solvers with the argv-element-exact
`pids_of '.*/claude'`, then killed orphans with `pgrep -f claude` -- the
substring form, in the same function, eleven lines below the comment explaining
why it is wrong.

The substring is not marginally looser, it is unrelated: the scratchpad lives at
`/tmp/claude-0/...`, so ANY process carrying that path in its argv matches.
Measured on the live box 2026-08-09: `pgrep -f claude` -> 6 pids (the session's
own CLI, context_watch.py, the environment manager, a shell);
`pids_of '.*/claude'` -> 0.

Only the cwd gate kept those alive, and it is the wrong thing to rely on: it
admits any process whose cwd sits under the work tree -- a proofread pass, an
analysis script, an operator's shell -- and kills it while reporting "orphaned
solver", which is also a lie about what died.
"""

from __future__ import annotations

import os
import pathlib
import re
import signal
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "tools" / "supervisor.sh"
SRC = SUPERVISOR.read_text(encoding="utf-8")


def _fn(name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{.*?^\}}$", SRC, re.S | re.M)
    assert m, f"supervisor.sh no longer defines {name}()"
    return m.group(0)


def _orphan_sweep() -> str:
    """The kill loop out of stop_politely, without its two-hour wait.

    **Anchored on the kill, not on the loop header.** Both loops in
    stop_politely now open with `for pid in $(pids_of '.*/claude'); do` -- that
    is the fix -- so slicing from the FIRST one swallowed the wait loop and its
    `for _ in $(seq 1 240); do ... sleep 30`. Four cases here then passed
    without the kill loop ever running: the sweep sat in the wait, killed
    nothing, and "the bystander survived" was true for the wrong reason.

    So: find the kill itself, then take the loop that encloses it. The `sleep`
    assertion below is what stops this from silently regressing again.
    """
    body = _fn("stop_politely")
    kill_at = body.index("killed orphaned solver")
    start = body.rindex("for pid in $(pids_of", 0, kill_at)
    start = body.rindex("\n", 0, start) + 1
    end = body.index("    done", kill_at)
    snippet = body[start:end] + "    done\n"
    assert "sleep" not in snippet, (
        "the extracted orphan sweep contains a sleep -- it has swallowed the "
        "two-hour wait loop, and every case below would pass by not running:\n"
        + snippet
    )
    assert "seq 1 240" not in snippet, "the wait loop was captured"
    return snippet


def _spawn(exe: pathlib.Path, cwd: pathlib.Path, *args: str) -> subprocess.Popen:
    proc = subprocess.Popen([str(exe), *args], cwd=str(cwd),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    return proc


def _alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


@pytest.fixture
def bench(tmp_path: pathlib.Path):
    ws = tmp_path / "clean_rollouts" / "g1-fake" / "attempt_1" / "ws"
    ws.mkdir(parents=True)
    binv = tmp_path / "bin"
    binv.mkdir()

    solver = binv / "claude"            # argv[0] ends in /claude -> a solver
    solver.write_text("#!/bin/bash\nsleep 120\n", encoding="utf-8")
    solver.chmod(0o755)

    bystander = binv / "python3"        # NOT a solver, but argv mentions claude
    bystander.write_text("#!/bin/bash\nsleep 120\n", encoding="utf-8")
    bystander.chmod(0o755)

    procs: list[subprocess.Popen] = []
    yield ws, solver, bystander, procs
    for p in procs:
        try:
            os.kill(p.pid, signal.SIGKILL)
        except OSError:
            pass


def _sweep(work: str) -> subprocess.CompletedProcess:
    script = f'WORK={work}\n' + _fn("pids_of") + "\n" + _orphan_sweep()
    return subprocess.run(["/bin/bash", "-c", script],
                          capture_output=True, text=True, timeout=60)


def test_an_orphaned_solver_is_killed(bench) -> None:
    """The other direction, or the sweep could kill nothing and pass."""
    ws, solver, _bystander, procs = bench
    proc = _spawn(solver, ws)
    procs.append(proc)
    out = _sweep("clean_rollouts")
    time.sleep(0.5)
    assert not _alive(proc), (
        f"an orphaned solver survived the sweep: {out.stdout!r} {out.stderr!r}"
    )
    assert "killed orphaned solver" in out.stdout


def test_a_bystander_naming_the_scratchpad_is_not_killed(bench) -> None:
    """The bug: its argv contains 'claude' only because of the scratchpad path.

    This is exactly the shape of every real process on the box -- the scratchpad
    is at /tmp/claude-0/... so it appears in the command line of anything the
    harness runs there.
    """
    ws, _solver, bystander, procs = bench
    proc = _spawn(bystander, ws, "--scratch=/tmp/claude-0/-home-user-athanor/x")
    procs.append(proc)
    out = _sweep("clean_rollouts")
    time.sleep(0.5)
    assert _alive(proc), (
        "a non-solver whose argv merely mentions the scratchpad path was killed "
        f"and reported as an orphaned solver: {out.stdout!r}"
    )


def test_a_finished_solver_is_left_alone(bench) -> None:
    """`result.json` present means it is finishing up, not orphaned."""
    ws, solver, _bystander, procs = bench
    (ws / "result.json").write_text("{}", encoding="utf-8")
    proc = _spawn(solver, ws)
    procs.append(proc)
    _sweep("clean_rollouts")
    time.sleep(0.5)
    assert _alive(proc), "a solver that had already written result.json was killed"


def test_a_solver_outside_the_work_tree_is_left_alone(bench, tmp_path) -> None:
    ws, solver, _bystander, procs = bench
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    proc = _spawn(solver, elsewhere)
    procs.append(proc)
    _sweep("clean_rollouts")
    time.sleep(0.5)
    assert _alive(proc), "a solver working outside the sweep directory was killed"


def test_the_sweep_never_uses_pgrep() -> None:
    """Pinned by name: this trap has bitten the project three times."""
    code = "\n".join(ln for ln in SRC.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "pgrep" not in code, (
        "supervisor.sh uses pgrep again -- substring matching on the full "
        "command line selects this session's own processes"
    )
