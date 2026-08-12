"""`kill_orphan_solvers` must find every orphan of *this* sweep and nothing else.

Two defects, both found by coverage showing the whole `/proc` loop never
executed.

**1. It matched sweeps by substring.** `if str(OUT) not in str(cwd)` — and
`clean_rollouts` is a prefix of `clean_rollouts_validate` and of
`clean_rollouts_submission`. A driver on the default sweep therefore matched, and
would `SIGTERM` the process group of, an orphan belonging to a differently-named
sweep running beside it. `CCARC3_SWEEP_DIR` exists precisely to let two sweeps
share a box, and the supervisor's own comments explain that a driver walking the
wrong directory "sees nothing pending, kills mid-game".

**2. It split `/proc/<pid>/stat` across the whole line.** `comm` is field 2, it
is arbitrary bytes chosen by the process, and node sets it from `process.title`.
One space in it shifts every field right, so `stat[3]` reads a fragment of the
name instead of the ppid, `!= "1"` never matches, and the orphan is left running.
The cleanup then fails by finding nothing, which looks exactly like a clean box —
and the field beside the one it misreads is the pgid handed to `killpg`.

The consequence measured before this fix: restarting the driver leaked two
orphaned trees in one session, on `ka59` and `wa30`, both still writing to their
traces minutes later, one of them duplicating a game the new driver had already
restarted — two solvers on one environment, two scorecards, no benefit.

These spawn real processes and let the real function decide.
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    not pathlib.Path("/proc").is_dir(),
    reason="orphan process ownership is implemented against Linux procfs",
)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import clean_rollouts as cr  # noqa: E402


def _spawn(cwd: pathlib.Path, *, orphan: bool, title: str = "solver") -> subprocess.Popen:
    """A process sitting in `cwd`. `orphan=True` reparents it to init.

    The double fork is what makes it an orphan: `setsid` alone gives a child a
    new session, not a new parent.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    script = cwd / f"{title}.sh"
    script.write_text("#!/bin/bash\nsleep 120\n", encoding="utf-8")
    script.chmod(0o755)
    if orphan:
        # The launcher gets its own session too. It has to stay alive -- an
        # exiting launcher orphans both forms and the fixture stops
        # discriminating -- and while alive its cwd is under the sweep, so a
        # mutant that drops the `ppid == 1` guard would `killpg` whatever group
        # it is in. Leaving it in the runner's group means that mutant kills
        # pytest instead of failing a test.
        return subprocess.Popen(
            ["bash", "-c", f'( setsid "{script}" >/dev/null 2>&1 & ) ; sleep 120'],
            cwd=cwd, start_new_session=True)
    # `start_new_session=True`, matching how a real solver is spawned
    # (`session.py:730` — "makes the child a group leader, so one `killpg`"
    # takes the whole tree). It also keeps a mutant that drops the `ppid == 1`
    # guard from reaching *this* process group: without it the tested function
    # SIGTERMs the pytest run that is testing it, the failure is real but the
    # runner dies before reporting it, and the mutated source is left in the
    # tree. Measured, once, the hard way.
    return subprocess.Popen([str(script)], cwd=cwd, start_new_session=True)


def _pids_under(root: pathlib.Path) -> set[int]:
    out = set()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cwd = (entry / "cwd").resolve()
        except (OSError, PermissionError):
            continue
        if cwd == root or root in cwd.parents:
            out.add(int(entry.name))
    return out


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    out = tmp_path / "scratchpad" / "clean_rollouts"
    out.mkdir(parents=True)
    monkeypatch.setattr(cr, "OUT", out)
    procs: list[subprocess.Popen] = []
    yield out, procs
    for p in procs:
        try:
            p.kill()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
    for pid in _pids_under(tmp_path):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def test_a_neighbouring_sweep_is_not_swept(sweep):
    """`clean_rollouts` is a prefix of `clean_rollouts_validate`."""
    out, procs = sweep
    neighbour = out.parent / "clean_rollouts_validate" / "bp35-0a0ad940"
    procs.append(_spawn(neighbour, orphan=True))
    time.sleep(1.0)
    before = _pids_under(neighbour)
    assert before, "the fixture orphan did not start"

    killed = cr.kill_orphan_solvers()

    time.sleep(0.7)
    assert killed == 0, (
        f"the default sweep killed {killed} process group(s) belonging to "
        f"clean_rollouts_validate"
    )
    assert _pids_under(neighbour), (
        "an orphan of the neighbouring sweep was terminated; two sweeps cannot "
        "share a box under a substring match"
    )


def test_an_orphan_of_this_sweep_is_killed(sweep):
    out, procs = sweep
    ws = out / "zz99-deadbeef"
    procs.append(_spawn(ws, orphan=True))
    time.sleep(1.0)
    assert _pids_under(ws), "the fixture orphan did not start"

    killed = cr.kill_orphan_solvers()

    time.sleep(0.7)
    assert killed >= 1, "the sweep's own orphan was not found"


def test_a_live_child_of_this_process_is_left_alone(sweep):
    """Only trees reparented to init qualify, so a running solver is safe."""
    out, procs = sweep
    ws = out / "yy88-cafebabe"
    live = _spawn(ws, orphan=False)
    procs.append(live)
    time.sleep(0.7)

    killed = cr.kill_orphan_solvers()

    time.sleep(0.7)
    assert killed == 0, "a live child was counted as an orphan"
    assert live.poll() is None, (
        "the driver killed its own running solver; the ppid==1 test is what "
        "keeps a mid-game process alive"
    )


def test_an_orphan_whose_name_holds_a_space_is_still_found(sweep):
    """`comm` is arbitrary bytes, and it is not argv[0].

    Splitting the whole `stat` line on whitespace shifts every field right, the
    ppid test stops matching, and the orphan survives a cleanup that reports
    success.

    The name has to come from the *executable*, not from `exec -a`: the kernel
    takes `comm` from the binary's basename and `exec -a` only rewrites argv[0].
    A first version of this test used `exec -a` and both split mutants survived
    it — the fixture could not express the failure it was named for, which is the
    exact defect class this file documents. So the binary is copied to a name
    carrying a space and a `)`, the two characters that break a naive parse.
    """
    import shutil

    out, procs = sweep
    ws = out / "xx77-feedface"
    ws.mkdir(parents=True)
    exe = ws / "sl ee)p"
    shutil.copy2("/bin/sleep", exe)
    exe.chmod(0o755)

    procs.append(subprocess.Popen(
        ["bash", "-c", f'( setsid "{exe}" 120 >/dev/null 2>&1 & ) ; sleep 120'],
        cwd=ws, start_new_session=True))
    time.sleep(1.0)

    named = [pid for pid in _pids_under(ws)
             if b" " in pathlib.Path(f"/proc/{pid}/comm").read_bytes()]
    assert named, "the fixture never produced a process whose comm holds a space"

    killed = cr.kill_orphan_solvers()

    assert killed >= 1, (
        "an orphan whose process name contains a space was invisible to the "
        "sweep; the stat fields must be split after the last ')'"
    )

    def _still_running() -> list[int]:
        # `/proc/<pid>` outlives the process while it is a zombie awaiting its
        # reaper, so read the state rather than the directory.
        alive = []
        for pid in named:
            try:
                raw = pathlib.Path(f"/proc/{pid}/stat").read_text()
            except OSError:
                continue
            if raw[raw.rindex(")") + 2:].split()[0] != "Z":
                alive.append(pid)
        return alive

    deadline = time.time() + 5
    while _still_running() and time.time() < deadline:
        time.sleep(0.2)
    assert not _still_running(), "the space-named orphan is still running"
