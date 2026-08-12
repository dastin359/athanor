"""`elapsed_s` on the live panel must be the process's age, not an inode's.

`live_games()` computed the start of a running solver as
`(entry / "stat").stat().st_mtime` -- the modification time of a file inside
`/proc/<pid>/`. That is the timestamp of a procfs inode, which the kernel stamps
when the inode is instantiated in the dcache, i.e. whenever something last looked
the process up. It is not when the process started, and it agrees with the start
only for a process nobody has looked at since it forked -- which is every process
you check this by hand on, and no process in a box that runs a heartbeat.

Measured across this box's 97 live processes on 2026-08-09, the proxy understated
the true age of **every one of them**:

    pid   4192   real 4h42m   mtime-derived 4h13m    (29 minutes younger)
    pid  15520   real 4h35m   mtime-derived 4h13m    (22 minutes younger)
    pid   4818   real 9m58s   mtime-derived 8m16s    (17% younger)

The direction is what makes it more than untidy. `elapsed_s` is the one field an
operator reads to decide whether a solver is stalled, and the error only ever
argues *against* intervening.

Same shape as the rest of this file's siblings: a field that names a thing and
reads a proxy for it, where the proxy was only ever checked where it agrees.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    not pathlib.Path("/proc").is_dir(),
    reason="live-panel process ages require Linux procfs",
)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_trace_audit as bta  # noqa: E402

HZ = os.sysconf("SC_CLK_TCK")


def _btime() -> float:
    for line in pathlib.Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    pytest.skip("/proc/stat carries no btime on this kernel")


def _fake_proc(root: pathlib.Path, starttime_ticks: int, comm: str = "claude") -> pathlib.Path:
    """A directory shaped like `/proc/<pid>/`, holding one crafted `stat`.

    Its own mtime is *now* by construction, which is the whole point: a reader
    that takes the file's mtime returns "started just now" for a process the
    fields say booted long ago.
    """
    entry = root / "1234"
    entry.mkdir(parents=True, exist_ok=True)
    # Fields 3..52, so index 0 is `state`. Field 22 -- starttime -- is index 19.
    fields = [str(n) for n in range(3, 53)]
    fields[0] = "S"
    fields[19] = str(starttime_ticks)
    (entry / "stat").write_text(f"1234 ({comm}) " + " ".join(fields), encoding="utf-8")
    return entry


def test_start_time_comes_from_the_kernel_not_the_inode(tmp_path):
    """A process the fields say is an hour old must not read as newborn."""
    an_hour_ago = time.time() - 3600
    ticks = int((an_hour_ago - _btime()) * HZ)
    if ticks <= 0:
        pytest.skip("this kernel booted less than an hour ago")
    entry = _fake_proc(tmp_path, ticks)

    started = bta._process_started(entry)

    age = time.time() - started
    assert 3590 <= age <= 3610, (
        f"the crafted process is an hour old; _process_started put it at "
        f"{age:.0f}s. The file's own mtime is {time.time() - (entry / 'stat').stat().st_mtime:.0f}s "
        f"old, which is what a mtime reader would have returned."
    )


def test_a_comm_with_spaces_and_parens_does_not_shift_the_fields(tmp_path):
    """`comm` is arbitrary bytes; splitting the whole line on whitespace is wrong.

    A solver launched through a wrapper can carry a name like `claude (worker 2)`,
    and a naive `split()` then reads field 22 from the wrong offset -- silently,
    producing an age that is merely implausible rather than obviously broken.
    """
    an_hour_ago = time.time() - 3600
    ticks = int((an_hour_ago - _btime()) * HZ)
    if ticks <= 0:
        pytest.skip("this kernel booted less than an hour ago")

    plain = bta._process_started(_fake_proc(tmp_path / "a", ticks, comm="claude"))
    weird = bta._process_started(_fake_proc(tmp_path / "b", ticks, comm="cla ude) (x"))

    assert plain == weird, (
        "a process name containing spaces and parens moved the start time; "
        "the fields must be split after the last ')' rather than across the line"
    )


def test_a_real_process_reads_close_to_when_it_was_spawned():
    """The unit above is arithmetic; this one checks it against a real fork."""
    proc = subprocess.Popen(["sleep", "30"])
    spawned = time.time()
    try:
        time.sleep(1.0)
        started = bta._process_started(pathlib.Path(f"/proc/{proc.pid}"))
    finally:
        proc.kill()
        proc.wait()
    assert abs(started - spawned) < 2.0, (
        f"a process spawned at {spawned:.1f} was reported as starting at "
        f"{started:.1f}"
    )


def test_the_panel_ages_a_solver_from_its_own_start(tmp_path, monkeypatch):
    """End to end: the number the page renders is the process's real age.

    `live_games()` is the caller, and it is what regresses if the helper is
    correct but wired up to the wrong field.
    """
    scratch = tmp_path / "scratchpad"
    ws = scratch / "clean_rollouts" / "zz99-deadbeef"
    ws.mkdir(parents=True)
    (ws / "trace.jsonl").write_text('{"level": 2}\n', encoding="utf-8")
    (ws / "meta.json").write_text('{"levels": 6}', encoding="utf-8")
    monkeypatch.setenv("CCARC3_SCRATCH", str(scratch))

    binv = tmp_path / "bin"
    binv.mkdir()
    exe = binv / "codex"
    exe.write_text("#!/bin/bash\nsleep 60\n", encoding="utf-8")
    exe.chmod(0o755)

    proc = subprocess.Popen([str(exe)], cwd=ws)
    spawned = time.time()
    try:
        # Long enough that a stamp taken "now" and a stamp taken at spawn are
        # tellable apart, short enough not to slow the suite.
        time.sleep(3.0)
        rows = [r for r in bta.live_games() if r["id"] == "zz99-deadbeef"]
    finally:
        proc.kill()
        proc.wait()

    assert rows, "the fixture solver was not seen at all"
    elapsed = rows[0]["elapsed_s"]
    assert elapsed is not None, "the panel reported an unknown age for a live solver"
    assert abs(elapsed - (time.time() - spawned)) < 2.0, (
        f"the panel aged the solver at {elapsed}s against a real "
        f"{time.time() - spawned:.1f}s"
    )
