"""A relaunched heartbeat must reparent to init, not hang off the watchdog.

The heartbeat used to run AS the harness Monitor's command, so the Monitor's
timeout was the heartbeat's lifetime. Moving it behind a watchdog fixed the
reporting but not the lifetime: `setsid` gives a process its own SESSION, not a
new parent, and whatever reaps a finished Monitor walks descendants. A heartbeat
started by a watchdog is a live child of that watchdog, so it died on every
cycle -- measured at 16:26, 16:58 and 17:28 PDT, one per 30-minute Monitor
timeout, while supervisor.sh and preserve_evidence.sh (both ppid 1) sailed
through all three.

The first version of the fix claimed the relaunched heartbeat "outlived the
watchdog's own death". That was a true observation of the wrong process: the
instance measured had been started from a shell that exited, so it was already
an orphan. The mechanism this pins is the double fork -- the subshell exits at
once and the heartbeat reparents to init.

The launch line is taken from the real file and run for real. Asserting that the
source contains "setsid" would have passed the whole time the bug was live.
"""

from __future__ import annotations

import os
import pathlib
import re
import signal
import shutil
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
WATCHDOG = REPO / "tools" / "heartbeat_watch.sh"
SRC = WATCHDOG.read_text(encoding="utf-8")


def _launch_line() -> str:
    for line in SRC.splitlines():
        if "setsid" in line and not line.lstrip().startswith("#"):
            return line.strip()
    raise AssertionError("heartbeat_watch.sh no longer launches anything with setsid")


def _find(stub: pathlib.Path) -> int | None:
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(a.decode("utf-8", "replace") == str(stub) for a in argv if a):
            return int(d.name)
    return None


def _stat_fields(pid: int) -> list[str]:
    """`/proc/<pid>/stat` after the last `)`.

    `comm` is field 2 and is arbitrary bytes, so one space in a process name
    shifts every later field right -- splitting the whole line then reads a
    fragment of the name as the ppid. The same misparse has already been fixed in
    `clean_rollouts.kill_orphan_solvers` and `build_trace_audit`; it was still
    here.
    """
    raw = pathlib.Path(f"/proc/{pid}/stat").read_text()
    return raw[raw.rindex(")") + 2:].split()


def _ppid(pid: int) -> int:
    return int(_stat_fields(pid)[1])


def _sid(pid: int) -> int:
    return int(_stat_fields(pid)[3])


def _is_descendant_of(pid: int, ancestor: int) -> bool:
    """Walk the parent chain, so an intermediate subshell still counts as
    'hanging off the launcher' while it lives."""
    seen: set[int] = set()
    cur = pid
    while cur and cur not in seen:
        seen.add(cur)
        try:
            cur = _ppid(cur)
        except (OSError, ValueError):
            return False
        if cur == ancestor:
            return True
    return False


def test_the_relaunch_orphans_the_heartbeat(tmp_path: pathlib.Path) -> None:
    """Run the real launch line and read the resulting parent.

    **The launcher is kept ALIVE, which is the whole point.** The first version
    let it exit, and a parent that exits orphans its child under either form --
    so `setsid nohup bash "$HB" &` scored ppid 1 exactly like the subshell, and
    the runtime half of this file discriminated nothing. The real watchdog does
    not exit; it loops every 60s. Only against a living parent does the double
    fork differ from a bare background job.

    It also read ppid the instant the process appeared, before reparenting had
    settled, which made it flake roughly one run in three.
    """
    if not pathlib.Path("/proc").is_dir() or shutil.which("setsid") is None:
        pytest.skip("the production runner's orphan/setsid contract is Linux-specific")

    stub = tmp_path / "stub_heartbeat.sh"
    stub.write_text("#!/bin/bash\nsleep 120\n", encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "heartbeat.log"

    # The real line, with HB and LOG aimed at the fixture, then a parent that
    # stays alive exactly as heartbeat_watch.sh does between polls.
    line = _launch_line()
    script = f'HB={stub}\nLOG={log}\n{line}\nsleep 60\n'
    launcher = subprocess.Popen(
        ["/bin/bash", "-c", script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    pid = None
    try:
        deadline = time.time() + 10
        while time.time() < deadline and pid is None:
            pid = _find(stub)
            if pid is None:
                time.sleep(0.1)
        assert pid is not None, "the launch line started nothing"

        # Wait for reparenting to settle rather than reading it once.
        settle = time.time() + 5
        while time.time() < settle and _is_descendant_of(pid, launcher.pid):
            time.sleep(0.1)

        # **"Orphaned" is not spelled `ppid == 1` on every box.** pid 1 reaps
        # orphans only where it is the sole reaper. Under a user `systemd`, a
        # container init shim, or WSL -- where `/sbin/init` is pid 1 but a
        # separate `/init` process is the subreaper -- a correctly detached
        # process reparents to that reaper instead, and this assertion failed on
        # a launch line that had behaved exactly as designed. Measured on the
        # desktop box 2026-08-12: ppid 28233, launcher alive, double fork
        # working.
        #
        # The number was never the property. What the bug was about is that the
        # heartbeat must NOT be a live descendant of its launcher, so a sweep of
        # the Monitor task's descendants cannot take it down -- so assert that,
        # and keep the launcher-alive precondition that makes it discriminating.
        assert launcher.poll() is None, (
            "the launcher exited, so the child would be orphaned either way and "
            "this test cannot tell a double fork from a bare background job"
        )
        assert not _is_descendant_of(pid, launcher.pid), (
            f"the relaunched heartbeat (pid {pid}, ppid {_ppid(pid)}) is still a "
            f"descendant of its launcher {launcher.pid}, which is running -- so a "
            f"descendant sweep of the Monitor task takes it down on every timeout"
        )
        assert _sid(pid) == pid, (
            f"pid {pid} is not a session leader -- setsid did not take effect"
        )
    finally:
        for victim in (pid, launcher.pid):
            if victim:
                try:
                    os.kill(victim, signal.SIGKILL)
                except OSError:
                    pass


def test_the_watchdog_does_not_exec_the_heartbeat_directly() -> None:
    """The bug shape, pinned: a bare backgrounded launch keeps it as a child.

    `setsid nohup bash "$HB" ... &` looks detached and is not. The subshell
    wrapper is what orphans it.
    """
    line = _launch_line()
    assert line.startswith("("), (
        f"the launch is not wrapped in a subshell: {line!r} -- without the "
        f"double fork the heartbeat stays a child of the watchdog"
    )
    assert re.search(r"&\s*\)", line), (
        f"the subshell does not background the heartbeat before exiting: {line!r}"
    )
