"""The watchdog's liveness check must be argv-exact, and it must be one check.

Two defects motivated this file, both found by running the thing rather than
reading it:

1. **The single-instance guard was a second copy of the liveness check**, and the
   copy matched only the absolute path while the original matched the absolute
   path *and* the relative one. The watchdog is started as
   ``bash tools/daemon_watchdog.sh``, so the guard could not see a running
   watchdog and a second instance launched straight past it. Two implementations
   of one predicate fail the way two enumerated lists fail: the copy nobody
   exercises drifts. ``already_watching`` is now ``running`` applied to itself.

2. **A ``pgrep -f`` liveness check matches its own command line**, and this
   script names all three daemons -- so ``pgrep -f supervisor.sh`` finds the
   watchdog and reports a dead supervisor as alive. That is the defect this
   project has hit most often, most recently in a probe that excluded ``*grep*``
   to drop self-matches and thereby dropped the real watchdog, whose script
   contains ``pgrep``.
"""

from __future__ import annotations

import os
import signal
import uuid
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("/proc").is_dir(),
    reason="the production watchdog is a Linux procfs daemon",
)

REPO = Path(__file__).resolve().parents[1]
WATCHDOG = REPO / "tools" / "daemon_watchdog.sh"


def _check(env: dict[str, str]) -> str:
    out = subprocess.run(["bash", str(WATCHDOG), "--check"], cwd=REPO, text=True,
                         capture_output=True, env={**os.environ, **env}, timeout=60)
    return out.stdout


@pytest.fixture()
def probe(tmp_path):
    """A throwaway daemon in tools/, so the watchdog's own path logic applies.

    **The name is unique per test, not shared.** A shared name made cleanup a
    race rather than isolation: this watchdog relaunches daemons with `setsid`,
    so a relaunch outlives the test that caused it, and a leak read as "alive"
    to the next test and failed a test that was perfectly correct. The suite
    passed or failed depending on the order pytest happened to choose. A name
    nothing else can collide with removes the race instead of narrowing it.
    """
    script = REPO / "tools" / f"_wd_test_probe_{uuid.uuid4().hex[:8]}.sh"
    script.write_text("#!/usr/bin/env bash\nwhile true; do sleep 5; done\n")
    script.chmod(0o755)
    started: list[subprocess.Popen] = []
    try:
        yield script, started
    finally:
        for proc in started:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        # **The watchdog's relaunches are not ours to kill by group.** It starts
        # a daemon with `setsid`, so the child leaves our process group and
        # reparents to init: killpg cannot reach it, and it outlives the test
        # that spawned it. A leaked probe then reads as "alive" to the *next*
        # test, which fails a test that is perfectly correct. Sweep by argv.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if any(a.decode(errors="ignore").endswith(script.name) for a in argv if a):
                try:
                    os.kill(int(entry.name), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        script.unlink(missing_ok=True)


def test_a_daemon_that_is_not_running_reads_as_gone(probe):
    script, _ = probe
    assert "GONE" in _check({"CCARC3_WATCHDOG_DAEMONS": script.name})


def test_a_daemon_that_is_running_reads_as_alive(probe):
    script, started = probe
    started.append(subprocess.Popen(["bash", str(script)], cwd=REPO,
                                    start_new_session=True))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if "alive" in _check({"CCARC3_WATCHDOG_DAEMONS": script.name}):
            break
    assert f"{script.name} alive" in _check({"CCARC3_WATCHDOG_DAEMONS": script.name})


def test_merely_naming_a_daemon_does_not_make_it_look_alive(probe):
    """The ``pgrep -f`` defect, made concrete.

    A process whose command line *contains* the daemon's path but does not have
    it as an argv element must not count. ``pgrep -f`` would match this; an
    argv-exact check does not -- and the watchdog's own script names every
    daemon it watches, so it would otherwise always see them as alive.
    """
    script, started = probe
    mention = f"# watching {script} here\nwhile true; do sleep 5; done"
    started.append(subprocess.Popen(["bash", "-c", mention], cwd=REPO,
                                    start_new_session=True))
    time.sleep(0.5)
    report = _check({"CCARC3_WATCHDOG_DAEMONS": script.name})
    assert f"{script.name} GONE" in report, (
        "a process that only mentions the daemon was counted as the daemon"
    )


def test_a_watchdog_watching_a_different_daemon_set_is_not_a_duplicate():
    """Otherwise the live instance makes every other one unstartable.

    The first guard refused any second watchdog at all, which is both wrong --
    two watchdogs over disjoint daemon sets do not double anything -- and the
    reason the behaviour could not be tested while the production watchdog ran.
    """
    body = WATCHDOG.read_text()
    assert "CCARC3_WATCHDOG_DAEMONS=" in body and "environ" in body, (
        "the guard must compare daemon sets, read from the candidate's own env"
    )


@pytest.mark.parametrize("spelling", ["absolute", "relative"])
def test_a_second_watchdog_refuses_whichever_spelling_started_the_first(spelling, probe, tmp_path):
    """The actual bug, tested by doing it rather than by reading the source.

    A watchdog started as ``bash tools/daemon_watchdog.sh`` was invisible to a
    guard that only knew the absolute path, so the second instance sailed past
    and ran forever. The failure is *behavioural* -- the source read fine -- so
    this starts a real watchdog under each spelling and asserts the next one
    exits promptly and says why.
    """
    script, started = probe
    log = tmp_path / "wd.log"
    env = {"CCARC3_WATCHDOG_DAEMONS": script.name,
           "CCARC3_WATCHDOG_LOG": str(log),
           "CCARC3_WATCHDOG_INTERVAL": "2"}
    target = str(WATCHDOG) if spelling == "absolute" else "tools/daemon_watchdog.sh"

    first = subprocess.Popen(["bash", target], cwd=REPO, start_new_session=True,
                             env={**os.environ, **env},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started.append(first)
    deadline = time.monotonic() + 20
    log.touch()
    while time.monotonic() < deadline and "watchdog up" not in log.read_text(errors="ignore"):
        time.sleep(0.2)
    assert "watchdog up" in log.read_text(), "the first watchdog never started"

    # A second launch must return, not loop. The timeout IS the assertion.
    second = subprocess.run(["bash", target], cwd=REPO, timeout=25,
                            env={**os.environ, **env},
                            capture_output=True, text=True)
    assert second.returncode == 0
    assert "already running" in log.read_text(), (
        f"the guard did not recognise a watchdog started by {spelling} path"
    )


def test_the_watchdog_never_relaunches_itself():
    """It watches the three shell daemons. Adding itself to that list would make
    every cycle spawn a new watchdog."""
    body = WATCHDOG.read_text()
    default = [line for line in body.splitlines() if line.startswith("DAEMONS=")]
    assert len(default) == 1
    assert "daemon_watchdog" not in default[0]


def test_a_lone_watchdog_starts_instead_of_mistaking_itself_for_another(tmp_path, probe):
    """The guard's other direction, and the one that had no test.

    ``$(pids_of ...)`` is a **forked copy of this script** -- same argv, same
    environment, different pid -- so a scan that excludes only ``$$`` finds that
    copy and calls it another watchdog. The guard then refused *every* launch
    including the first, and ``--check`` reported "watchdog already running"
    while the box in fact had none: broken closed, and reporting the opposite.

    Every earlier test here asserted a second launch refuses, which stayed true
    the whole time. Nothing asserted a first launch succeeds, so nothing failed.
    """
    script, started = probe
    log = tmp_path / "lone.log"
    log.touch()
    env = {"CCARC3_WATCHDOG_DAEMONS": script.name,
           "CCARC3_WATCHDOG_LOG": str(log),
           "CCARC3_WATCHDOG_INTERVAL": "60"}
    proc = subprocess.Popen(["bash", str(WATCHDOG)], cwd=REPO, start_new_session=True,
                            env={**os.environ, **env},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started.append(proc)

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and "watchdog up" not in log.read_text(errors="ignore"):
        time.sleep(0.2)
    body = log.read_text(errors="ignore")
    assert "watchdog up" in body, (
        "a lone watchdog refused to start -- it matched its own forked subshell"
    )
    assert "already running" not in body


def test_the_check_subcommand_agrees_with_the_process_table(probe):
    """``--check`` said a watchdog was running when none was. A status line that
    can be wrong in the reassuring direction is worse than none."""
    script, started = probe
    env = {"CCARC3_WATCHDOG_DAEMONS": script.name}
    assert "no other watchdog" in _check(env), (
        "--check claims a watchdog is running with none started"
    )


def _count_running(name: str) -> int:
    n = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(a.decode(errors="ignore").endswith(name) for a in argv if a):
            n += 1
    return n


def test_a_daemon_that_is_already_alive_is_not_relaunched(tmp_path, probe):
    """Otherwise every cycle doubles it.

    The watchdog exists to restart what has died. A version that restarts what
    is *running* turns a two-minute loop into an unbounded fork bomb against the
    supervisor -- and the supervisor launches solvers, so the blast radius is
    real money. Nothing asserted the negative until this.
    """
    script, started = probe
    started.append(subprocess.Popen(["bash", str(script)], cwd=REPO,
                                    start_new_session=True))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _count_running(script.name) < 1:
        time.sleep(0.2)
    assert _count_running(script.name) == 1, "fixture did not start exactly one probe"

    log = tmp_path / "nodupe.log"
    log.touch()
    watchdog = subprocess.Popen(
        ["bash", str(WATCHDOG)], cwd=REPO, start_new_session=True,
        env={**os.environ, "CCARC3_WATCHDOG_DAEMONS": script.name,
             "CCARC3_WATCHDOG_LOG": str(log), "CCARC3_WATCHDOG_INTERVAL": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started.append(watchdog)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and "watchdog up" not in log.read_text(errors="ignore"):
        time.sleep(0.2)
    time.sleep(3)                      # three cycles at a one-second interval

    assert _count_running(script.name) == 1, (
        f"the watchdog relaunched a live daemon: {_count_running(script.name)} copies"
    )
    assert "RELAUNCHED" not in log.read_text(errors="ignore")


def test_a_health_check_is_not_mistaken_for_a_running_watchdog(tmp_path, probe):
    """`--check` carries this script's argv for the half-second it lives.

    A watchdog launched in the same breath as a health check saw the check as a
    rival and refused. That is not hypothetical sequencing: it is exactly how an
    autopilot loop calls this — launch if missing, then report — and it failed
    ten times out of ten while `--check` simultaneously reported no watchdog
    running. The two invocations were looking at each other. Reporting on the
    daemons is not being one.
    """
    script, started = probe
    log = tmp_path / "check.log"
    log.touch()
    env = {"CCARC3_WATCHDOG_DAEMONS": script.name,
           "CCARC3_WATCHDOG_LOG": str(log),
           "CCARC3_WATCHDOG_INTERVAL": "60"}

    checker = subprocess.Popen(
        ["bash", str(WATCHDOG), "--check"], cwd=REPO, start_new_session=True,
        env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started.append(checker)
    daemon = subprocess.Popen(
        ["bash", str(WATCHDOG)], cwd=REPO, start_new_session=True,
        env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started.append(daemon)

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and "watchdog up" not in log.read_text(errors="ignore"):
        time.sleep(0.2)
    body = log.read_text(errors="ignore")
    assert "watchdog up" in body, "a concurrent --check blocked the launch"
    assert "already running" not in body


def test_a_pid_that_vanishes_mid_scan_is_not_counted_as_a_rival():
    """`pgid_of` reads /proc/PID/stat, and the pid most likely to disappear is
    our own command-substitution subshell — which carries this script's argv and
    exits the moment `pids_of` returns. An empty pgid compared unequal to
    MY_PGID, so the corpse counted as another watchdog and the guard refused with
    nothing running. Intermittent, which is why it read as contention for hours.
    """
    body = WATCHDOG.read_text()
    assert 'pg="$(pgid_of "$pid")"' in body and '[ -z "$pg" ] && continue' in body, (
        "an unreadable pgid must skip the candidate, not fall through to a compare"
    )
    assert '[ -r "/proc/$pid/environ" ] || continue' in body, (
        "an unreadable environ must skip too, not default to the standard set"
    )
