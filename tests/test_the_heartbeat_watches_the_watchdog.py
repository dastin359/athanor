"""Every daemon that must outlive this session is on one list, and it is one list.

`daemon_watchdog.sh` relaunches `supervisor.sh`, `heartbeat.sh` and
`preserve_evidence.sh` when they die. Nothing reported the watchdog's own death —
it was added on 2026-08-09 and not added to the heartbeat's watch list, so the
process that repairs the other three could disappear silently and the first
symptom would be a second daemon dying with nobody to restart it.

The watchdog watches them; the heartbeat watches the watchdog. That closes the
loop without either one depending on the session staying alive.

The list was also written **twice** — once to record what was seen, once to
report what was missing — so a name added to one and not the other is silently
unwatched. That is the eighth stale enumerated list in this project, and it had
already gone stale. It is now one array read in both places.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HEARTBEAT = REPO / "tools" / "heartbeat.sh"
WATCHDOG = REPO / "tools" / "daemon_watchdog.sh"


def _watched() -> list[str]:
    body = HEARTBEAT.read_text()
    match = re.search(r"^WATCHED_DAEMONS=\(([^)]*)\)", body, re.M)
    assert match, "WATCHED_DAEMONS not found"
    return match.group(1).split()


def test_the_watchdog_is_watched():
    assert "daemon_watchdog.sh" in _watched(), (
        "nothing reports the death of the process that restarts everything else"
    )


def test_every_daemon_the_watchdog_restarts_is_also_reported():
    """The two lists must agree on everything except each other.

    The watchdog restarts the daemons; the heartbeat reports them. A daemon the
    watchdog revives but the heartbeat never mentions is one whose repeated
    death nobody sees.
    """
    body = WATCHDOG.read_text()
    match = re.search(r'^DAEMONS="\$\{CCARC3_WATCHDOG_DAEMONS:-([^}]*)\}"', body, re.M)
    assert match, "the watchdog's default daemon list was not found"
    restarted = set(match.group(1).split())
    reported = set(_watched())

    # heartbeat.sh is restarted by the watchdog and cannot report itself.
    unreported = restarted - reported - {"heartbeat.sh"}
    assert not unreported, f"restarted but never reported: {sorted(unreported)}"


def test_the_watch_list_is_read_from_one_place():
    """Not "both copies agree today" — a property two copies can satisfy right
    up until someone edits one of them."""
    body = HEARTBEAT.read_text()
    assert body.count("WATCHED_DAEMONS=(") == 1, "more than one definition"
    assert body.count('"${WATCHED_DAEMONS[@]}"') == 1, (
        "the report loop must consume the one canonical list"
    )
    assert "for name in supervisor.sh" not in body, "an inline copy of the list survives"


def _daemon_check(watch: list[str], cwd: Path) -> str:
    """Run `daemon_check` against a chosen watch list.

    The list is overridden rather than the process table faked: this function
    scans the real `/proc`, and on this box the daemons genuinely are running —
    a first draft asserted they were all missing and failed for that reason,
    which is the fixture disagreeing with the world rather than the code being
    wrong.
    """
    body = HEARTBEAT.read_text()
    start = body.index("WATCHED_DAEMONS=(")
    end = body.index("\n}\n", body.index("daemon_check() {")) + 3
    snippet = body[start:end]
    assert "DAEMON DOWN" in snippet, "extraction lost the report"
    names = " ".join(watch)
    script = f"{snippet}\nWATCHED_DAEMONS=({names})\ndaemon_check"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         timeout=60, cwd=cwd)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_daemon_that_is_not_running_is_reported(tmp_path):
    absent = ["definitely_absent_aardvark.sh", "definitely_absent_badger.sh"]
    report = _daemon_check(absent, tmp_path)
    for name in absent:
        assert f"DAEMON DOWN: {name}" in report, f"{name} not reported: {report!r}"


def test_a_daemon_that_is_running_is_not_reported(tmp_path):
    """The positive control: if the check reported everything regardless, the
    test above would pass while the alarm meant nothing."""
    fake = tmp_path / "probe_daemon.sh"
    fake.write_text("#!/usr/bin/env bash\nwhile true; do sleep 5; done\n")
    fake.chmod(0o755)
    proc = subprocess.Popen(["bash", str(fake)], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        report = ""
        while time.monotonic() < deadline:
            report = _daemon_check(["probe_daemon.sh", "definitely_absent_civet.sh"], tmp_path)
            if "DAEMON DOWN: probe_daemon.sh" not in report:
                break
            time.sleep(0.3)
        assert "DAEMON DOWN: probe_daemon.sh" not in report, (
            f"a running daemon was reported down: {report!r}"
        )
        assert "DAEMON DOWN: definitely_absent_civet.sh" in report, (
            "the absent one must still be reported, or this proves nothing"
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_the_real_watch_list_is_all_running_right_now(tmp_path):
    """A live check of the box, not of the code: every daemon the heartbeat
    watches should be up while a sweep is sanctioned.

    The configured scratch directory is the source of truth: a positive
    concurrency brake authorizes a sweep and therefore requires its daemons.
    The explicit flag remains available for managed runners that keep their
    brake elsewhere.  No implicit home-directory lookup is used in the Codex
    path, so an ordinary test run cannot reach host ARC state by accident.
    """
    sanctioned = os.environ.get("CCARC3_LIVE_DAEMON_CHECK") == "1"
    scratch = os.environ.get("CCARC3_SCRATCH")
    brake = Path(scratch) / "concurrency" if scratch else None
    if not sanctioned and brake and brake.is_file():
        try:
            sanctioned = int(brake.read_text().strip() or "0") > 0
        except ValueError:
            sanctioned = False
    if not sanctioned:
        pytest.skip(
            f"no sweep is sanctioned (brake {brake} is absent or 0), so the "
            "daemons are correctly down; nothing to assert about a parked box"
        )
    report = _daemon_check(_watched(), tmp_path)
    assert "DAEMON DOWN" not in report, report
