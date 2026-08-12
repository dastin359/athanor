"""The supervisor's start/stop decision, driven directly.

`supervisor.sh` is the only thing standing between a 25-game sweep and a
weekly-quota blackout, and exhausting that window is a hard kill of in-flight
solvers rather than a slowdown. It had 47 tests and no mutation battery; 10 of
18 mutants survived, including these two:

* **the brake never engaging at all** — deleting the stop entirely
  (`if [ -n "$running" ] && false`) passed every test;
* **launching with no quota reading** — the documented policy is "neither
  starting nor stopping" when `quota.sh` is silent, and nothing held it to that.

The rest are the hysteresis (a ceiling stop must not restart until utilisation
falls back below `RESUME`, or the loop churns a game per cycle at the ceiling)
and the ceiling announcement.

The decision is a block inside an infinite loop, so it is extracted and run
standalone against stubbed `start`/`stop_politely`. That is the same technique
`test_supervisor_quota_gate.py` uses for `util_now`, and it carries the same
hazard: an extraction that silently comes back empty tests nothing, so the
balance of the block is asserted before it is used.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "tools" / "supervisor.sh"

def _tmp_marker() -> str:
    fd, path = tempfile.mkstemp(prefix="supervisor_launch_")
    import os as _os
    _os.close(fd)
    return path


_BLOCK_START = '    if [ -n "$u" ] && [ "${u#*[0-9]}" != "$u" ]; then'
_BLOCK_END = "    sleep 600"


def _decision_block() -> str:
    text = SUPERVISOR.read_text()
    assert text.count(_BLOCK_START) == 1, "decision block anchor is not unique"
    assert text.count(_BLOCK_END) == 1, "loop-tail anchor is not unique"
    block = text[text.index(_BLOCK_START):text.index(_BLOCK_END)]
    assert block.count("if ") >= 3 and block.count("fi") >= 3, "extraction looks truncated"
    assert "stop_politely" in block and "start " in block, "extraction lost its actions"
    return block


def _ge() -> str:
    for line in SUPERVISOR.read_text().splitlines():
        if line.startswith("ge()"):
            return line
    raise AssertionError("ge() not found")


def decide(*, u: str, running: str, ceiling_stopped: str = "0",
           limit: str = "0.98", resume: str = "0.90") -> tuple[str, str]:
    """Run one pass of the decision. Returns (action, ceiling_stopped-after)."""
    script = "\n".join([
        "#!/bin/bash",
        _ge(),
        'start() { echo "ACTION=START"; }',
        'stop_politely() { echo "ACTION=STOP"; }',
        f'u="{u}"; running="{running}"; ceiling_stopped="{ceiling_stopped}"',
        f'LIMIT="{limit}"; RESUME="{resume}"',
        _decision_block(),
        'echo "CEILING_STOPPED=$ceiling_stopped"',
    ])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    action = next((l.split("=", 1)[1] for l in out.stdout.splitlines()
                   if l.startswith("ACTION=")), "NONE")
    flag = next(l.split("=", 1)[1] for l in out.stdout.splitlines()
                if l.startswith("CEILING_STOPPED="))
    return action, flag


# --------------------------------------------------------------------------- #
# the brake itself
# --------------------------------------------------------------------------- #


def test_a_running_arm_at_the_ceiling_is_stopped():
    """Deleting this entirely passed all 47 existing tests."""
    assert decide(u="0.99", running="1234")[0] == "STOP"


def test_the_ceiling_is_inclusive():
    """`>=`, not `>`. Exactly at the ceiling the operator's limit is reached, and
    the margin there is about one game — a game that starts may be lost outright
    when the window closes mid-run."""
    assert decide(u="0.98", running="1234")[0] == "STOP"


def test_a_running_arm_below_the_ceiling_is_left_alone():
    assert decide(u="0.97", running="1234")[0] == "NONE"


def test_nothing_running_means_nothing_to_stop():
    """The stop is conditioned on a live runner; firing it otherwise logs a
    stop that did not happen and sets the hysteresis flag from nowhere."""
    action, flag = decide(u="0.99", running="")
    assert action == "NONE" and flag == "0"


def test_a_ceiling_stop_records_itself():
    assert decide(u="0.99", running="1234")[1] == "1"


# --------------------------------------------------------------------------- #
# hysteresis — why a ceiling stop does not restart immediately
# --------------------------------------------------------------------------- #


def test_after_a_ceiling_stop_the_arm_waits_for_the_window_to_recover():
    """Without the gap the loop kills at the ceiling and restarts a second
    later, churning a game per cycle at exactly the point where a game is most
    likely to be lost."""
    assert decide(u="0.95", running="", ceiling_stopped="1")[0] == "NONE"


def test_the_wait_is_against_resume_and_not_against_the_ceiling():
    """0.95 is below the 0.98 ceiling and above the 0.90 resume: resuming here
    would mean there was never any hysteresis at all."""
    assert decide(u="0.95", running="", ceiling_stopped="1")[0] == "NONE"
    assert decide(u="0.89", running="", ceiling_stopped="1")[0] == "START"


def test_recovering_below_resume_clears_the_flag():
    action, flag = decide(u="0.50", running="", ceiling_stopped="1")
    assert action == "START" and flag == "0"


# --------------------------------------------------------------------------- #
# a runner that exited on its own
# --------------------------------------------------------------------------- #


def test_a_finished_queue_restarts_while_any_quota_is_left():
    """Deliberately NOT gated on RESUME: a runner that exited because it emptied
    its queue must be restartable at 0.95, or extending the queue strands the
    extra games until the weekly reset."""
    assert decide(u="0.95", running="", ceiling_stopped="0")[0] == "START"


def test_a_finished_queue_does_not_restart_above_the_ceiling():
    assert decide(u="0.99", running="", ceiling_stopped="0")[0] == "NONE"


# --------------------------------------------------------------------------- #
# no reading is not a low reading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reading", ["", "unknown", "n/a", "-"])
def test_silence_neither_starts_nor_stops(reading):
    """`quota.sh` reads quota out of solver streams, so a build-only stretch
    reports nothing at all. Treating that as "plenty left" launches into an
    unknown window; treating it as "exhausted" stalls forever. The documented
    policy is to do neither and say so, leaving the decision with an operator.
    """
    assert decide(u=reading, running="")[0] == "NONE"
    assert decide(u=reading, running="1234")[0] == "NONE"


def test_a_real_reading_is_still_acted_on():
    """The positive control for the test above: if the numeric check rejected
    everything, "neither starts nor stops" would pass vacuously."""
    assert decide(u="0.99", running="1234")[0] == "STOP"
    assert decide(u="0.10", running="")[0] == "START"


# --------------------------------------------------------------------------- #
# a raised ceiling has to be audible
# --------------------------------------------------------------------------- #


def _announcement(limit: str, resume: str) -> str:
    text = SUPERVISOR.read_text()
    marker = 'if [ "$LIMIT" != "0.98" ] || [ "$RESUME" != "0.90" ]; then'
    assert text.count(marker) == 1
    body = text[text.index(marker):]
    body = body[:body.index("\nfi\n") + 4]
    script = f'LIMIT="{limit}"; RESUME="{resume}"\n' + body
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, timeout=30).stdout


def test_a_raised_ceiling_says_so():
    assert "CEILING RAISED" in _announcement("0.999", "0.90")


def test_a_changed_resume_says_so_too():
    """Announcing only a changed LIMIT passed. The resume threshold decides how
    long a stopped arm stays stopped, so moving it silently changes how the
    brake behaves with nothing in the log to explain it."""
    assert "CEILING RAISED" in _announcement("0.98", "0.50")


def test_the_defaults_are_silent():
    """An always-on announcement is one people stop reading."""
    assert _announcement("0.98", "0.90").strip() == ""


# --------------------------------------------------------------------------- #
# what `start` must do before it launches anything
# --------------------------------------------------------------------------- #


def _start_body() -> str:
    text = SUPERVISOR.read_text()
    assert text.count("start() {") == 1
    body = text[text.index("start() {"):]
    return body[:body.index("\n}\n") + 3]


@pytest.mark.parametrize("guard", ["refresh_proxy", "key_ok"])
def test_start_refuses_when_a_precondition_fails(guard):
    """Both guards must be able to stop the launch.

    Dropping `refresh_proxy` passed the suite. The proxy is what confines the
    solver's network reach, so launching without refreshing it starts a real
    game with the gate in whatever state the last run left it.
    """
    other = "key_ok" if guard == "refresh_proxy" else "refresh_proxy"
    # **The launch writes to $LOG, not to stdout.** The first draft stubbed
    # `setsid` to echo and pointed LOG at /dev/null, so the evidence went
    # straight to the bit bucket and the test passed against the very mutant it
    # was written to kill. Give it a real file and read that.
    marker = Path(_tmp_marker())
    script = "\n".join([
        f"{guard}() {{ return 1; }}",
        f"{other}() {{ return 0; }}",
        'setsid() { echo "LAUNCHED"; }',
        'nohup() { echo "LAUNCHED"; }',
        'REPO="/tmp"; RUNNER="/tmp/none.py"; RESUME="0.9"',
        f'LOG="{marker}"',
        _start_body(),
        'start 0.5',
        'wait',
    ])
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    body = marker.read_text() if marker.exists() else ""
    marker.unlink(missing_ok=True)
    assert "LAUNCHED" not in body, f"a failing {guard} did not stop the launch"


def test_start_launches_when_both_preconditions_pass():
    """The positive control: if either stub broke the function outright, the
    test above would pass for the wrong reason."""
    marker = Path(_tmp_marker())
    script = "\n".join([
        "refresh_proxy() { return 0; }",
        "key_ok() { return 0; }",
        'setsid() { echo "LAUNCHED"; }',
        'nohup() { echo "LAUNCHED"; }',
        'REPO="/tmp"; RUNNER="/tmp/none.py"; RESUME="0.9"',
        f'LOG="{marker}"',
        _start_body(),
        'start 0.5',
        'wait',
    ])
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    body = marker.read_text() if marker.exists() else ""
    marker.unlink(missing_ok=True)
    assert "LAUNCHED" in body


def test_scanning_procfs_is_silent_when_a_process_vanishes(tmp_path):
    """`2>/dev/null` on `tr` does not cover the redirection — the shell opens
    `< "$d/cmdline"` before `tr` exists, so a pid that exits between the glob and
    the read reports on the shell's own stderr. Measured: that line reached the
    heartbeat's output and the Monitor forwarded it as an alert. An alarm channel
    that emits noise is one people stop reading.
    """
    text = SUPERVISOR.read_text()
    assert text.count("pids_of() {") == 1
    body = text[text.index("pids_of() {"):]
    body = body[:body.index("\n}\n") + 3]

    # **Deterministic, not a race, and not a directory either.** Two earlier
    # drafts failed to reproduce the bug: deleting entries mid-scan never landed
    # in the window between `[ -r ]` and the redirection, and a `cmdline` that is
    # a DIRECTORY turns out to print nothing at all. The real failure is an open
    # that returns an errno -- measured as "No such file or directory" on a pid
    # that exited. A unix socket reproduces exactly that shape on every run: it
    # passes `-r`, and opening it fails with "No such device or address".
    import socket

    # AF_UNIX paths are capped at 104 bytes on macOS; pytest's tmp_path is
    # intentionally descriptive and can exceed that before "/cmdline".
    with tempfile.TemporaryDirectory(prefix="ccarc3-proc-", dir="/tmp") as short:
        fake = Path(short)
        keep = []
        for pid in range(100, 105):
            (fake / str(pid)).mkdir()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(fake / str(pid) / "cmdline"))
            keep.append(sock)                          # bound until the test ends
        script = "\n".join([
            body.replace("/proc/[0-9]*", f"{fake}/[0-9]*"),
            'pids_of ".*/claude" >/dev/null',
        ])
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        assert out.stderr == "", f"the scan printed to stderr: {out.stderr!r}"
