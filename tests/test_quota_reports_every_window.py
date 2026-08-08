"""Regression tests for tools/quota.sh -- the rate-limit window reporter.

Every test here drives the REAL script end to end via subprocess, with
CCARC3_SCRATCH pointed at a fixture tree of stream.jsonl files, and asserts on
what the script PRINTED. Exit codes are not enough: quota.sh exits 0 from its
"no reading on disk" early-out too, so a fixture that never reaches the scan
loop would satisfy a returncode-only assertion while testing nothing. Each test
below pins a distinctive utilization value that can only appear in the output if
the scan actually reached the event in question.

Four fixed defects are covered:

  (a) LAST write wins within one file. Two rate_limit_events of the same type
      in one stream.jsonl must report the SECOND. The old code compared file
      mtimes, which are equal within a file, so ``mt > latest[t][0]`` was False
      by construction and the FIRST (oldest, lowest) event won -- a later
      ``rejected`` was discarded in favour of an earlier ``allowed``.
  (b) No fixed scan cap. A seven_day event must be found even when buried
      behind 50+ newer five_hour-only streams. The old code scanned
      ``streams[:40]``.
  (c) An absent expected window prints MISSING and
      ``status=unknown reason=window-not-observed``. The old code printed only
      the window it did find, plus a clean status.
  (d) The normal case still prints both rows and a status line.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest


def _find_quota_sh() -> Path:
    """Locate tools/quota.sh by walking up from THIS FILE, never from cwd.

    Resolving from ``__file__`` (and never from the working directory or a
    command substitution evaluated after a chdir) is what keeps the test honest
    about which copy of the script it is exercising. The walk also lets the file
    run from a git worktree, where the untracked script only exists in the main
    checkout further up the path.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "quota.sh"
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"tools/quota.sh not found above {here}")


QUOTA_SH = _find_quota_sh()

#: Windows quota.sh is required to report on.
FIVE_HOUR = "five_hour"
SEVEN_DAY = "seven_day"

#: Comfortably inside the script's FRESH=3600 window, so every fixture reading
#: is "fresh" and the staleness branches stay out of the way of what we assert.
FRESH_AGE = 30

#: Comfortably in the future, so ``left > 0`` and no reading is judged VOID.
RESET_AHEAD = 3 * 3600


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------


def _event(window: str, status: str, utilization: float, *, resets_at: float) -> str:
    """One rate_limit_event line, shaped the way a solver stream writes it."""
    return json.dumps(
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "rateLimitType": window,
                "status": status,
                "utilization": utilization,
                "resetsAt": resets_at,
            },
        }
    )


def _noise() -> list[str]:
    """Lines the scan must skip: a wrong type, a truncated line that still
    contains the marker string the fast-path substring check looks for, and a
    line that merely mentions the marker in prose."""
    return [
        json.dumps({"type": "assistant", "text": "hello"}),
        '{"type": "rate_limit_event", "rate_limit_info": {truncated',
        json.dumps({"type": "tool_result", "note": "rate_limit_event in prose"}),
    ]


def write_stream(root: Path, name: str, events: list[str], *, age: float) -> Path:
    """Write ``<root>/<name>/stream.jsonl`` and stamp its mtime ``age`` seconds ago.

    mtime is what quota.sh sorts on, so the test sets scan order explicitly
    rather than relying on write order or filesystem timestamp granularity.
    """
    path = root / name / "stream.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = _noise()
    lines = noise[:1] + events + noise[1:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def run_quota(scratch: Path) -> subprocess.CompletedProcess:
    """Run the real quota.sh against ``scratch`` with a built-from-scratch env.

    The environment is constructed rather than inherited so that ARC_API_KEY --
    or anything else the launching shell happens to export -- cannot change the
    result. quota.sh needs no credentials and no network.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "PYTHONIOENCODING": "utf-8",
        "CCARC3_SCRATCH": str(scratch),
    }
    assert "ARC_API_KEY" not in env
    return subprocess.run(
        ["bash", str(QUOTA_SH)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(scratch),
    )


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------

_ROW = re.compile(r"^\s+(\S+)\s+(\S.*)$")


def rows(out: str) -> dict[str, str]:
    """Indented per-window rows, keyed by window name."""
    found: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("status="):
            continue
        m = _ROW.match(line)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def status_line(out: str) -> str:
    lines = [ln for ln in out.splitlines() if ln.startswith("status=")]
    assert len(lines) == 1, f"expected exactly one status= line, got {lines!r}\n{out}"
    return lines[0]


def util_of(row: str) -> str:
    m = re.search(r"util=(\S+)", row)
    assert m, f"no util= field in row {row!r}"
    return m.group(1)


@pytest.fixture(autouse=True)
def _script_exists():
    assert QUOTA_SH.is_file(), f"script under test not found at {QUOTA_SH}"


# ---------------------------------------------------------------------------
# control: the CCARC3_SCRATCH override really redirects the scan
# ---------------------------------------------------------------------------


def test_scratch_override_redirects_the_scan(tmp_path: Path):
    """An empty fixture root yields the no-reading early-out.

    This is the control for every other test: it shows CCARC3_SCRATCH is what
    the scan reads, so a later assertion about a fixture value cannot be
    satisfied by readings from the live scratchpad.
    """
    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "status=unknown reason=no-reading-on-disk", proc.stdout
    assert not rows(proc.stdout), proc.stdout


# ---------------------------------------------------------------------------
# (d) the normal case
# ---------------------------------------------------------------------------


def test_normal_case_prints_both_windows_and_a_status(tmp_path: Path):
    """Both expected windows present and fresh -> two rows plus status=allowed."""
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "run-a",
        [_event(FIVE_HOUR, "allowed", 0.42, resets_at=resets)],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "run-b",
        [_event(SEVEN_DAY, "allowed", 0.11, resets_at=resets)],
        age=FRESH_AGE + 5,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)

    # Both rows present, and the values prove the scan reached OUR events.
    assert set(got) == {FIVE_HOUR, SEVEN_DAY}, proc.stdout
    assert util_of(got[FIVE_HOUR]) == "0.42", proc.stdout
    assert util_of(got[SEVEN_DAY]) == "0.11", proc.stdout
    assert "fresh" in got[FIVE_HOUR] and "fresh" in got[SEVEN_DAY], proc.stdout
    assert "MISSING" not in proc.stdout, proc.stdout
    assert status_line(proc.stdout) == "status=allowed", proc.stdout


def test_normal_case_escalates_status_to_the_worst_window(tmp_path: Path):
    """The status line reflects the worst window, not the last one printed.

    The positive counterpart to the clean-status case: a seven_day warning must
    not be swallowed by a healthy five_hour.
    """
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "run-a",
        [_event(FIVE_HOUR, "allowed", 0.05, resets_at=resets)],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "run-b",
        [_event(SEVEN_DAY, "allowed_warning", 0.79, resets_at=resets)],
        age=FRESH_AGE + 5,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)
    assert util_of(got[SEVEN_DAY]) == "0.79", proc.stdout
    assert "allowed_warning" in got[SEVEN_DAY], proc.stdout
    assert status_line(proc.stdout) == "status=allowed_warning", proc.stdout


# ---------------------------------------------------------------------------
# (a) last write wins WITHIN one file
# ---------------------------------------------------------------------------


def test_second_event_in_one_file_beats_the_first(tmp_path: Path):
    """Two five_hour events in ONE stream.jsonl -> the SECOND is reported.

    Regression for the mtime comparison: both events carry the same file mtime,
    so ``mt > latest[t][0]`` was False and the earlier, lower ``allowed``
    reading won while the later ``rejected`` was silently discarded.
    """
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "run-a",
        [
            _event(FIVE_HOUR, "allowed", 0.83, resets_at=resets),
            _event(FIVE_HOUR, "rejected", 0.84, resets_at=resets),
        ],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "run-b",
        [_event(SEVEN_DAY, "allowed", 0.20, resets_at=resets)],
        age=FRESH_AGE + 5,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)

    assert FIVE_HOUR in got, f"five_hour row never printed:\n{proc.stdout}"
    assert util_of(got[FIVE_HOUR]) == "0.84", f"first-event-wins regression:\n{proc.stdout}"
    assert "rejected" in got[FIVE_HOUR], proc.stdout
    assert "0.83" not in proc.stdout, f"reported the superseded event:\n{proc.stdout}"
    assert status_line(proc.stdout) == "status=rejected", proc.stdout


def test_second_event_in_one_file_wins_even_when_it_is_milder(tmp_path: Path):
    """The rule is LAST wins, not WORST wins.

    The other direction of the same fix: a ``rejected`` followed in the same
    file by an ``allowed`` (the window rolled over mid-run) must report the
    allowed reading. A "keep the max severity" implementation would pass the
    test above and fail this one.
    """
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "run-a",
        [
            _event(FIVE_HOUR, "rejected", 0.99, resets_at=resets),
            _event(FIVE_HOUR, "allowed", 0.07, resets_at=resets),
        ],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "run-b",
        [_event(SEVEN_DAY, "allowed", 0.20, resets_at=resets)],
        age=FRESH_AGE + 5,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)

    assert FIVE_HOUR in got, f"five_hour row never printed:\n{proc.stdout}"
    assert util_of(got[FIVE_HOUR]) == "0.07", proc.stdout
    assert "0.99" not in proc.stdout, f"reported the superseded event:\n{proc.stdout}"
    assert status_line(proc.stdout) == "status=allowed", proc.stdout


def test_newer_file_still_beats_an_older_file(tmp_path: Path):
    """Across files, mtime order still decides.

    The within-file sequence tiebreak must not let an older file's late-line
    event outrank a newer file's first-line event.
    """
    resets = time.time() + RESET_AHEAD
    # Older file, events on late lines (high sequence numbers).
    write_stream(
        tmp_path,
        "run-old",
        [
            _event(FIVE_HOUR, "rejected", 0.95, resets_at=resets),
            _event(FIVE_HOUR, "rejected", 0.96, resets_at=resets),
            _event(FIVE_HOUR, "rejected", 0.97, resets_at=resets),
        ],
        age=FRESH_AGE + 600,
    )
    # Newer file, event on the first line (sequence 0).
    write_stream(
        tmp_path,
        "run-new",
        [_event(FIVE_HOUR, "allowed", 0.31, resets_at=resets)],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "run-seven",
        [_event(SEVEN_DAY, "allowed", 0.20, resets_at=resets)],
        age=FRESH_AGE + 5,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)
    assert util_of(got[FIVE_HOUR]) == "0.31", proc.stdout
    assert "0.97" not in proc.stdout, proc.stdout
    assert status_line(proc.stdout) == "status=allowed", proc.stdout


# ---------------------------------------------------------------------------
# (b) no fixed scan cap
# ---------------------------------------------------------------------------

DECOYS = 55  # comfortably past the old streams[:40] cap


def test_seven_day_is_found_behind_fifty_newer_streams(tmp_path: Path):
    """Regression for ``streams[:40]``: the rare window must still be reported."""
    resets = time.time() + RESET_AHEAD
    for i in range(DECOYS):
        # The newest decoy carries a value nothing else uses, so the five_hour
        # row also proves mtime ordering survived the uncapped scan.
        util = 0.11 if i == 0 else 0.50
        write_stream(
            tmp_path,
            f"decoy-{i:03d}",
            [_event(FIVE_HOUR, "allowed", util, resets_at=resets)],
            age=FRESH_AGE + i,
        )
    write_stream(
        tmp_path,
        "deep-seven-day",
        [_event(SEVEN_DAY, "allowed_warning", 0.77, resets_at=resets)],
        age=FRESH_AGE + DECOYS + 1,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)

    assert SEVEN_DAY in got, f"seven_day lost behind {DECOYS} newer streams:\n{proc.stdout}"
    # 0.77 exists only in the deepest file: the scan demonstrably got there.
    assert util_of(got[SEVEN_DAY]) == "0.77", proc.stdout
    assert "MISSING" not in proc.stdout, proc.stdout
    # And the common window is still the newest reading, not a middle decoy.
    assert util_of(got[FIVE_HOUR]) == "0.11", proc.stdout
    assert status_line(proc.stdout) == "status=allowed_warning", proc.stdout


def test_scan_stops_once_both_windows_are_found(tmp_path: Path):
    """The uncapped scan must not become an unconditional full-tree walk.

    Both windows sit in the two newest files; the 30 older files below them hold
    a value that can only surface if the loop kept going past its break.
    """
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "top-a",
        [_event(FIVE_HOUR, "allowed", 0.12, resets_at=resets)],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "top-b",
        [_event(SEVEN_DAY, "allowed", 0.13, resets_at=resets)],
        age=FRESH_AGE + 1,
    )
    for i in range(30):
        write_stream(
            tmp_path,
            f"buried-{i:03d}",
            [_event(FIVE_HOUR, "rejected", 0.98, resets_at=resets)],
            age=FRESH_AGE + 10 + i,
        )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = rows(proc.stdout)
    assert util_of(got[FIVE_HOUR]) == "0.12", proc.stdout
    assert util_of(got[SEVEN_DAY]) == "0.13", proc.stdout
    assert "0.98" not in proc.stdout, f"scan ran past its break:\n{proc.stdout}"


# ---------------------------------------------------------------------------
# (c) an absent expected window is a finding, not a silence
# ---------------------------------------------------------------------------


def test_absent_seven_day_reports_missing_and_unknown(tmp_path: Path):
    """Only five_hour on disk -> a MISSING row and status=unknown.

    Regression for the defect where a per-window absence printed the one window
    that was found plus a clean status, which downstream (supervisor.sh's
    ``util_now``) reads as "no constraint".
    """
    resets = time.time() + RESET_AHEAD
    for i in range(3):
        write_stream(
            tmp_path,
            f"run-{i}",
            [_event(FIVE_HOUR, "allowed", 0.42, resets_at=resets)],
            age=FRESH_AGE + i,
        )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    got = rows(out)
    assert SEVEN_DAY in got, f"no seven_day line at all:\n{out}"
    assert "MISSING" in got[SEVEN_DAY], out
    assert status_line(out) == f"status=unknown reason=window-not-observed:{SEVEN_DAY}", out

    # The scan really ran over all three files rather than short-circuiting on
    # an empty tree...
    m = re.search(r"any of (\d+) stream\.jsonl scanned", got[SEVEN_DAY])
    assert m, out
    assert int(m.group(1)) == 3, out
    # ...and the window that WAS found must not be reported as a clean status.
    assert "0.42" not in out, f"printed a normal row despite a missing window:\n{out}"
    assert "status=allowed" not in out, out


def test_absent_five_hour_reports_missing_and_unknown(tmp_path: Path):
    """The mirror case: the check is not hardcoded to seven_day."""
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "run-a",
        [_event(SEVEN_DAY, "allowed", 0.09, resets_at=resets)],
        age=FRESH_AGE,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    got = rows(out)
    assert FIVE_HOUR in got and "MISSING" in got[FIVE_HOUR], out
    assert status_line(out) == f"status=unknown reason=window-not-observed:{FIVE_HOUR}", out
    assert "0.09" not in out, out


def test_both_windows_absent_reports_no_reading(tmp_path: Path):
    """Streams exist but carry no parseable rate_limit_event at all -> the
    distinct no-reading message, not window-not-observed."""
    write_stream(tmp_path, "run-a", [], age=FRESH_AGE)
    write_stream(tmp_path, "run-b", [], age=FRESH_AGE + 1)

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "status=unknown reason=no-reading-on-disk", proc.stdout
    assert "MISSING" not in proc.stdout, proc.stdout


def test_unrelated_window_does_not_satisfy_an_expected_one(tmp_path: Path):
    """A rate_limit_event of some other type must not count as coverage."""
    resets = time.time() + RESET_AHEAD
    write_stream(
        tmp_path,
        "run-a",
        [_event(FIVE_HOUR, "allowed", 0.33, resets_at=resets)],
        age=FRESH_AGE,
    )
    write_stream(
        tmp_path,
        "run-b",
        [_event("one_minute", "allowed", 0.66, resets_at=resets)],
        age=FRESH_AGE + 1,
    )

    proc = run_quota(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "MISSING" in proc.stdout, proc.stdout
    assert (
        status_line(proc.stdout) == f"status=unknown reason=window-not-observed:{SEVEN_DAY}"
    ), proc.stdout


def test_a_stream_at_its_real_depth_is_found(tmp_path):
    """**Every fixture above sits at depth 1; no real stream does.**

    `write_stream` builds `<root>/<name>/stream.jsonl`, which a non-recursive
    `glob(f"{S}/*/stream.jsonl")` matches just as well as the recursive one — so
    dropping `**` and `recursive=True` left all twelve tests green. Real solver
    streams live at `<sweep>/<game>/attempt_N/<game>/stream.jsonl`, four levels
    down, and under that mutation quota.sh would find none of them and report
    `no-reading-on-disk` for a box that is spending quota right now.
    """
    root = tmp_path / "scratch"
    deep = root / "clean_rollouts" / "ar25-0c556536" / "attempt_1" / "ar25-0c556536"
    deep.mkdir(parents=True)
    soon = time.time() + 7200
    (deep / "stream.jsonl").write_text(
        _event("seven_day", "allowed_warning", 0.61, resets_at=soon) + "\n"
        + _event("five_hour", "allowed", 0.12, resets_at=soon) + "\n",
        encoding="utf-8")

    out = run_quota(root).stdout
    assert "seven_day" in out and "0.61" in out, (
        f"a stream at its real depth was not scanned:\n{out}"
    )
    assert "MISSING" not in out and "no-reading-on-disk" not in out
