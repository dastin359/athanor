"""Regression tests for ``util_now()`` in ``tools/supervisor.sh``.

``util_now`` is the whole quota gate: the supervisor loop reads one number from
it and decides, every ten minutes, whether to keep launching solvers. Five
defects were fixed in it and none had a test:

(a) it filtered on ``/seven_day/`` and threw the five-hour line away, so a
    rejected five-hour window (a HARD KILL of in-flight solvers) was invisible
    as long as seven-day sat below the ceiling. It must consider BOTH windows
    and return the WORSE.
(b) ``rejected`` / ``out_of_credits`` on EITHER window must pin the reading to
    ``1.00`` -- a window can refuse before its own utilization reaches the
    ceiling.
(c) a VOID window (``resetsAt`` already past) counts as ``0.00``, not the stale
    ``utilization`` field quota.sh still prints for it. Reading the stale number
    deadlocked the arm: parked -> no fresh reading -> util stays high -> never
    restarts -> still parked.
(d) it must emit EXACTLY ONE line. awk's ``exit`` still runs ``END``, so the
    refusal branch used to print the refusal *and* the accumulated worst -- two
    lines into a variable every caller treats as one number.
(e) with no reading at all the output is empty, which the loop's ``[ -n "$u" ]``
    guard turns into "neither start nor stop".

How these are driven
--------------------
``supervisor.sh`` ends in an infinite ``while true`` loop and hardcodes
``QUOTA`` to the repo's ``quota.sh``, so it cannot be run end to end and cannot
be pointed at a fixture. ``util_now`` (and ``ge``) are therefore extracted with
``sed`` from the real script -- located by absolute path up front, never after a
``cd`` -- and run under ``bash`` against a fake QUOTA script that prints canned
``quota.sh`` output.

Two habits every test here keeps, because their absence produced four tests
that passed without ever reaching the check they named:

* the fake QUOTA script touches a marker file, and every run asserts the marker
  exists. An empty reading must mean "quota ran and said nothing", never "the
  fixture made the function exit before it got there".
* every pattern is tested in BOTH directions. ``rejected`` on a five_hour line
  must trip; ``rejected`` on another window's line must not. ``VOID`` must zero
  a window; the same utilization without ``VOID`` must not.

The environment for every subprocess is built from scratch, so no test depends
on the shell that launched pytest. No network, and no ARC_API_KEY.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest


def _find_supervisor() -> Path:
    """Locate the real supervisor.sh without ever depending on the cwd.

    Everything is derived from this file's own resolved path. When these tests
    run from a git worktree that does not carry the (untracked) script, the
    main checkout is found through git's common dir -- still absolute, still
    computed before any subprocess runs somewhere else.
    """
    here = Path(__file__).resolve()
    candidates = [here.parents[1] / "tools" / "supervisor.sh"]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(here.parent),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            candidates.append(
                Path(proc.stdout.strip()).parent / "tools" / "supervisor.sh"
            )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        pass
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return candidates[0]


#: Resolved once, at import, BEFORE any subprocess runs with a different cwd.
#: A command substitution evaluated after cd'ing into a sandbox is how a
#: previous test ended up reading an unpatched copy of the file under test.
SUPERVISOR = _find_supervisor()

pytestmark = pytest.mark.skipif(
    not SUPERVISOR.is_file(), reason=f"supervisor.sh not found (looked at {SUPERVISOR})"
)

VOID_NOTE = "VOID (window reset since this reading)"


# --------------------------------------------------------------------------
# extracting the function under test
# --------------------------------------------------------------------------
def _sed(expr: str) -> str:
    return subprocess.run(
        ["sed", "-n", expr, str(SUPERVISOR)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout


def _extract(func: str) -> str:
    """Pull one shell function out of supervisor.sh with sed.

    Handles both shapes the script uses: a block (``util_now``) and a one-liner
    (``ge``). Getting this wrong is silent -- an empty extraction makes the
    driver fail identically for every fixture, so a missing subject would read
    as a uniform behavioural failure. Hence the assertions.
    """
    header = f"{func}() {{"
    text = _sed(f"/^{func}() {{.*}}$/p")  # one-liner form
    if not text.strip():
        text = _sed(f"/^{func}() {{$/,/^}}$/p")  # block form
    assert text.startswith(header), f"sed did not find {func}() in {SUPERVISOR}"
    assert text.rstrip().endswith("}"), f"extraction of {func}() is unbalanced"
    body = text.rstrip()[len(header) : -1].strip()
    assert body, f"extraction of {func}() has no body"
    return text


def _limit() -> str:
    """The operator's ceiling, read from the script instead of hardcoded here.

    **Evaluated, not pattern-matched.** This was `re.search(r"^LIMIT=([0-9.]+)")`,
    which reads the number only while the number is written literally. The moment
    the ceiling became a per-run knob -- `LIMIT="${CCARC3_QUOTA_LIMIT:-0.98}"` --
    the regex matched nothing and four tests failed on an assertion about the
    file's *spelling* rather than about the ceiling.

    Running the line is the version that survives its form: it yields the real
    default with no override set, and it would keep working if the default moved
    into a function or a config file tomorrow. The env is emptied deliberately so
    a launching shell that happens to export the override cannot change what the
    suite believes the default is.
    """
    line = next((ln for ln in SUPERVISOR.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("LIMIT=")), "")
    assert line, "no LIMIT= in supervisor.sh"
    proc = subprocess.run(["bash", "-c", line + '\nprintf "%s" "$LIMIT"'],
                          capture_output=True, text=True, check=True,
                          env={"PATH": "/usr/bin:/bin"})
    assert proc.stdout.strip(), f"LIMIT line yielded nothing: {line!r}"
    return proc.stdout.strip()


def _clean_env(home: Path) -> dict[str, str]:
    """A fixed environment. Nothing is inherited from the launching shell."""
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(home),
        "LC_ALL": "C",
        "TZ": "America/Los_Angeles",
    }


def _fake_quota(tmp_path: Path, canned: str) -> tuple[Path, Path]:
    """A stand-in quota.sh that prints ``canned`` and records that it ran."""
    marker = tmp_path / "quota_was_invoked"
    script = tmp_path / "fake_quota.sh"
    script.write_text(
        "#!/bin/bash\n"
        f"printf 'ran\\n' >> {shlex.quote(str(marker))}\n"
        "cat <<'CANNED_QUOTA_OUTPUT_EOF'\n"
        f"{canned}\n"
        "CANNED_QUOTA_OUTPUT_EOF\n",
        encoding="utf-8",
    )
    return script, marker


def run_util_now(tmp_path: Path, canned: str) -> str:
    """Run the real ``util_now`` against canned quota.sh output.

    Returns raw stdout with its line structure intact -- fix (d) is about how
    many lines come out, so nothing here strips or joins them.
    """
    quota, marker = _fake_quota(tmp_path, canned)
    driver = tmp_path / "driver.sh"
    driver.write_text(
        "#!/bin/bash\n" 'QUOTA="$1"\n' + _extract("util_now") + "util_now\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(driver), str(quota)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=_clean_env(tmp_path),
        timeout=60,
    )
    assert proc.returncode == 0, f"driver failed: {proc.stderr}"
    # The check must actually have run. An early-out that never invoked quota
    # would produce the same empty stdout as a genuine no-reading.
    assert marker.is_file(), (
        "util_now never invoked QUOTA -- the fixture short-circuited the check, "
        f"so this assertion proves nothing. stdout={proc.stdout!r}"
    )
    return proc.stdout


def line(window: str, status: str, util: str, note: str = "fresh") -> str:
    """One quota.sh window row, in quota.sh's exact print format."""
    return f"  {window:10} {status:16} util={util}  reset_in=2h13m  age=45s  [{note}]"


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------
BOTH_LOW = "\n".join(
    [
        line("five_hour", "allowed", "0.40"),
        line("seven_day", "allowed", "0.30"),
        "status=allowed",
    ]
)

FIVE_HOT = "\n".join(
    [
        line("five_hour", "allowed_warning", "0.99"),
        line("seven_day", "allowed", "0.12"),
        "status=allowed_warning",
    ]
)

SEVEN_HOT = "\n".join(
    [
        line("five_hour", "allowed", "0.10"),
        line("seven_day", "allowed_warning", "0.97"),
        "status=allowed_warning",
    ]
)

FIVE_REJECTED = "\n".join(
    [
        line("five_hour", "rejected", "0.99"),
        line("seven_day", "allowed", "0.10"),
        "status=rejected",
    ]
)

SEVEN_REJECTED = "\n".join(
    [
        line("five_hour", "allowed", "0.10"),
        line("seven_day", "rejected", "0.99"),
        "status=rejected",
    ]
)

FIVE_OUT_OF_CREDITS = "\n".join(
    [
        line("five_hour", "out_of_credits", "0.61"),
        line("seven_day", "allowed", "0.10"),
        "status=rejected",
    ]
)

FIVE_VOID = "\n".join(
    [
        line("five_hour", "allowed_warning", "0.98", note=VOID_NOTE),
        line("seven_day", "allowed", "0.20"),
        "status=allowed",
    ]
)

BOTH_VOID = "\n".join(
    [
        line("five_hour", "allowed_warning", "0.98", note=VOID_NOTE),
        line("seven_day", "allowed_warning", "0.91", note=VOID_NOTE),
        "status=allowed",
    ]
)

FIVE_LIVE_AT_98 = "\n".join(
    [
        line("five_hour", "allowed_warning", "0.98"),
        line("seven_day", "allowed", "0.20"),
        "status=allowed_warning",
    ]
)

# quota.sh prints every rateLimitType it found, not only the two it requires.
# Another window's refusal is a real line that must NOT be read as a refusal of
# ours -- and must not contribute its utilization either.
OTHER_WINDOW_REJECTED = "\n".join(
    [
        line("five_hour", "allowed", "0.40"),
        line("opus_weekly", "rejected", "0.99"),
        line("seven_day", "allowed", "0.55"),
        "status=rejected",
    ]
)

NO_READING = "status=unknown reason=no-reading-on-disk"

# `utilization` is absent below the first threshold; quota.sh prints the literal
# `<threshold>`. That is not zero, and it is not a number.
FIVE_BELOW_THRESHOLD = "\n".join(
    [
        line("five_hour", "allowed", "<threshold>"),
        line("seven_day", "allowed", "0.55"),
        "status=allowed",
    ]
)

#: Every fixture with the single line util_now owes for it ("" = no line).
TABLE = [
    ("both-low", BOTH_LOW, "0.40"),
    ("five-hot", FIVE_HOT, "0.99"),
    ("seven-hot", SEVEN_HOT, "0.97"),
    ("five-rejected", FIVE_REJECTED, "1.00"),
    ("seven-rejected", SEVEN_REJECTED, "1.00"),
    ("five-out-of-credits", FIVE_OUT_OF_CREDITS, "1.00"),
    ("five-void", FIVE_VOID, "0.20"),
    ("both-void", BOTH_VOID, "0.00"),
    ("five-live-at-98", FIVE_LIVE_AT_98, "0.98"),
    ("other-window-rejected", OTHER_WINDOW_REJECTED, "0.55"),
    ("five-below-threshold", FIVE_BELOW_THRESHOLD, "0.55"),
    ("no-reading", NO_READING, ""),
]

TABLE_PARAMS = [(c, e) for _, c, e in TABLE]
TABLE_IDS = [i for i, _, _ in TABLE]

REFUSALS = [FIVE_REJECTED, SEVEN_REJECTED, FIVE_OUT_OF_CREDITS]
REFUSAL_IDS = ["five-rejected", "seven-rejected", "five-out-of-credits"]


@pytest.mark.parametrize("canned,expected", TABLE_PARAMS, ids=TABLE_IDS)
def test_reading_table(tmp_path: Path, canned: str, expected: str) -> None:
    """Every window combination yields the one reading the loop should act on."""
    out = run_util_now(tmp_path, canned)
    assert out.strip() == expected, f"got {out!r} for:\n{canned}"


# --------------------------------------------------------------------------
# (a) both windows, and the worse of the two
# --------------------------------------------------------------------------
def test_five_hour_is_not_discarded(tmp_path: Path) -> None:
    """A hot five-hour window is reported even when seven-day is cool.

    The pre-fix code filtered on /seven_day/, so this fixture returned 0.12 --
    comfortably under the ceiling -- while five_hour sat at 0.99 and every
    launch was about to be killed mid-game.
    """
    out = run_util_now(tmp_path, FIVE_HOT).strip()
    assert out != "0.12", "five_hour line was discarded; only seven_day was read"
    assert out == "0.99"


def test_seven_day_is_not_discarded(tmp_path: Path) -> None:
    """The other direction: a hot seven-day window still wins over a cool five."""
    out = run_util_now(tmp_path, SEVEN_HOT).strip()
    assert out != "0.10", "seven_day line was discarded"
    assert out == "0.97"


def test_worse_window_wins_regardless_of_order(tmp_path: Path) -> None:
    """The maximum is taken, not the first or the last line seen."""
    forward = "\n".join(
        [line("five_hour", "allowed", "0.88"), line("seven_day", "allowed", "0.11")]
    )
    reverse = "\n".join(
        [line("seven_day", "allowed", "0.11"), line("five_hour", "allowed", "0.88")]
    )
    assert run_util_now(tmp_path, forward).strip() == "0.88"
    assert run_util_now(tmp_path, reverse).strip() == "0.88"


# --------------------------------------------------------------------------
# (b) a refusal on either window is a stop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("canned", REFUSALS, ids=REFUSAL_IDS)
def test_refusal_on_either_window_pins_to_one(tmp_path: Path, canned: str) -> None:
    """rejected/out_of_credits reports 1.00 whatever the utilization says.

    Every fixture here keeps BOTH windows' utilization below the ceiling: if the
    refusal were ignored the reading would come back a comfortable 0.99 or 0.61
    and the loop would launch straight back into a refusing window.
    """
    assert run_util_now(tmp_path, canned).strip() == "1.00"


@pytest.mark.parametrize("canned", REFUSALS, ids=REFUSAL_IDS)
def test_refusal_reading_trips_the_ceiling(tmp_path: Path, canned: str) -> None:
    """1.00 is not just a number: it must satisfy the loop's own `ge $u $LIMIT`.

    Uses supervisor.sh's real `ge` and its real LIMIT, so a ceiling raised past
    1.00 -- or a reading that stopped being numeric -- fails here.
    """
    limit = _limit()
    quota, marker = _fake_quota(tmp_path, canned)
    driver = tmp_path / "gate.sh"
    driver.write_text(
        "#!/bin/bash\n"
        'QUOTA="$1"\n'
        + _extract("util_now")
        + _extract("ge")
        + f"LIMIT={limit}\n"
        "u=$(util_now)\n"
        # the loop's own numeric guard, in the same shape
        'if [ -z "$u" ] || [ "${u#*[0-9]}" = "$u" ]; then echo "NO-READING"; exit 0; fi\n'
        'if ge "$u" "$LIMIT"; then echo "STOP $u"; else echo "LAUNCH $u"; fi\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(driver), str(quota)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=_clean_env(tmp_path),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert marker.is_file(), "quota was never consulted"
    assert (
        proc.stdout.strip() == "STOP 1.00"
    ), f"a refused window did not trip the ceiling (LIMIT={limit}): {proc.stdout!r}"


def test_healthy_reading_does_not_trip_the_ceiling(tmp_path: Path) -> None:
    """The other direction of the same gate: a low reading still launches."""
    limit = _limit()
    quota, marker = _fake_quota(tmp_path, BOTH_LOW)
    driver = tmp_path / "gate.sh"
    driver.write_text(
        "#!/bin/bash\n"
        'QUOTA="$1"\n'
        + _extract("util_now")
        + _extract("ge")
        + f"LIMIT={limit}\n"
        "u=$(util_now)\n"
        'if [ -z "$u" ] || [ "${u#*[0-9]}" = "$u" ]; then echo "NO-READING"; exit 0; fi\n'
        'if ge "$u" "$LIMIT"; then echo "STOP $u"; else echo "LAUNCH $u"; fi\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(driver), str(quota)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=_clean_env(tmp_path),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert marker.is_file(), "quota was never consulted"
    assert proc.stdout.strip() == "LAUNCH 0.40", proc.stdout


def test_refusal_on_another_window_does_not_trip(tmp_path: Path) -> None:
    """The other direction: 'rejected' must be anchored to OUR two windows.

    quota.sh prints every rate-limit type it found. A refusal on an unrelated
    window is not a refusal of five_hour or seven_day, and its 0.99 must not
    become the reported worst either.
    """
    out = run_util_now(tmp_path, OTHER_WINDOW_REJECTED).strip()
    assert out != "1.00", "an unrelated window's refusal was read as ours"
    assert out == "0.55"


def test_the_word_rejected_alone_does_not_trip(tmp_path: Path) -> None:
    """A trailer line is not a window row."""
    canned = "\n".join(
        [
            line("five_hour", "allowed", "0.40"),
            line("seven_day", "allowed", "0.55"),
            "status=allowed reason=nothing-rejected-here",
        ]
    )
    assert run_util_now(tmp_path, canned).strip() == "0.55"


# --------------------------------------------------------------------------
# (c) VOID counts as zero
# --------------------------------------------------------------------------
def test_void_window_reads_as_zero(tmp_path: Path) -> None:
    """A window whose resetsAt has passed contributes 0.00, not its stale 0.98.

    Pre-fix the stale 0.98 stayed the worst forever once the arm was parked (no
    solver -> no fresh reading), so the supervisor never restarted.
    """
    assert run_util_now(tmp_path, FIVE_VOID).strip() == "0.20"
    assert run_util_now(tmp_path, BOTH_VOID).strip() == "0.00"


def test_live_window_keeps_its_utilization(tmp_path: Path) -> None:
    """The other direction: without the VOID note the same 0.98 still counts.

    Zeroing on anything but VOID would be the far worse failure -- launching
    into an exhausted window instead of waiting one out.
    """
    assert run_util_now(tmp_path, FIVE_LIVE_AT_98).strip() == "0.98"


def test_nonnumeric_utilization_is_skipped_not_zeroed(tmp_path: Path) -> None:
    """`util=<threshold>` is unreadable, so the other window decides."""
    assert run_util_now(tmp_path, FIVE_BELOW_THRESHOLD).strip() == "0.55"


# --------------------------------------------------------------------------
# (d) exactly one line
# --------------------------------------------------------------------------
@pytest.mark.parametrize("canned,expected", TABLE_PARAMS, ids=TABLE_IDS)
def test_emits_at_most_one_line(tmp_path: Path, canned: str, expected: str) -> None:
    """Callers assign this to `u` and compare it as a number.

    awk's `exit` still runs END, so the refusal branch printed the refusal AND
    the accumulated worst -- `u` became "1.00\\n0.99" and every numeric
    comparison downstream was garbage.
    """
    out = run_util_now(tmp_path, canned)
    lines = out.splitlines()
    assert len(lines) == (0 if expected == "" else 1), f"expected one line, got {out!r}"


@pytest.mark.parametrize("canned", REFUSALS, ids=REFUSAL_IDS)
def test_refusal_output_is_a_single_bare_number(tmp_path: Path, canned: str) -> None:
    """The refusal branch specifically -- this is the one that printed twice."""
    out = run_util_now(tmp_path, canned)
    assert out == "1.00\n", f"refusal branch emitted {out!r}"


# --------------------------------------------------------------------------
# (e) no reading at all
# --------------------------------------------------------------------------
def test_no_reading_yields_empty_output(tmp_path: Path) -> None:
    """quota.sh reporting no data on disk produces no number at all.

    The marker assertion inside run_util_now is what makes this mean anything:
    empty stdout has to come from quota.sh having been run and having said
    nothing usable, not from util_now bailing before it ran.
    """
    assert run_util_now(tmp_path, NO_READING) == ""


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, not covered by today's five fixes. quota.sh's other "
        "unknown path names the window it could not observe -- both the "
        "'MISSING' row and the 'reason=window-not-observed:seven_day' trailer "
        "contain 'seven_day' -- so util_now sets seen=1, finds no util= field, "
        "and END prints an uninitialised worst as '0.00'. The supervisor then "
        "reads a maximally permissive 0.00 at exactly the moment quota.sh is "
        "saying it cannot see the window whose exhaustion costs days. The "
        "no-reading path (e) is silent as designed; this sibling path is not. "
        "When util_now is fixed this test XPASSes: delete the xfail marker."
    ),
)
def test_window_not_observed_yields_no_number(tmp_path: Path) -> None:
    """A window quota.sh could not observe must not read as a healthy 0.00."""
    canned = "\n".join(
        [
            "  seven_day  MISSING — no event in any of 61 stream.jsonl scanned",
            "status=unknown reason=window-not-observed:seven_day",
        ]
    )
    assert run_util_now(tmp_path, canned) == ""


def test_completely_empty_quota_output_yields_empty(tmp_path: Path) -> None:
    """And a quota.sh that prints nothing at all is not a reading of zero."""
    assert run_util_now(tmp_path, "") == ""


# --------------------------------------------------------------------------
# harness self-checks
# --------------------------------------------------------------------------
def test_fixture_reaches_the_check(tmp_path: Path) -> None:
    """The marker mechanism itself works, so its use above is not decorative."""
    _quota, marker = _fake_quota(tmp_path, BOTH_LOW)
    assert not marker.exists()
    run_util_now(tmp_path, BOTH_LOW)
    assert marker.is_file()


def test_extraction_targets_the_real_script() -> None:
    """These tests read the repo's supervisor.sh, and read the awk gate itself."""
    src = _extract("util_now")
    assert "awk" in src, "util_now no longer runs awk; this test file is stale"
    assert 'bash "$QUOTA"' in src, "util_now no longer shells out to QUOTA"


# ==========================================================================
# Gap found by mutation-testing this file, closed.
# ==========================================================================

def test_a_partially_numeric_utilization_is_rejected_not_half_read(tmp_path: Path) -> None:
    """`if (t ~ /^[0-9.]+$/)` — the guard that makes util_now reject a malformed
    number rather than let awk coerce its numeric prefix.

    `test_nonnumeric_utilization_is_skipped_not_zeroed` names this behaviour and
    cannot see it: its only non-numeric fixture is `util=<threshold>`, which awk
    coerces to 0, and awk's uninitialised accumulator is already 0 — so skipping
    and zeroing are indistinguishable there. The mutation is only observable on a
    value with a numeric PREFIX: `util=0.99junk` coerces to 0.99 and would win
    over a healthy sibling window, turning a garbled line into a stop.
    """
    # **Newline-joined, because `line()` does not terminate.** Concatenating
    # them put both windows on ONE record, and util_now accumulates once per
    # record from whatever `u` holds at the end of the field loop -- so only the
    # last util= counted and the mutation was invisible. A fixture that cannot
    # express the failure is the defect this suite exists for.
    canned = "\n".join([line("five_hour", "allowed", "0.99junk"),
                        line("seven_day", "allowed", "0.10"),
                        "status=allowed"])
    got = run_util_now(tmp_path, canned)
    assert got.strip() == "0.10", (
        f"a garbled utilization was half-read as 0.99 instead of skipped: {got!r}"
    )
    assert len(got.strip().splitlines()) == 1, f"util_now must emit one line: {got!r}"
