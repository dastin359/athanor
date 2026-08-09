"""Four holes the reach and inbound scans left open, found by mutation.

Each of these mutants survived the existing suite -- the tests exercise the
detectors only where the leak sits near the start of a short fixture, or on a
path shape that happens not to be `/tmp`. Two of the four are live safety holes:
a contaminated run would be reported clean.

* **Only the first window is scanned.** `for start in range(len(seen))` reduced
  to `range(1)` passed every test. A solver that dumps the medians on turn 300 of
  a 400 kB transcript -- which is what a real leak looks like, not turn 1 --
  would then read clean.
* **`/tmp` was exempted wholesale.** The exemption is for ordinary temp files and
  carries `"/scratchpad" not in path` precisely so another run's evidence is not
  exempt. Dropping that clause passed every test, and the scratchpad under `/tmp`
  is where every banked card, ledger and `meta.json` on this box lives.
* **The window bound is load-bearing in the other direction.** Removing it
  entirely also passed. Then any eight of a game's medians occurring in order
  anywhere across 400 kB is a "leak", and the scan starts voiding clean runs --
  the failure mode the bound was written to prevent, with nothing testing it.
* **Four-digit medians.** `\\d{1,4}` narrowed to `\\d{1,3}` passed, because the
  largest per-level median in the whole corpus is 442. That makes it an
  equivalent mutant *on today's games only*; the regex is written for four
  digits on purpose and nothing held it there.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import proofread_trace as pt  # noqa: E402

LP85 = [17, 38, 31, 16, 41, 60, 26, 159]
WS = pathlib.Path("/tmp/scratchpad/clean_rollouts/zz99-deadbeef/attempt_1/zz99-deadbeef")

# Filler with no run of LP85 in it. Deliberately built from values that appear in
# LP85, so the scan cannot pass by ignoring the neighbourhood -- it has to fail to
# assemble the *sequence*.
NOISE = " ".join(str(n) for n in [159, 26, 60, 41, 16, 31, 38, 17] * 400)


def test_a_leak_late_in_a_long_transcript_is_still_caught():
    """A real leak lands mid-run, not in the first hundred bytes."""
    text = NOISE + " " + ", ".join(str(n) for n in LP85)
    assert pt.array_arrived(text, LP85), (
        "the medians arrived at the end of the transcript and the scan missed them"
    )


def test_a_leak_in_the_middle_is_caught_too():
    half = len(NOISE) // 2
    text = NOISE[:half] + " " + str(LP85)[1:-1] + " " + NOISE[half:]
    assert pt.array_arrived(text, LP85)


def test_medians_scattered_across_the_whole_transcript_are_not_a_leak():
    """The window bound is the thing that keeps this a check and not an alarm.

    Each value in order, but hundreds of unrelated numbers between them: that is
    a 400 kB transcript containing the digits, not a solver being handed the
    array.
    """
    # Spaces on both sides of the gap. Without them `gap.join` welds "17" onto
    # "200" and the scan sees the five-digit token "17200", which its own regex
    # discards -- so the fixture would assert nothing while looking like it
    # asserts everything. Exactly the shape this file is about.
    gap = " " + " ".join(str(n) for n in range(200, 320)) + " "
    text = gap.join(str(n) for n in LP85)
    assert not pt.array_arrived(text, LP85), (
        "values spread across the whole transcript were read as a leak; the "
        "window bound is what stops the scan crying wolf"
    )


def test_a_four_digit_median_is_not_invisible():
    """Nothing in the corpus exceeds 442, so only a test holds the regex at four."""
    wanted = [17, 1024, 31, 2048]
    assert pt.array_arrived(", ".join(str(n) for n in wanted), wanted)


def test_a_reach_into_another_runs_scratchpad_under_tmp_is_flagged():
    """`/tmp` is exempt for temp files, never for the evidence store.

    Every banked card, ledger and `meta.json` on this box sits under
    `/tmp/claude-.../scratchpad/`, and a scorecard holds complete median arrays
    for every game played on it.
    """
    others = [
        "cat /tmp/claude-0/-home-user-athanor/abc/scratchpad/clean_rollouts/bp35-0a0ad940/card.json",
        "ls /tmp/claude-0/-home-user-athanor/abc/scratchpad/arc3/.env",
        "python3 -c \"open('/tmp/claude-0/x/scratchpad/best_or_last/card.json')\"",
    ]
    for command in others:
        assert pt.strayed(command, WS), f"reach into the evidence store unseen: {command}"


def test_an_ordinary_temp_file_is_still_exempt():
    """The positive control: the exemption itself must keep working."""
    for command in ["cat /tmp/pip-build-abc/log.txt", "ls /tmp/tmpf83jd", "mktemp -d /tmp/xx"]:
        assert not pt.strayed(command, WS), f"an ordinary temp path was flagged: {command}"


# ---------------------------------------------------------------------------
# The two calibration points below are not invariants -- they are choices, and
# both were checked against the whole preserved corpus rather than argued.
#
# 77 preserved streams x 16 recoverable median arrays, 2026-08-09:
#
#     window = 3n   (shipped)     43 fires
#     window = 10n                46 fires   (+3)
#     window = 30n                99 fires   (+56)
#     order-insensitive, 3n       53 fires   (+10)
#
# The +56 at 30n include `bp35`, `cd82`, `lp85`, `tn36`, `vc33` and `su15` --
# most of the bank -- each "leaking" a *different* game's medians by chance. The
# +10 for order-insensitivity land on four banked clean rollouts. So both
# widenings would void runs that are not contaminated, which is the failure this
# scan is least able to survive: a false void discards a real result, and unlike
# a false clean nobody ever goes looking for it.
# ---------------------------------------------------------------------------


def test_a_median_every_eighth_number_is_not_a_leak():
    """The window tolerates labels between values, not paragraphs.

    `level 0: 17, level 1: 38` is about three numbers per median, which is where
    `3n` comes from. Eight unrelated numbers between each is prose that happens
    to contain the values in order, and at `10n` or `30n` the scan starts calling
    that a leak -- 56 extra fires across the corpus at `30n`, on runs that are
    not contaminated.
    """
    parts = []
    filler = iter(range(200, 400))
    for n in LP85:
        parts.append(str(n))
        parts.extend(str(next(filler)) for _ in range(8))
    assert not pt.array_arrived(" ".join(parts), LP85), (
        "eight unrelated numbers between each median was read as a leak; the "
        "window multiplier has been widened past what the corpus tolerates"
    )


def test_the_values_have_to_arrive_in_order():
    """Order is what makes eight small integers an identification.

    Ignoring it -- `value in wanted` rather than `value == wanted[i]` -- looks
    strictly stronger and is not: measured over the corpus it fires on ten
    streams the ordered scan calls clean, four of them banked clean rollouts,
    each matching a permutation of some *other* game's array inside an 18-number
    window of a 400 kB transcript.

    The gap this leaves is real and accepted: a solver that sorts the array
    before printing it evades the scan. That is a narrower hole than voiding four
    good runs, and the inbound `FOREIGN_MEDIANS` pattern still catches the
    labelled form.
    """
    shuffled = [159, 17, 60, 38, 26, 31, 41, 16]
    assert sorted(shuffled) == sorted(LP85), "fixture drifted from LP85"
    assert not pt.array_arrived(" ".join(str(n) for n in shuffled), LP85), (
        "a permutation matched; the scan identifies the array by its order"
    )
    assert pt.array_arrived(" ".join(str(n) for n in LP85), LP85), (
        "positive control: the array in its published order must still match"
    )
