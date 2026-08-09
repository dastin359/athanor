"""`MAX_PASSES` bounds how many times an interrupted game is paid for again.

A game cut short by a container replacement is retried, and this is the only
thing between one authorised game and twelve attempts at it. Twelve is right for
a 25-game sweep that has to finish; it is wrong for a single validation run, and
the difference belongs to whoever launches it.

`bp35`'s prior is 3.9 hours of wall clock and $57-79. This box was replaced 2.5
hours before the run this knob was added for.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = (REPO / "tools" / "clean_rollouts.py").read_text(encoding="utf-8")


def _max_passes(value: str | None) -> tuple[int, str]:
    """The real function, lifted out and run under the given environment."""
    start = SRC.index("def _max_passes")
    end = SRC.index("MAX_PASSES = _max_passes()")
    env = {**os.environ}
    env.pop("CCARC3_MAX_PASSES", None)
    if value is not None:
        env["CCARC3_MAX_PASSES"] = value
    proc = subprocess.run(
        [sys.executable, "-c",
         "import os\n" + SRC[start:end] + "\nprint(_max_passes())"],
        capture_output=True, text=True, env=env, timeout=60, check=True)
    lines = proc.stdout.strip().splitlines()
    return int(lines[-1]), "\n".join(lines[:-1])


def test_unset_keeps_the_sweep_default():
    """The 25-game sweep must not change behaviour because a knob appeared."""
    assert _max_passes(None) == (12, "")


@pytest.mark.parametrize("value,expected", [("1", 1), ("2", 2), ("25", 25)])
def test_a_number_is_honoured(value, expected):
    assert _max_passes(value)[0] == expected


def test_zero_is_raised_to_one_and_said_out_loud():
    """Zero passes is a driver that launches nothing and reports finishing."""
    passes, said = _max_passes("0")
    assert passes == 1
    assert "below one pass" in said


def test_a_nonsense_value_falls_back_loudly_rather_than_to_one_pass():
    """A typo must not quietly halve a sweep, and must not be silent either."""
    passes, said = _max_passes("banana")
    assert passes == 12, "a typo silently changed the retry bound"
    assert "not a number" in said


def test_an_empty_value_is_not_a_typo():
    """`CCARC3_MAX_PASSES=` from an unset shell variable is absence, not error."""
    passes, said = _max_passes("")
    assert passes == 12 and said == "", f"an empty value warned: {said!r}"
