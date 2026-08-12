"""The leak guard discarded a winning run because a document had a line 426.

`proofread_trace` proves two numbers never reached the solver: the human
baseline, and the solver's own action cap. It checks the cap by computing the
value and searching everything the solver read for those digits.

Claude Code's `Read` returns files as `<lineno><TAB><content>`, so **every line
number of every document a solver opens is an integer in that haystack**.

Measured 2026-08-11 on the submission sweep. sb26 cleared 8 of 8 levels and was
discarded on `FAIL LEAK: budget 426 present in tool results`. sb26's baseline
total is 213; 213 x 2 = 426, the cap under `ablate_baselines`' multiple, which
that run never used. The match was line 426 of the doctrine. The same verdict
block reported the run's actual cap, 1065, `absent`.

Nothing had leaked. The check named "the budget reached the solver" and read
"these digits occur somewhere" -- and the gap between the two claims is filled
by the line numbers of every file on disk.

**Long files make it worse rather than better.** A real cap of 1065 collides
with line 1065 of any document over a thousand lines, so the correct number was
heading for the same false positive; the wrong multiple merely got there first.
"""
from __future__ import annotations

import re
import sys

sys.path.insert(0, "tools")
import proofread_trace as pt


READ_OUTPUT = (
    "   422\t\n"
    "   423\tThis bites in a plain list comprehension exactly as hard as in a rule:\n"
    "   424\t\n"
    "   425\t```python\n"
    "   426\tts = client.transitions()\n"
    "   427\t[t for t in ts if t.action == \"ACTION1\"]\n"
)


def test_a_line_number_is_stripped_before_numeric_checks():
    numeric = pt._LINE_GUTTER.sub("", READ_OUTPUT)
    assert not re.search(r"\b426\b", numeric), (
        "line 426 still reads as the value 426; this is the false positive that "
        "discarded an 8-of-8 run"
    )
    # The content itself must survive -- stripping a gutter is not stripping text.
    assert "ts = client.transitions()" in numeric
    assert "```python" in numeric


def test_the_real_cap_in_content_is_still_caught():
    """The guard must still fire when the number arrives as a value, not a gutter."""
    leaked = "Your action budget for this game is 1065 actions.\n"
    numeric = pt._LINE_GUTTER.sub("", leaked)
    assert re.search(r"\b1065\b", numeric), "a genuine cap leak stopped being detected"


def test_a_gutter_number_and_a_real_leak_are_told_apart():
    """Both in one blob: the line number must vanish, the value must remain."""
    blob = READ_OUTPUT + "the cap is 426 actions, spend them well\n"
    numeric = pt._LINE_GUTTER.sub("", blob)
    assert re.search(r"\b426\b", numeric), (
        "stripping the gutter also swallowed a real mention of the same number"
    )
    assert numeric.count("426") == 1, "the line-number occurrence survived"


def test_the_gutter_pattern_does_not_eat_ordinary_content():
    """A tab after a number mid-line is not a gutter, and must be left alone."""
    body = "level\t3\nactions\t426\n"
    assert pt._LINE_GUTTER.sub("", body) == body


def test_a_six_digit_line_number_is_still_a_gutter():
    assert pt._LINE_GUTTER.sub("", "123456\tcontent\n") == "content\n"


def test_a_seven_digit_number_is_not_treated_as_a_gutter():
    """Bounded on purpose, so the pattern cannot start eating long values."""
    assert pt._LINE_GUTTER.sub("", "1234567\tcontent\n") == "1234567\tcontent\n"
