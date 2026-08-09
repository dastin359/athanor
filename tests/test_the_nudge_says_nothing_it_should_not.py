"""The give-up nudge must not hand back what the harness spends machinery hiding.

When a solver quits with its allowance untouched, `run_game` resumes its session
and tells it to carry on. That message goes straight into the solver's context,
and the module holding it sits on the solver's own `PYTHONPATH` — the file's own
comments record that `inspect.getsource` is one of the first things a careful
solver runs, and that an action count written beside a baseline total is a human
median in two subtractions.

So both the prompt *and* the prose around it are solver-reachable surface. The
first draft of this feature passed every existing leak test while its docstring
explained, in order: that a hidden cap exists, that it equals the withheld
per-level total times `budget_multiple`, and that `CCARC3_MAX_ACTIONS` is
stripped from the child's environment and enforced in the proxy instead. That is
a map to three things the harness hides, and no test objected — which is why this
file exists rather than a note saying "be careful".
"""

from __future__ import annotations

import inspect
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from athanor.ccarc3 import session as sess  # noqa: E402

# What a solver must not be handed. `budget_multiple` and the env var are the
# two halves of the recovery: cap / multiple is the withheld total.
FORBIDDEN = [
    "budget_multiple",
    "CCARC3_MAX_ACTIONS",
    "baseline_actions",
    "level_baseline_actions",
    "human median",
]


def _nudge_surface() -> str:
    """The prompt plus its own documentation — both reach the solver."""
    doc = ""
    for name, value in vars(sess).items():
        if name == "NUDGE_PROMPT":
            # A module-level string's docstring is the source line after it.
            src = inspect.getsource(sess)
            i = src.index("NUDGE_PROMPT = ")
            doc = src[i:i + 2400]
    return sess.NUDGE_PROMPT + "\n" + doc


def test_the_prompt_states_no_count():
    """"You have actions left" is fair. A number is not."""
    numbers = re.findall(r"(?<![\w.])\d+(?![\w.])", sess.NUDGE_PROMPT)
    assert numbers == [], (
        f"the nudge names {numbers}; a solver reading a figure beside its own "
        f"spend learns the ceiling, which the proxy exists to withhold"
    )


@pytest.mark.parametrize("term", FORBIDDEN)
def test_neither_the_prompt_nor_its_documentation_names_the_mechanism(term):
    surface = _nudge_surface()
    assert term.lower() not in surface.lower(), (
        f"{term!r} appears in the nudge's solver-reachable surface. This module "
        f"is on the solver's PYTHONPATH; naming the concealment mechanism is the "
        f"same disclosure as naming the number."
    )


def test_the_prompt_still_says_the_useful_thing():
    """A positive control: reticence must not have emptied the message.

    Without this, deleting the prompt entirely passes every assertion above.
    """
    text = sess.NUDGE_PROMPT.lower()
    assert "stop" in text, "the nudge does not name what the solver did wrong"
    assert "actions" in text, "the nudge does not tell the solver it has budget"
    assert "0b" in text, "the nudge does not point at the doctrine it is enforcing"
    assert len(sess.NUDGE_PROMPT.split()) > 40, "the nudge is too thin to act on"


def test_the_scan_can_fail():
    """The rule has to be able to catch something, or it asserts nothing."""
    planted = "you have 2614 actions remaining of your budget_multiple cap"
    assert re.findall(r"(?<![\w.])\d+(?![\w.])", planted), "the number scan is blind"
    assert any(t.lower() in planted.lower() for t in FORBIDDEN), "the term scan is blind"
