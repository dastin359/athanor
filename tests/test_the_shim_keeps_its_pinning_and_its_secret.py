"""Two invariants of the proxy's per-game state that nothing was holding.

Both surfaced as surviving mutants in the 2026-08-09 audit of `arc_proxy`.

**1. `set_budget` drops the session, and must not drop the adopted one.**
A card is reachable only from a session carrying its four `AWSALBAPP-*`
stickiness cookies -- measured live, 8 of 8 card reads succeed direct and 1 of 8
through a shim without them. `proxy_for()` adopts the driver's cookies when a
sweep shares one card, and `build_without_baselines` then calls `set_budget`,
which resets the session on purpose so a new game cannot inherit the previous
one's pinning. The two only coexist because `_adopted` is stored apart from the
live jar and re-seeded afterwards. Its own docstring says storing them in the jar
alone "would make the order of these two calls silently load-bearing" -- and
nothing tested either half, so the reset could have been deleted (or the
separation collapsed) with the suite green and the whole 25-game shared-card
sweep failing its first RESET with `game <id> not found`.

**2. The exhaustion refusal must not name the cap.** The cap is the withheld
per-level total times `budget_multiple`, and `budget_multiple` defaults to 5.0 in
this package's source on the solver's `PYTHONPATH`. `CCARC3_MAX_ACTIONS` is
popped from the child's environment precisely so the number stays out of reach;
printing it in the refusal hands it straight back, and division recovers the
medians' total. A mutant that formatted the ceiling into the message passed all
110 proxy tests.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from athanor.ccarc3.arc_proxy import ProxyState  # noqa: E402

COOKIES = (
    {"name": "AWSALBAPP-0", "value": "abc", "domain": "three.arcprize.org"},
    {"name": "AWSALBAPP-1", "value": "def", "domain": "three.arcprize.org"},
)


def _jar_names(state: ProxyState) -> set[str]:
    opener = state.upstream()
    for handler in opener.handlers:
        jar = getattr(handler, "cookiejar", None)
        if jar is not None:
            return {c.name for c in jar}
    return set()


def test_arming_the_budget_keeps_an_adopted_pinning():
    """The order of `adopt_session` and `set_budget` must not matter."""
    adopt_first = ProxyState()
    adopt_first.adopt_session(COOKIES)
    adopt_first.set_budget(500)
    assert _jar_names(adopt_first) == {"AWSALBAPP-0", "AWSALBAPP-1"}, (
        "set_budget dropped the shared card's stickiness cookies; every game in "
        "the sweep would route to a backend that never heard of the card"
    )

    budget_first = ProxyState()
    budget_first.set_budget(500)
    budget_first.adopt_session(COOKIES)
    assert _jar_names(budget_first) == _jar_names(adopt_first), (
        "the two call orders disagree, which is the failure the separate "
        "`_adopted` store exists to prevent"
    )


def test_arming_the_budget_drops_a_session_that_was_not_adopted():
    """A shim reused for another game must not carry the old pinning."""
    state = ProxyState()
    first = state.upstream()
    state.set_budget(500)
    assert state.upstream() is not first, (
        "the session survived set_budget; a new game would inherit the previous "
        "game's pinning"
    )


def test_a_lent_card_is_marked_as_lent():
    """`card_is_lent` is what stops a game closing the driver's card."""
    state = ProxyState()
    assert not state.card_is_lent
    state.adopt_session(COOKIES)
    assert state.card_is_lent
    state.set_budget(500)
    assert state.card_is_lent, "arming the budget forgot the card was borrowed"


def test_the_refusal_never_names_the_ceiling():
    """The cap divided by `budget_multiple` is the withheld total, exactly."""
    state = ProxyState()
    state.set_budget(1234)
    assert state.exhausted() is None, "the ceiling bound before any action"

    for i in range(1234):
        assert state.reserve() is None, f"the ceiling bound early, at action {i}"
    message = state.exhausted()

    assert message, "the ceiling did not bind at the cap"
    for forbidden in ("1234", "1,234", str(1234 // 5), str(1234 / 5)):
        assert forbidden not in message, (
            f"the refusal names {forbidden!r}; the cap is the withheld per-level "
            f"total times budget_multiple, so printing it returns the secret "
            f"that popping CCARC3_MAX_ACTIONS was meant to withhold"
        )
    assert not any(ch.isdigit() for ch in message), (
        f"the refusal carries a figure at all: {message!r}"
    )


def test_the_ceiling_binds_exactly_at_the_cap_not_one_past_it():
    state = ProxyState()
    state.set_budget(3)
    for expected in (None, None, None):
        assert state.exhausted() is expected
        assert state.reserve() is None, "a slot inside the cap was refused"
    assert state.exhausted() is not None, "a fourth action was allowed through"
    assert state.reserve() is not None, "a fourth action was allowed through"
