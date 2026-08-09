"""Every ALLOW pattern must be anchored, because `_allowed` relies on it.

Found while mutation-auditing the proxy: swapping `p.match(path)` for
`p.search(path)` in `_allowed` passed all 110 tests. That is an **equivalent
mutant today** and only because every pattern in `ALLOW` begins with `^` and ends
with `$` -- with no `re.MULTILINE`, `^` matches at position 0 and nowhere else, so
`search` cannot start later than `match` would.

The equivalence is a property of the four patterns, not of the code, and nothing
held it there. Add one pattern without a leading `^` and `match` still refuses
`/evil/api/whatever` while `search` forwards it; add one without a trailing `$`
and both forward `/api/scorecard/open/../../games`. The allowlist is the whole
network gate -- `/api/games` is excluded from it deliberately because it returns
`baseline_actions` for all 25 environments -- so the anchoring is load-bearing and
should be asserted rather than assumed.

This is the same shape as the rest: a check that holds by a coincidence of the
data it was written against.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from athanor.ccarc3.arc_proxy import ALLOW, _allowed  # noqa: E402


@pytest.mark.parametrize("pattern", ALLOW, ids=lambda p: p.pattern)
def test_every_allowed_pattern_is_anchored(pattern):
    src = pattern.pattern
    assert src.startswith("^"), (
        f"{src!r} has no leading anchor: a path with anything prepended to it "
        f"would be forwarded"
    )
    assert src.endswith("$"), (
        f"{src!r} has no trailing anchor: a path with anything appended to it "
        f"would be forwarded"
    )
    assert not pattern.flags & re.MULTILINE, (
        f"{src!r} is MULTILINE, so '^' matches after any newline and the "
        f"leading anchor stops meaning what it reads as"
    )


@pytest.mark.parametrize("path", [
    "/evil/api/cmd/RESET",
    "https://example.com/api/cmd/RESET",
    "/redirect?to=/api/scorecard/open",
    "/api/cmd/RESET/../../games",
    "/api/scorecard/open.json",
    "/api/scorecard/openX",
])
def test_a_prefix_or_suffix_does_not_open_an_endpoint(path):
    """What the anchors buy, stated as behaviour rather than as regex syntax."""
    assert not _allowed(path), f"anchoring did not hold for {path!r}"


def test_api_games_stays_out_however_it_is_dressed():
    """The one exclusion the whole allowlist exists for."""
    for path in ["/api/games", "/api/games/", "/x/api/games", "/api/games?x=1",
                 "/api/cmd/RESET/../games", "//api/games"]:
        assert not _allowed(path), f"/api/games reachable as {path!r}"
