"""The proxy allowlist, checked without a network or a live game.

The whole point of `arc_proxy` is that `/api/games` -- the endpoint that carries
`baseline_actions` for all 25 environments -- is unreachable from a solver. That
property lives entirely in one regex tuple, so it is worth pinning directly.
"""
import pytest

from athanor.ccarc3.arc_proxy import _allowed

ALLOWED = [
    "/api/cmd/RESET",
    "/api/cmd/ACTION1",
    "/api/cmd/ACTION6",
    "/api/scorecard/open",
    "/api/scorecard/close",
    "/api/scorecard/f597d2be-e064-41ef-8c2e-703b8a32ffec/bp35-0a0ad940",
]

REFUSED = [
    "/api/games",                      # the one that carries baseline_actions
    "/api/games/",
    "/api/GAMES",                      # case-folding must not open it
    "/api/cmd/../games",               # traversal is not a cmd
    "/api/scorecard/open/../../games",
    "/api/environments",               # unknown endpoint: closed by default
    "/api",
    "/",
]


@pytest.mark.parametrize("path", ALLOWED)
def test_solver_endpoints_are_forwarded(path):
    assert _allowed(path)


@pytest.mark.parametrize("path", REFUSED)
def test_everything_else_is_refused(path):
    assert not _allowed(path)


def test_query_string_is_stripped_before_matching():
    # _forward() splits on "?" first; a query must not be a way to smuggle a
    # non-matching path past the anchored regexes.
    assert not _allowed("/api/games?x=1".split("?", 1)[0])
    assert _allowed("/api/cmd/RESET?x=1".split("?", 1)[0])
