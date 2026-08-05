"""Allowlist behaviour, without a network or a live game."""
import sys; sys.path.insert(0, "/home/user/athanor/src")
from athanor.ccarc3.arc_proxy import _allowed
CASES = [
    ("/api/cmd/RESET", True), ("/api/cmd/ACTION6", True), ("/api/cmd/ACTION1", True),
    ("/api/scorecard/open", True), ("/api/scorecard/close", True),
    ("/api/scorecard/f597d2be-e064-41ef-8c2e-703b8a32ffec/bp35-0a0ad940", True),
    ("/api/games", False),                 # the one that carries baseline_actions
    ("/api/games/", False), ("/api/games?x=1".split("?")[0], False),
    ("/api/GAMES", False), ("/api/cmd/../games", False),
    ("/api/scorecard/open/../../games", False), ("/", False), ("/api", False),
    ("/api/environments", False),          # unknown endpoint: closed by default
]
def test_allowlist():
    bad = [(p, w, _allowed(p)) for p, w in CASES if _allowed(p) != w]
    assert not bad, bad


bad = [(p, w, _allowed(p)) for p, w in CASES if _allowed(p) != w]
for p, w, g in bad: print(f"  FAIL {p!r}: want {w}, got {g}")
print(f"  {len(CASES)-len(bad)}/{len(CASES)} allowlist cases correct")
sys.exit(1 if bad else 0)
