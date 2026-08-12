"""A shared scorecard is unreachable without its stickiness cookies.

`shared_card.py` opens with a measured probe table saying exactly this: a card is
state on **one backend instance**, and the load balancer's `AWSALB*` cookies are
the only thing that routes a request back to it. Carry them and any process can
play any game against the card; lose them and the card is unreachable while the
API key keeps working perfectly — so the failure reads `game <id> not found` and
points at the wrong thing entirely.

The suite tested the *pinning check* thoroughly and the *cookie carrying* not at
all. Nine of seventeen mutants survived, and they are all one family: export the
names without their values, export nothing, adopt nothing, adopt under the wrong
domain or path, or reach the card from a session that was never pinned. Every one
of them yields a card that opens cleanly, passes its own pinning assertion, and
then fails on the first RESET of the second game — a 25-game sweep that dies
after game one with a message naming the game.

Which is to say: the module documents the mechanism at length and nothing held
the code to it.
"""

from __future__ import annotations

import http.cookiejar
import json
import stat
import urllib.request
from pathlib import Path

import pytest

from athanor.ccarc3 import shared_card as sc


def _opener_with(cookies: list[tuple[str, str, str]]) -> urllib.request.OpenerDirector:
    """An opener whose jar already holds `(name, value, domain)` triples."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    for name, value, domain in cookies:
        jar.set_cookie(http.cookiejar.Cookie(
            0, name, value, None, False, domain, True, domain.startswith("."),
            "/", True, False, None, True, None, None, {}))
    return opener


PINNED = [("AWSALBAPP-0", "sticky-value-0", "three.arcprize.org"),
          ("GAMESESSION", "session-value", "three.arcprize.org")]


# --------------------------------------------------------------------------- #
# export: the values are the credential, not the names
# --------------------------------------------------------------------------- #


def test_export_carries_the_cookie_values():
    """Names alone route nothing. This is the whole payload."""
    exported = sc.export_cookies(_opener_with(PINNED))
    assert exported, "nothing exported at all"
    by_name = {c["name"]: c for c in exported}
    assert by_name["AWSALBAPP-0"]["value"] == "sticky-value-0"
    assert by_name["GAMESESSION"]["value"] == "session-value"


def test_export_carries_the_domain_and_path():
    exported = sc.export_cookies(_opener_with(PINNED))
    assert all(c["domain"] == "three.arcprize.org" for c in exported)
    assert all(c["path"] == "/" for c in exported)


def test_export_of_an_opener_with_no_jar_is_empty_not_a_crash():
    assert sc.export_cookies(urllib.request.build_opener()) == ()


# --------------------------------------------------------------------------- #
# adopt: the round trip is the mechanism
# --------------------------------------------------------------------------- #


def test_a_jar_that_adopted_the_export_holds_the_same_cookies():
    """Export then adopt is how the card crosses a process boundary. If the
    round trip loses anything, every game after the first fails."""
    exported = sc.export_cookies(_opener_with(PINNED))
    jar = http.cookiejar.CookieJar()
    sc.adopt(jar, exported)

    got = {(c.name, c.value, c.domain, c.path) for c in jar}
    want = {(c["name"], c["value"], c["domain"], c["path"]) for c in exported}
    assert got == want, "the round trip changed the jar"


def test_adopt_puts_something_in_an_empty_jar():
    """The positive control for the comparison above: two empty sets are equal."""
    jar = http.cookiejar.CookieJar()
    assert len(list(jar)) == 0
    sc.adopt(jar, sc.export_cookies(_opener_with(PINNED)))
    assert len(list(jar)) == len(PINNED)


def test_an_adopted_cookie_is_sent_to_the_cards_own_host():
    """Domain and path are not decoration — a cookie filed under the wrong
    domain is never sent, and the request lands on a random backend."""
    jar = http.cookiejar.CookieJar()
    sc.adopt(jar, sc.export_cookies(_opener_with(PINNED)))
    request = urllib.request.Request("https://three.arcprize.org/api/scorecard/x/y")
    jar.add_cookie_header(request)
    header = request.get_header("Cookie") or ""
    assert "AWSALBAPP-0=sticky-value-0" in header, f"not sent: {header!r}"


def test_a_cookie_for_another_host_is_not_sent():
    """The control: if `add_cookie_header` sent everything regardless, the test
    above would pass with any domain at all."""
    jar = http.cookiejar.CookieJar()
    sc.adopt(jar, ({"name": "AWSALBAPP-0", "value": "v",
                    "domain": "example.invalid", "path": "/"},))
    request = urllib.request.Request("https://three.arcprize.org/api/scorecard/x/y")
    jar.add_cookie_header(request)
    assert "AWSALBAPP-0" not in (request.get_header("Cookie") or "")


# --------------------------------------------------------------------------- #
# every call that touches the card must be pinned to it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("call", ["close_card", "read_card"])
def test_reaching_the_card_uses_a_session_carrying_its_cookies(monkeypatch, call):
    """A card read or closed from an unpinned session lands on a random backend
    and reports the card as missing — the same misleading failure as the RESET."""
    seen: dict = {}

    def capture(url, *args, **kw):
        opener = kw.get("opener")
        seen["cookies"] = {c.name for c in sc._jar_of(opener)}
        return {}

    monkeypatch.setattr(sc, "_post", lambda url, payload, key, **kw: capture(url, **kw))
    monkeypatch.setattr(sc, "_get", lambda url, key, **kw: capture(url, **kw))
    monkeypatch.setenv("ARC_API_KEY", "k")

    card = sc.SharedCard("card-1", sc.export_cookies(_opener_with(PINNED)))
    if call == "close_card":
        sc.close_card(card)
    else:
        sc.read_card(card, "g1")
    assert "AWSALBAPP-0" in seen["cookies"], (
        f"{call} used a session with no stickiness cookie: {seen['cookies']}"
    )


def test_an_opener_without_a_jar_cannot_hold_a_card():
    """Refusing loudly beats returning a fresh empty jar, which would silently
    produce exactly the unpinned session this module exists to prevent."""
    with pytest.raises(RuntimeError, match="cannot hold a card"):
        sc._jar_of(urllib.request.build_opener())


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def test_the_saved_card_is_never_briefly_world_readable(tmp_path, monkeypatch):
    """The mode is set on the temp file BEFORE the rename. Setting it after
    leaves a window at the final path where the credential half is readable —
    small, and this box is shared with a solver told to read the filesystem.
    """
    observed = []
    real_replace = sc.os.replace

    def watching_replace(src, dst):
        real_replace(src, dst)
        observed.append(stat.S_IMODE(Path(dst).stat().st_mode))

    monkeypatch.setattr(sc.os, "replace", watching_replace)
    sc.save(sc.SharedCard("c", ({"name": "AWSALBAPP-0", "value": "v",
                                 "domain": "d", "path": "/"},)), tmp_path / "card.json")
    assert observed, "os.replace was never called — the write was not atomic"
    assert observed[0] == 0o600, (
        f"the card was mode {observed[0]:o} at the moment it appeared at its "
        "final path, before any later chmod"
    )


def test_a_saved_card_round_trips_with_its_cookies(tmp_path):
    card = sc.SharedCard("card-9", sc.export_cookies(_opener_with(PINNED)))
    back = sc.load(sc.save(card, tmp_path / "card.json"))
    assert back.card_id == "card-9"
    assert back.cookies == card.cookies
    assert back.pinned


# --------------------------------------------------------------------------- #
# the jar must not live where the agent works
# --------------------------------------------------------------------------- #


def test_the_cookie_jar_is_not_written_into_the_agents_workspace(tmp_path):
    """What the leak of 2026-08-11 actually was.

    The jar was a field of `trace.state.json`, which sits in the directory the
    solver works in. An agent ran `cat` on it to check its own progress, the tool
    result went into `stream.jsonl` verbatim, and the preserver pushed a live
    64-character `GAMESESSION` and a full `AWSALBAPP-0` token to a public repo.
    Nothing malfunctioned -- the credential was simply left where the agent works.
    """
    import json as _json
    from athanor.ccarc3 import client as _client

    ws = tmp_path / "attempt_1" / "gm01-abcd"
    ws.mkdir(parents=True)
    c = _client.ArcClient(game_id="gm01-abcd", trace_path=str(ws / "trace.jsonl"),
                          api_key="k", root="http://127.0.0.1:1")
    c._cookiejar().set_cookie(http.cookiejar.Cookie(
        0, "GAMESESSION", "a" * 64, None, False, "127.0.0.1", True, False,
        "/", True, False, None, True, None, None, {}))
    c._save_state()

    state = _json.loads((ws / "trace.state.json").read_text())
    blob = (ws / "trace.state.json").read_text()
    assert "a" * 64 not in blob, "the value is still in the file the agent can cat"
    assert [e["name"] for e in state["cookies"]] == ["GAMESESSION"], (
        "the NAMES must stay -- a vanished field is indistinguishable from a run "
        "that never had a session")

    # Nothing anywhere in the workspace carries it.
    for f in ws.rglob("*"):
        if f.is_file():
            assert "a" * 64 not in f.read_text(errors="ignore"), f"leaked into {f.name}"

    sidecar = c.session_path
    assert sidecar.parent == ws.parent, "the jar must live outside the workspace"
    assert sidecar.name.startswith("."), "and be a dotfile, so `cat *` misses it"
    assert oct(sidecar.stat().st_mode)[-3:] == "600"
    assert "a" * 64 in sidecar.read_text(), "the jar still has to be persisted"


def test_a_resume_still_gets_its_cookies_back(tmp_path):
    """The jar moved; resumption is the reason it exists at all."""
    from athanor.ccarc3 import client as _client

    ws = tmp_path / "attempt_1" / "gm01-abcd"
    ws.mkdir(parents=True)
    kw = dict(game_id="gm01-abcd", trace_path=str(ws / "trace.jsonl"),
              api_key="k", root="http://127.0.0.1:1")
    a = _client.ArcClient(**kw)
    a._cookiejar().set_cookie(http.cookiejar.Cookie(
        0, "GAMESESSION", "z" * 64, None, False, "127.0.0.1", True, False,
        "/", True, False, None, True, None, None, {}))
    a._save_state()

    b = _client.ArcClient(**kw)
    assert b._restore_state() is True
    assert [c.value for c in b._cookiejar()] == ["z" * 64]


def test_a_state_file_from_before_the_move_still_resumes(tmp_path):
    """Backward compatibility, and it is load-bearing during a sweep.

    A run interrupted mid-sweep has a state file written by the old code, with
    real values under `cookies` and no sidecar. Relaunching it under the new code
    must still rebind the card -- otherwise the fix strands exactly the runs it
    was meant to protect.
    """
    import json as _json
    from athanor.ccarc3 import client as _client

    ws = tmp_path / "attempt_1" / "gm01-abcd"
    ws.mkdir(parents=True)
    (ws / "trace.state.json").write_text(_json.dumps({
        "game_id": "gm01-abcd", "card_id": "c" * 8, "actions_used": 7,
        "cookies": [{"name": "GAMESESSION", "value": "y" * 64,
                     "domain": "127.0.0.1", "path": "/"}],
    }))
    (ws / "trace.jsonl").write_text("")

    c = _client.ArcClient(game_id="gm01-abcd", trace_path=str(ws / "trace.jsonl"),
                          api_key="k", root="http://127.0.0.1:1")
    assert c._restore_state() is True
    assert [k.value for k in c._cookiejar()] == ["y" * 64], (
        "an old state file must still restore its jar")
