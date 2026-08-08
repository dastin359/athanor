"""The proxy allowlist, checked without a network or a live game.

The whole point of `arc_proxy` is that `/api/games` -- the endpoint that carries
`baseline_actions` for all 25 environments -- is unreachable from a solver. That
property lives entirely in one regex tuple, so it is worth pinning directly.
"""
import json

import pytest

from athanor.ccarc3 import arc_proxy
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


# --- response filtering ---------------------------------------------------- #
# Closing the allowlist was not sufficient: /api/scorecard/close is a call every
# solver makes, and its body carries level_baseline_actions for every level.

CLOSE_BODY = {
    "card_id": "26ac1c56-e8b9-4f0b-862c-22271e201316",
    "score": 2.7777777777777777,
    "total_actions": 28,
    "tags_scores": [{"id": "click", "score": 2.777, "actions": 7}],
    "environments": [{
        "id": "lp85-305b61c3",
        "score": 2.7777777777777777,
        "actions": 28,
        "levels_completed": 1,
        "runs": [{
            "guid": "0f964409-4919-4322-b244-1556786515f2",
            "actions": 7,
            "state": "NOT_FINISHED",
            "level_actions": [7, 0, 0],
            "level_baseline_actions": [17, 38, 31],
            "level_scores": [115.0, 0.0, 0.0],
            "score": 2.777,
        }],
    }],
}


def _round_trip(payload):
    return json.loads(arc_proxy._filtered(json.dumps(payload).encode()))


# **`CLOSE_BODY` names 4 of the 6 hidden fields, so 2 were pinned by nothing.**
# Deleting `baseline_actions` from `HIDDEN_FIELDS` left the whole suite green; so
# did deleting `scores`. `baseline_actions` is the exact key `/api/games` returns
# for all 25 environments -- the one field name this module exists to remove --
# and it was the one no fixture contained.
#
# Two tests are needed and neither substitutes for the other. Parametrising over
# `HIDDEN_FIELDS` proves the filter reaches every entry at every depth, but it
# *shrinks* when an entry is deleted, so on its own it is green for a frozenset
# that has quietly lost a name. The roll-call below is what fails then.

REQUIRED_HIDDEN = {
    # field                   emitted by
    "baseline_actions":       "GET /api/games, for all 25 environments",
    "level_baseline_actions": "POST /api/scorecard/close, per run",
    "level_scores":           "POST /api/scorecard/close, per run",
    "score":                  "close, at card / environment / run depth",
    "scores":                 "card-level aggregate",
    "tags_scores":            "close, per tag",
}


def test_no_hidden_field_may_be_dropped_from_the_frozenset():
    """The roll-call. Every name here is a field the API really sends, so
    removing one from `HIDDEN_FIELDS` hands it to the solver."""
    missing = sorted(set(REQUIRED_HIDDEN) - set(arc_proxy.HIDDEN_FIELDS))
    assert not missing, (
        "no longer withheld: "
        + "; ".join(f"{f} ({REQUIRED_HIDDEN[f]})" for f in missing)
    )


@pytest.mark.parametrize("field", sorted(arc_proxy.HIDDEN_FIELDS))
def test_every_hidden_field_is_stripped_at_every_depth(field):
    """Depth coverage, for whatever the frozenset currently holds. ARC's score is
    100*min(1.15, (h/a)^2) and the solver knows its own `a`, so any one of these
    surviving at any depth inverts back to the human median."""
    body = {field: [17, 38, 31], "keep": 1,
            "environments": [{field: 1, "id": "lp85-305b61c3",
                              "runs": [{field: [1, 2], "actions": 7}]}]}
    out = _round_trip(body)
    assert field not in json.dumps(out), f"{field} survived the filter"
    assert out["keep"] == 1 and out["environments"][0]["runs"][0]["actions"] == 7


def test_close_response_loses_every_baseline_field():
    out = _round_trip(CLOSE_BODY)
    flat = json.dumps(out)
    for banned in ("level_baseline_actions", "level_scores", "tags_scores"):
        assert banned not in flat, f"{banned} survived the filter"


def test_score_fields_are_stripped_at_every_depth():
    out = _round_trip(CLOSE_BODY)
    assert "score" not in out
    assert "score" not in out["environments"][0]
    assert "score" not in out["environments"][0]["runs"][0]


def test_the_fields_the_client_needs_survive():
    out = _round_trip(CLOSE_BODY)
    assert out["card_id"] == CLOSE_BODY["card_id"]
    assert out["total_actions"] == 28
    env = out["environments"][0]
    assert env["id"] == "lp85-305b61c3"
    assert env["levels_completed"] == 1
    run = env["runs"][0]
    assert run["level_actions"] == [7, 0, 0]     # our own actions are not secret
    assert run["guid"] and run["state"] == "NOT_FINISHED"


def test_actions_by_level_survives():
    # snapshot_scorecard reads this and RHAE is checked against it.
    body = {"cards": {"lp85": {"actions_by_level": [[[1, 7], [2, 45]]],
                               "total_actions": 45, "states": ["WIN"]}}}
    assert _round_trip(body) == body


def test_non_json_body_passes_through_unchanged():
    # A truncated or empty 5xx body must reach the client untouched; the client
    # has careful error handling and cannot contain the fields anyway.
    for raw in (b"", b"upstream exploded", b"{not json"):
        assert arc_proxy._filtered(raw) == raw


def test_a_shim_refuses_another_game_and_does_not_bill_its_own_budget(monkeypatch):
    """The cap was per shim but not per game, so a sibling paid for your actions.

    `ProxyState` held only `(max_actions, actions_used)` and `_forward` never
    looked at which game a request named, so a shim armed for game B would
    forward game A's `/api/cmd/*` and charge B. Games run concurrently and every
    sibling's port is a loopback scan away, so this is a cap that a solver can
    spend out of someone else's pocket — and the ceiling was moved out of the
    child's environment in the first place so it could not be raised.
    """
    import json as _json
    import urllib.error
    import urllib.request

    from athanor.ccarc3 import arc_proxy

    # The shim refuses to start without a key, and this test never reaches the
    # network. Setting one keeps the verdict from depending on whether the shell
    # that launched pytest had sourced `.env`.
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-used-offline")

    mine = arc_proxy.Proxy(game_id="aaaa-1111")
    theirs = arc_proxy.Proxy(game_id="bbbb-2222")
    theirs.set_budget(10)
    try:
        req = urllib.request.Request(
            f"{theirs.url}/api/cmd/ACTION1",
            data=_json.dumps({"game_id": "aaaa-1111"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=8)
            raise AssertionError("a sibling's shim forwarded another game's action")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, f"expected a refusal, got {exc.code}"
        assert theirs.actions_used == 0, "the sibling was billed for a foreign action"
    finally:
        for p in (mine, theirs):
            try:
                p.close()
            except Exception:                      # noqa: BLE001
                pass



def test_a_lent_card_cannot_be_closed_through_the_shim(monkeypatch):
    """**The one forwarded endpoint that can end the whole sweep.**

    `/api/scorecard/close` is on the allowlist because a per-game run closes its
    own card through this shim. Under the sweep's shared card it would finalize
    a submission artifact carrying 25 games' scores, mid-run. `ArcClient.close`
    already declines for a card it does not own — but that guard lives in the
    client, and a solver reaches the shim with nine lines of urllib.
    """
    import json as _json
    import urllib.error
    import urllib.request

    from athanor.ccarc3 import arc_proxy

    monkeypatch.setenv("ARC_API_KEY", "test-key-not-used-offline")

    lent = arc_proxy.Proxy(game_id="aaaa-1111")
    lent.adopt_session(({"name": "AWSALBAPP-0", "value": "x"},))
    try:
        req = urllib.request.Request(
            f"{lent.url}/api/scorecard/close",
            data=_json.dumps({"card_id": "the-sweep-card"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=8)
            raise AssertionError("the shim closed a card lent by the driver")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, f"expected a refusal, got {exc.code}"
    finally:
        lent.shutdown()


def test_a_shim_that_owns_its_card_may_still_close_it(monkeypatch):
    """The refusal must be conditional, and this has to DRIVE the request.

    Its first version only inspected `card_is_lent` and the allowlist, so making
    the refusal unconditional -- which would break every per-game run, the reason
    `close` is on the allowlist at all -- left it green. It now posts a close
    through a shim with no adopted session and requires the failure NOT to be the
    lent-card refusal. The upstream call cannot succeed offline; what matters is
    which wall it hits.
    """
    import json as _json
    import urllib.error
    import urllib.request

    from athanor.ccarc3 import arc_proxy

    monkeypatch.setenv("ARC_API_KEY", "test-key-not-used-offline")
    own = arc_proxy.Proxy(game_id="bbbb-2222")
    try:
        assert own.state.card_is_lent is False, (
            "a shim with no adopted session must not think its card is lent"
        )
        assert arc_proxy._allowed("/api/scorecard/close"), (
            "close must stay on the allowlist for runs that own their card"
        )
        req = urllib.request.Request(
            f"{own.url}/api/scorecard/close",
            data=_json.dumps({"card_id": "my-own-card"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        body, code = "", None
        try:
            body = urllib.request.urlopen(req, timeout=15).read().decode()
        except urllib.error.HTTPError as exc:
            code, body = exc.code, exc.read().decode(errors="ignore")
        except urllib.error.URLError:
            body = "(upstream unreachable, which is fine offline)"
        assert "lent by the driver" not in body, (
            f"a shim that owns its card was refused as if the card were lent "
            f"(code={code}): {body[:200]}"
        )
    finally:
        own.shutdown()


def test_a_refused_request_with_a_body_does_not_desync_the_connection(monkeypatch):
    """**A defence that corrupts the traffic it permits.**

    `_refuse` answered and returned with the request body still unread, so on a
    keep-alive connection the leftover bytes were parsed as the next request
    line — and the victim is the NEXT call, which is a legitimate one. The body
    is drained before any response is emitted now.
    """
    import http.client
    import json as _json

    from athanor.ccarc3 import arc_proxy

    monkeypatch.setenv("ARC_API_KEY", "test-key-not-used-offline")
    px = arc_proxy.Proxy(game_id="aaaa-1111")
    try:
        host, port = px.url.split("//", 1)[1].split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=10)
        # A refused path, carrying a body — /api/games is the one the allowlist exists for.
        conn.request("POST", "/api/games", body=_json.dumps({"x": "y" * 200}),
                     headers={"Content-Type": "application/json"})
        first = conn.getresponse()
        first.read()
        assert first.status == 403, f"expected a refusal, got {first.status}"

        # The next call on the SAME connection must still be parsed as a request.
        conn.request("POST", "/api/cmd/RESET",
                     body=_json.dumps({"game_id": "aaaa-1111"}),
                     headers={"Content-Type": "application/json"})
        second = conn.getresponse()
        second.read()
        assert second.status != 400, (
            "the refused request's body was parsed as the next request line — "
            "the connection is desynced and a legitimate call paid for it"
        )
        conn.close()
    finally:
        px.shutdown()
