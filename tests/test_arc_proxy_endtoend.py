"""The proxy end to end, against a stub upstream instead of ARC.

The allowlist and the response filter are unit-tested in
``test_arc_proxy_allowlist.py``. What was never tested is the thing that
actually runs: a real ``ThreadingHTTPServer`` forwarding a real request and
recomputing ``Content-Length`` over a body it just rewrote. Those are the two
places where a filter that works in isolation still breaks a solver -- a stale
length truncates the body, and a keep-alive client then reads the next
response's bytes as the tail of this one.

Everything here talks to a local stub. No ARC traffic, so this is safe to run
while games are in flight.
"""
from __future__ import annotations

import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from athanor.ccarc3 import arc_proxy

# A close-response shaped like the real one: per-run human medians and the
# scores that invert back to them.
UPSTREAM_BODY = {
    "card_id": "26ac1c56-e8b9-4f0b-862c-22271e201316",
    "score": 2.7777777777777777,
    "total_actions": 28,
    "environments": [{
        "id": "lp85-305b61c3",
        "score": 2.7777777777777777,
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


class _Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # keep pytest output clean
        pass

    def _reply(self):
        body = json.dumps(UPSTREAM_BODY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


def _serve(handler):
    """Start a threaded server on an ephemeral port; return (server, port)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    stub, stub_port = _serve(_Stub)
    monkeypatch.setattr(arc_proxy, "UPSTREAM", f"http://127.0.0.1:{stub_port}")
    srv, port = _serve(arc_proxy.Handler)
    try:
        yield port
    finally:
        srv.shutdown()
        stub.shutdown()


def _request(port, path, conn=None):
    own = conn is None
    c = conn or http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        if own:
            c.close()


def test_an_allowed_path_is_forwarded_and_filtered(proxy):
    status, body = _request(proxy, "/api/scorecard/close")
    assert status == 200
    flat = body.decode()
    for banned in ("level_baseline_actions", "level_scores"):
        assert banned not in flat
    got = json.loads(body)
    assert got["card_id"] == UPSTREAM_BODY["card_id"]
    assert got["environments"][0]["runs"][0]["level_actions"] == [7, 0, 0]


def test_content_length_matches_the_filtered_body(proxy):
    """The header must describe the body after filtering, not before.

    Stripping shrinks the payload; a stale Content-Length would leave the client
    blocking for bytes that never arrive, or reading into the next response.
    """
    c = http.client.HTTPConnection("127.0.0.1", proxy, timeout=10)
    try:
        c.request("GET", "/api/scorecard/close")
        r = c.getresponse()
        declared = int(r.getheader("Content-Length"))
        body = r.read()
    finally:
        c.close()
    assert declared == len(body)
    assert declared < len(json.dumps(UPSTREAM_BODY))     # filtering did shrink it


def test_keep_alive_reuse_stays_in_frame(proxy):
    """Several requests down one connection must not bleed into each other."""
    c = http.client.HTTPConnection("127.0.0.1", proxy, timeout=10)
    try:
        for _ in range(8):
            status, body = _request(proxy, "/api/scorecard/close", conn=c)
            assert status == 200
            assert json.loads(body)["card_id"] == UPSTREAM_BODY["card_id"]
    finally:
        c.close()


def test_refused_and_forwarded_paths_interleave_under_concurrency(proxy):
    """The filter is shared state only if someone made it so. Prove it is not."""
    def hit(i):
        if i % 3 == 0:
            status, body = _request(proxy, "/api/games")
            return status, "level_baseline_actions" in body.decode()
        status, body = _request(proxy, "/api/scorecard/close")
        return status, "level_baseline_actions" in body.decode()

    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(hit, range(120)))

    assert not any(leaked for _, leaked in results), "a baseline escaped the filter"
    assert {s for s, _ in results} == {403, 200}
    assert sum(1 for s, _ in results if s == 403) == 40


def test_games_is_still_refused_end_to_end(proxy):
    status, body = _request(proxy, "/api/games")
    assert status == 403
    assert b"level_baseline_actions" not in body


# --- session cookies ------------------------------------------------------- #
# ARC binds a scorecard to the HTTP session, not the API key: open a card on one
# connection and RESET on another and the server answers "game <id> not found".
# A proxy that builds a clean request per call breaks every game it touches.

SEEN_COOKIES: list[str | None] = []


class _CookieStub(_Stub):
    """Records the Cookie it was sent and always issues a stickiness cookie."""

    def _reply(self):
        SEEN_COOKIES.append(self.headers.get("Cookie"))
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", "GAMESESSION=abc123; Path=/")
        self.send_header("Set-Cookie", "AWSALBAPP-0=sticky; Path=/")
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


@pytest.fixture
def cookie_proxy(monkeypatch):
    SEEN_COOKIES.clear()
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    stub, stub_port = _serve(_CookieStub)
    monkeypatch.setattr(arc_proxy, "UPSTREAM", f"http://127.0.0.1:{stub_port}")
    srv, port = _serve(arc_proxy.Handler)
    try:
        yield port
    finally:
        srv.shutdown()
        stub.shutdown()


def test_set_cookie_headers_reach_the_client(cookie_proxy):
    """Both of them: send_header must be called per cookie, not once."""
    c = http.client.HTTPConnection("127.0.0.1", cookie_proxy, timeout=10)
    try:
        c.request("GET", "/api/scorecard/open")
        r = c.getresponse()
        cookies = r.getheader("Set-Cookie") or ""
        r.read()
    finally:
        c.close()
    assert "GAMESESSION=abc123" in cookies
    assert "AWSALBAPP-0=sticky" in cookies


def test_the_clients_cookie_is_never_forwarded_upstream(cookie_proxy):
    """This used to assert the opposite, and the opposite broke the scorecard.

    Relaying the client's `Cookie` header meant `http.cookiejar` would not touch
    the request — it refuses to set a header that is already present — so the
    proxy's own session was silenced and the client's cookies drove the routing
    instead. The client's jar holds the ALB's `AWSALBAPP-N=_remove_` tombstones
    as though they were values and sends them back, which unpins the session.
    ARC then answers `404 card_id not found` for a card it created moments
    earlier.

    Measured on the first clean rollout: 9 of 12 scorecard reads 404'd, the
    snapshot froze at level 6 of 8, and the run still finished 8/8 — actions
    carry a `guid` and are not card-scoped, so nothing else showed a symptom.

    The proxy owns the upstream session. It holds the key, and the session
    belongs with the credential.
    """
    c = http.client.HTTPConnection("127.0.0.1", cookie_proxy, timeout=10)
    try:
        c.request("GET", "/api/cmd/RESET",
                  headers={"Cookie": "GAMESESSION=xyz789; AWSALBAPP-0=_remove_"})
        c.getresponse().read()
    finally:
        c.close()
    assert SEEN_COOKIES, "upstream was never reached"
    assert "xyz789" not in (SEEN_COOKIES[-1] or "")
    assert "_remove_" not in (SEEN_COOKIES[-1] or "")


def test_the_proxy_keeps_one_session_across_calls(cookie_proxy):
    """What replaces it: the cookie the *stub* issued comes back on the next call.

    That is the pinning the scorecard depends on, now held here instead of
    round-tripped through a client that mangles it.
    """
    import urllib.request

    arc_proxy._reset_session()
    root = f"http://127.0.0.1:{cookie_proxy}"
    urllib.request.urlopen(f"{root}/api/scorecard/open", timeout=10).read()
    urllib.request.urlopen(f"{root}/api/cmd/RESET", timeout=10).read()
    assert "GAMESESSION=abc123" in (SEEN_COOKIES[-1] or ""), (
        "the proxy did not carry its own session into the second call"
    )


def test_a_cookie_jar_client_keeps_its_session_across_calls(cookie_proxy):
    """The real client's shape: an opener with a jar, several calls, one session.

    This is the end-to-end property that matters -- open a card, then act, and
    have the second call carry what the first was given.
    """
    import http.cookiejar
    import urllib.request

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    root = f"http://127.0.0.1:{cookie_proxy}"

    opener.open(urllib.request.Request(f"{root}/api/scorecard/open")).read()
    assert {c.name for c in jar} == {"GAMESESSION", "AWSALBAPP-0"}

    opener.open(urllib.request.Request(f"{root}/api/cmd/RESET")).read()
    assert "GAMESESSION=abc123" in (SEEN_COOKIES[-1] or "")


# --- the action cap --------------------------------------------------------- #
# The cap lives here rather than in the workspace client because the cap *is* the
# secret: it is `budget_multiple` times the baseline total, and `budget_multiple`
# is a default in this package's source, on the solver's PYTHONPATH. A solver that
# reads `CCARC3_MAX_ACTIONS` out of its own environment divides by five and has
# the number the whole baseline-free arm exists to withhold.


@pytest.fixture(autouse=True)
def _clear_budget():
    """Module globals, so an armed cap would leak into every later test."""
    arc_proxy.set_budget(0)
    yield
    arc_proxy.set_budget(0)


def test_actions_are_refused_once_the_cap_is_reached(proxy):
    arc_proxy.set_budget(3)
    for i in range(3):
        status, _ = _request(proxy, "/api/cmd/ACTION1")
        assert status == 200, f"action {i + 1} of 3 should be inside the budget"

    status, body = _request(proxy, "/api/cmd/ACTION1")
    assert status == 403, "the fourth action is over the cap"
    # 4xx and not 429: `client._send` raises on 4xx without retrying, so the
    # solver gets one terminal error rather than three rounds of backoff.
    assert "budget exhausted" in json.loads(body)["error"]


def test_reset_is_charged_like_any_other_action(proxy):
    """`ArcClient.actions_used` increments on RESET, so the cap must too.

    Both counters increment on the same `/api/cmd/*` call. If the proxy skipped
    RESET the two would drift by one per replay, and a solver replaying six
    levels would quietly earn six extra actions.
    """
    arc_proxy.set_budget(1)
    assert _request(proxy, "/api/cmd/RESET")[0] == 200
    assert _request(proxy, "/api/cmd/ACTION1")[0] == 403


def test_scorecard_calls_are_not_charged(proxy):
    """Opening and closing a card is bookkeeping, not play."""
    arc_proxy.set_budget(1)
    for path in ("/api/scorecard/open", "/api/scorecard/close"):
        assert _request(proxy, path)[0] == 200
    assert _request(proxy, "/api/cmd/ACTION1")[0] == 200, "the one action survived"


def test_a_resumed_game_does_not_get_its_budget_back(proxy):
    """The count is in memory here and on disk there.

    A container replacement restarts the driver, and with it the proxy, while the
    solver's `trace.state.json` still records what it spent. Seeding from that
    file is what stops the cap binding at `used + max` and growing with every
    interruption -- which would have made the budget a function of how often the
    box was replaced.
    """
    arc_proxy.set_budget(3, used=2)
    assert _request(proxy, "/api/cmd/ACTION1")[0] == 200, "one action left"
    assert _request(proxy, "/api/cmd/ACTION1")[0] == 403


def test_an_unarmed_proxy_does_not_cap_anything(proxy):
    """`set_budget` is per game; a bare proxy must forward without limit."""
    for _ in range(5):
        assert _request(proxy, "/api/cmd/ACTION1")[0] == 200


def test_a_refused_action_is_not_charged(proxy):
    """A path off the allowlist never reaches upstream, so it costs nothing."""
    arc_proxy.set_budget(1)
    assert _request(proxy, "/api/games")[0] == 403
    assert _request(proxy, "/api/cmd/ACTION1")[0] == 200, "the refusal spent budget"
