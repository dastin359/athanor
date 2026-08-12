"""The action ceiling has to bind under concurrency, not only in sequence.

`_forward` asked the ceiling and billed it in **two separate lock
acquisitions**, with the upstream round-trip sitting between them:

    over = self.state.exhausted()      # lock taken, released
    ...  self.state.upstream().open(req, timeout=120)  ...
    self.state.charge()                # lock taken, released

Each acquisition is individually correct, which is why reading the code does not
show the bug -- `ProxyState` is scrupulous about its lock everywhere. The gap is
between them. Every request already in flight has passed the check and not yet
paid, so N concurrent actions at the ceiling all read "not exhausted" and all
proceed. The cap overshoots by N-1, and N is however many connections the caller
opens.

**The caller is the party the cap exists to constrain.** `arc_proxy`'s own header
explains that the ceiling was moved out of the solver's environment because "a
limit the solver can read is a limit the solver can invert", and that the solver
reaches this shim over a loopback URL with no key. Nothing about that design
stops it opening fifty connections at once, and the window is not a few
instructions -- it is a whole HTTP round-trip to ARC, up to the 120-second
timeout.

The fix is to take the slot and the decision under one lock, and hand it back if
upstream did not accept -- preserving the documented rule that "a 502 costs the
solver nothing at either end".
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from athanor.ccarc3 import arc_proxy

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _SlowStub(BaseHTTPRequestHandler):
    """Upstream that dawdles, so the check-to-charge window is wide enough to aim at.

    The window is real at any speed; the sleep only removes the need to win a
    race by luck. A test that reproduces a concurrency bug one run in twenty is
    indistinguishable from a test that does not reproduce it at all.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self):
        time.sleep(0.25)
        body = json.dumps({"score": 0, "state": "NOT_FINISHED"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


class _FailingStub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self):
        body = b'{"error": "upstream fell over"}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


def _serve(handler, state=None):
    cls = ThreadingHTTPServer if state is None else arc_proxy._Server
    srv = cls(("127.0.0.1", 0), handler)
    if state is not None:
        srv.state = state
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


@pytest.fixture
def shim(request, monkeypatch):
    """A proxy with its own ProxyState, so nothing leaks between tests."""
    upstream_cls = getattr(request, "param", _SlowStub)
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    stub, stub_port = _serve(upstream_cls)
    monkeypatch.setattr(arc_proxy, "UPSTREAM", f"http://127.0.0.1:{stub_port}")
    state = arc_proxy.ProxyState()
    srv, port = _serve(arc_proxy.Handler, state=state)
    try:
        yield port, state
    finally:
        srv.shutdown()
        stub.shutdown()


def _action(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        body = json.dumps({"game_id": "x"}).encode()
        c.request("POST", "/api/cmd/ACTION1", body=body,
                  headers={"Content-Length": str(len(body))})
        r = c.getresponse()
        r.read()
        return r.status
    finally:
        c.close()


def test_concurrent_actions_cannot_overshoot_the_ceiling(shim):
    port, state = shim
    state.set_budget(3)

    with ThreadPoolExecutor(max_workers=12) as pool:
        codes = list(pool.map(lambda _: _action(port), range(12)))

    served = sum(1 for c in codes if c == 200)
    refused = sum(1 for c in codes if c == 403)
    assert served == 3, (
        f"the ceiling was 3 and {served} actions were served. Every request that "
        "passed the check before any of them paid was let through."
    )
    assert refused == 9
    assert state.actions_used == 3


def test_the_ceiling_still_binds_one_request_at_a_time(shim):
    """The sequential path is the one that already worked. Keep it working."""
    port, state = shim
    state.set_budget(2)
    assert [_action(port) for _ in range(4)] == [200, 200, 403, 403]
    assert state.actions_used == 2


def test_a_seeded_counter_leaves_only_the_remainder(shim):
    """A resumed game starts part-spent, and the ceiling is absolute, not fresh."""
    port, state = shim
    state.set_budget(5, used=4)
    assert [_action(port) for _ in range(3)] == [200, 403, 403]
    assert state.actions_used == 5


@pytest.mark.parametrize("shim", [_FailingStub], indirect=True)
def test_an_upstream_failure_costs_the_solver_nothing(shim):
    """A reserved slot that upstream refused has to come back.

    `charge`'s docstring promises "a 502 costs the solver nothing at either
    end", and taking the slot up front would quietly break that promise -- the
    solver would be billed for actions ARC never performed, which is the
    ceiling drifting down instead of up.
    """
    port, state = shim
    state.set_budget(3)
    assert [_action(port) for _ in range(3)] == [500, 500, 500]
    assert state.actions_used == 0, "upstream failures were billed to the solver"
    assert state.exhausted() is None


class _EchoStub(BaseHTTPRequestHandler):
    """Reports back the exact path it was asked for, so the hop is observable."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self):
        body = json.dumps({"seen": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


def _get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


@pytest.mark.parametrize("shim", [_EchoStub], indirect=True)
def test_the_query_string_does_not_ride_along_upstream(shim):
    """The allowlist matched a query-stripped path; the forward sent the full one.

    `_forward` computed `path = self.path.split("?", 1)[0]`, matched the
    allowlist against `path`, and then built the upstream request from
    `self.path`. So the string that was checked and the string that was sent
    were different strings, and everything after the `?` reached ARC without
    passing any check at all.

    Whether ARC honours a parameter that widens a response is not knowable from
    here, and that is the point: `arc_proxy`'s stated doctrine is "allowlist,
    never denylist... closed by default rather than open until someone
    notices". Unvalidated bytes on a forwarded request are open-by-default
    surface. No call the harness makes carries a query -- all five URLs
    `client.py` builds are bare.
    """
    port, state = shim
    status, body = _get(port, "/api/scorecard/close?leak=level_scores&all=1")
    assert status == 200
    assert json.loads(body)["seen"] == "/api/scorecard/close", (
        "the query string was forwarded upstream; the request that was checked "
        "is not the request that was sent"
    )


@pytest.mark.parametrize("shim", [_EchoStub], indirect=True)
def test_a_bare_allowed_path_is_forwarded_byte_for_byte(shim):
    """Stripping must not damage the ordinary case it exists to protect."""
    port, state = shim
    status, body = _get(port, "/api/scorecard/close")
    assert status == 200
    assert json.loads(body)["seen"] == "/api/scorecard/close"


@pytest.mark.parametrize("shim", [_EchoStub], indirect=True)
def test_a_query_cannot_smuggle_a_refused_path_past_the_allowlist(shim):
    """The check still runs on the real path, so `/api/games` stays closed."""
    port, state = shim
    status, _ = _get(port, "/api/games?tags=whatever")
    assert status == 403


def test_a_refund_cannot_drive_the_counter_below_zero():
    """A refund that lands after the shim is re-armed must not credit the next game.

    `refund()` is guarded with `if self.actions_used > 0`, and the guard looks
    like defensive noise: every refund follows a successful `reserve()` on the
    same request, so the counter is at least 1. That is true only while nothing
    resets it in between — and `set_budget()` does exactly that, zeroing the
    counter to arm the next game.

    An in-flight action whose upstream call is still outstanding when the shim
    is re-armed will refund into the *new* game's counter. Without the guard it
    goes to -1, and a negative counter is not a cosmetic problem: the ceiling is
    `actions_used >= max_actions`, so the next game silently gets an extra
    action for every such straggler. The cap drifting up is the one direction
    that matters, since it is the number a submission is scored against.

    The mutation battery found this by surviving. It had been reported CAUGHT by
    the broken verdict logic — see the retraction in
    `docs/ccarc3_open_findings.md`.
    """
    state = arc_proxy.ProxyState()
    state.set_budget(3)
    assert state.reserve() is None          # an action goes out
    assert state.actions_used == 1

    state.set_budget(3)                     # next game armed while it is in flight
    assert state.actions_used == 0

    state.refund()                          # the straggler's upstream call fails
    assert state.actions_used == 0, (
        "the refund credited the next game; its ceiling is now one action higher "
        "than the budget it was armed with"
    )

    for _ in range(3):
        assert state.reserve() is None
    assert state.reserve() is not None, "the new game got a fourth action"
