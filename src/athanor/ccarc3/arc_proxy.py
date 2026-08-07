"""A local shim that forwards the four ARC endpoints this harness uses.

Runs in the harness process, not the solver's. Holds the credential and the
per-game ceiling, and refuses every path outside its allowlist.
"""
# **The rest of this file's rationale is a comment, not a docstring, because the
# solver can import this module.** `athanor.ccarc3.arc_proxy` sits on the child's
# `PYTHONPATH`, so `pydoc athanor.ccarc3.arc_proxy` or `arc_proxy.__doc__` prints
# whatever is up there -- and what used to be up there was a complete description
# of the defence: which environment variable closes the in-process route, that
# `baselines_for` refuses and `list_games` filters, that the ceiling is
# `budget_multiple` times the withheld total and therefore inverts to it, and
# which two runs had already gone and fetched the numbers by hand. A solver that
# read it would know exactly what is hidden, why, and where the seams are. The
# reasoning is worth keeping; printing it to the person it guards against is not.
#
# **Why this exists.** `CCARC3_HIDE_BASELINES` closes the in-process route to the
# human medians. It cannot close the deliberate one, because the solver holds
# `ARC_API_KEY` in its own environment and `GET /api/games` returns
# `baseline_actions` for all 25 environments. Nine lines of `urllib` defeat the
# whole arm, and two runs did exactly that: `tu93` printed its own medians
# verbatim, `bp35` printed the game-id list. Detection caught both; detection is
# not prevention. The fix is to move the credential out of reach. This process
# holds the real key and is the only thing that talks to ARC; the solver gets a
# loopback URL and no key at all, so a hand-rolled request has nothing to
# authenticate with.
#
# **Allowlist, never denylist.** Only the four endpoints a solver legitimately
# needs are forwarded. Anything else is refused, so a new ARC endpoint that
# happens to carry baselines is closed by default rather than open until someone
# notices.
#
# **It also holds the action ceiling**, for the same reason it holds the key: a
# limit the solver can read is a limit the solver can invert. The ceiling is
# `budget_multiple` times the baseline total, so telling a solver its ceiling
# tells it the number the baseline-free arm exists to withhold -- and
# `budget_multiple`'s value is a default in this package's own source, on the
# solver's `PYTHONPATH`. Counting here means the child's environment carries no
# such number at all, and it makes the ceiling unforgeable besides: the
# in-process guard read a count from a file in the solver's own workspace.
#
# **What this does not claim.** The key is still readable from this process's
# `/proc/<pid>/environ` by the same uid, so a solver determined to escalate could
# find it. That is a far deeper reach than calling a documented endpoint and it
# leaves an obvious trace in the refusal log. The goal is to make the easy path
# impossible, not to sandbox an adversary.
from __future__ import annotations

import json
import os
import re
import http.cookiejar
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://three.arcprize.org"

# Every path a solver legitimately needs, and nothing else. Derived from the only
# four call sites in client.py that reach the network.
ALLOW = (
    re.compile(r"^/api/cmd/[A-Z0-9_]+$"),                     # RESET, ACTION1..7
    re.compile(r"^/api/scorecard/open$"),
    re.compile(r"^/api/scorecard/close$"),
    re.compile(r"^/api/scorecard/[0-9a-f-]+/[a-z0-9-]+$"),    # read one card
)


def _allowed(path: str) -> bool:
    return any(p.match(path) for p in ALLOW)


# Every action the client sends -- RESET is `/api/cmd/RESET` and counts, exactly
# as `ArcClient.actions_used` counts it, because both increment on the same call.
_CMD = re.compile(r"^/api/cmd/[A-Z0-9_]+$")

class ProxyState:
    """One game's cap and one game's HTTP session to ARC.

    **Per game, because both are per game.** These were module globals, which is
    fine for a driver that runs one environment at a time and silently wrong the
    moment two share a process: the action counter would bill one game for the
    other's moves, and -- worse -- a single cookie jar would pin both games'
    scorecards to one ARC session. Card reads are session-bound, so the second
    game's card would answer 404 while its actions kept succeeding, which is
    exactly the failure that finished `sb26` 8/8 against a card frozen at level
    3 and showed no other symptom.

    A shared counter is a bug you would eventually see in the numbers. A shared
    session is one you would not.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.max_actions = 0        # 0 disables the cap
        self.actions_used = 0
        # Which game this shim serves. Empty means "do not check", which is what
        # the module-level default and the connectivity probe want.
        self.game_id = ""
        self._opener: urllib.request.OpenerDirector | None = None

    def set_budget(self, max_actions: int, *, used: int = 0) -> None:
        """Arm the cap for one game.

        `used` seeds the counter from a run already in progress. The proxy's
        count lives in memory and the solver's does not, so a resumed game whose
        proxy restarted would otherwise start from zero -- the ceiling would bind
        at `used + max_actions` and grow with every interruption.
        Callers read the figure from the workspace's `trace.state.json`.
        """
        with self._lock:
            self.max_actions, self.actions_used = int(max_actions), int(used)
        # A new game means a new card; carrying the previous game's pinning over
        # is how a stale session survives into a run that did not open it.
        self.reset_session()

    def exhausted(self) -> str | None:
        """The refusal once the ceiling is reached, or ``None`` before then."""
        with self._lock:
            if self.max_actions and self.actions_used >= self.max_actions:
                # No figure. `actions_used` equals `max_actions` at the moment
                # this fires, and the cap is the withheld total times
                # `budget_multiple` -- so printing it hands back the number the
                # ceiling was moved out of the child's environment to withhold.
                return ("action budget exhausted. The environment is over; "
                        "nothing further can be scored.")
        return None

    def charge(self) -> None:
        """Bill one action, after upstream accepted it.

        Charged on the response rather than the request so the count tracks
        `ArcClient.actions_used`, which increments while processing a frame the
        server actually returned. A 502 costs the solver nothing at either end.
        """
        with self._lock:
            self.actions_used += 1

    def upstream(self) -> urllib.request.OpenerDirector:
        """This game's persistent session to ARC, cookie jar and all.

        **Forwarding the client's cookies is not enough.** ARC binds a scorecard
        to the HTTP session and its load balancer pins that session with four
        `AWSALBAPP-*` cookies. Building a fresh request per call -- which
        `urllib.request.urlopen` does -- lets the balancer re-pin on every hop,
        so the card lands on a backend that never heard of it. Measured live on
        one card: 8 of 8 reads succeed direct, **1 of 8** through a shim without
        this. The jar lives here rather than in the client because the client is
        the thing being kept at arm's length.
        """
        with self._lock:
            if self._opener is None:
                self._opener = urllib.request.build_opener(
                    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
                )
            return self._opener

    def reset_session(self) -> None:
        with self._lock:
            self._opener = None


# A server built straight from `Handler` -- as the tests do -- has no game
# attached, so it falls back to this. Real runs always carry their own.
_DEFAULT = ProxyState()


def set_budget(max_actions: int, *, used: int = 0) -> None:
    """Arm the default state. Per-game callers use `Proxy.set_budget`."""
    _DEFAULT.set_budget(max_actions, used=used)


# **An allowlisted endpoint still leaks the baselines, so responses are filtered
# too.** Closing the allowlist was not enough: `POST /api/scorecard/close` is a
# call every solver legitimately makes, and its response body carries, per run,
#
#     "level_actions":          [a0, a1, ...]
#     "level_baseline_actions": [h0, h1, ...]
#     "level_scores":           [s0, s1, ...]
#
# -- the human medians for every level of the game, handed over verbatim. That is
# exactly what `CCARC3_HIDE_BASELINES` withholds.
#
# `level_scores` has to go with them: ARC's score is 100*min(1.15, (h/a)^2) and the
# solver knows its own `a`, so a score inverts straight back to `h`. The aggregate
# `score` fields go too -- a solver has no legitimate use for its own RHAE while
# the run is in progress, and that is the number the arm is built to withhold.
#
# **Written symbolically because this file is on the solver's PYTHONPATH.** It
# carried a real card's arrays as the worked example, which put one environment's
# complete median list inside the module whose whole job is to withhold it --
# reachable by `inspect.getsource(arc_proxy)` from a stripped workspace, which is
# how it was found. Twice over: `level_actions[0]` beside `level_scores[0]` invert
# to `h0` on their own. A sample that demonstrates a leak must not be one.
#
# Nothing in client.py reads any of these; `close()` is called for its side effect
# and `snapshot_scorecard` reads `actions_by_level`, which is untouched.
HIDDEN_FIELDS = frozenset({
    "level_baseline_actions",
    "baseline_actions",
    "level_scores",
    "score",
    "scores",
    "tags_scores",
})


def _strip(node):
    """Recursively drop every baseline-bearing field from a decoded JSON body."""
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k not in HIDDEN_FIELDS}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _filtered(body: bytes) -> bytes:
    """Strip baselines from a response, passing anything unparseable through.

    Deliberately fail-open on a body that is not JSON: the client has careful
    error handling and an empty or truncated 5xx body must reach it unchanged.
    A non-JSON body cannot contain the fields anyway.
    """
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    return json.dumps(_strip(decoded)).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # noqa: A003 - stdlib signature
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _refuse(self, path: str) -> None:
        # Loud on purpose: a refusal is the signal that a solver went looking, and
        # it belongs in the log where baseline_watch and the operator can see it.
        sys.stderr.write(f"REFUSED {path} — not on the solver allowlist\n")
        sys.stderr.flush()
        body = json.dumps({"error": "endpoint not available to solvers"}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def state(self) -> ProxyState:
        return getattr(self.server, "state", _DEFAULT)

    def _forward(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        if not _allowed(path):
            self._refuse(path)
            return
        charge = bool(_CMD.match(path))
        n = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(n) if n else None

        # **A shim is dedicated to one game, and until 2026-08-07 it did not
        # check.** `ProxyState` held only `(max_actions, actions_used)` and this
        # never looked at which game a request named, so a shim armed for game B
        # would forward game A's `/api/cmd/*` and charge B -- and with games in
        # flight concurrently, every sibling's port is a loopback scan away. The
        # cap moved out of the child's environment precisely so the solver could
        # not raise it; billing it to a neighbour raises it just as effectively.
        wanted = self.state.game_id
        if wanted and payload:
            try:
                asked = (json.loads(payload) or {}).get("game_id")
            except ValueError:
                asked = None
            if asked and asked != wanted:
                sys.stderr.write(f"WRONG GAME {path}: shim serves {wanted}, "
                                 f"request names {asked}\n")
                sys.stderr.flush()
                self._deny(403, f"this shim serves {wanted}, not {asked}")
                return

        if charge:
            # 403 and not 429: `client._send` raises on 4xx without retrying, so
            # the solver gets one clear terminal error rather than three rounds
            # of backoff against a wall that will not move.
            over = self.state.exhausted()
            if over:
                sys.stderr.write(f"BUDGET {path} — {over}\n")
                sys.stderr.flush()
                self._deny(403, over)
                return
        headers = {
            "X-API-Key": os.environ["ARC_API_KEY"],
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        # **Cookies must survive the hop, or nothing works.** ARC binds a
        # scorecard to the HTTP *session*, not to the API key -- open a card on
        # one connection and RESET on another and the server answers
        # ``game <id> not found`` (see :func:`athanor.ccarc3.client.new_session`).
        # The saved sessions carry a ``GAMESESSION`` cookie and an
        # ``AWSALBAPP-0`` load-balancer stickiness cookie, so a proxy that builds
        # a clean request per call makes every solver call a new session and
        # every game unreachable the moment it is opened.
        #
        # As first written this proxy dropped cookies, which would have broken
        # every game it touched, silently and in a way that looks exactly like
        # ARC dropping the card. That is why it sat unwired for so long. It is
        # fixed and covered end to end by `test_arc_proxy_endtoend.py`.
        # **The client's cookies are deliberately NOT forwarded.** They were, and
        # that is what broke the scorecard: `http.cookiejar` refuses to set a
        # `Cookie` header on a request that already has one, so relaying the
        # client's header silenced this process's own jar entirely. The client's
        # jar meanwhile holds the ALB's `AWSALBAPP-N=_remove_` tombstones as if
        # they were values and sends them back, which unpins the session. The
        # card then lands on a backend that has never heard of it and every read
        # answers `404 card_id not found`.
        #
        # The proxy owns the upstream session. That is the whole point of it
        # holding the key, and the session belongs with the credential.
        req = urllib.request.Request(
            UPSTREAM + self.path, data=payload, method=method, headers=headers,
        )
        set_cookies: list[str] = []
        try:
            with self.state.upstream().open(req, timeout=120) as r:
                body, code = r.read(), r.getcode()
                set_cookies = r.headers.get_all("Set-Cookie") or []
        except urllib.error.HTTPError as exc:
            # Pass the upstream status and body through unchanged. The client has
            # careful 4xx/5xx handling and retry logic; flattening errors here
            # would break it in ways that look like game bugs. Cookies come back
            # on error responses too -- an ALB re-pins on a 4xx like any other.
            body, code = exc.read(), exc.code
            set_cookies = exc.headers.get_all("Set-Cookie") or []
        except (urllib.error.URLError, TimeoutError) as exc:
            body, code = json.dumps({"error": f"upstream: {exc}"}).encode(), 502
        if charge and 200 <= code < 300:
            self.state.charge()
        # Filter on the way back, not just on the way in. Content-Length is
        # recomputed below from the filtered body, so this must happen first.
        body = _filtered(body)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for value in set_cookies:
            self.send_header("Set-Cookie", value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):       # noqa: N802
        self._forward("GET")

    def do_POST(self):      # noqa: N802
        self._forward("POST")


class _Server(ThreadingHTTPServer):
    """A server that knows which game it is serving."""

    daemon_threads = True
    state: ProxyState


class Proxy:
    """One listening shim, dedicated to one game.

    In-process rather than a subprocess, because the caller is the one thing
    that already holds the key legitimately -- the runner builds the queue and
    scores the result from the same baselines it keeps away from the solver. A
    subprocess would need the key handed to it anyway and would outlive a driver
    that died.

    Port 0 always: with several games in flight there is no single well-known
    port to claim, and a fixed one turns a not-yet-reaped predecessor into
    `Address already in use` at exactly the wrong moment.
    """

    def __init__(self, port: int = 0, game_id: str = "") -> None:
        if not os.environ.get("ARC_API_KEY"):
            raise RuntimeError("arc_proxy: no ARC_API_KEY in env; refusing to start")
        self.state = ProxyState()
        self.state.game_id = game_id
        self._server = _Server(("127.0.0.1", port), Handler)
        self._server.state = self.state
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, bound = self._server.server_address[:2]
        self.url = f"http://{host}:{bound}"

    def set_budget(self, max_actions: int, *, used: int = 0) -> None:
        self.state.set_budget(max_actions, used=used)

    @property
    def actions_used(self) -> int:
        return self.state.actions_used

    @property
    def max_actions(self) -> int:
        return self.state.max_actions

    def shutdown(self) -> None:
        """Stop listening and free the port. Safe to call twice."""
        try:
            self._server.shutdown()
            self._server.server_close()
        except OSError:
            pass


def serve_in_background(port: int = 0) -> tuple[str, Proxy]:
    """Back-compat: a single shim, for callers that only ever run one game."""
    proxy = Proxy(port)
    return proxy.url, proxy


def main() -> int:
    if not os.environ.get("ARC_API_KEY"):
        sys.stderr.write("arc_proxy: no ARC_API_KEY in env; refusing to start\n")
        return 2
    port = int(os.environ.get("CCARC3_PROXY_PORT", "8787"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"arc_proxy: listening on 127.0.0.1:{port} -> {UPSTREAM}\n")
    sys.stderr.flush()
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
