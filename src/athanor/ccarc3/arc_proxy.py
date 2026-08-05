"""A local shim that holds the ARC key so the solver never has to.

**Why this exists.** `CCARC3_HIDE_BASELINES` closes the in-process route to the
human medians -- `baselines_for` refuses, `list_games` filters. It cannot close
the deliberate one, because the solver holds `ARC_API_KEY` in its own environment
and `GET /api/games` returns `baseline_actions` for all 25 environments. Nine
lines of `urllib` defeat the whole arm. Two runs did exactly that: `tu93` printed
its own medians verbatim, `bp35` printed the game-id list. Detection caught both;
detection is not prevention.

The fix is to move the credential out of reach. This process holds the real key
and is the only thing that talks to ARC. The solver gets a loopback URL and no key
at all, so a hand-rolled request has nothing to authenticate with.

**Allowlist, never denylist.** Only the four endpoints a solver legitimately needs
are forwarded. Anything else is refused, so a new ARC endpoint that happens to
carry baselines is closed by default rather than open until someone notices.

**What this does not claim.** The key is still readable from this process's
`/proc/<pid>/environ` by the same uid, so a solver determined to escalate could
find it. That is a far deeper reach than calling a documented endpoint, it leaves
an obvious trace, and `baseline_watch.py` looks for it. The goal is to make the
easy path impossible, not to sandbox an adversary.
"""
from __future__ import annotations

import json
import os
import re
import sys
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


# **An allowlisted endpoint still leaks the baselines, so responses are filtered
# too.** Closing the allowlist was not enough: `POST /api/scorecard/close` is a
# call every solver legitimately makes, and its response body carries, per run,
#
#     "level_actions":          [21, 0, 0, ...]
#     "level_baseline_actions": [17, 38, 31, 16, 41, 60, 26, 159]
#     "level_scores":           [65.53, 0.0, ...]
#
# -- the human medians for every level of the game, handed over verbatim. That is
# exactly what `CCARC3_HIDE_BASELINES` withholds. Measured on a real card kept at
# scratchpad/best_or_last/card.json.
#
# `level_scores` has to go with them: ARC's score is 100*min(1.15, (h/a)^2) and the
# solver knows its own `a`, so a score inverts straight back to `h`. The aggregate
# `score` fields go too -- a solver has no legitimate use for its own RHAE while
# the run is in progress, and that is the number the arm is built to withhold.
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

    def _forward(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        if not _allowed(path):
            self._refuse(path)
            return
        n = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(n) if n else None
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
        # This is why the proxy is not yet wired into the runner launch path: as
        # first written it would have broken every game it touched, silently and
        # in a way that looks exactly like ARC dropping the card.
        cookie = self.headers.get("Cookie")
        if cookie:
            headers["Cookie"] = cookie
        req = urllib.request.Request(
            UPSTREAM + self.path, data=payload, method=method, headers=headers,
        )
        set_cookies: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
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
