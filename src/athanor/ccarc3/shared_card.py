"""One scorecard spanning many games -- what a leaderboard submission needs.

**The blocker this exists to remove.** :meth:`ArcClient.open` opens a card per
client, so a 25-game sweep mints 25 cards. The ARC community leaderboard takes
exactly one ``scorecard_url`` per submission and reads the score off ARC's own
card rather than anything self-reported, so 25 cards is not a submission at all
-- it is 25 of them, none complete.

**What actually binds a card to a caller, measured 2026-08-08.** Not the API
key, and not the ``GAMESESSION`` cookie. Four probes against the live API:

===================================================  =======================
attempt                                              result
===================================================  =======================
card opened on jar A, RESET a second game on jar B   ``game <id> not found``
same, but jar B pre-loaded with jar A's cookies      ok
``GAMESESSION`` alone transplanted (two games)       ``game <id> not found``
``AWSALBAPP-*`` alone transplanted (two games)       ok
jar that fetched its OWN ``AWSALBAPP-*`` first       ``game <id> not found``
===================================================  =======================

So a scorecard is state on **one backend instance**, and the load balancer's
``AWSALBAPP-*`` stickiness cookies are the only thing that routes a request back
to it. Carry those cookies and any process, thread or jar can play any game
against the card; lose them and the card is unreachable while the API key keeps
working perfectly -- which is why the failure reads as ``game <id> not found``
and points at the wrong thing entirely.

Two consequences worth stating plainly:

* **Injecting ``card_id`` alone cannot work.** That was the plan of record
  before these probes; it fails on the first RESET, in every process.
* **The card is only as durable as one backend.** A sweep that takes days is
  betting that instance is not recycled in the meantime. Nothing here can
  prevent that, so the local ledger stays the source of truth for scoring and
  the shared card is the artifact for submission -- not the other way round.

The stickiness cookies were observed with a ~7-day expiry and are re-issued on
every response, so a sweep that keeps playing keeps its card reachable. The
``GAMESESSION`` cookie expires in ~1 day and, per the table above, does not
matter.
"""
from __future__ import annotations

import http.cookiejar
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .client import ROOT_URL, _api_key, _get, _post, new_session

__all__ = [
    "SharedCard",
    "adopt",
    "close_card",
    "export_cookies",
    "load",
    "open_card",
    "save",
]

# The cookies that actually matter. `GAMESESSION` is carried too -- it costs
# nothing and the server may start relying on it -- but the probe above is clear
# that stickiness is what does the work, so this is the set whose ABSENCE is an
# error rather than a curiosity.
PINNING_PREFIX = "AWSALB"


@dataclass(frozen=True)
class SharedCard:
    """A card plus the session that can reach it. Neither half is any use alone."""

    card_id: str
    cookies: tuple[dict[str, str], ...] = ()

    @property
    def pinned(self) -> bool:
        return any(c["name"].startswith(PINNING_PREFIX) for c in self.cookies)

    def to_json(self) -> str:
        return json.dumps({"card_id": self.card_id, "cookies": list(self.cookies)}, indent=2)


def export_cookies(opener: urllib.request.OpenerDirector) -> tuple[dict[str, str], ...]:
    """Lift a jar out of an opener as plain data, so it can cross a process."""
    for h in opener.handlers:
        if isinstance(h, urllib.request.HTTPCookieProcessor):
            return tuple(
                {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
                for c in h.cookiejar
            )
    return ()


def adopt(jar: http.cookiejar.CookieJar, cookies: Iterable[dict[str, str]]) -> None:
    """Put exported cookies into a jar, so this caller lands on the card's backend."""
    for c in cookies:
        domain = c["domain"]
        jar.set_cookie(
            http.cookiejar.Cookie(
                0, c["name"], c["value"], None, False,
                domain, True, domain.startswith("."),
                c.get("path", "/"), True, False, None, True, None, None, {},
            )
        )


def open_card(
    *,
    tags: Sequence[str] = ("ccarc3",),
    api_key: str | None = None,
    root: str = ROOT_URL,
) -> SharedCard:
    """Open one card and keep the session that can reach it.

    Called by the driver, which holds the ARC key legitimately -- it builds the
    queue and scores the result from the same baselines it keeps from the solver.
    """
    opener = new_session()
    key = _api_key(api_key)
    card = _post(f"{root}/api/scorecard/open", {"tags": list(tags)}, key, opener=opener)
    cookies = export_cookies(opener)
    shared = SharedCard(card["card_id"], cookies)
    if not shared.pinned:
        # Better to stop here than to hand every game a card none of them can
        # reach. Without a stickiness cookie each shim lands on a random backend
        # and the first RESET fails with a message that names the game.
        raise RuntimeError(
            f"scorecard {shared.card_id} opened but the response carried no "
            f"{PINNING_PREFIX}* stickiness cookie, so nothing can be routed back "
            f"to it. Cookies seen: {[c['name'] for c in cookies] or 'none'}."
        )
    return shared


def close_card(card: SharedCard, *, api_key: str | None = None, root: str = ROOT_URL) -> dict[str, Any]:
    """Close the shared card from a session that can actually see it."""
    opener = new_session()
    adopt(_jar_of(opener), card.cookies)
    return _post(f"{root}/api/scorecard/close", {"card_id": card.card_id},
                 _api_key(api_key), opener=opener)


def read_card(
    card: SharedCard,
    game_id: str,
    *,
    api_key: str | None = None,
    root: str = ROOT_URL,
) -> Any:
    """Read one game's row off the shared card, from a session pinned to it."""
    opener = new_session()
    adopt(_jar_of(opener), card.cookies)
    return _get(f"{root}/api/scorecard/{card.card_id}/{game_id}",
                _api_key(api_key), opener=opener)


def _jar_of(opener: urllib.request.OpenerDirector) -> http.cookiejar.CookieJar:
    for h in opener.handlers:
        if isinstance(h, urllib.request.HTTPCookieProcessor):
            return h.cookiejar
    raise RuntimeError("opener has no cookie jar; it cannot hold a card")


def save(card: SharedCard, path: str | Path) -> Path:
    """Persist the card for the driver's own restarts.

    Deliberately NOT written into a workspace: the cookies are the credential
    half of the card, and the workspace is the one directory the solver is told
    to read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.to_json(), encoding="utf-8")
    path.chmod(0o600)
    return path


def load(path: str | Path) -> SharedCard:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SharedCard(raw["card_id"], tuple(raw.get("cookies") or ()))
