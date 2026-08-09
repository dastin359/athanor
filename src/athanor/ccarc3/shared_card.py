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

**Idle survival, measured 2026-08-08.** A *game* left idle is reaped inside
``(13.6, 18.2]`` minutes -- eight resumes bracket it, see
:func:`athanor.ccarc3.session.snapshot_scorecard`. A *card* is not on that
clock, which had to be measured rather than assumed because the sweep pauses
between games:

===============================================  ===================
probe                                            result
===============================================  ===================
card polled every 5 min                          alive past +65 min
card opened, one game, then **untouched**        alive at +28 min
card untouched 32 min, then a **new game RESET**  **accepted**
===============================================  ===================

The middle row is the control for the first: polling could have kept its own
card warm, in which case the first row measures nothing. It did not.

The third row is the one the sweep actually needs. "Readable" is not "playable"
-- they are different operations against different server state -- and the sweep
does not read an idle card, it adds game N+1 to it. So the property is: **a
shared card accepts a new game after a sweep-sized gap.**

**Still unmeasured: a multi-hour gap.** 32 minutes covers solver startup and
scoring between games. It does not cover a quota pause, which on the seven-day
window can be days. Treat a long pause as a card at risk: snapshot before it and
verify with a read after it.

The stickiness cookies were observed with a ~7-day expiry and are re-issued on
every response, so a sweep that keeps playing keeps its card reachable. The
``GAMESESSION`` cookie expires in ~1 day and, per the table above, does not
matter.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
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

    **Written atomically, like the run state in client.py.** A plain
    ``write_text`` leaves a window in which the file exists and is truncated,
    and this container is reaped every 10-50 minutes. The cost of landing in
    that window is not one lost write: ``sweep_card`` sees the file exists,
    ``load`` raises on the partial JSON, and the driver dies -- then the
    supervisor relaunches it ten minutes later into the same crash, forever,
    with the card carrying the banked games stranded and unreachable.

    The mode is set on the temporary file BEFORE the rename, so the card is
    never briefly world-readable at its final path. The cookies in it are the
    credential half of the card.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(card.to_json(), encoding="utf-8")
    tmp.chmod(0o600)
    os.replace(tmp, path)                     # atomic on POSIX
    return path


def load(path: str | Path) -> SharedCard:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SharedCard(raw["card_id"], tuple(raw.get("cookies") or ()))
