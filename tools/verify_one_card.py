#!/usr/bin/env python3
"""Assert a sweep put every game on ONE scorecard. Exit 1 if it did not.

**Why this exists.** A community-leaderboard submission is a single
`scorecard_url`, so the whole value of a 25-game sweep is that all 25 land on one
card. Nothing checked it. Every run already records the card it scored on --
`session._card_facts` writes `card_id` into `result.json` -- but a recorded fact
nobody reads is not a check, and this project has repeatedly shipped exactly that
shape: the guard that detects instead of enforcing, the brake that reads as
engaged, the watcher blind to the run it watched.

A split is silent by construction. Every game still plays, still scores, still
banks a clean result; the sweep prints success and the artifact at the end is
worth nothing. At roughly $650 a sweep, finding that out afterwards is the
expensive way.

Two ways a sweep splits, both seen in this project on 2026-08-08:

* the card is **reaped** mid-sweep and a game opens its own replacement -- an
  idle *game* dies inside (13.6, 18.2] minutes;
* a game **resumes** onto the card its trace was played on, which is correct for
  that game and wrong for the artifact. `ArcClient.foreign_card` records it.

Usage::

    python3 tools/verify_one_card.py <sweep_dir> [--expect <card_id>]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def cards_in(sweep: pathlib.Path) -> dict[str, list[str]]:
    """Map card id -> games recorded on it, from each game's banked attempt."""
    found: dict[str, list[str]] = collections.defaultdict(list)
    for banked in sorted(sweep.glob("*/clean_result.json")):
        gid = banked.parent.name
        want = json.loads(banked.read_text())
        # **Match the attempt that produced the banked result.** Taking any
        # `attempt_*` reads an abandoned run: su15 banked attempt_3 (9/9) while a
        # glob returned attempt_2 (2/9), which scored 0.0567 for a run that won
        # every level. Same trap, twice, in one day.
        for a in sorted(banked.parent.glob("attempt_*/%s" % gid)):
            rp = a / "result.json"
            if not rp.exists():
                continue
            r = json.loads(rp.read_text())
            if (r.get("actions_used"), r.get("levels_reached")) != (
                want.get("actions_used"), want.get("levels_reached")
            ):
                continue
            # **Fall back to the state file.** `card_id` only entered `result.json`
            # on 2026-08-08; every run before that recorded it solely in
            # `trace.state.json`, which is still on disk. Without this the check
            # reports "cannot confirm" for all 25 banked runs and is useless
            # exactly where it could have told us something.
            card = r.get("card_id") or ""
            if not card:
                st = a / "trace.state.json"
                if st.exists():
                    try:
                        card = json.loads(st.read_text()).get("card_id", "") or ""
                    except (OSError, ValueError):
                        card = ""
            found[card].append(gid)
            if r.get("foreign_card"):
                found.setdefault("__foreign__", []).append(
                    "%s (played on %s, sweep wanted %s)"
                    % (gid, r.get("card_id"), r["foreign_card"])
                )
            break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--expect", default="", help="the card id the sweep opened")
    args = ap.parse_args()

    sweep = pathlib.Path(args.sweep)
    if not sweep.is_dir():
        print("no such sweep directory: %s" % sweep)
        return 2

    found = cards_in(sweep)
    foreign = found.pop("__foreign__", [])
    unknown = found.pop("", [])
    total = sum(len(v) for v in found.values()) + len(unknown)

    for card, games in sorted(found.items(), key=lambda kv: -len(kv[1])):
        print("%s  %2d games  %s" % (card, len(games),
                                     " ".join(sorted(g.split("-")[0] for g in games))))
    if unknown:
        print("(no card_id recorded)  %d games  %s" % (
            len(unknown), " ".join(sorted(g.split("-")[0] for g in unknown))))

    ok = True
    if len(found) > 1:
        print("\nSPLIT: %d distinct cards across %d games. This sweep cannot be "
              "submitted as one scorecard_url." % (len(found), total))
        ok = False
    if unknown:
        print("\n%d games recorded no card_id -- they predate `_card_facts` or the "
              "state file was removed. Their card cannot be confirmed." % len(unknown))
        ok = False
    if foreign:
        print("\nNOT ON THE SWEEP CARD (resumed onto their own):")
        for f in foreign:
            print("  " + f)
        ok = False
    if args.expect and list(found) not in ([args.expect], []):
        print("\nEXPECTED card %s, found %s" % (args.expect, ", ".join(found) or "none"))
        ok = False

    print("\n%s" % ("one card, %d games — submittable as a single scorecard_url" % total
                    if ok and total else "NOT submittable as one scorecard"))
    return 0 if ok and total else 1


if __name__ == "__main__":
    sys.exit(main())
