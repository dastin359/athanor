"""No human median may appear in source the solver can import.

**This is the first leak found here that disclosed the treasure rather than the
map.** Every earlier one told a solver that baselines existed and were withheld;
these handed over the numbers. Eight sites, across four modules:

* `arc_proxy.py` documented its response filter with a real scorecard as the
  worked example, so the module whose entire job is to withhold
  `level_baseline_actions` contained one environment's complete list. Twice
  over — the same comment showed `level_actions[0]` beside `level_scores[0]`,
  and `100*min(1.15,(h/a)^2)` inverts.
* `client.py`, four times, in prose explaining why each fix was made: a per-game
  total for `bp35`, two per-level medians for `su15`, a per-level median and a
  budget for `tn36`. One was added *by* the `bp35` repair.
* `gate.py` quoted a per-game total inside a comment explaining that it had
  removed that very number from the docstring above it.
* `cli.py` and `session.py` gave the range across all 25 environments.

The strip cannot reach any of it. It rewrites four files in a workspace; these
are on `PYTHONPATH`, imported by the workspace's own `session.py`, so reading
them is ordinary in-bounds behaviour — and `m0r0` was observed doing exactly
that, with `dir()` and `inspect.getsource`.

**Why the obvious test does not work.** Checking every published median against
every reachable file collides constantly: with 25 games of roughly eight levels
there are ~200 medians, and `159` is both `lp85`'s last level and a palette index
in `grids.py`. Eleven such collisions came back on the first run, none of them
leaks. Value alone cannot identify a median, and a test that fails on a palette
index gets deleted rather than fixed.

What identifies a leak is the *association* — a number sitting next to the word
that says what it is. Hence two structural checks and one exact check narrowed to
the same window. Between them they catch all eight original sites.
"""
from __future__ import annotations

import io
import os
import pathlib
import re
import tokenize

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "src" / "athanor" / "ccarc3"

# What a solver reaches by importing `session`, plus what `dir()` walks from
# there. Not the assets -- those go through the strip.
REACHABLE = sorted(PKG.rglob("*.py"))

# "651-action baseline", "31-baseline level", "a 171 baseline", "baseline of 7",
# "baseline runs to 1843 actions", and a literal `level_baseline_actions` list.
MEDIAN_SHAPES = re.compile(
    r"\b\d+\s*-?\s*(?:action\s*-?\s*)?baseline\b"
    r"|\bbaseline[s]?\s+(?:of|is|was|are|at|runs\s+to|totall?ing)\s+\d+"
    r"|level_baseline_actions\"?\s*:\s*\[\s*\d",
    re.IGNORECASE,
)

# `scoring.py` explains the rubric with a toy game. A worked example needs *some*
# number, and one that belongs to no real environment discloses nothing.
ALLOWED_SHAPES = {"baseline of 7"}

# Numbers that appear near baseline vocabulary for reasons that are not medians.
ALLOWED_NEAR = {"401"}          # the HTTP status `/api/games` returns without a key

# The environment count. It is in half these modules as "baseline_actions for all
# 25 environments", which is the leak being described rather than a leak. Cost of
# allowing it: a genuine median of exactly 25 would pass this check -- but not the
# shape check, which reads "25-baseline level" the same as any other.
ALLOWED_VALUES = {25}

# `{'game':24}` and `{ratio:.1f}` are layout, not narrative. Their digits collide
# with two-digit medians constantly and mean nothing.
FORMAT_SPEC = re.compile(r"\{[^{}]*\}")

BASELINE_WORD = re.compile(r"baseline|human median", re.I)
WINDOW = 70                     # characters either side -- one wrapped line


def prose(path: pathlib.Path) -> str:
    """Comments and string literals only — the parts that explain rather than run.

    Every leak found here was in prose, which is the natural home for "and the
    baseline was N". Code that needed a median would have to have been handed
    one, and none of this code is.
    """
    out: list[str] = []
    with path.open("rb") as fh:
        try:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    out.append(tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return path.read_text(encoding="utf-8")   # unparseable: check it all
    return "\n".join(out)


def windows(text: str) -> list[str]:
    """The neighbourhoods of every mention of a baseline."""
    return [FORMAT_SPEC.sub(" ", text[max(0, m.start() - WINDOW):m.end() + WINDOW])
            for m in BASELINE_WORD.finditer(text)]


@pytest.mark.parametrize("path", REACHABLE, ids=lambda p: p.name)
def test_no_median_shaped_phrase_in_importable_source(path):
    """`651-action baseline` says what it is. That is the whole tell."""
    hits = [m.group(0) for m in MEDIAN_SHAPES.finditer(prose(path))
            if m.group(0).strip().lower() not in ALLOWED_SHAPES]
    assert not hits, (
        f"{path.name} carries what reads as a human median: {hits}. "
        "Say the ratio, not the number -- the lesson survives, the leak does not."
    )


@pytest.mark.parametrize("path", REACHABLE, ids=lambda p: p.name)
def test_no_large_number_beside_baseline_vocabulary(path):
    """Catches the looser phrasings the shape regex was written too tight for.

    Three digits, because that is where the noise stops: format widths, level
    counts, ratios and dates are all shorter, and every per-game total on record
    is longer. Per-level medians can be two digits, which is what the shape test
    above is for -- the two are complementary, not redundant.
    """
    hits = sorted({n for win in windows(prose(path))
                   for n in re.findall(r"(?<![\w.])\d{3,}(?![\w.%])", win)
                   if n not in ALLOWED_NEAR and not n.startswith("20")})
    assert not hits, (
        f"{path.name} has {hits} within {WINDOW} characters of the word "
        "'baseline'. If it is a median it must go; if it is not, move it away "
        "from the word or add it to ALLOWED_NEAR with a reason."
    )


@pytest.mark.skipif(not os.environ.get("ARC_API_KEY"),
                    reason="needs the API key to know what the real medians are")
def test_no_published_median_sits_beside_baseline_vocabulary():
    """The exact check, narrowed to the window so it can be believed.

    Unnarrowed this reports a dozen collisions and is worthless (see the module
    docstring). Narrowed, it is the only check that catches a *two-digit*
    per-level median written in a phrasing the shape regex does not match --
    `su15`'s were 31 and 8.
    """
    from athanor.ccarc3 import list_games

    games = list_games()
    assert games, "no games returned; the key is present but the call failed"

    leaks: list[str] = []
    for path in REACHABLE:
        for win in windows(prose(path)):
            present = {int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.%])", win)}
            for game in games:
                published = {*game.baseline_actions, sum(game.baseline_actions)}
                # 7 is the toy example in `scoring.py`; anything under 10 is a
                # level index as often as a median.
                for value in sorted(published & present):
                    if value >= 10 and value not in ALLOWED_VALUES:
                        leaks.append(f"{path.name}: {value} ({game.game_id})")
    assert not leaks, ("published medians beside baseline vocabulary: "
                       + "; ".join(sorted(set(leaks))))
