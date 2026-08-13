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

    **f-strings are prose too, and under Python 3.12 they stopped being
    `STRING`.** PEP 701 re-tokenised them: an f-string now arrives as
    `FSTRING_START` / `FSTRING_MIDDLE` / `FSTRING_END`, and only the literal
    text between the braces is `FSTRING_MIDDLE`. Collecting `STRING` alone
    therefore captured `"the baseline was 651"` and silently dropped
    `f"the baseline was {n} (651)"`.

    Nothing announced the change: the scan kept returning text, the tests kept
    passing, and the coverage simply shrank when the interpreter moved. That is
    this project's recurring shape reached through a dependency upgrade rather
    than an edit -- a check that names one thing (prose) and reads a proxy for
    it (one token type that used to mean prose). The `.venv` here is 3.12.7, so
    it is live, not hypothetical.

    **An f-string is put back together from the source, not concatenated from
    its tokens.** Appending the `FSTRING_*` pieces individually looked like the
    obvious fix and quietly broke `windows()`: `FORMAT_SPEC` strips `{...}` as a
    unit, and the tokeniser hands back the braces as separate `OP` tokens that
    prose never collects. So `f"{'game':24} {'levels':>14}"` arrived as the bare
    text `24 … 14`, and column widths in a table header were reported as leaked
    medians for `tu93` and `lp85`. Slicing the original source between
    `FSTRING_START` and `FSTRING_END` keeps the braces, so the existing
    suppression works unchanged and only the literal prose is examined.

    The names are looked up rather than referenced directly so this still runs
    on an interpreter that predates them.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def off(pos) -> int:
        row, col = pos
        return starts[row - 1] + col if 0 < row <= len(lines) else len(text)

    f_start = getattr(tokenize, "FSTRING_START", None)
    f_end = getattr(tokenize, "FSTRING_END", None)

    out: list[str] = []
    depth, begin = 0, 0
    with path.open("rb") as fh:
        try:
            for tok in tokenize.tokenize(fh.readline):
                if f_start is not None and tok.type == f_start:
                    if depth == 0:
                        begin = off(tok.start)
                    depth += 1
                elif f_end is not None and tok.type == f_end and depth:
                    depth -= 1
                    if depth == 0:
                        out.append(text[begin:off(tok.end)])
                elif depth == 0 and tok.type in (tokenize.COMMENT, tokenize.STRING):
                    out.append(tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return text                               # unparseable: check it all
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


# --------------------------------------------------------------------------- #
# The key: found where it LIVES, not only where a shell happened to export it.
# --------------------------------------------------------------------------- #
#: **These four checks were skipping on every box that has the key.**
#:
#: They gate on `os.environ["ARC_API_KEY"]`, which is exported by
#: `desktop_bootstrap.sh`, by `supervisor.sh`'s `key_ok`, by the driver and by
#: `ccarc3_resume.sh` -- by every launcher except the one that runs the tests.
#: `pytest` is started from an ordinary shell, so the variable is absent and all
#: four skipped, on a box where the key is sitting at mode 600 two directories
#: away. `pytest -rs` reported it every run and a skip reads like a decision.
#:
#: These are the value-based half of the withholding audit: the half that knows
#: what the real medians ARE and can therefore find one that leaked. The
#: structural half cannot substitute for it. So this is the single most costly
#: instance of the defect this file exists to guard against -- a check that names
#: one thing (no published median in solver-reachable source) and reads a proxy
#: for it (an environment variable), passing by not running.
#:
#: The key's real home is `$CCARC3_SCRATCH/arc3/.env`, deliberately outside the
#: repo because `evidence/` is public. Read it only when that scratch root was
#: explicitly configured. The Codex path never searches the operator's home
#: directory implicitly; an ordinary offline test run must not acquire a live
#: credential merely because one exists elsewhere on the host.
def _arc_key() -> str:
    key = os.environ.get("ARC_API_KEY", "").strip()
    if key:
        return key
    scratch = os.environ.get("CCARC3_SCRATCH", "").strip()
    if not scratch:
        return ""
    envf = pathlib.Path(scratch) / "arc3" / ".env"
    try:
        for line in envf.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "ARC_API_KEY":
                key = value.strip()
                if key:
                    # **Export it, exactly as every launcher does.** `.env` carries
                    # no `export`, and `client._auth()` reads `os.environ` rather
                    # than taking the key from here -- so finding the file is not
                    # the same as the call working. Sourcing it into the process is
                    # the whole of what `set -a; . "$ENVF"` does in
                    # `supervisor.sh`'s `key_ok` and in `desktop_bootstrap.sh`;
                    # that exact gap ("a SHELL variable a python child cannot
                    # see") is already recorded in the bootstrap, and it reappeared
                    # here the moment these tests stopped skipping.
                    os.environ["ARC_API_KEY"] = key
                return key
    except OSError:
        pass
    return ""


#: Resolved once, at import: `skipif` is evaluated at collection time.
ARC_KEY = _arc_key()
_NO_KEY_REASON = (
    "no ARC_API_KEY in the environment and none at $CCARC3_SCRATCH/arc3/.env, "
    "so the real medians are unknowable here"
)


@pytest.mark.skipif(not ARC_KEY, reason=_NO_KEY_REASON)
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


# ---------------------------------------------------------------------------
# The second audit, 2026-08-07. Three more criticals, all invisible to
# everything above, because none of them is a median: they are *functions* of
# one. A cap is `h_total * budget_multiple`; a level score is
# `100*min(1.15,(h/a)^2)`. Both invert, both were sitting in docstrings, and
# `help()` renders docstrings.
#
# The worst of them cleared every purpose-built guard in this repo. The
# reachable-docs test walked `athanor.ccarc3:score_run`, whose docstring
# contained `21 actions` beside `level_scores [65.5, ...]` -- and returned no
# hits, because 65.5 is a score and 21 is an action count and neither is
# "baseline"-shaped. Checking for the secret is not enough when the secret's
# published transforms are lying next to it.
# ---------------------------------------------------------------------------

# Every multiplier this harness has shipped. A cap is the only number in the
# system that is a published constant times the thing being withheld.
BUDGET_MULTIPLES = (2.0, 4.0, 5.0)

DECIMAL = re.compile(r"(?<![\w.])\d{1,3}\.\d{1,4}(?![\w.])")
INTEGER = re.compile(r"(?<![\w.])\d{2,5}(?![\w.%])")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def numeric(text: str) -> str:
    """Join thousands separators so `12,416` is one number and not also `416`.

    `416` is `ft09`'s median total doubled, and `12,416` is a row count in
    `grids.py`. The first version of the cap check reported it and was wrong.
    """
    return THOUSANDS.sub("", text)


@pytest.mark.skipif(not ARC_KEY, reason=_NO_KEY_REASON)
def test_no_action_cap_appears_in_importable_prose():
    """`52 actions of a 1040 budget` is `ft09`'s median total, times five.

    The strip removes `action_budget` from `meta.json` and `CCARC3_MAX_ACTIONS`
    from the child's environment precisely because a cap inverts. Then three
    docstrings handed one back: two as a `used/cap` pair, one outright.
    """
    from athanor.ccarc3 import list_games

    leaks = []
    for path in REACHABLE:
        present = {int(n) for n in INTEGER.findall(numeric(prose(path)))}
        for game in list_games():
            total = sum(game.baseline_actions)
            for mult in BUDGET_MULTIPLES:
                cap = int(total * mult)
                if cap in present:
                    leaks.append(f"{path.name}: {cap} = {game.game_id} total x {mult}")
    assert not leaks, "action caps invert to medians: " + "; ".join(sorted(set(leaks)))


# A level score, however it is dressed. `S = 100*min(1.15,(h/a)^2)` and the
# doctrine prints that formula in the solver's own workspace, so a score in
# reachable prose is a median once the action count is known -- and the action
# count is usually in the same sentence.
LEVEL_SCORE_SHAPE = re.compile(
    # A populated array, not a subscript: `level_scores [65.5, ...]` leaks and
    # `level_scores[0]` is prose about the leak.
    r"level_scores?\s*[\"']?\s*[:=]?\s*[\[(]\s*\d+(?!\s*\])"
    r"|\blevel[ _]\d+\b[^.\n]{0,30}?\bscored?\b[^.\n]{0,20}?\d{1,3}\.\d",
    re.IGNORECASE,
)

# **What this deliberately does not match.** A looser rule -- any decimal within
# 40 characters of the word "score" -- fires on the rubric itself
# (`100*min(1.15,(h/a)^2)`), on the `[0.0, 1.15]` range assertions, and on this
# project's own environment scores (`0.2748` lost by `bp35`). None of those
# inverts to a median: `E` is an aggregate and `1.15` is a constant. Only a
# *level* score does, and only that shape is banned here.
#
# The gap this leaves is a level score written in prose that names neither
# "level_scores" nor a level number. Nothing automatic covers that; the rule to
# hold in review is simply that a worked example uses `higher` and `lower`.
ALLOWED_SCORE_PHRASES = ()


@pytest.mark.parametrize("path", REACHABLE, ids=lambda p: p.name)
def test_no_level_score_appears_in_importable_prose(path):
    """Ban the shape, because no value-based test can catch this one.

    **The obvious check does not work and it is worth recording why.** Given a
    score `s` and an action count `a`, the implied median is `a*sqrt(s/100)`.
    With ~200 published medians spread over 1..400, almost every (s, a) pair in
    ordinary prose lands within rounding distance of one: `1.1 beside 403` yields
    42.3, and both 42 and 43 are real medians. A first attempt reported sixteen
    such "leaks", every one a coincidence — the same density problem that made
    the bare-value scan useless (see the module docstring).

    So this bans the presentation instead of hunting the value. A level score has
    no reason to appear in importable prose at all: the argument a worked example
    is making survives with `higher`/`lower` in place of the numbers, which is
    how `DOCTRINE.md` §0a was already written and how `score_run`'s docstring
    should have been.
    """
    text = prose(path)
    hits = [m.group(0) for m in LEVEL_SCORE_SHAPE.finditer(text)
            if not any(a in m.group(0) for a in ALLOWED_SCORE_PHRASES)]
    assert not hits, (
        f"{path.name} prints level-score-shaped values: {hits}. Say `higher` and "
        "`lower` — a score plus its action count is the median in disguise."
    )


@pytest.mark.skipif(not ARC_KEY, reason=_NO_KEY_REASON)
def test_the_unstripped_doctrine_asset_carries_no_published_median():
    """The master doctrine is reachable, and §6 held two games' medians.

    `athanor.ccarc3.session.ASSETS` is a module-level attribute resolving to the
    assets directory, so the *unstripped* `CCARC3_DOCTRINE.md` — §6 included — is
    one `read_text()` away from any solver, whatever the strip did to the copy in
    its workspace. §6's `client.pace()` example printed `{0: (17, 22, 0.77),
    1: (59, 123, 0.48)}`, and 22 and 123 are `ls20`'s first two per-level medians;
    the `status()` example beside it printed `190/55`, which is `tn36`'s.

    The strip removing §6 is what made this invisible: the workspace copy is
    clean, so every check that reads a *workspace* passes. This one reads the
    asset.
    """
    from athanor.ccarc3 import list_games
    from athanor.ccarc3 import session as sess

    doc = pathlib.Path(sess.ASSETS, "CCARC3_DOCTRINE.md").read_text(encoding="utf-8")
    published = {n for g in list_games() for n in g.baseline_actions
                 if n >= 20 and n not in ALLOWED_VALUES}   # 25 = the arm size

    leaks = []
    for line in doc.splitlines():
        if not re.search(r"baseline|pace\(|status\(|median", line, re.I):
            continue
        for token in re.findall(r"(?<![\w.])\d{2,4}(?![\w.%])", numeric(line)):
            if int(token) in published:
                leaks.append(f"{token}: {line.strip()[:80]}")
    assert not leaks, ("published medians in the reachable master doctrine: "
                       + "; ".join(leaks))


@pytest.mark.skipif(not ARC_KEY, reason=_NO_KEY_REASON)
def test_no_published_array_appears_anywhere_in_importable_source():
    """Every other check here reads `prose()`, which is comments and strings only.

    That scoping is right for the value checks — `159` is a palette index in
    `grids.py` as well as `lp85`'s last level, and matching bare integers against
    code produces nothing but noise. But it leaves the most direct leak of all
    completely invisible: `LP85_MEDIANS = [17, 38, 31, 16, 41, 60, 26, 159]` at
    module level passes all of them, verified by inserting exactly that into
    `grids.py` and watching 42 tests go green.

    An array is not a value. The full published sequence, in order, is not a
    coincidence at any length worth checking, so this one runs on the whole file.
    """
    from athanor.ccarc3 import list_games

    leaks = []
    for path in REACHABLE:
        text = path.read_text(encoding="utf-8")
        for game in list_games():
            arr = list(game.baseline_actions)
            if len(arr) < 4:
                continue
            pattern = r"[\[(]\s*" + r"\s*,\s*".join(str(n) for n in arr) + r"\s*[\])]"
            if re.search(pattern, text):
                leaks.append(f"{path.name}: {game.game_id}'s array")
    assert not leaks, "published median arrays in source: " + "; ".join(sorted(set(leaks)))


def test_the_withholding_note_does_not_end_by_reinstating_a_withdrawn_result():
    """**Its last word used to be "The 25-environment result stands."**

    That sentence closed a section about leaks in the *package*, where it was
    true. Forty lines above, the same file records that the *workspace* leaked a
    real median for four days and that eight banked runs read it — which is why
    the pooled 25-environment figure is withdrawn. A reader who reached the end
    and stopped got the opposite of the finding.
    """
    import pathlib as _p

    doc = (_p.Path(__file__).resolve().parent.parent
           / "docs/ccarc3_withholding.md").read_text()

    assert "The 25-environment result stands" not in doc, (
        "the pooled figure is withdrawn; only the 17-run cohort survives"
    )
    assert "withdrawn" in doc
