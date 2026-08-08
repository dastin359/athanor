"""The audit page must pick its best play by the rubric, not by a proxy.

`arc_actions_per_level` chose "furthest first, then cheapest at that depth". RHAE
weights level `i` by `i`, so a play that spent more actions in total can still
score higher when it spent them on early levels — the proxy and the maximum are
different functions. That proxy produced every `E` and `raw` in
`evidence/ccarc3/trace_audit/runs.json.gz` and on the published page.

On the 29 preserved scorecards (23 multi-play) the two never disagreed, so no
recorded number moved when this was fixed. These are the constructed cases where
they do, so the proxy cannot come back unnoticed.

**The test that stood here could not have caught it.** It imported only
`athanor.ccarc3.scoring` and asserted `score_environment` prefers a play that
overran an early level — a property of the RHAE weighting, never of the audit's
selector, which it did not import. `tools/build_trace_audit.py` ran 0 of its 542
statements under the whole suite, and swapping the selector back to the proxy
left every test green. Worse, its fixture could not express the failure even
wired up: the two plays had equal depth *and* equal totals, so the proxy's key
ties and `max` returns the first — which is also the rubric's winner. The two
selectors agree on it.

So these drive the real function, off a real scorecard on disk, on a pair where
the selectors genuinely diverge.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import build_trace_audit as bta          # noqa: E402
from athanor.ccarc3 import client, scoring  # noqa: E402

GID = "zz99-0123456789ab"

# Cumulative, as ARC reports them. Same depth; A is dearer overall but spends the
# overrun on level 1, where the weight is smallest.
PLAY_A = [[1, 30], [2, 40], [3, 50]]     # per-level 30, 10, 10
PLAY_B = [[1, 10], [2, 20], [3, 45]]     # per-level 10, 10, 25
MEDIANS = (10, 10, 10)


def _game_dir(tmp_path, plays) -> pathlib.Path:
    d = tmp_path / GID
    d.mkdir(parents=True, exist_ok=True)
    (d / "scorecard.json").write_text(json.dumps(
        {"cards": {GID: {"actions_by_level": plays}}}))
    return d


@pytest.fixture
def medians(monkeypatch):
    monkeypatch.setattr(client, "baselines_for", lambda gid: MEDIANS)


def _proxy_pick(plays):
    """The selector this replaced: furthest first, then cheapest at that depth."""
    return max(range(len(plays)),
               key=lambda i: (len(plays[i]), -(plays[i][-1][1] if plays[i] else 0)))


def test_the_fixture_is_one_the_two_selectors_disagree_about():
    """Guards the guard. If a future edit made these agree, every assertion below
    would pass while testing nothing — which is how the last version of this file
    failed."""
    assert _proxy_pick([PLAY_A, PLAY_B]) == 1, "the proxy takes the cheaper play"
    a = scoring.score_environment(MEDIANS, [30, 10, 10]).score
    b = scoring.score_environment(MEDIANS, [10, 10, 25]).score
    assert a > b, f"the rubric takes the other one: {a} vs {b}"


def test_the_audit_returns_the_play_the_rubric_scores_highest(tmp_path, medians):
    got = bta.arc_actions_per_level(_game_dir(tmp_path, [PLAY_A, PLAY_B]), GID)
    assert got == [30, 10, 10], (
        "the audit took the cheaper play; RHAE scores the other one higher"
    )


def test_the_order_of_the_plays_does_not_decide_it(tmp_path, medians):
    """`max` keeps the first maximal element, so a selector that is really doing
    nothing passes whichever ordering happens to be tested."""
    got = bta.arc_actions_per_level(_game_dir(tmp_path, [PLAY_B, PLAY_A]), GID)
    assert got == [30, 10, 10]


def test_a_tie_breaks_toward_the_cheaper_play(tmp_path, medians):
    """Every level of a good run pins at the 1.15 ceiling, so equal scores are
    routine — `su15` displayed 42 actions on a level another play took 12 in."""
    dear = [[1, 5], [2, 10], [3, 19]]     # per-level 5, 5, 9 — every level at 1.15
    cheap = [[1, 5], [2, 10], [3, 15]]    # per-level 5, 5, 5 — likewise, but cheaper
    assert (scoring.score_environment(MEDIANS, [5, 5, 9]).score
            == scoring.score_environment(MEDIANS, [5, 5, 5]).score), "a real tie"
    assert bta.arc_actions_per_level(_game_dir(tmp_path, [dear, cheap]), GID) == [5, 5, 5]


def test_without_medians_it_falls_back_rather_than_scoring_nothing(tmp_path, monkeypatch):
    """No key, no proxy: the rubric is not computable and the old proxy is the
    honest answer. Pinned so the fallback cannot quietly become the mainline."""
    monkeypatch.setattr(client, "baselines_for", lambda gid: None)
    got = bta.arc_actions_per_level(_game_dir(tmp_path, [PLAY_A, PLAY_B]), GID)
    assert got == [10, 10, 25], "the proxy's pick, which is what the fallback is"


def test_a_missing_or_unusable_scorecard_returns_none(tmp_path, medians):
    assert bta.arc_actions_per_level(tmp_path / "nope", GID) is None
    d = tmp_path / GID
    d.mkdir(parents=True)
    (d / "scorecard.json").write_text("{not json")
    assert bta.arc_actions_per_level(d, GID) is None
