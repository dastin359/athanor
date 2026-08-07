"""The audit page must pick its best play by the rubric, not by a proxy.

`arc_actions_per_level` chose "furthest first, then cheapest at that depth". RHAE
weights level `i` by `i`, so a play that spent more actions in total can still
score higher when it spent them on early levels — the proxy and the maximum are
different functions. That proxy produced every `E` and `raw` in
`evidence/ccarc3/trace_audit/runs.json.gz` and on the published page.

On the 29 preserved scorecards (23 multi-play) the two never disagreed, so no
recorded number moved when this was fixed. This test is the constructed case
where they do, so the proxy cannot come back unnoticed.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from athanor.ccarc3 import scoring


def test_the_proxy_and_the_rubric_disagree_on_a_constructible_case():
    """Two plays, same depth; the one with MORE total actions scores higher.

    Three levels, medians (10, 10, 10). Play A spends 30 on level 1 and 10 each
    after; play B spends 10, 10, 30. Same depth, same total — but the weights are
    1, 2, 3, so wasting the actions on level 3 costs three times as much.
    """
    baselines = (10, 10, 10)
    play_a = [30, 10, 10]          # slow early, fast late
    play_b = [10, 10, 30]          # fast early, slow late

    a = scoring.score_environment(baselines, play_a)
    b = scoring.score_environment(baselines, play_b)

    assert sum(play_a) == sum(play_b), "the proxy cannot tell these apart"
    assert len(play_a) == len(play_b), "same depth, so the proxy tie-breaks on total"
    assert a.score > b.score, (
        "RHAE must prefer the play that overran an EARLY level: weights are the "
        f"level indices, so {a.score} should beat {b.score}"
    )
