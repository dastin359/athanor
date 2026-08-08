"""The card check must read only the plays *this attempt* produced.

**The guard silently changed meaning when the shared scorecard landed.**
`levels_completed` holds one entry per play, and a shared card keeps appending to
the same `cards[game_id]` entry across every *attempt* of a game -- so
`max(levels_completed)`, which all three copies of this guard used, carries the
high-water mark of attempts that were thrown away. A game whose attempt_1 reached
level 8 and was discarded, and whose attempt_2 has a card that stopped updating
at level 3, reads `best=8 >= reached=8` and banks as corroborated. That is the
exact `sb26` failure the guard exists to catch, made invisible by the fix to an
unrelated problem.

Three tools carry this check and none of them had a test. They now share one
implementation, and this drives each of them from disk.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import clean_rollouts as cr            # noqa: E402
import restore_clean_rollouts as rcr   # noqa: E402
from athanor.ccarc3 import scoring     # noqa: E402

GID = "sb26-7fbdac44"

# attempt_1 got to level 8 and was discarded; attempt_2 is the one being banked
# and its card froze at 3. Both live in one card entry.
POISONED = {"cards": {GID: {"levels_completed": [8, 3], "total_actions": 400}}}
BANKED = {"game_id": GID, "levels_reached": 8, "playthroughs": 1}


def _card_dir(tmp_path, card, name="ws") -> pathlib.Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "scorecard.json").write_text(json.dumps(card))
    return d


def _evidence_dir(tmp_path, card, name="ev") -> pathlib.Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "scorecard.json.gz", "wt") as fh:
        json.dump(card, fh)
    return d


def test_a_frozen_card_is_not_excused_by_a_discarded_attempt():
    """`max()` over the whole entry passes this; the attempt's own plays do not."""
    assert max(POISONED["cards"][GID]["levels_completed"]) == 8, "the old reading"
    assert scoring.card_disagreement(POISONED, GID, BANKED) == "card 3 vs result 8 levels"


def test_the_sweep_driver_refuses_it(tmp_path):
    assert cr.uncorroborated(_card_dir(tmp_path, POISONED), BANKED)


def test_the_restore_refuses_it(tmp_path):
    assert rcr.card_corroborates(_evidence_dir(tmp_path, POISONED), BANKED)


def test_a_card_holding_fewer_plays_than_the_run_is_a_disagreement():
    """A card that stopped listening mid-run leaves fewer entries than the trace
    recorded resets. That is the named failure, not a shrug."""
    card = {"cards": {GID: {"levels_completed": [3]}}}
    result = {"game_id": GID, "levels_reached": 3, "playthroughs": 3}
    assert scoring.card_disagreement(card, GID, result) == (
        "card holds 1 play(s), result claims 3")


def test_a_real_multi_play_attempt_still_corroborates(tmp_path):
    """The control, from preserved evidence: `su15` attempt_3 ran three plays and
    its card holds exactly those three. Nothing here may reject it."""
    ev = pathlib.Path("evidence/ccarc3/clean_rollouts/su15-1944f8ab/attempt_3/su15-1944f8ab")
    if not (ev / "scorecard.json.gz").exists():
        return                            # evidence not checked out in this tree
    card = json.loads(gzip.open(ev / "scorecard.json.gz").read())
    result = json.loads(gzip.open(ev / "result.json.gz").read())
    assert result["playthroughs"] == 3 and result["levels_reached"] == 9
    assert scoring.card_disagreement(card, result["game_id"], result) == ""
    assert cr.uncorroborated(_card_dir(tmp_path, card), result) == ""


def test_the_three_copies_share_one_implementation():
    """They drifted apart once already. Each must reach the same function."""
    import ast, inspect                   # noqa: PLC0415
    for fn in (cr.uncorroborated, rcr.card_corroborates):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        body = ast.unparse(tree.body[0].body[1:])   # drop the docstring; it
        assert "card_disagreement" in body           # quotes the broken reading
        assert "max(" not in body, "a local re-implementation is how this broke"
