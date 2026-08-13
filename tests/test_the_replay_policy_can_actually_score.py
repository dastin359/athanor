"""The retry policy has never once fired, because scoring always raised.

`_rhae` computes a run's RHAE so `_wants_replay` can send a short score back to
the queue. It contained two independent errors on one line:

  * arguments swapped -- `score_run(base, transitions)` against a signature of
    `score_run(transitions, baselines)`, so scoring received a tuple of ints
    where a ledger belonged and died on `'int' object has no attribute
    'full_reset'`;
  * the wrong attribute -- `.E` on an `EnvironmentScore`, which exposes `raw`,
    `cap` and `score` and never `E`.

Neither was ever observed, because the only caller wrapped it in
`except Exception` and turned every failure into `None` -- which
`_wants_replay` reads as "cannot be scored, leave it banked". So a policy whose
whole purpose is to refuse to let a bad score stand quietly refused to run, and
`lf52` sat banked at 7/10 with `won=False` and was never requeued.

The blanket catch is right for its stated purpose: a run that genuinely cannot be
scored must still bank rather than be lost. It is wrong as a place for a
`TypeError` to land, and this file pins that distinction.

**Baselines here are synthetic.** The real per-level medians are the one number
the solver must never see, and this repository is on its `PYTHONPATH`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))


def _trace(tmp_path, rows):
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def _row(i, level, state="NOT_FINISHED", full_reset=False):
    return {"i": i, "level": level, "action": "ACTION1", "params": {},
            "hypothesis": None, "score": level, "state": state,
            "full_reset": full_reset, "available_actions": ["ACTION1"],
            "frames": [[[0]]]}


def test_rhae_returns_a_number_for_a_scoreable_run(tmp_path, monkeypatch):
    """The regression that matters: a real ledger must score, not return None."""
    import clean_rollouts as cr
    from athanor.ccarc3 import client as _client

    # two levels, cleared in 4 and 4 actions
    rows = [_row(0, 0), _row(1, 0), _row(2, 0), _row(3, 1),
            _row(4, 1), _row(5, 1), _row(6, 1), _row(7, 2)]
    ws = _trace(tmp_path, rows)
    monkeypatch.setattr(_client, "baselines_for", lambda *a, **k: (4, 4))

    e = cr._rhae(ws, "gm01-abcd")
    assert e is not None, (
        "scoring returned None for a perfectly good ledger — the replay policy "
        "is inert whenever this happens, and it happened on every run")
    assert isinstance(e, float) and 0.0 <= e <= 1.0, e


def test_a_programming_error_is_loud_not_banked_quietly(tmp_path, monkeypatch, capsys):
    """A TypeError cannot come from bad data; it means this function is wrong."""
    import clean_rollouts as cr
    from athanor.ccarc3 import client as _client, scoring as _scoring

    ws = _trace(tmp_path, [_row(0, 0), _row(1, 1)])
    monkeypatch.setattr(_client, "baselines_for", lambda *a, **k: (2,))
    monkeypatch.setattr(_scoring, "score_run",
                        lambda *a, **k: (_ for _ in ()).throw(TypeError("boom")))

    assert cr._rhae(ws, "gm01-abcd") is None       # still banks
    out = capsys.readouterr().out
    assert "SCORING IS BROKEN" in out, (
        "a programming error printed the same routine message as an unscoreable "
        "run — which is exactly how two bugs survived on one line:\n" + out)
    assert "inert" in out


def test_an_unscoreable_run_still_banks_quietly(tmp_path, monkeypatch, capsys):
    """The blanket catch keeps its job: no baselines is a fact about the game,
    not a bug, and must not cost the run its bank."""
    import clean_rollouts as cr
    from athanor.ccarc3 import client as _client

    ws = _trace(tmp_path, [_row(0, 0)])
    monkeypatch.setattr(_client, "baselines_for", lambda *a, **k: ())
    assert cr._rhae(ws, "gm01-abcd") is None
    assert "SCORING IS BROKEN" not in capsys.readouterr().out


def test_a_short_score_is_requeued_and_a_full_one_is_not(tmp_path, monkeypatch):
    """The behaviour the whole mechanism exists for, end to end."""
    import clean_rollouts as cr

    monkeypatch.setattr(cr, "OUT", tmp_path)
    for gid, e in (("short-1111", 0.4), ("full-2222", 1.0), ("unknown-3333", None)):
        (tmp_path / gid).mkdir()
        (tmp_path / gid / "clean_result.json").write_text(json.dumps({"E": e}))
    monkeypatch.setattr(cr, "completed_attempts", lambda *a, **k: 1)

    assert cr._wants_replay("short-1111") is True
    assert cr._wants_replay("full-2222") is False
    assert cr._wants_replay("unknown-3333") is False, (
        "unknown must mean 'leave it banked', never 'it scored zero'")
