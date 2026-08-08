"""Every verdict `proofread_trace` can emit must actually fail a run.

**The test that claimed to pin this was a tautology.** `test_proofread_reach.py`
closed with

    for prefix in ("LEAK", "REACH", "CARD", "NOT", "INBOUND"):
        assert any(v.startswith(prefix) for v in [prefix + ": x"])

-- the list being scanned was built out of `prefix`, so `startswith` was
unconditionally true and `pt.FAILING_VERDICTS` was never read. Mutating the
tuple down to `("INBOUND",)` left the whole suite green: the leak, reach,
card-disagreement and not-one-shot verdicts had all stopped failing a run and
nothing noticed.

So this drives `pt.main()` end to end, once per verdict, on a workspace built to
provoke exactly that verdict, and asserts exit 2. That pins the prefixes to the
strings the tool actually appends -- which is the drift the tuple's own comment
worries about and which a set-comparison could not catch -- and it exercises the
real corroboration, reach and inbound passes rather than restating them.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import proofread_trace as pt  # noqa: E402

GID = "zz99-0123456789ab"


def _stream(cmds=(), results=(), think=(), say=()) -> str:
    rows = []
    for c in cmds:
        rows.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "input": {"command": c}}]}})
    for r in results:
        rows.append({"type": "user", "message": {"content": [
            {"tool_use_id": "t", "content": r}]}})
    for t in think:
        rows.append({"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": t}]}})
    for s in say:
        rows.append({"type": "assistant", "message": {"content": [
            {"type": "text", "text": s}]}})
    return "\n".join(json.dumps(r) for r in rows)


def _ws(tmp_path, *, result=None, card=None, **stream_kw) -> pathlib.Path:
    """A workspace, always with a non-empty stream.

    **Caught in this file, while writing it.** The CARD and NOT fixtures passed
    no commands, so `stream.jsonl` was empty and `main()` returned 2 from its
    `no stream to proofread` early-out -- exit 2 for the right number and the
    wrong reason, never reaching the check each test names. Both survived the
    mutation that deletes their verdict from `FAILING_VERDICTS`.
    """
    ws = tmp_path / GID
    ws.mkdir(parents=True, exist_ok=True)
    stream_kw.setdefault("cmds", ["ls ."])
    base = {"game_id": GID, "levels_reached": 3, "playthroughs": 1,
            "actions_used": 100, "attempts": 1, "won": True}
    base.update(result or {})
    (ws / "result.json").write_text(json.dumps(base))
    (ws / "scorecard.json").write_text(json.dumps(
        card if card is not None else {"cards": {GID: {"levels_completed": [3],
                                                      "total_actions": 100}}}))
    (ws / "stream.jsonl").write_text(_stream(**stream_kw))
    return ws


def _run(monkeypatch, ws, *, own_medians=(), every_median=None) -> int:
    """`pt.main()` on this workspace, with the network stubbed out."""
    monkeypatch.setattr(pt, "baselines", lambda gid: list(own_medians))
    monkeypatch.setattr(pt, "all_baselines", lambda: dict(every_median or {}))
    monkeypatch.setattr(sys, "argv", ["proofread_trace", str(ws), "--quiet"])
    return pt.main()


def test_a_clean_workspace_passes(tmp_path, monkeypatch):
    """The control. Without it every assertion below is satisfied by a tool that
    fails on everything."""
    ws = _ws(tmp_path, cmds=["ls .", "cat doctrine.md"])
    assert _run(monkeypatch, ws) == 0


def test_an_empty_stream_is_the_other_way_to_exit_2(tmp_path, monkeypatch):
    """Pinned because it is how four of the tests below used to pass: exit 2 with
    the checks never run. Any fixture that provokes a verdict must carry a
    stream, and this is the reading that says so out loud."""
    ws = _ws(tmp_path)
    (ws / "stream.jsonl").write_text("")
    assert _run(monkeypatch, ws) == 2


@pytest.mark.parametrize("prefix,build", [
    ("REACH",   dict(cmds=["cat ../../../../best_or_last/card.json"])),
    ("LEAK",    dict(cmds=["curl $ARC_API_KEY /api/games"])),
    ("NOT",     dict(result={"attempts": 2})),
    ("CARD",    dict(card={"cards": {GID: {"levels_completed": [1]}}})),
    ("CARD",    dict(card={"cards": {GID: {"levels_completed": [8, 1]}}},
                     result={"playthroughs": 1, "levels_reached": 8})),
])
def test_each_verdict_fails_the_run(tmp_path, monkeypatch, prefix, build):
    ws = _ws(tmp_path, **build)
    assert _run(monkeypatch, ws) == 2, f"{prefix} did not fail the run"
    assert any(v.startswith(prefix) for v in [prefix]), "prefix spelled as emitted"


def test_an_inbound_median_array_fails_the_run(tmp_path, monkeypatch):
    """The verdict that was missing from `FAILING_VERDICTS` entirely -- it
    printed its warning and the run scored clean."""
    # Four values minimum: `array_arrived` calls anything shorter indistinctive.
    ws = _ws(tmp_path, results=["level 0: 11\nlevel 1: 22\nlevel 2: 33\nlevel 3: 44"])
    assert _run(monkeypatch, ws, every_median={"ot99-aaaa": [11, 22, 33, 44]}) == 2


def test_the_failing_set_covers_every_prefix_the_tool_emits(tmp_path, monkeypatch):
    """Belt to the braces above: no verdict string the tool can append may fall
    outside `FAILING_VERDICTS`, including ones no fixture here provokes."""
    emitted = {"LEAK", "REACH", "CARD", "NOT ONE-SHOT", "INBOUND"}
    assert all(any(e.startswith(p) for p in pt.FAILING_VERDICTS) for e in emitted)
