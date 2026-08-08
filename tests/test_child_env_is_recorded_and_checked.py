"""Whether the ARC key reached the solver is recorded, and failing it is fatal.

**This surface cost two runs and was covered by a sentence.** `tu93` printed its
own median array and `bp35` made the same call six minutes in, both because
`ARC_API_KEY` sat in the child's environment; both were caught afterwards by a
watcher reading the transcript. `build_trace_audit.surface_digest` excluded the
environment from its digest — correctly, since it cannot be recovered from a
finished run — and justified the exclusion by saying `proofread_trace.py` "reads
it from the live process rather than inferring it". It never did: it takes a
workspace path and reads `stream.jsonl`, and in `--gz` mode there is no process
to read. Its only environment rule flags that the solver *looked*.

So `session._env_facts` records the key names — never the values — at the one
moment they are knowable, and `proofread_trace` fails a run whose record shows
the key or the cap reaching the child.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import proofread_trace as pt                     # noqa: E402
from athanor.ccarc3 import session as S          # noqa: E402
from athanor.ccarc3.client import GameInfo       # noqa: E402
from athanor.ccarc3 import Ccarc3Config          # noqa: E402

GID = "zz99-0123456789ab"


def _ws(tmp_path, env):
    return S.Workspace(root=tmp_path, config=Ccarc3Config(game_id=GID),
                       info=GameInfo(GID, "Test", ("click",), (10, 20, 30)), env=env)


def test_the_names_are_recorded_and_the_values_never_are(tmp_path):
    facts = S._env_facts(_ws(tmp_path, {"ARC_API_KEY": "sk-secret-value"}))
    assert facts["child_env"]["ARC_API_KEY"] is True
    assert facts["child_env"]["CCARC3_MAX_ACTIONS"] is False
    assert "sk-secret-value" not in json.dumps(facts), (
        "a value in result.json is a leak with a longer half-life than the run"
    )


def test_a_really_stripped_workspace_records_the_absences(tmp_path, monkeypatch):
    """**The strip path, built for real.** `build_without_baselines` is the
    function every scored run goes through and no test called it — so the record
    here is of the thing itself, not of an env dict assembled by the test. It
    runs offline: the shim binds a loopback port and nothing contacts ARC.
    """
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-used-offline")
    sys.path.insert(0, str(REPO / "tools"))
    import ablate_baselines as ab                     # noqa: PLC0415

    cfg = Ccarc3Config(game_id=GID, out_dir=tmp_path)
    ws = ab.build_without_baselines(
        cfg, GameInfo(GID, "Test", ("click",), (10, 20, 30)))

    assert S._env_facts(ws)["child_env"] == {
        "ARC_API_KEY": False, "ARCPRIZE_API_KEY": False, "CCARC3_MAX_ACTIONS": False,
        "CCARC3_HIDE_BASELINES": True, "CCARC3_ARC_ROOT": True}
    # The layers the record is evidence about, checked at the same time.
    assert "baseline_actions=()," in (ws.root / "session.py").read_text()
    assert "action_budget" not in json.loads((ws.root / "meta.json").read_text())


def test_the_recorded_names_cover_both_spellings_of_the_key():
    """`ablate_baselines` refuses a workspace carrying either `ARC_API_KEY` or
    `ARCPRIZE_API_KEY`. A record watching one of the two is a check with a hole
    in the shape of its own subject."""
    assert {"ARC_API_KEY", "ARCPRIZE_API_KEY"} <= set(S.RECORDED_ENV)


def _proofread(tmp_path, monkeypatch, child_env) -> int:
    ws = tmp_path / GID
    ws.mkdir(parents=True, exist_ok=True)
    result = {"game_id": GID, "levels_reached": 3, "playthroughs": 1,
              "actions_used": 100, "attempts": 1, "won": True}
    if child_env is not None:
        result["child_env"] = child_env
    (ws / "result.json").write_text(json.dumps(result))
    (ws / "scorecard.json").write_text(json.dumps(
        {"cards": {GID: {"levels_completed": [3], "total_actions": 100}}}))
    (ws / "stream.jsonl").write_text(json.dumps(
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "input": {"command": "ls ."}}]}}))
    monkeypatch.setattr(pt, "baselines", lambda gid: [])
    monkeypatch.setattr(pt, "all_baselines", dict)
    monkeypatch.setattr(sys, "argv", ["proofread_trace", str(ws), "--quiet"])
    return pt.main()


@pytest.mark.parametrize("leaked", ["ARC_API_KEY", "ARCPRIZE_API_KEY",
                                    "CCARC3_MAX_ACTIONS"])
def test_a_run_whose_child_saw_the_key_or_the_cap_fails(tmp_path, monkeypatch, leaked):
    env = {"ARC_API_KEY": False, "CCARC3_MAX_ACTIONS": False}
    env[leaked] = True
    assert _proofread(tmp_path, monkeypatch, env) == 2


def test_a_clean_child_env_passes(tmp_path, monkeypatch):
    assert _proofread(tmp_path, monkeypatch,
                      {"ARC_API_KEY": False, "CCARC3_MAX_ACTIONS": False}) == 0


def test_a_run_predating_the_record_is_reported_not_voided(tmp_path, monkeypatch):
    """Every run banked before 2026-08-08 has no record. A rule that
    retroactively voids the corpus is a rule that gets switched off."""
    assert _proofread(tmp_path, monkeypatch, None) == 0


def test_the_env_verdict_is_in_the_failing_set():
    assert "ENV" in pt.FAILING_VERDICTS, (
        "INBOUND was appended, printed, and not recognised by the predicate for "
        "weeks; this is that mistake's second chance"
    )
