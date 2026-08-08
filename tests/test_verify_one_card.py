"""One sweep, one scorecard — checked, not assumed.

A split sweep is silent: every game plays, scores and banks a clean result, the
driver prints success, and the artifact is worth nothing. At ~$650 a sweep that
is an expensive thing to discover afterwards.
"""
import json
import pathlib
import subprocess
import sys

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "verify_one_card.py"


def _game(sweep, gid, card, *, acts=10, levels=3, foreign=""):
    ws = sweep / gid / "attempt_1" / gid
    ws.mkdir(parents=True)
    banked = {"actions_used": acts, "levels_reached": levels}
    (sweep / gid / "clean_result.json").write_text(json.dumps(banked))
    r = dict(banked, card_id=card)
    if foreign:
        r["foreign_card"] = foreign
    (ws / "result.json").write_text(json.dumps(r))
    return ws


def _run(sweep, *extra):
    return subprocess.run([sys.executable, str(TOOL), str(sweep), *extra],
                          capture_output=True, text=True)


def test_one_card_across_every_game_passes(tmp_path):
    _game(tmp_path, "aa11-x", "CARD")
    _game(tmp_path, "bb22-y", "CARD")

    out = _run(tmp_path)

    assert out.returncode == 0, out.stdout
    assert "one card, 2 games" in out.stdout


def test_a_split_sweep_fails(tmp_path):
    """The failure being guarded: two games, two cards, nothing else wrong."""
    _game(tmp_path, "aa11-x", "CARD_A")
    _game(tmp_path, "bb22-y", "CARD_B")

    out = _run(tmp_path)

    assert out.returncode == 1
    assert "SPLIT" in out.stdout and "cannot be submitted" in out.stdout


def test_a_game_resumed_onto_its_own_card_is_named(tmp_path):
    """Correct for that game, wrong for the artifact — so it must be visible."""
    _game(tmp_path, "aa11-x", "CARD")
    _game(tmp_path, "bb22-y", "CARD", foreign="CARD")

    out = _run(tmp_path)

    assert out.returncode == 1
    assert "NOT ON THE SWEEP CARD" in out.stdout and "bb22" in out.stdout


def test_the_expected_card_is_enforced_when_given(tmp_path):
    _game(tmp_path, "aa11-x", "CARD")

    assert _run(tmp_path, "--expect", "CARD").returncode == 0
    out = _run(tmp_path, "--expect", "OTHER")
    assert out.returncode == 1 and "EXPECTED card OTHER" in out.stdout


def test_the_card_is_read_from_the_state_file_when_the_result_lacks_one(tmp_path):
    """`card_id` only entered result.json on 2026-08-08. Without this fallback
    the check reports "cannot confirm" for every run banked before then."""
    ws = _game(tmp_path, "aa11-x", "")
    (ws / "trace.state.json").write_text(json.dumps({"card_id": "FROM_STATE"}))

    out = _run(tmp_path)

    assert out.returncode == 0, out.stdout
    assert "FROM_STATE" in out.stdout


def test_the_banked_attempt_is_the_one_checked(tmp_path):
    """Globbing any attempt_* reads an abandoned run — the trap that scored
    su15's attempt_2 (2/9) for a run that banked attempt_3 (9/9)."""
    gid = "aa11-x"
    (tmp_path / gid).mkdir()
    (tmp_path / gid / "clean_result.json").write_text(
        json.dumps({"actions_used": 99, "levels_reached": 9}))
    for n, (acts, lv, card) in enumerate([(5, 1, "ABANDONED"), (99, 9, "BANKED")], 1):
        ws = tmp_path / gid / ("attempt_%d" % n) / gid
        ws.mkdir(parents=True)
        (ws / "result.json").write_text(
            json.dumps({"actions_used": acts, "levels_reached": lv, "card_id": card}))

    out = _run(tmp_path)

    assert "BANKED" in out.stdout and "ABANDONED" not in out.stdout
