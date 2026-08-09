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
    """Run the checker against a fixture.

    Declares the fixture's own size unless the test says otherwise: without it
    every card-identity test would also trip the completeness gate, which
    defaults to the driver's 25-game queue. The completeness dimension has its
    own tests below rather than riding along in all of them.
    """
    extra = list(extra)
    if "--games" not in extra and "--partial" not in extra:
        extra += ["--games", str(len(list(pathlib.Path(sweep).glob("*/clean_result.json"))))]
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


def test_a_sweep_missing_games_is_not_submittable(tmp_path):
    """**A gate that counts only what arrived cannot see what did not.**

    This enumerated `*/clean_result.json` and reported "one card, N games —
    submittable" for whatever N happened to be there, so a sweep that lost five
    games to crashes passed: every game that *did* bank was on one card.
    Verified on a three-game fixture with one unbanked — exit 0, "submittable".
    """
    _game(tmp_path, "aa11-x", "CARD")
    _game(tmp_path, "bb22-y", "CARD")

    out = _run(tmp_path, "--games", "3")

    assert out.returncode == 1
    assert "INCOMPLETE" in out.stdout and "2 of 3" in out.stdout


def test_partial_is_an_explicit_choice(tmp_path):
    """Mid-sweep the incompleteness is expected; it must be asked for, not the
    default, or the gate is back where it started."""
    _game(tmp_path, "aa11-x", "CARD")

    assert _run(tmp_path, "--games", "3").returncode == 1
    assert _run(tmp_path, "--games", "3", "--partial").returncode == 0


def test_the_expected_count_comes_from_the_drivers_own_queue():
    """**`_expected_games` had no test, and was silently set to `return 0`.**

    Every other test here passes `--games N` explicitly, so none of them ever
    called this function — and `return 0` makes `expected` falsy, which disables
    the completeness gate entirely while leaving the suite green. The edit
    reached a commit because `git add -A` swept it in unread.

    The count has to come from the driver's queue so the two cannot drift, and
    it has to be non-zero or the gate it feeds does nothing.
    """
    import importlib
    import sys as _s

    _s.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
    voc = importlib.import_module("verify_one_card")
    cr = importlib.import_module("clean_rollouts")

    assert voc._expected_games() == len(cr.GAMES)
    assert voc._expected_games() > 0, (
        "a zero expectation turns the completeness check into a no-op"
    )


def test_a_sweep_missing_games_fails_without_being_told_how_many(tmp_path):
    """The gate must work on the default path, which is the one a submission
    actually uses — no `--games`, no `--partial`."""
    _game(tmp_path, "aa11-x", "CARD")
    _game(tmp_path, "bb22-y", "CARD")

    out = subprocess.run([sys.executable, str(TOOL), str(tmp_path)],
                         capture_output=True, text=True)

    assert out.returncode == 1, out.stdout
    assert "INCOMPLETE" in out.stdout


def test_a_game_with_no_card_id_is_refused(tmp_path):
    """A result that records no card cannot be shown to be on the sweep's card.

    Found by mutation: deleting the `if unknown:` branch passed the whole suite.
    Such a game could be on any card at all -- it predates `_card_facts`, or its
    state file was removed -- so a sweep containing one is not *provably* single
    card, which is the only thing this tool exists to establish.
    """
    _game(tmp_path, "aa11-x", "CARD")
    ws = _game(tmp_path, "bb22-y", "CARD")
    r = json.loads((ws / "result.json").read_text())
    r.pop("card_id")
    (ws / "result.json").write_text(json.dumps(r))

    out = _run(tmp_path)
    assert out.returncode != 0, (
        f"a game recording no card_id was accepted as on the sweep card:\n"
        f"{out.stdout}"
    )
    assert "no card_id" in out.stdout


def test_an_empty_sweep_is_not_submittable(tmp_path):
    """Zero games must not read as success.

    `return 0 if ok and total else 1` -- dropping `and total` survived every
    other case here, and the failure is this file's own subject: nothing went
    wrong, so `ok` stays True, and an empty directory reports submittable. It is
    the same shape as the bug the module docstring records fixing, one step on:
    a gate that only counts what arrived cannot see that nothing did.
    """
    (tmp_path / "empty_sweep").mkdir()
    out = _run(tmp_path / "empty_sweep", "--partial")
    assert out.returncode != 0, (
        f"an empty sweep reported submittable:\n{out.stdout}"
    )
    assert "NOT submittable" in out.stdout


def test_a_sweep_of_only_uncarded_games_is_not_submittable(tmp_path):
    """Both survivors at once: nothing carded, nothing to submit."""
    ws = _game(tmp_path, "aa11-x", "CARD")
    r = json.loads((ws / "result.json").read_text())
    r.pop("card_id")
    (ws / "result.json").write_text(json.dumps(r))
    out = _run(tmp_path)
    assert out.returncode != 0, out.stdout
    assert "NOT submittable" in out.stdout
