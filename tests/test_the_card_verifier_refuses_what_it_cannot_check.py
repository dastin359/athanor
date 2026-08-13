"""`verify_one_card.py` must not report "submittable" by failing to look.

The file's own docstring is about this defect class -- "the guard that detects
instead of enforcing", "a gate that only counts what arrived cannot see what did
not" -- and it carried the same hole one layer down.

`_expected_games()` imports the driver to learn how many games a full sweep
holds, and returned **0** if that import raised. `main()` gates completeness on
`if expected and total < expected`, so 0 did not mean "zero games required"; it
meant the check did not run. A card holding 5 of 25 games printed
``one card, 5 games -- submittable as a single scorecard_url`` and exited 0.

At roughly $650 a sweep, and with a submission being a single `scorecard_url`,
that is the expensive way to find out.

Fixed by making "I cannot tell" a distinct answer from "nothing is required":
`_expected_games()` returns None and says why, and `main()` REFUSES rather than
passing. `--games N` and `--partial` remain the two ways to say what is expected.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
VERIFY = REPO / "tools" / "verify_one_card.py"

CARD = "1944f8ab-0000-4000-8000-00000000beef"


def _sweep(tmp_path: pathlib.Path, games: int, card: str = CARD) -> pathlib.Path:
    """A sweep directory holding `games` banked results, all on one card.

    Mirrors the real layout, because `cards_in` does NOT read the card out of
    `clean_result.json` -- it walks `attempt_*/<gid>/result.json` and takes only
    the attempt whose (actions_used, levels_reached) match the banked result.
    (That matching exists because a glob once read su15's abandoned attempt_2 at
    2/9 while attempt_3 had won 9/9.) A fixture that skips the attempt tree
    verifies as zero games and would have made every assertion here vacuous.
    """
    sweep = tmp_path / "clean_rollouts"
    for i in range(games):
        gid = f"g{i:02d}-{card[:8]}"
        d = sweep / gid
        facts = {"game_id": gid, "actions_used": 100 + i, "levels_reached": 1}
        (d / "attempt_1" / gid).mkdir(parents=True)
        (d / "clean_result.json").write_text(json.dumps(facts), encoding="utf-8")
        (d / "attempt_1" / gid / "result.json").write_text(
            json.dumps({**facts, "card_id": card}), encoding="utf-8")
    return sweep


def _run(sweep: pathlib.Path, *args: str, env: dict | None = None):
    return subprocess.run(
        [sys.executable, str(VERIFY), str(sweep), *args],
        capture_output=True, text=True, timeout=120,
        env=env, cwd=str(REPO))


# ── the hole ─────────────────────────────────────────────────────────────────

def test_an_unreadable_game_list_is_refused_not_ignored(tmp_path):
    """The regression, end to end. Break the import; must NOT say submittable.

    **How the import is broken matters, and the first draft got it wrong.** It
    set a hostile `CCARC3_SWEEP_DIR`, which makes `clean_rollouts` raise
    SystemExit at import -- a BaseException, so the old `except Exception` never
    saw it and the tool died with a traceback. `returncode != 0` was satisfied by
    the crash, so the test passed against the pre-fix code and proved nothing.
    A check that agrees with the truth for the wrong reason, in a file about
    exactly that.

    So: copy the verifier alone into a directory with no `clean_rollouts.py`
    beside it. The import then fails with ModuleNotFoundError -- an ordinary
    Exception, the one the old handler swallowed into `return 0`.
    """
    lonely = tmp_path / "bin"
    lonely.mkdir()
    solo = lonely / "verify_one_card.py"
    solo.write_text(VERIFY.read_text(encoding="utf-8"), encoding="utf-8")

    sweep = _sweep(tmp_path, games=3)
    proc = subprocess.run([sys.executable, str(solo), str(sweep)],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(tmp_path))
    assert proc.returncode != 0, (
        "the verifier exited 0 with no way to know whether the sweep is "
        "complete:\n" + proc.stdout + proc.stderr)
    assert "submittable as a single scorecard_url" not in proc.stdout, proc.stdout
    assert "UNVERIFIABLE" in proc.stdout, (
        "it refused, but not for this reason -- check it is not just crashing:\n"
        + proc.stdout + proc.stderr)


def test_the_expected_count_says_it_cannot_tell(capsys):
    """`_expected_games()` returns None, never 0, when it cannot read the list.

    0 and None are the same falsy value to `if expected and ...`, which is
    exactly how "nothing is required" and "I do not know what is required" got
    conflated. Asserting identity, not truthiness.
    """
    sys.path.insert(0, str(REPO / "tools"))
    import importlib
    voc = importlib.import_module("verify_one_card")

    saved = sys.modules.get("clean_rollouts")
    sys.modules["clean_rollouts"] = None      # `import x` on a None entry raises
    try:
        got = voc._expected_games()
    finally:
        if saved is None:
            sys.modules.pop("clean_rollouts", None)
        else:
            sys.modules["clean_rollouts"] = saved

    assert got is None, f"expected None (cannot tell), got {got!r}"
    assert "completeness" in capsys.readouterr().out.lower()


# ── the paths that must keep working ─────────────────────────────────────────

def test_an_explicit_count_still_gates_completeness(tmp_path):
    sweep = _sweep(tmp_path, games=3)
    proc = _run(sweep, "--games", "5")
    assert proc.returncode != 0, proc.stdout
    assert "INCOMPLETE: 3 of 5" in proc.stdout, proc.stdout


def test_an_explicit_count_that_is_met_passes(tmp_path):
    sweep = _sweep(tmp_path, games=3)
    proc = _run(sweep, "--games", "3")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "submittable as a single scorecard_url" in proc.stdout, proc.stdout


def test_partial_is_an_explicit_operator_choice(tmp_path):
    """`--partial` must still pass -- it is the operator saying so out loud."""
    sweep = _sweep(tmp_path, games=3)
    proc = _run(sweep, "--partial")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UNVERIFIABLE" not in proc.stdout, proc.stdout


def test_a_split_card_is_still_caught(tmp_path):
    """The check this file was written for must survive the change."""
    sweep = _sweep(tmp_path, games=2)
    other = "cafe0000-0000-4000-8000-00000000cafe"
    gid = f"g99-{other[:8]}"
    facts = {"game_id": gid, "actions_used": 999, "levels_reached": 1}
    (sweep / gid / "attempt_1" / gid).mkdir(parents=True)
    (sweep / gid / "clean_result.json").write_text(json.dumps(facts), encoding="utf-8")
    (sweep / gid / "attempt_1" / gid / "result.json").write_text(
        json.dumps({**facts, "card_id": other}), encoding="utf-8")
    proc = _run(sweep, "--games", "3")
    assert proc.returncode != 0, proc.stdout
    assert "SPLIT" in proc.stdout, proc.stdout


def test_the_fixture_is_actually_read(tmp_path):
    """Guards the guard: three banked games must be SEEN as three.

    Without this, every assertion above passes just as well against a fixture
    the verifier cannot parse -- "INCOMPLETE: 0 of 3" satisfies a
    `returncode != 0` check while proving nothing. That is how the first draft
    of this file passed its refusal test and failed everything else.
    """
    sweep = _sweep(tmp_path, games=3)
    proc = _run(sweep, "--games", "3")
    assert "3 games" in proc.stdout, proc.stdout
    assert CARD in proc.stdout, proc.stdout
