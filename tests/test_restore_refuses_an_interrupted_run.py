"""The fresh-box restore must not bank a run that ended in an error.

`restore_clean_rollouts.py` repopulates the sweep directory from preserved
evidence after a container replacement, and its own docstring says it restores
"only attempts that finished without an error". Nothing tested that: mutating
`result.get("error")` to a key that never exists passed the whole suite.

The consequence is on the unattended path. `_run_one` skips any game with a
banked `clean_result.json`, so an interrupted attempt restored as if it were
clean makes the driver treat a failed game as done -- it is never re-run, and
the sweep finishes with a scorecard that silently omits a real result. That is
the same shape as the skip-if-banked guard this restore feeds.

Fixtures are built as `preserve_evidence.sh` writes them: gzipped `result.json`
under `<game>/attempt_N/<game>/`.
"""

from __future__ import annotations

import gzip
import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PY_BIN = REPO / ".venv" / "bin" / "python"
TOOL = REPO / "tools" / "restore_clean_rollouts.py"


def _evidence(root: pathlib.Path, gid: str, *, attempt: int, payload: dict,
              card: bool = True) -> None:
    """Evidence as `preserve_evidence.sh` writes it: gzipped, under attempt_N.

    **The scorecard is not optional decoration.** The restore refuses anything
    it cannot corroborate against ARC's own card ("no scorecard preserved"), so
    a fixture without one is skipped for that reason and every assertion about
    the error branch passes vacuously. The first draft of this file omitted it
    and the guard test passed while nothing was restored at all -- caught only
    because the clean-run control failed beside it.
    """
    d = root / gid / f"attempt_{attempt}" / gid
    d.mkdir(parents=True)
    with gzip.open(d / "result.json.gz", "wt") as fh:
        json.dump(payload, fh)
    with gzip.open(d / "trace.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"level": 1}) + "\n")
    if card:
        # `levels_completed` is ONE ENTRY PER PLAY holding that play's level
        # count -- not a list of level indices. The first draft wrote
        # `range(levels)`, which the corroboration read as 3 plays whose best was
        # 2, and every restore was refused as "card 2 vs result 3 levels".
        levels = payload.get("levels_reached") or 0
        with gzip.open(d / "scorecard.json.gz", "wt") as fh:
            json.dump({"cards": {gid: {"levels_completed": [levels],
                                       "plays": 1}}}, fh)


def _restore(evidence: pathlib.Path, out: pathlib.Path) -> subprocess.CompletedProcess:
    """Run the real tool with EVIDENCE pointed at a fixture."""
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(REPO / 'tools')!r})\n"
        "import restore_clean_rollouts as r\n"
        f"r.EVIDENCE = pathlib.Path({str(evidence)!r})\n"
        f"sys.argv = ['restore', '--out', {str(out)!r}]\n"
        "raise SystemExit(r.main())"
    )
    return subprocess.run([str(PY_BIN), "-c", code], capture_output=True,
                          text=True, timeout=180,
                          env={**os.environ, "ARC_API_KEY": "dummy"})


def test_a_clean_run_is_restored(tmp_path) -> None:
    """The control: without this a tool that restores nothing would pass below."""
    ev, out = tmp_path / "ev", tmp_path / "out"
    _evidence(ev, "aa11-0000beef", attempt=1,
              payload={"game_id": "aa11-0000beef", "levels_reached": 3, "won": True})
    r = _restore(ev, out)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (out / "aa11-0000beef" / "clean_result.json").exists(), (
        f"a clean preserved run was not restored:\n{r.stdout}"
    )


def test_an_errored_run_is_not_restored(tmp_path) -> None:
    """The guard. A restored failure is a game the driver will never re-run."""
    ev, out = tmp_path / "ev", tmp_path / "out"
    _evidence(ev, "bb22-0000cafe", attempt=1,
              payload={"game_id": "bb22-0000cafe", "levels_reached": 2,
                       "error": "interrupted by a container replacement"})
    r = _restore(ev, out)
    assert r.returncode == 0, r.stderr[-1500:]
    assert not (out / "bb22-0000cafe" / "clean_result.json").exists(), (
        "an interrupted run was banked as a clean result. The driver skips any "
        "game with one, so this game would never be re-run and the sweep would "
        f"finish missing it:\n{r.stdout}"
    )
    assert "interrupted" in r.stdout


def test_a_later_clean_attempt_supersedes_an_earlier_failure(tmp_path) -> None:
    """Newest attempt first: the retry that succeeded is the one to restore."""
    ev, out = tmp_path / "ev", tmp_path / "out"
    gid = "cc33-0000f00d"
    _evidence(ev, gid, attempt=1, payload={"game_id": gid, "error": "boom"})
    _evidence(ev, gid, attempt=2,
              payload={"game_id": gid, "levels_reached": 4, "won": True})
    r = _restore(ev, out)
    assert r.returncode == 0, r.stderr[-1500:]
    banked = out / gid / "clean_result.json"
    assert banked.exists(), f"the successful retry was not restored:\n{r.stdout}"
    assert "error" not in json.loads(banked.read_text()), (
        "the failed first attempt was restored over the successful second"
    )


def test_the_newest_of_two_clean_attempts_wins(tmp_path) -> None:
    """Ordering is only observable when BOTH attempts are restorable.

    `test_a_later_clean_attempt_supersedes_an_earlier_failure` above does not
    test it: attempt_1 carries an error, so the loop `continue`s past it in
    either direction and the outcome is identical. Reversing the sort survived
    that case. Two clean attempts with different results is what makes
    newest-first mean something -- a re-run exists precisely because the earlier
    result was not the one to keep.
    """
    ev, out = tmp_path / "ev", tmp_path / "out"
    gid = "dd44-0000abcd"
    _evidence(ev, gid, attempt=1,
              payload={"game_id": gid, "levels_reached": 2, "won": False})
    _evidence(ev, gid, attempt=2,
              payload={"game_id": gid, "levels_reached": 5, "won": True})
    r = _restore(ev, out)
    assert r.returncode == 0, r.stderr[-1500:]
    banked = json.loads((out / gid / "clean_result.json").read_text())
    assert banked["levels_reached"] == 5, (
        f"the earlier attempt was restored over the later one: {banked}. A "
        f"re-run exists because the first result was not the one to keep."
    )


def test_a_run_its_card_contradicts_is_not_restored(tmp_path) -> None:
    """Corroboration is the only independent check, and it was untested.

    A card that stopped updating mid-game leaves a plausible `result.json` --
    the first clean rollout finished 8/8 while its card froze at level 3.
    Bypassing `card_corroborates` entirely passed every other case here.
    """
    ev, out = tmp_path / "ev", tmp_path / "out"
    gid = "ee55-0000dead"
    _evidence(ev, gid, attempt=1,
              payload={"game_id": gid, "levels_reached": 8, "won": True},
              card=False)
    # a card that saw only 3 levels against a result claiming 8
    d = ev / gid / "attempt_1" / gid
    with gzip.open(d / "scorecard.json.gz", "wt") as fh:
        json.dump({"cards": {gid: {"levels_completed": [3], "plays": 1}}}, fh)

    r = _restore(ev, out)
    assert r.returncode == 0, r.stderr[-1500:]
    assert not (out / gid / "clean_result.json").exists(), (
        "a run whose own scorecard contradicts it was restored as clean:\n"
        + r.stdout
    )
    assert "uncorroborated" in r.stdout
