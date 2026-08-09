"""The results ledger must cover the arm actually running.

`snapshot_results.py` exists because a review subagent once destroyed six won
environments' traces and the figures had to be rebuilt by hand from a committed
table. It is the durability mechanism -- and it had recorded nothing at all for
the current arm.

Measured on the live box 2026-08-09: 25 banked games under `clean_rollouts`,
**zero** rows for it in `results.jsonl`. Two independent causes, either alone
sufficient:

 (a) the batch list read `("runs", "runs2", "runs3", "ablate_nobaseline")`,
     written when those were the arms and never updated;
 (b) `glob("*/result.json")` cannot reach a sweep result, which lives at
     `<game>/attempt_N/<gid>/result.json`.

Fixing only one leaves the ledger just as empty, so both are pinned separately.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "tools" / "snapshot_results.py"


def _run(scratch: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), str(SNAPSHOT)],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "CCARC3_SCRATCH": str(scratch), "ARC_API_KEY": "dummy"},
    )


def _rows(scratch: pathlib.Path) -> list[dict]:
    ledger = scratch / "results.jsonl"
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def _flat_result(scratch: pathlib.Path, batch: str, gid: str) -> None:
    d = scratch / batch / gid
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"game_id": gid, "won": True}),
                                   encoding="utf-8")


def _nested_result(scratch: pathlib.Path, batch: str, gid: str) -> None:
    """The sweep layout: the result sits two levels down, beside its trace."""
    d = scratch / batch / gid / "attempt_1" / gid
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"game_id": gid, "won": False}),
                                   encoding="utf-8")
    (scratch / batch / gid / "clean_result.json").write_text(
        json.dumps({"game_id": gid, "won": False}), encoding="utf-8")


def test_the_sweep_layout_is_recorded(tmp_path: pathlib.Path) -> None:
    """Cause (b): the deep pattern must reach `attempt_N/<gid>/result.json`."""
    _nested_result(tmp_path, "clean_rollouts", "ar25-fixture1")
    out = _run(tmp_path)
    assert out.returncode == 0, out.stderr[-1500:]
    batches = {r.get("batch") for r in _rows(tmp_path)}
    assert "clean_rollouts" in batches, (
        f"a sweep result was not recorded; ledger holds {batches!r}. "
        f"stdout={out.stdout!r}"
    )


def test_the_flat_layout_still_works(tmp_path: pathlib.Path) -> None:
    """The older arms must not regress while fixing the newer one."""
    _flat_result(tmp_path, "ablate_nobaseline", "ar25-fixture2")
    out = _run(tmp_path)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "ablate_nobaseline" in {r.get("batch") for r in _rows(tmp_path)}


def test_batches_stay_discriminable(tmp_path: pathlib.Path) -> None:
    """The arms saw different information and must never be summed together.

    `ablate_nobaseline` and `clean_rollouts` runs did not see baselines; the
    `runs*` arms did. The `batch` field is the only thing keeping one headline
    number from silently blending two experiments.
    """
    _flat_result(tmp_path, "runs", "ar25-old")
    _nested_result(tmp_path, "clean_rollouts", "ar25-new")
    _run(tmp_path)
    rows = _rows(tmp_path)
    assert {r["batch"] for r in rows} == {"runs", "clean_rollouts"}
    assert all(r.get("batch") for r in rows), "a row carries no batch field"


def test_an_unlisted_directory_is_reported_not_skipped(tmp_path: pathlib.Path) -> None:
    """The bug's real shape: an omission that produced no output at all.

    The explicit list is deliberate -- a `runs*` glob once swept in a directory
    of synthetic fixtures, and a ledger mixing fabricated rows with real ones is
    worse than none. So the fix is not to auto-discover; it is to make the
    omission visible.
    """
    _flat_result(tmp_path, "some_new_arm", "ar25-fixture3")
    out = _run(tmp_path)
    assert "UNRECORDED" in out.stdout, (
        f"a results directory outside BATCHES was skipped silently: {out.stdout!r}"
    )
    assert "some_new_arm" in out.stdout
    assert "some_new_arm" not in {r.get("batch") for r in _rows(tmp_path)}, (
        "an unlisted directory was recorded anyway -- the explicit list is the "
        "guard against fabricated fixtures entering the ledger"
    )


def test_it_stays_idempotent(tmp_path: pathlib.Path) -> None:
    """Called every heartbeat forever, so a second run must add nothing."""
    _nested_result(tmp_path, "clean_rollouts", "ar25-fixture4")
    _run(tmp_path)
    first = len(_rows(tmp_path))
    second_run = _run(tmp_path)
    assert len(_rows(tmp_path)) == first, (
        f"a repeat run grew the ledger from {first} rows: {second_run.stdout!r}"
    )
    assert "0 new snapshot row(s)" in second_run.stdout


def test_the_scratchpad_copy_is_a_symlink_to_the_repo() -> None:
    """It lived only in the scratchpad, which is the store that reverts.

    Skipped where the live scratchpad is absent (a checkout elsewhere), because
    this asserts the state of this box's bring-up, not of the source tree.
    """
    sp = pathlib.Path(
        os.environ.get("CCARC3_SCRATCH")
        or "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
    )
    target = sp / "snapshot_results.py"
    if not target.exists():
        pytest.skip("no live scratchpad on this box")
    assert target.is_symlink(), (
        f"{target} is a real file again -- it will drift from the repo exactly "
        f"as heartbeat.sh did, and vanish with the next container replacement"
    )
    assert target.resolve() == SNAPSHOT.resolve()


def test_rehydrate_symlinks_it_on_a_fresh_box() -> None:
    """Otherwise the symlink above exists only until the next replacement."""
    text = (REPO / "tools" / "rehydrate_box.sh").read_text(encoding="utf-8")
    line = next((ln for ln in text.splitlines()
                 if ln.startswith("for tool in") and "heartbeat.sh" in ln), None)
    assert line, "rehydrate_box.sh no longer symlinks a tool list"
    assert "snapshot_results.py" in line, (
        f"bring-up does not restore snapshot_results.py: {line!r}"
    )
