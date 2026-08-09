"""The restore must write where the driver reads, under any sweep name.

`clean_rollouts.py` computes `OUT = SP / (CCARC3_SWEEP_DIR or "clean_rollouts")`.
`restore_clean_rollouts.py` read `evidence/ccarc3/clean_rollouts` and wrote
`$SCRATCH/clean_rollouts`, both hard-coded -- and `rehydrate_box.sh` invokes it
with no arguments.

So for the submission sweep (`CCARC3_SWEEP_DIR=clean_rollouts_submission`) a
container replacement mid-run restores banked games into the WRONG directory,
the driver reads its own and finds nothing, and it re-runs games that are
already on the shared scorecard. That costs money twice and corrupts the
artifact: a scorecard cannot un-play a game, so the second play lands beside the
first and the per-game row count stops matching the result.

The invariant is the pairing, so it is asserted as a pairing -- both values read
from the real modules under the same environment -- rather than as two separate
"is it clean_rollouts" checks, which is how the two drifted apart in the first
place while each looked right on its own.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PY_BIN = REPO / ".venv" / "bin" / "python"

SWEEPS = [None, "clean_rollouts", "clean_rollouts_submission", "clean_rollouts_v2"]


def _probe(code: str, sweep: str | None, scratch: pathlib.Path) -> str:
    env = {k: v for k, v in os.environ.items() if k != "CCARC3_SWEEP_DIR"}
    env["CCARC3_SCRATCH"] = str(scratch)
    env["ARC_API_KEY"] = "dummy"
    if sweep is not None:
        env["CCARC3_SWEEP_DIR"] = sweep
    out = subprocess.run(
        [str(PY_BIN), "-c",
         f"import sys; sys.path[:0] = [{str(REPO / 'tools')!r}, {str(REPO / 'src')!r}]\n" + code],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("sweep", SWEEPS)
def test_restore_destination_matches_driver_output(sweep, tmp_path) -> None:
    driver = _probe("import clean_rollouts as cr; print(cr.OUT)", sweep, tmp_path)
    restore = _probe(
        "import restore_clean_rollouts as r, argparse, pathlib\n"
        "print(r.SCRATCH / r.SWEEP_DIR)", sweep, tmp_path)
    assert driver == restore, (
        f"CCARC3_SWEEP_DIR={sweep!r}: the driver reads {driver!r} but the "
        f"restore writes {restore!r}. After a container replacement the driver "
        f"would find no banked games and re-run them onto the shared card."
    )


@pytest.mark.parametrize("sweep", SWEEPS)
def test_restore_reads_the_evidence_for_that_sweep(sweep, tmp_path) -> None:
    """The source half. preserve_evidence mirrors $SP/<dir> to $DEST/<dir>.

    It globs `clean_rollouts*`, which is why evidence/ccarc3/ already holds
    clean_rollouts_stale_card and clean_rollouts_void. Only the restore was
    pinned to one name, so it would read an empty or unrelated directory.
    """
    name = sweep or "clean_rollouts"
    evidence = _probe("import restore_clean_rollouts as r; print(r.EVIDENCE)",
                      sweep, tmp_path)
    assert evidence.endswith(f"evidence/ccarc3/{name}"), (
        f"CCARC3_SWEEP_DIR={sweep!r}: restore reads {evidence!r}, which is not "
        f"where preserve_evidence.sh mirrors that sweep"
    )


def test_the_default_argument_is_not_a_frozen_literal() -> None:
    """`--out`'s default is what rehydrate_box.sh gets, because it passes none."""
    src = (REPO / "tools" / "restore_clean_rollouts.py").read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if '"--out"' in ln)
    assert '"clean_rollouts"' not in line, (
        f"--out defaults to a literal again: {line.strip()!r} -- bring-up calls "
        f"this with no arguments, so that default IS the behaviour"
    )
    assert "SWEEP_DIR" in line


def test_bring_up_still_passes_no_arguments() -> None:
    """Pinned, because it is why the default matters at all.

    If rehydrate_box.sh ever passed --out explicitly this test should be
    revisited rather than deleted: the invariant would move, not disappear.
    """
    text = (REPO / "tools" / "rehydrate_box.sh").read_text(encoding="utf-8")
    line = next((ln for ln in text.splitlines() if "$restore.py" in ln), None)
    assert line, "rehydrate_box.sh no longer runs the restore tools"
    assert "--out" not in line, (
        "bring-up now passes --out; the default is no longer load-bearing and "
        "these tests need rethinking rather than deleting"
    )
