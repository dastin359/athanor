"""The rebuild command must point at the workspace, under any sweep name.

`stamps()` emits keys prefixed with `$CLEAN_ROOT` (= `CCARC3_SWEEP_DIR`), but
the `case` that turns a key into a `--ingest` path matched the literal
`clean_rollouts`. Under `CCARC3_SWEEP_DIR=clean_rollouts_submission` the key
fell through to the flat `*)` arm and produced `--ingest $SP/<sweep>/<gid>` --
the GAME directory, which holds only `clean_result.json`. `build_trace_audit
--ingest` needs the attempt workspace, with its `result.json` and `trace.jsonl`.

The script's own header records fixing exactly this hazard in `stamps()`; the
same literal survived 55 lines further down. A rebuild command that ingests
nothing is that silence one step on -- the audit announces new results and the
artifact never changes.

Run end to end against a fixture scratchpad: the real script, the real output.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
AUDIT = REPO / "tools" / "refresh_audit.sh"

SWEEPS = ["clean_rollouts", "clean_rollouts_submission", "clean_rollouts_v2"]


def _sweep_fixture(scratch: pathlib.Path, sweep: str, gid: str) -> pathlib.Path:
    """A finished nested sweep result, laid out as the driver writes it."""
    game = scratch / sweep / gid
    ws = game / "attempt_1" / gid
    ws.mkdir(parents=True)
    payload = {"game_id": gid, "won": True, "levels_reached": 3}
    (ws / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (ws / "trace.jsonl").write_text(json.dumps({"level": 1}) + "\n", encoding="utf-8")
    (game / "clean_result.json").write_text(json.dumps(payload), encoding="utf-8")
    # ablate_nobaseline is in ROOTS and must exist or the scan skips it silently
    (scratch / "ablate_nobaseline").mkdir(parents=True, exist_ok=True)
    return ws


def _run(scratch: pathlib.Path, sweep: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CCARC3_SCRATCH": str(scratch), "CCARC3_SWEEP_DIR": sweep}
    return subprocess.run(["/bin/bash", str(AUDIT)],
                          capture_output=True, text=True, env=env, timeout=180)


@pytest.mark.parametrize("sweep", SWEEPS)
def test_the_rebuild_ingests_the_attempt_workspace(sweep, tmp_path) -> None:
    gid = "ar25-fixture"
    ws = _sweep_fixture(tmp_path, sweep, gid)
    out = _run(tmp_path, sweep)

    assert "new results since last refresh" in out.stdout, (
        f"the audit did not see the finished game at all:\n{out.stdout}\n{out.stderr}"
    )
    assert f"--ingest {ws}" in out.stdout, (
        f"CCARC3_SWEEP_DIR={sweep!r}: the rebuild command does not point at the "
        f"attempt workspace {ws}.\nGot:\n{out.stdout}"
    )
    assert f"--ingest {tmp_path / sweep / gid} " not in out.stdout, (
        "the rebuild points at the game directory, which holds only "
        "clean_result.json -- build_trace_audit has nothing to ingest there"
    )


@pytest.mark.parametrize("sweep", SWEEPS)
def test_the_clean_suffix_is_applied(sweep, tmp_path) -> None:
    """The `@clean` tag is what keeps a sweep run distinct from its arm run.

    It is emitted by the same branch, so a fallthrough loses it silently and two
    different runs of one game collapse onto a single tile in the page.
    """
    gid = "ar25-fixture"
    _sweep_fixture(tmp_path, sweep, gid)
    out = _run(tmp_path, sweep)
    assert f"--as {gid}@clean" in out.stdout, (
        f"CCARC3_SWEEP_DIR={sweep!r}: the run was not tagged @clean, so it would "
        f"overwrite the arm run of the same game.\nGot:\n{out.stdout}"
    )


def test_an_errored_result_is_not_offered_for_rebuild(tmp_path) -> None:
    """The other direction: a crashed run is retried by the driver, not scored."""
    sweep, gid = "clean_rollouts", "ar25-broken"
    ws = _sweep_fixture(tmp_path, sweep, gid)
    (ws / "result.json").write_text(
        json.dumps({"game_id": gid, "error": "boom"}), encoding="utf-8")
    out = _run(tmp_path, sweep)
    assert f"--ingest {ws}" not in out.stdout, (
        f"a result carrying an error was offered for rebuild:\n{out.stdout}"
    )
