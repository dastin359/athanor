"""`tools/refresh_audit.sh` is an hourly safety net whose entire protocol is that
silence means "nothing new". Every defect it has had was a way for silence to
mean something else.

It was pinned to the literal `clean_rollouts` while the submission sweep writes
to `CCARC3_SWEEP_DIR=clean_rollouts_submission` — so it would have watched an
empty directory and reported "nothing new" through all 25 games of the one run
it exists to catch. And a missing sweep directory used to fall out of `grep -f`
into an empty result and exit 0, which is the *same signature as a healthy quiet
hour* on the one run where the state it compares against is gone.

So each test here checks both directions: that the check fires when it should,
AND that the quiet case it is being compared against is quiet for the right
reason. A test that only asserts an exit code cannot tell "no new games" from
"never looked" — and that confusion is the bug.

The script is copied into a sandbox repo because it derives its marker path from
`readlink -f "${BASH_SOURCE[0]}"`; running it in place would write the real
`evidence/ccarc3/trace_audit/.last_refresh` and leak state between tests. The
source path is resolved at import, before any fixture exists, so no test can
accidentally read a sandbox copy of the file under test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _script_source() -> Path:
    """The real `tools/refresh_audit.sh`, resolved from this file's location.

    Walks up rather than assuming `parents[1]`, so the file works both in
    `tests/` of the checkout and in a git worktree that does not carry the
    script on its own branch.
    """
    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "tools" / "refresh_audit.sh"
        if candidate.is_file():
            return candidate
    raise AssertionError(f"tools/refresh_audit.sh not found above {here}")


SCRIPT_SRC = _script_source()

EXIT_NOTHING_NEW = 0
EXIT_CANNOT_RUN = 3
EXIT_NEW_RESULTS = 10


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A repo-shaped copy of the script plus an empty scratchpad.

    Returns the sandbox root; `sandbox/repo/tools/refresh_audit.sh` is the copy
    under test and `sandbox/scratch` is what CCARC3_SCRATCH points at.
    """
    tools = tmp_path / "repo" / "tools"
    tools.mkdir(parents=True)
    dest = tools / SCRIPT_SRC.name
    shutil.copy2(SCRIPT_SRC, dest)
    # Guard against testing a stale or truncated copy of the very file whose
    # behaviour is the subject.
    assert dest.read_bytes() == SCRIPT_SRC.read_bytes()
    (tmp_path / "scratch").mkdir()
    return tmp_path


def script(sandbox: Path) -> Path:
    return sandbox / "repo" / "tools" / "refresh_audit.sh"


def marker(sandbox: Path) -> Path:
    return sandbox / "repo" / "evidence" / "ccarc3" / "trace_audit" / ".last_refresh"


def run(sandbox: Path, *args: str, sweep: str | None = None) -> subprocess.CompletedProcess:
    """Run the script with an environment built from scratch.

    Nothing is inherited but PATH: a suite that reads its inputs from the
    ambient environment passes or fails depending on the launching shell, and
    CCARC3_SWEEP_DIR is exactly the sort of variable an operator leaves set.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(sandbox),
        "LC_ALL": "C",
        "CCARC3_SCRATCH": str(sandbox / "scratch"),
    }
    if sweep is not None:
        env["CCARC3_SWEEP_DIR"] = sweep
    assert "ARC_API_KEY" not in env
    assert ("CCARC3_SWEEP_DIR" in env) == (sweep is not None)
    return subprocess.run(
        ["bash", str(script(sandbox)), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(sandbox),
        timeout=120,
    )


def clean_game(sandbox: Path, root: str, gid: str, *, score: int = 3, indent: int | None = None) -> Path:
    """A finished clean rollout in the real on-disk layout.

    The banked `clean_result.json` sits at the game directory while the run
    itself is two levels down at <game>/attempt_1/<game>/ — the mismatch that
    made three finished rollouts invisible to a flat `*/result.json` glob.
    """
    game = sandbox / "scratch" / root / gid
    workspace = game / "attempt_1" / gid
    workspace.mkdir(parents=True, exist_ok=True)
    payload = {"game_id": gid, "final_score": score, "levels": 2}
    workspace.joinpath("result.json").write_text(json.dumps(payload), encoding="utf-8")
    game.joinpath("clean_result.json").write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return game


def arm_game(sandbox: Path, gid: str, *, score: int = 1) -> Path:
    """A finished game in the flat <root>/<gid>/result.json arm layout."""
    game = sandbox / "scratch" / "ablate_nobaseline" / gid
    game.mkdir(parents=True, exist_ok=True)
    game.joinpath("result.json").write_text(
        json.dumps({"game_id": gid, "final_score": score}), encoding="utf-8"
    )
    return game


def reported_ids(stdout: str) -> set[str]:
    """The `<root>/<game_id>` keys the script printed as new."""
    out: set[str] = set()
    for line in stdout.splitlines():
        if line.startswith("rebuild with:"):
            break
        if line.startswith("  ") and "/" in line:
            out.add(line.strip())
    return out


def test_the_sweep_directory_scanned_is_the_one_ccarc3_sweep_dir_names(sandbox: Path) -> None:
    """(a) Both directions, because only the pair distinguishes the bug.

    The submission sweep's games live in `clean_rollouts_submission`. Pinned to
    the literal `clean_rollouts`, the script sees an empty default directory and
    reports "nothing new" — so the silent case alone proves nothing. The same
    fixture must go quiet without the variable and loud with it.
    """
    clean_game(sandbox, "clean_rollouts_submission", "sub01-aa")
    # The default root exists but is empty: this is the state during a
    # submission sweep, and it is why the old code's silence was plausible.
    (sandbox / "scratch" / "clean_rollouts").mkdir()

    pinned = run(sandbox)
    assert pinned.returncode == EXIT_NOTHING_NEW, pinned.stderr
    assert pinned.stdout == ""

    honoured = run(sandbox, sweep="clean_rollouts_submission")
    assert honoured.returncode == EXIT_NEW_RESULTS, (honoured.stdout, honoured.stderr)
    # Not merely "some output": the game from the submission sweep, keyed under
    # the submission root, must be the thing it reported.
    assert "clean_rollouts_submission/sub01-aa" in reported_ids(honoured.stdout)

    # And the marker was written inside the sandbox, proving REPO resolved from
    # the copy under test rather than the real repo's evidence directory.
    assert marker(sandbox).is_file()


def test_a_missing_sweep_directory_exits_3_loudly_instead_of_0_silently(sandbox: Path) -> None:
    """(b) Silence must never mean "the check could not run".

    Two shapes, because they fail differently. With nothing else on disk the old
    code produced exit 0 and no output — byte-identical to a healthy quiet hour.
    With a genuinely new arm result it would report a partial answer as if it
    were the whole one. Neither is acceptable: the directory it was told to
    watch is gone, so the only honest reply is that the check did not run.
    """
    empty = run(sandbox, sweep="clean_rollouts_submission")
    assert empty.returncode == EXIT_CANNOT_RUN, (empty.returncode, empty.stdout)
    assert empty.stdout == "", "exit 3 must not look like a report either"
    assert "does not exist" in empty.stderr
    assert "CCARC3_SWEEP_DIR" in empty.stderr

    arm_game(sandbox, "arm01-zz")

    missing = run(sandbox, sweep="clean_rollouts_submission")
    assert missing.returncode == EXIT_CANNOT_RUN, (missing.returncode, missing.stdout)
    assert "does not exist" in missing.stderr
    # It must not also emit the report; an hourly caller reads stdout, and a
    # rebuild recipe printed beside exit 3 would be acted on as a full answer.
    assert "new results since last refresh" not in missing.stdout

    # The guard was reachable and the fixture is live: create the directory and
    # the same scratchpad reports. Without this, the exit code above could have
    # come from any early-out on the way to the check being named.
    clean_game(sandbox, "clean_rollouts_submission", "sub02-bb")
    now_scans = run(sandbox, sweep="clean_rollouts_submission")
    assert now_scans.returncode == EXIT_NEW_RESULTS, now_scans.stderr
    assert reported_ids(now_scans.stdout) == {
        "ablate_nobaseline/arm01-zz",
        "clean_rollouts_submission/sub02-bb",
    }


def test_a_genuine_no_op_stays_completely_silent_and_exits_0(sandbox: Path) -> None:
    """(c) The other half of (b): loudness must not become the default.

    A fix that makes the script announce itself would break the protocol just as
    thoroughly, because the hourly caller treats any output as an event.
    """
    clean_game(sandbox, "clean_rollouts", "cln01-cc")
    arm_game(sandbox, "arm02-dd")

    # Live fixture: before recording there is something to report. This is what
    # makes the silence below meaningful rather than an empty scratchpad.
    before = run(sandbox)
    assert before.returncode == EXIT_NEW_RESULTS, before.stderr
    assert reported_ids(before.stdout) == {"clean_rollouts/cln01-cc", "ablate_nobaseline/arm02-dd"}

    recorded = run(sandbox, "--record")
    assert recorded.returncode == 0, recorded.stderr
    assert recorded.stdout.startswith("recorded 2 results"), recorded.stdout
    assert len(marker(sandbox).read_text().splitlines()) == 2

    quiet = run(sandbox)
    assert quiet.returncode == EXIT_NOTHING_NEW, (quiet.stdout, quiet.stderr)
    assert quiet.stdout == "", f"a no-op must print nothing, got: {quiet.stdout!r}"
    assert quiet.stderr == "", f"a no-op must print nothing, got: {quiet.stderr!r}"


def test_a_newly_banked_clean_result_is_reported_with_a_usable_rebuild(sandbox: Path) -> None:
    """(d) The event the net exists to catch, end to end.

    A clean rollout finishes by banking `clean_result.json` at the game
    directory. That must surface as new, must not drag the already-published
    game along with it, and must come with an ingest path pointing at the
    attempt workspace that actually holds the run.
    """
    clean_game(sandbox, "clean_rollouts", "cln01-cc")
    assert run(sandbox, "--record").returncode == 0

    clean_game(sandbox, "clean_rollouts", "cln02-ee")
    fired = run(sandbox)

    assert fired.returncode == EXIT_NEW_RESULTS, (fired.stdout, fired.stderr)
    assert reported_ids(fired.stdout) == {"clean_rollouts/cln02-ee"}
    assert "cln01-cc" not in fired.stdout, "an already-recorded game was re-flagged"
    expected_ingest = sandbox / "scratch" / "clean_rollouts" / "cln02-ee" / "attempt_1" / "cln02-ee"
    assert f"--ingest {expected_ingest} --as cln02-ee@clean" in fired.stdout


def test_reformatting_a_recorded_result_is_not_a_new_game(sandbox: Path) -> None:
    """The stamp keys on canonical content, so restoring evidence is not an event.

    Restoring banked results from evidence rewrites them with different
    indentation and fresh mtimes. Under a byte hash or an mtime key every
    already-published game re-flags as new, and the hourly net fires a full
    rebuild for an hour in which nothing ran.
    """
    clean_game(sandbox, "clean_rollouts", "cln03-ff", score=5, indent=None)
    assert run(sandbox, "--record").returncode == 0

    # Same result, restored: identical content, different bytes, new mtime.
    clean_game(sandbox, "clean_rollouts", "cln03-ff", score=5, indent=2)
    restored = run(sandbox)
    assert restored.returncode == EXIT_NOTHING_NEW, restored.stdout
    assert restored.stdout == ""

    # But a real change in what the run did is still caught — the hash must be
    # insensitive to formatting only, not to content.
    clean_game(sandbox, "clean_rollouts", "cln03-ff", score=6, indent=2)
    changed = run(sandbox)
    assert changed.returncode == EXIT_NEW_RESULTS, changed.stdout
    assert reported_ids(changed.stdout) == {"clean_rollouts/cln03-ff"}
