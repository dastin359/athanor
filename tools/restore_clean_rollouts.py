"""Put banked clean rollouts back after a container replacement.

`tools/clean_rollouts.py` decides what still needs running by looking for
`clean_result.json` at each game directory, and falls back to scanning
`attempt_*/<game>/result.json`. Both live in the scratchpad, which reverts to an
image snapshot when the container is replaced — so a replacement makes four
finished, scored games look unstarted, and the driver re-runs them.

The evidence preserver pushes each rollout workspace to
`evidence/ccarc3/clean_rollouts/<game>/attempt_N/<game>/` every five minutes, so
the results survive in the repo. This restores the workspace files and rewrites
the `clean_result.json` marker from the preserved `result.json`.

**Only attempts that finished cleanly are restored.** An attempt carrying an
`error` was interrupted — by a signal, a crash, or the wall clock with budget
unspent — and the whole point of the rollout experiment is that such an attempt
is discarded and re-run, never resumed. Restoring one as if it were banked would
quietly admit a non-clean run into a set defined by cleanliness.

**And only attempts whose workspace has no baselines in it.** The first seven
rollouts ran with the strip uninstalled, so their preserved `meta.json` still
carries `baseline_actions` and `action_budget`. Restoring one of those re-banks a
void run as finished, and the driver then skips the game it most needs to re-run
— which is how a contaminated result survives being noticed. The check is on the
preserved files rather than a list of ids, so it keeps working for a leak nobody
has thought of yet.

Run after any container replacement, before relaunching the driver. Idempotent:
a game already banked on disk is left alone.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# **Both ends follow CCARC3_SWEEP_DIR, because the driver's do.** This restore
# read `evidence/ccarc3/clean_rollouts` and wrote `$SCRATCH/clean_rollouts`, both
# hard-coded, while `clean_rollouts.py` computes
# `OUT = SP / (CCARC3_SWEEP_DIR or "clean_rollouts")` -- and `rehydrate_box.sh`
# invokes this with no arguments at all.
#
# So for the submission sweep (`CCARC3_SWEEP_DIR=clean_rollouts_submission`) a
# container replacement mid-run would restore banked games into the WRONG
# directory, the driver would read its own and find nothing, and it would re-run
# games that are already on the shared scorecard. That costs money twice and
# corrupts the artifact: a scorecard cannot un-play a game, so the second play
# lands beside the first and the per-game row count stops matching the result.
#
# `preserve_evidence.sh` has always handled this correctly -- it globs
# `clean_rollouts*` and mirrors `$SP/<dir>` to `$DEST/<dir>`, which is why
# `evidence/ccarc3/` already holds `clean_rollouts_stale_card` and
# `clean_rollouts_void`. Only the restore half was pinned to one name.
SWEEP_DIR = os.environ.get("CCARC3_SWEEP_DIR") or "clean_rollouts"
EVIDENCE = REPO / "evidence" / "ccarc3" / SWEEP_DIR
# Overridable via `CCARC3_SCRATCH`: the path encodes a session UUID and the
# container is recycled every 10-50 minutes, so a hard-coded copy points at a
# directory that stops existing. See tools/clean_rollouts.py for the full note.
SCRATCH = pathlib.Path(
    os.environ.get("CCARC3_SCRATCH")
    or "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)


def _scoring():
    """`athanor.ccarc3.scoring`, imported on use.

    This tool runs against a checked-out tree with no install step, so the
    package is reached the same way `proofread_trace.py` reaches it: by path, at
    the call site, so a missing `src/` breaks the one function that needs it
    rather than the whole restore.
    """
    import sys  # noqa: PLC0415
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from athanor.ccarc3 import scoring  # noqa: PLC0415
    return scoring


def live_workspaces() -> set[str]:
    """Every workspace a running process is currently sitting in.

    **This restore wrote a banked result on top of a run that was in flight.**
    Attempt directories are numbered per game, so a re-run of a game whose
    previous attempt was preserved reuses `attempt_1` — and the restore, seeing
    no `clean_result.json`, faithfully copied the old run's `result.json`,
    `trace.jsonl` and `scorecard.json` over the live solver's own. The result
    was a workspace holding one run's outcome and another run's trace, and a
    `clean_result.json` that told the driver to skip the game it was mid-way
    through playing.

    Idempotence is not enough when the destination is being written by someone
    else. Nothing here touches a directory that a live process has open.
    """
    live = set()
    for proc in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            cwd = (proc / "cwd").resolve()
        except (OSError, RuntimeError):
            continue
        live.add(str(cwd))
    return live


def card_corroborates(src: pathlib.Path, result: dict) -> str:
    """Empty when ARC's own card agrees with the result, else why it does not.

    The scorecard is the authoritative per-level count and the only independent
    check on a run. A card that stopped updating mid-game still leaves a
    plausible-looking `result.json` — the first clean rollout finished 8/8 while
    its card froze at level 3, because a proxy bug was 404ing the reads and
    actions are not card-scoped. Banking that means banking a score no source
    but ourselves can confirm.

    Two sources that should agree, compared. It is the only failure mode that
    has actually been caught here. The comparison is `scoring.card_disagreement`
    -- shared with `clean_rollouts.uncorroborated` and `proofread_trace`, because
    when each kept its own copy all three read `max(levels_completed)` and all
    three broke together the day the shared scorecard landed.
    """
    card_gz = src / "scorecard.json.gz"
    if not card_gz.exists():
        return "no scorecard preserved"
    try:
        card = json.loads(gzip.open(card_gz).read())
    except (OSError, ValueError):
        return "scorecard unreadable"
    return _scoring().card_disagreement(card, result.get("game_id"), result)


def baselines_in(src: pathlib.Path) -> str:
    """Name the first preserved file that still hands over a baseline, if any.

    Matches the *values*, not the identifier: `baseline_actions=()` is the
    blanked form a stripped workspace legitimately contains, so the pattern
    requires a container that opens onto a digit. `action_budget` goes too — it
    is the baseline total times five, which is not a strip.
    """
    leak = re.compile(
        r"""baseline_actions["'\s:=]*[(\[]\s*\d|"action_budget"|baseline actions per level"""
    )
    for gz in sorted(src.glob("*.gz")):
        if gz.name in {"trace.jsonl.gz", "stream.jsonl.gz"}:
            continue
        try:
            text = gzip.open(gz).read().decode("utf-8", "ignore")
        except OSError:
            continue
        if leak.search(text):
            return gz.name[:-3]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(SCRATCH / SWEEP_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    if not EVIDENCE.is_dir():
        print("no preserved clean rollouts yet")
        return 0

    live = live_workspaces()
    restored, present, skipped = [], 0, []
    for game_dir in sorted(EVIDENCE.iterdir()):
        if not game_dir.is_dir():
            continue
        gid = game_dir.name
        if (out / gid / "clean_result.json").exists():
            present += 1
            continue

        # Newest attempt first: a later attempt supersedes an earlier one.
        for attempt in sorted(game_dir.glob("attempt_*"), reverse=True):
            src = attempt / gid
            result_gz = src / "result.json.gz"
            if not result_gz.exists():
                continue
            try:
                result = json.loads(gzip.open(result_gz).read())
            except (OSError, ValueError):
                continue
            if result.get("error"):
                skipped.append(f"{gid}/{attempt.name} (interrupted)")
                continue
            if leaked := baselines_in(src):
                skipped.append(f"{gid}/{attempt.name} (contaminated: {leaked})")
                continue
            if why := card_corroborates(src, result):
                skipped.append(f"{gid}/{attempt.name} (uncorroborated: {why})")
                continue

            dst = out / gid / attempt.name / gid
            if str(dst.resolve()) in live:
                skipped.append(f"{gid}/{attempt.name} (a solver is running there)")
                break
            if not args.dry_run:
                dst.mkdir(parents=True, exist_ok=True)
                for gz in src.glob("*.gz"):
                    (dst / gz.name[:-3]).write_bytes(gzip.open(gz).read())
                (out / gid / "clean_result.json").write_text(json.dumps(result, indent=1))
            restored.append(f"{gid} ({result.get('levels_reached')}/"
                            f"{result.get('levels_total')}, {result.get('actions_used')} actions)")
            break

    verb = "would restore" if args.dry_run else "restored"
    print(f"{verb} {len(restored)}; {present} already on disk; {len(skipped)} skipped")
    for line in restored:
        print(f"  {line}")
    for line in skipped:
        print(f"  skipped {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
