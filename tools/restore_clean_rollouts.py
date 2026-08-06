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
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "evidence" / "ccarc3" / "clean_rollouts"
SCRATCH = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)


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
    ap.add_argument("--out", default=str(SCRATCH / "clean_rollouts"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    if not EVIDENCE.is_dir():
        print("no preserved clean rollouts yet")
        return 0

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

            dst = out / gid / attempt.name / gid
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
