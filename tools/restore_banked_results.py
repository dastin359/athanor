"""Put banked `result.json` files back where the arm runner looks for them.

`scratchpad/ablate_baselines.py` decides what to run by testing
`ablate_nobaseline/<game>/result.json` on disk, and it does that *without*
restoring from `evidence/` first -- the same ordering bug that was fixed in
`tools/rerun_losses.py`, still present here.

That check is only sound while the scratchpad is intact. On 2026-08-06, 13 of the
25 banked arm results had been lost with their directories, so the runner would
have counted 13 finished, scored games as unstarted and re-run all of them. One
was `sk48-d8078629`, which a separate driver was re-running at that moment: two
solvers would have been issuing actions against the same ARC card, on a ledger
that does not reset.

The results survive in the trace-audit store, which carries a full `result` block
per game. This writes the missing ones back, so relaunching the supervisor is
genuinely idempotent again -- which is what the autopilot instructions already
assume it is.

Run after any container replacement, before relaunching the arm supervisor.
Re-runs are skipped: they live under `rerun_losses/` and are keyed `<game>@rerun`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
STORE = REPO / "evidence" / "ccarc3" / "trace_audit" / "spans.json.gz"
SCRATCH = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=str(SCRATCH / "ablate_nobaseline"),
                    help="directory the arm runner checks for result.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = pathlib.Path(args.runs)
    store = json.load(gzip.open(STORE))

    restored, present, skipped = [], 0, 0
    for gid, entry in store.items():
        if "@" in gid:            # a re-run, stored elsewhere
            skipped += 1
            continue
        target = runs / gid / "result.json"
        if target.exists():
            present += 1
            continue
        result = entry.get("result") or {}
        if not result.get("game_id"):
            print(f"  {gid}: store has no result block, cannot restore")
            continue
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=1))
        restored.append(gid)

    verb = "would restore" if args.dry_run else "restored"
    print(f"{verb} {len(restored)}; {present} already present; "
          f"{skipped} re-runs skipped")
    for gid in sorted(restored):
        print(f"  {gid}")

    missing = [g for g in store if "@" not in g
               and not (runs / g / "result.json").exists()]
    if missing and not args.dry_run:
        print(f"WARNING: {len(missing)} still missing: {', '.join(sorted(missing))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
