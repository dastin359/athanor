"""Re-run the three environments the arm lost, surviving container replacement.

All three failed the same way: cleared exactly five levels, hit a level they could
not read, and stopped voluntarily with 82-95% of the action budget unspent. §0b of
the doctrine -- "being stuck with budget left is a reason to change technique, not
to stop" -- is the change under test.

**Why this restores before it runs.** Containers here are replaced, not merely
restarted: boot_id changes and uptime resets. One measured box lived 1h22m, and
these three games took 1.5-3.2h in the arm. So a run that starts from nothing on
every new box can never finish -- it would re-learn the same mechanics until the
budget ran out.

`evidence/ccarc3/` on the remote holds gzipped ledgers, state and rule books,
pushed every five minutes by tools/preserve_evidence.sh. This restores whatever is
there and then resumes, so a replacement costs a replay, not the knowledge. What
carries is rules.json: the mechanics the run already paid to work out.

**Measured correction (2026-08-06).** An earlier version of this note claimed the
ARC card "is always gone by then -- the gap exceeds the ~12 minute window", so
every replay would start at level 0. Our own `sk48` run falsifies that. It crossed
the 02:48:15Z container replacement on card
``57690598-daed-4fac-8b18-e8bb34734288`` and came out the other side on the *same*
card: zero ``full_reset`` markers in the ledger and one continuous play, level
0 -> 1 -> 2. So the card survives a replacement provided the relaunch is prompt,
and the restore buys back the *game*, not only the notes. See
`session.snapshot_scorecard` for the reap bracket this tightened.

Deliberately NOT `fresh=True`. The point is accumulation across boxes.
"""
import gzip
import json
import pathlib
import shutil
import sys
import traceback

SP = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
REPO = pathlib.Path("/home/user/athanor")
EVIDENCE = REPO / "evidence" / "ccarc3" / "rerun_losses"
OUT = SP / "rerun_losses"

sys.path.insert(0, str(SP))

import ablate_baselines as ab            # noqa: E402 -- installs the baseline strip
from athanor.ccarc3 import Ccarc3Config  # noqa: E402
from athanor.ccarc3.client import list_games  # noqa: E402
from athanor.ccarc3.session import run_game   # noqa: E402

GAMES = ["sp80-589a99af", "tn36-ef4dde99", "sk48-d8078629"]
BUDGET_MULTIPLE = 5.0     # ARC's own ceiling: no game-wide pool, 5n per level


def rotated(games: list[str]) -> list[str]:
    """Start each pass at a different game, so one stalled run cannot starve the rest.

    **Measured need.** Containers are replaced every ~30 minutes, so a pass gets
    one short window and then dies. Running a fixed order meant `sp80` took every
    window: it reached level 5 of 6 and then sat there for three consecutive
    cycles at 365 actions, while `tn36` and `sk48` were never attempted once. A
    stalled game at the head of the list silently converts a three-game
    experiment into a one-game one.

    Rotation is by a counter on disk rather than a clock, so it advances once per
    pass and is unaffected by how long a pass survives.
    """
    counter = OUT / ".pass"
    try:
        n = int(counter.read_text().strip())
    except (OSError, ValueError):
        n = 0
    counter.write_text(str(n + 1))
    return games[n % len(games):] + games[:n % len(games)]



def restore(game_id: str) -> bool:
    """Put back what a replaced container took. True if anything was restored.

    Only fills gaps -- a file already on disk is newer than the preserved copy by
    construction, since the preserver runs behind the solver.
    """
    src, dst = EVIDENCE / game_id, OUT / game_id
    if not src.is_dir():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    restored = []
    for gz in sorted(src.glob("*.gz")):
        target = dst / gz.name[:-3]
        if target.exists():
            continue
        with gzip.open(gz, "rb") as fh, target.open("wb") as out:
            shutil.copyfileobj(fh, out)
        restored.append(target.name)
    if restored:
        print(f"    restored from evidence: {', '.join(restored)}", flush=True)
    return bool(restored)


OUT.mkdir(exist_ok=True)
infos = {g.game_id: g for g in list_games()}

print("RE-RUN OF THE THREE LOSSES — doctrine 0b active, baselines withheld",
      flush=True)
for gid in rotated(GAMES):
    info = infos.get(gid)
    if info is None:
        print(f"{gid}: not in list_games() any more, skipping", flush=True)
        continue

    print(f"\n=== {gid} — {info.levels} levels, cap "
          f"{info.suggested_budget(BUDGET_MULTIPLE)} actions ===", flush=True)

    # **Restore BEFORE deciding whether this game is done.** The check below reads
    # result.json out of the scratchpad, and on a freshly replaced container the
    # scratchpad is empty -- the only copy is the gzipped one in evidence/. Doing
    # the check first meant a finished game was never recognised as finished:
    # `sp80` won 6 of 6, and the very next pass restored its result.json and then
    # re-ran it anyway, spending a whole window re-playing a game that was already
    # in the bank while `tn36` and `sk48` waited.
    restore(gid)

    done = OUT / gid / "result.json"
    if done.exists():
        try:
            prior = json.loads(done.read_text())
        except json.JSONDecodeError:
            prior = None
        if prior and not prior.get("error"):
            print(f"    already finished ({prior.get('levels_reached')}/"
                  f"{prior.get('levels_total')}, won={prior.get('won')}), skipping",
                  flush=True)
            continue

    try:
        out = run_game(
            Ccarc3Config(
                gid,
                out_dir=OUT,
                budget_multiple=BUDGET_MULTIPLE,
                # Shorter than the box lives, so the harness writes a result and
                # the next launch resumes from it rather than being cut mid-action.
                wall_clock_timeout_s=1.0 * 3600,
            ),
            info,
        )
        print("   ", json.dumps(out), flush=True)
    except Exception:
        print(f"    FAILED {gid}", flush=True)
        traceback.print_exc()

print("\n=== pass complete ===", flush=True)
