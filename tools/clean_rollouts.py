"""Clean one-shot rollouts of the five environments that were never cleanly run.

`ft09`, `ka59`, `lf52`, `sb26` and `wa30` all scored 1.0000 in the arm, but every
one of them was interrupted by a container event and resumed. Under the operator's
criterion -- only a run that completes in a single solver process is a genuine
clean draw -- none of the five counts, because a resumed solver re-reads its own
`rules.json` and trace on the far side of a fresh context window.

This re-runs them to settle that. It is the opposite of `tools/rerun_losses.py` in
every way that matters:

===================  ==========================  ===========================
                     rerun_losses.py             this
===================  ==========================  ===========================
restore evidence     yes -- accumulate           **never**
resume               yes -- the point            **never**, ``fresh=True``
interrupted attempt  resume it                   **discard it**
what it measures     can the harness win at all  can it win in one pass
===================  ==========================  ===========================

**An interrupted attempt is discarded, not resumed.** Resuming is what disqualified
these five in the first place, so a driver that resumed them would reproduce the
very confound it exists to remove. Each attempt opens its own ARC scorecard, so a
discarded attempt costs quota and wall clock but never score -- the environment
keeps the best card, and an abandoned one is simply worse.

**The infrastructure is working against this and the odds are not uniform.**
Containers are replaced every 22-36 minutes at present, and a clean run has to fit
inside one window. Against the arm's own wall times -- `sb26` 0.29 h, `ft09`
0.54 h, `ka59` 1.34 h, `wa30` 2.34 h, `lf52` 3.55 h -- only the first two are
comfortably feasible. The re-runs did come in faster than their arm runs (`sk48`
0.95 h against 2.37 h), so those are pessimistic, but not by the factor `lf52`
would need. Shortest-first ordering below is deliberate: it banks the achievable
results before spending the window on a game that probably cannot finish.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback

SP = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
OUT = SP / "clean_rollouts"
sys.path.insert(0, str(SP))

import ablate_baselines as ab            # noqa: E402,F401 -- installs the baseline strip
from athanor.ccarc3 import Ccarc3Config  # noqa: E402
from athanor.ccarc3.client import list_games  # noqa: E402
from athanor.ccarc3.session import run_game   # noqa: E402

# Shortest arm wall time first: bank what can finish before betting a window on
# what probably cannot.
GAMES = [
    "sb26-7fbdac44",   # 8 levels, 0.29 h in the arm
    "ft09-0d8bbf25",   # 6 levels, 0.54 h
    "ka59-38d34dbb",   # 7 levels, 1.34 h
    "wa30-ee6fef47",   # 9 levels, 2.34 h
    "lf52-271a04aa",   # 10 levels, 3.55 h
]
BUDGET_MULTIPLE = 5.0


def verdict(game_dir: pathlib.Path) -> tuple[str, dict | None]:
    """Classify a finished attempt: clean, interrupted, or a real loss.

    The distinction is the whole experiment. `collect_outcome` already marks an
    interruption with an `error` field -- signal death, crash, or a wall clock
    that fired with budget unspent -- so anything carrying one is a discard rather
    than a datum. Everything else completed under its own steam and counts,
    including a genuine loss.
    """
    result = game_dir / "result.json"
    if not result.exists():
        return "no result", None
    try:
        data = json.loads(result.read_text())
    except json.JSONDecodeError:
        return "unreadable", None
    if data.get("error"):
        return "interrupted", data
    return "clean", data


def attempts_so_far(game_dir: pathlib.Path) -> int:
    return len(list(game_dir.glob("attempt_*")))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    infos = {g.game_id: g for g in list_games()}
    print("CLEAN ONE-SHOT ROLLOUTS — fresh every time, interrupted attempts discarded",
          flush=True)

    for gid in GAMES:
        banked = OUT / gid / "clean_result.json"
        if banked.exists():
            prior = json.loads(banked.read_text())
            print(f"\n=== {gid}: already has a clean run "
                  f"({prior.get('levels_reached')}/{prior.get('levels_total')}), skipping",
                  flush=True)
            continue

        info = infos.get(gid)
        if info is None:
            print(f"\n=== {gid}: not in list_games() any more, skipping", flush=True)
            continue

        n = attempts_so_far(OUT / gid) + 1 if (OUT / gid).exists() else 1
        run_dir = OUT / gid / f"attempt_{n}"
        print(f"\n=== {gid} — {info.levels} levels, cap "
              f"{info.suggested_budget(BUDGET_MULTIPLE)} actions — attempt {n} ===",
              flush=True)

        started = time.time()
        try:
            run_game(Ccarc3Config(
                gid,
                out_dir=run_dir,
                budget_multiple=BUDGET_MULTIPLE,
                # Never resume. A resumed attempt is exactly what disqualified
                # these five, so inheriting anything would defeat the point.
                fresh=True,
            ))
        except Exception:                      # noqa: BLE001 -- one game must not end the pass
            traceback.print_exc()
            print(f"    {gid} attempt {n} raised; treating as interrupted", flush=True)
            continue

        state, data = verdict(run_dir)
        mins = (time.time() - started) / 60
        if state == "clean":
            (OUT / gid / "clean_result.json").write_text(json.dumps(data, indent=1))
            print(f"    CLEAN in {mins:.0f} min — "
                  f"{data['levels_reached']}/{data['levels_total']}, "
                  f"won={data['won']}, {data.get('actions_used')} actions", flush=True)
        else:
            reason = (data or {}).get("error", state)
            print(f"    discarded after {mins:.0f} min — {reason}", flush=True)

    print("\n=== pass complete ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
