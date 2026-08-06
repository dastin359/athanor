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
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback

SP = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
OUT = SP / "clean_rollouts"

# **The strip is imported from the repo, not the scratchpad.** It used to come
# from `SP`, which is the one store that reverts to an image snapshot when the
# container is replaced -- so a driver relaunched after a replacement would pick
# up whatever version of the baseline strip the snapshot happened to hold, with
# nothing in any log to say so. The two copies were byte-identical when this was
# noticed, so no banked result turns on it; the hazard is that the code deciding
# whether a run is contaminated lived in the volatile store.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ablate_baselines as ab            # noqa: E402 -- strip installed below
from athanor.ccarc3 import Ccarc3Config  # noqa: E402
from athanor.ccarc3.client import list_games  # noqa: E402
from athanor.ccarc3.session import run_game   # noqa: E402

# **Install the baseline strip, and refuse to run without it.**
# The line above used to read "installs the baseline strip", which was false:
# `_install_patch()` is called from ablate_baselines.main() only, so importing
# did nothing and every workspace shipped the real per-level baselines and the
# action budget in meta.json. Three re-runs and eight clean rollouts were
# contaminated before a solver quoted the numbers back out of meta.json. The
# strip itself was never broken -- it raises if one file still holds a baseline.
# It was simply never wired in, which fails *successfully* and is invisible in
# every log. assert_installed() turns that into an abort.
#
# **Called from main(), not at import.** At module scope it made importing this
# file start a proxy and demand an API key, so `verdict()` -- the function that
# decides whether a run counts -- could not be imported to test. That is the same
# shape as the strip's own queue-at-import, and it is the reason the strip went
# untested for its whole life. Logic that decides what is admissible has to be
# reachable without launching anything.
def install_strip() -> None:
    ab.install()
    ab.assert_installed()

# The five interrupted-but-winning environments first, shortest arm wall time
# first within them: bank what can finish before betting a window on what
# probably cannot.
GAMES = [
    "sb26-7fbdac44",   # 8 levels, 0.29 h in the arm
    "ft09-0d8bbf25",   # 6 levels, 0.54 h
    "ka59-38d34dbb",   # 7 levels, 1.34 h
    "wa30-ee6fef47",   # 9 levels, 2.34 h
    "lf52-271a04aa",   # 10 levels, 3.55 h
    # **Then the three losses, which is the decisive experiment.** sp80, tn36 and
    # sk48 have been run on the current doctrine -- their re-runs carried 0b -- but
    # never *cleanly*: 12, 3 and 4 solver launches, each restoring rules.json from
    # the losing arm run. So "0b works" and "a second look at your own notes works"
    # are still confounded, and this is the only configuration that separates them.
    # Their arm losses (0.7143, 0.5357, 0.4167) predate 0b by a day and serve as
    # the control.
    "sp80-589a99af",   # 6 levels, 3.18 h in the arm, lost 5/6
    "tn36-ef4dde99",   # 7 levels, 1.48 h, lost 5/7
    "sk48-d8078629",   # 8 levels, 2.37 h, lost 5/8
]
BUDGET_MULTIPLE = 5.0

# **Long enough for the longest game, not the default 2 h.** The driver inherited
# Ccarc3Config's 7200 s, and `lf52` took 3.55 h in the arm -- so it would have hit
# the cap on every attempt, been correctly marked interrupted by the wall-clock
# guard, discarded, and retried forever without ever banking. A game that cannot
# finish should fail because the box died, not because the harness cut it off.
WALL_CLOCK_S = 6 * 3600


def kill_orphan_solvers() -> int:
    """Kill solvers left running by a previous driver, before starting work.

    **Stopping this driver does not stop its solver.** `run_game` waits on a
    `claude` subprocess; kill the parent and the child is reparented to init and
    keeps going -- still acting on the game, still spending quota, and now
    uncollectable, because `collect_outcome` runs in the parent that just died.
    Its `result.json` will never be written no matter how the run ends.

    Restarting the driver to pick up a code change leaked two such trees in one
    session, on `ka59` and `wa30`, both still writing to their traces minutes
    later. One of them was duplicating a game the new driver had already
    restarted, so two solvers were exploring the same environment on two
    scorecards for no benefit.

    Identified by working directory rather than by name: a solver's cwd is its
    workspace under `clean_rollouts`, and only trees reparented to init qualify,
    so the driver can never match its own live child.
    """
    killed = 0
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cwd = (entry / "cwd").resolve()
            stat = (entry / "stat").read_text().split()
        except (OSError, PermissionError):
            continue
        if str(OUT) not in str(cwd):
            continue
        ppid, pgid = stat[3], stat[4]
        if ppid != "1" or entry.name == str(os.getpid()):
            continue
        try:
            os.killpg(int(pgid), signal.SIGTERM)
            killed += 1
            print(f"    killed orphaned solver group {pgid} "
                  f"({str(cwd).split('clean_rollouts/')[-1]})", flush=True)
        except (ProcessLookupError, PermissionError, ValueError):
            pass
    return killed


def workspace_of(attempt_dir: pathlib.Path, game_id: str) -> pathlib.Path:
    """Where the harness actually puts the run.

    `Ccarc3Config(out_dir=X)` builds the workspace at `X/<game_id>`, not at `X`.
    The first version of this driver passed `out_dir=attempt_N` and then looked
    for `attempt_N/result.json`, which never exists -- so every attempt read as
    "no result", nothing was ever banked clean, and the retry loop would have run
    all 12 passes re-running games that had in fact completed. The solver was
    working the whole time; only the driver's idea of where to look was wrong.
    """
    return attempt_dir / game_id


def salvage(game_dir: pathlib.Path, game_id: str) -> dict | None:
    """Bank a clean result from an earlier attempt before starting a new one.

    Exists because the path bug above wasted finished runs: an attempt could
    complete cleanly and be discarded unread. Re-scanning previous attempt dirs
    means a restart of this driver picks those up instead of paying for them
    again.
    """
    for attempt in sorted(game_dir.glob("attempt_*")):
        state, data = verdict(workspace_of(attempt, game_id))
        if state == "clean":
            return data
    return None


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
    if why := uncorroborated(game_dir, data):
        return f"uncorroborated ({why})", data
    if why := fails_proofread(game_dir):
        return f"proofread failed ({why})", data
    return "clean", data


def fails_proofread(game_dir: pathlib.Path) -> str:
    """Run `proofread_trace.py` over the finished run; return why it failed.

    **A gate, not a report.** The mechanical half of a proofread -- did any
    command leave the workspace, did this game's own baselines or budget arrive
    in a tool result, does ARC's card agree -- is exactly the kind of thing that
    gets skipped when a run finishes at 3am and the score looks fine. So the
    driver refuses to bank a run that fails it.

    Exit 2 is a finding and discards the run. Exit 1 means passages mention a
    baseline or a budget and need a person to judge inference from reading; that
    cannot be automated and must not block banking, so it is printed and the run
    is kept. Exit 0 is clean with nothing to read.

    A crash in the proofreader is not a verdict on the run: it is reported and
    the run proceeds, because a broken checker silently discarding good runs is
    worse than one that occasionally lets a run through to be read by hand.
    """
    script = pathlib.Path(__file__).resolve().parent / "proofread_trace.py"
    if not script.exists():
        return ""
    try:
        p = subprocess.run([sys.executable, str(script), str(game_dir)],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"    proofread did not run ({exc}); banking unread", flush=True)
        return ""
    print("    " + "\n    ".join(p.stdout.strip().splitlines()[:40]), flush=True)
    if p.returncode == 2:
        fails = [ln for ln in p.stdout.splitlines() if ln.startswith("FAIL")]
        return "; ".join(f.removeprefix("FAIL ") for f in fails) or "see output"
    if p.returncode == 1:
        print("    ^ PROOFREAD: passages above need reading before this is trusted",
              flush=True)
    return ""


def uncorroborated(game_dir: pathlib.Path, data: dict) -> str:
    """Why ARC's own card disagrees with the result, or "" when it agrees.

    **A run nobody but us can confirm is not a clean run.** `scorecard.json` is
    the server's per-level count and the only source independent of our own
    trace; RHAE is scored from it. When a proxy bug started 404ing the reads, the
    first rollout of `sb26` finished 8/8 in 124 actions with a card frozen at
    level 3 -- and nothing else showed a symptom, because actions carry a `guid`
    and are not card-scoped. The game plays perfectly and the numbers you score
    from stop moving.

    So the card is checked against the result before a run is banked, rather than
    scored from later and hoped over. `levels_completed` is the comparison, not
    the action count: our ledger and ARC's have always differed by a few actions
    for reasons already documented, but a card that has seen fewer levels than we
    claim to have cleared is a card that stopped listening.
    """
    card_file = game_dir / "scorecard.json"
    if not card_file.exists():
        return "no scorecard"
    try:
        card = json.loads(card_file.read_text())
    except json.JSONDecodeError:
        return "scorecard unreadable"
    entry = (card.get("cards") or {}).get(data.get("game_id")) or {}
    done = entry.get("levels_completed") or []
    best = max(done) if done else 0
    reached = data.get("levels_reached") or 0
    if best < reached:
        return f"card {best} vs result {reached} levels"
    return ""


def attempts_so_far(game_dir: pathlib.Path) -> int:
    """Every attempt directory, used only to number the next one uniquely."""
    return len(list(game_dir.glob("attempt_*")))


def completed_attempts(game_dir: pathlib.Path, game_id: str) -> int:
    """Attempts that actually ran to a verdict, for prioritising the queue.

    Counting directories instead punishes a game for infrastructure it did not
    cause: an attempt killed by a container replacement -- or by this driver
    being restarted to pick up a code change -- leaves a directory behind with no
    result.json. `ka59` was demoted below two far less feasible games on exactly
    that basis, its only "attempt" being six seconds old when I stopped the
    driver. A game should fall down the queue for failing, not for being
    interrupted.
    """
    if not game_dir.is_dir():
        return 0
    return sum(1 for a in game_dir.glob("attempt_*")
               if (a / game_id / "result.json").exists())


def main() -> int:
    install_strip()
    OUT.mkdir(parents=True, exist_ok=True)
    infos = {g.game_id: g for g in list_games()}
    print("CLEAN ONE-SHOT ROLLOUTS — fresh every time, interrupted attempts discarded",
          flush=True)
    kill_orphan_solvers()

    # **Keep going until every game has a clean run.** The first version made a
    # single pass and exited, which quietly contradicted the whole design: a game
    # whose attempt was cut short by a container replacement got exactly one try
    # and was then abandoned, so "retry until one completes in a single process"
    # depended on a human relaunching the driver. Under a 35-minute median box
    # lifetime the long games would never have been retried at all.
    #
    # Bounded rather than infinite: a game that cannot fit in a window will burn
    # quota forever otherwise, and MAX_PASSES makes the give-up point explicit and
    # visible in the log rather than implicit in whoever stops watching.
    for pass_no in range(1, MAX_PASSES + 1):
        # **Fewest attempts first, GAMES order as the tie-break.** A plain GAMES
        # walk starves the tail: the driver dies with its container, and on
        # relaunch it starts again at the first outstanding game, so a game that
        # can never finish in one window takes every window and nothing behind it
        # is ever tried. That is exactly how a fixed order once converted a
        # three-game experiment into a one-game one in rerun_losses.py. With all
        # counts at zero this is identical to GAMES order, so the five keep their
        # priority until one of them starts failing repeatedly.
        outstanding = sorted(
            (g for g in GAMES if not (OUT / g / "clean_result.json").exists()),
            key=lambda g: (completed_attempts(OUT / g, g), GAMES.index(g)),
        )
        if not outstanding:
            print("\nall five have a clean run", flush=True)
            break
        print(f"\n--- pass {pass_no}/{MAX_PASSES}: {len(outstanding)} outstanding "
              f"({', '.join(g.split('-')[0] for g in outstanding)}) ---", flush=True)
        one_pass(outstanding, infos)

    print("\n=== driver finished ===", flush=True)
    remaining = [g for g in GAMES if not (OUT / g / "clean_result.json").exists()]
    if remaining:
        print(f"NO clean run after {MAX_PASSES} passes: {', '.join(remaining)}",
              flush=True)
    return 0


MAX_PASSES = 12


def one_pass(games: list[str], infos: dict) -> None:
    for gid in games:
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

        rescued = salvage(OUT / gid, gid) if (OUT / gid).exists() else None
        if rescued is not None:
            banked.write_text(json.dumps(rescued, indent=1))
            print(f"\n=== {gid}: salvaged a clean result from an earlier attempt "
                  f"({rescued.get('levels_reached')}/{rescued.get('levels_total')})",
                  flush=True)
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
                wall_clock_timeout_s=WALL_CLOCK_S,
                # Never resume. A resumed attempt is exactly what disqualified
                # these five, so inheriting anything would defeat the point.
                fresh=True,
            ))
        except Exception:                      # noqa: BLE001 -- one game must not end the pass
            traceback.print_exc()
            print(f"    {gid} attempt {n} raised; treating as interrupted", flush=True)
            continue

        state, data = verdict(workspace_of(run_dir, gid))
        mins = (time.time() - started) / 60
        if state == "clean":
            (OUT / gid / "clean_result.json").write_text(json.dumps(data, indent=1))
            print(f"    CLEAN in {mins:.0f} min — "
                  f"{data['levels_reached']}/{data['levels_total']}, "
                  f"won={data['won']}, {data.get('actions_used')} actions", flush=True)
        else:
            # `.get("error", state)` returns None rather than the fallback:
            # `collect_outcome` writes the key with a null value on a clean exit,
            # so the default never applies and a discard printed "— None".
            reason = (data or {}).get("error") or state
            print(f"    discarded after {mins:.0f} min — {reason}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
