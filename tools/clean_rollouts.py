"""Clean one-shot rollouts: every environment, run start to finish in one process.

**Started as five games and is now all 25.** It was built for `ft09`, `ka59`,
`lf52`, `sb26` and `wa30` -- each scored 1.0000 in the arm, and each was
interrupted by a container event and resumed, so under the operator's criterion
(only a run completing in a single solver process is a genuine clean draw) none
of them counted. Then the three arm losses, to separate "§0b works" from "a
second look at your own notes works". Then, on 2026-08-07, the remaining
seventeen, once the solver-surface proofread had landed.

Eight are banked at 7.4075/8. The rest are the queue.

It is the opposite of `tools/rerun_losses.py` in every way that matters:

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

**The infrastructure was expected to work against this, and it did not.** The
paragraph here used to say containers were being replaced every 22-36 minutes, so
a clean run had to fit inside one window and only the two shortest games were
comfortably feasible. That prediction was wrong in the event: all eight banked
runs completed one-shot, including `lf52` at 10 levels and `wa30` at 9, across a
10.78-hour continuous stretch on one `boot_id`. The container lifetime that
shaped this design was a phase, not a constant.

Shortest-first ordering is kept anyway, and still on its original reasoning: it
banks achievable results before spending a window on a game that probably cannot
finish, which costs nothing when windows are long and saves the queue when they
are short. For the seventeen with no comparable arm run, published baseline total
stands in for wall time -- the only length proxy available before the fact.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import traceback

SP = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
# **The submission sweep needs its own directory, and that is not cosmetic.**
# `_run_one` skips any game that already has a `clean_result.json`, so pointing a
# shared-card sweep at a directory holding 17 banked results would skip 17 games
# and put nothing of them on the card -- a sweep that runs to completion and
# produces an artifact missing two thirds of the benchmark, with no error
# anywhere. Give a sweep whose card matters a directory of its own:
#
#     CCARC3_SWEEP_DIR=clean_rollouts_submission tools/clean_rollouts.py
#
# The name must still contain `clean_rollouts`: `build_trace_audit` decides
# whether a run is one of ours by looking for it in the working directory, so a
# rename would silently drop the whole sweep out of the audit.
_SWEEP_DIR = os.environ.get("CCARC3_SWEEP_DIR") or "clean_rollouts"
if "clean_rollouts" not in _SWEEP_DIR:
    raise SystemExit(
        f"CCARC3_SWEEP_DIR={_SWEEP_DIR!r} does not contain 'clean_rollouts', so "
        f"build_trace_audit would not recognise the runs as ours. Rename it."
    )
OUT = SP / _SWEEP_DIR

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
from athanor.ccarc3 import shared_card as sc  # noqa: E402
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

    # **The remaining 17, opened 2026-08-07 once the solver-surface repair
    # landed.** Held closed until then deliberately: the audit's `surface_digest`
    # keys on `CLAUDE.md`, `DOCTRINE.md`, `session.py` and `meta.json`, and two
    # proofread passes were rewriting all four. A run started first would have been
    # marked superseded the moment the fix shipped -- real quota, no result. The
    # repair is `a63232a` and `f66cc91`; these are the first runs under it.
    #
    # Ordered by published baseline total, shortest first, for the same reason as
    # the block above: bank what can finish before betting a window on what
    # probably cannot. Baseline total is the only length proxy available for the
    # eleven of these that have no comparable arm run.
    "cd82-fb555c5d",   # 6 levels, baseline total 171
    "r11l-495a7899",   # 6 levels, 233
    "sc25-635fd71a",   # 6 levels, 350
    "su15-1944f8ab",   # 9 levels, 361
    "lp85-305b61c3",   # 8 levels, 388
    "tr87-cd924810",   # 6 levels, 414
    "vc33-5430563c",   # 7 levels, 447
    "tu93-0768757b",   # 9 levels, 462
    "s5i5-18d95033",   # 8 levels, 638
    "bp35-0a0ad940",   # 9 levels, 651
    "ar25-0c556536",   # 8 levels, 748
    "ls20-9607627b",   # 7 levels, 776
    "cn04-2fe56bfb",   # 6 levels, 789
    "g50t-5849a774",   # 7 levels, 879
    "m0r0-492f87ba",   # 6 levels, 1107
    "dc22-fdcac232",   # 6 levels, 1228
    "re86-8af5384d",   # 8 levels, 1255
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

    **But a crash must not be mistaken for exit 1.** It was, until 2026-08-07:
    an uncaught exception exits 1 in Python, so any failure after the reach pass
    banked the run *and printed* "PROOFREAD: passages above need reading", a
    sentence asserting that a proofread had happened. `proofread_trace.py` now
    exits 3 on its own failure, and anything outside {0, 1, 2} is reported here
    as "did not run" rather than as a verdict.
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
    elif p.returncode not in (0, 1, 2):
        # Loud and distinguishable: the run is banked, but nothing checked it.
        print(f"    ^ PROOFREAD DID NOT RUN (exit {p.returncode}) — banking "
              f"UNCHECKED, re-run tools/proofread_trace.py on this directory",
              flush=True)
        print("    " + "\n    ".join(p.stderr.strip().splitlines()[-8:]), flush=True)
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


SHARED_CARD_FILE = SP / "shared_card.json"
SHARED_CARD_HISTORY = SP / "shared_card_history.jsonl"


def sweep_card():
    """The one scorecard this sweep plays onto, across driver restarts.

    **Why a file and not a fresh card per driver.** The supervisor restarts this
    driver -- on a container replacement, on an abort, on a quota stop -- and a
    card minted per process would put the 25 games on as many cards as the sweep
    had restarts. That is the failure being fixed, arriving by a different door.

    **Why the liveness probe.** A card is state on one backend instance reached
    by its stickiness cookies (see :mod:`athanor.ccarc3.shared_card`), so it can
    become unreachable while the API key keeps working. Reading any game off the
    card tells the two apart cleanly: a live card answers 200 with an empty body
    for a game it has never seen, and only an unreachable one 404s. Measured
    2026-08-08 on a card opened for the purpose.

    **Why succession is recorded rather than silent.** If the card is gone the
    sweep cannot recover the games already on it, and continuing onto a new card
    means the submission covers only what came after. Minting quietly would make
    a partial artifact look whole -- so the old card is kept, the history file
    grows a line, and the log says it loudly.
    """
    if os.environ.get("CCARC3_NO_SHARED_CARD"):
        print("shared card DISABLED by env; one card per game", flush=True)
        return None

    if SHARED_CARD_FILE.exists():
        card = sc.load(SHARED_CARD_FILE)
        try:
            sc.read_card(card, GAMES[0])
            print(f"shared card {card.card_id} — still reachable", flush=True)
            return card
        except Exception as exc:                       # noqa: BLE001
            print(f"*** shared card {card.card_id} is UNREACHABLE ({str(exc)[-70:]}). "
                  f"Games already on it stay on it and are NOT in the new card. ***",
                  flush=True)
            dead = SHARED_CARD_FILE.with_suffix(f".{card.card_id[:8]}.dead.json")
            SHARED_CARD_FILE.replace(dead)
            with SHARED_CARD_HISTORY.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"card_id": card.card_id, "retired": time.time(),
                                     "reason": str(exc)[-160:]}) + "\n")

    card = sc.open_card(tags=("ccarc3", "clean-rollouts"))
    sc.save(card, SHARED_CARD_FILE)
    with SHARED_CARD_HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"card_id": card.card_id, "opened": time.time()}) + "\n")
    print(f"shared card {card.card_id} — opened, all games will score onto it",
          flush=True)
    return card


def main() -> int:
    install_strip()
    OUT.mkdir(parents=True, exist_ok=True)
    ab.use_shared_card(sweep_card())
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
    _aborted.clear()
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
            # Was "all five have a clean run", hardcoded when GAMES held five.
            # It kept printing after the queue grew to eight and then to 25, so
            # the one line that says "there is nothing left to do" was reporting
            # a count that had been wrong for two days -- and it read as a
            # completed queue while seventeen environments sat unrun.
            print(f"\nall {len(GAMES)} have a clean run", flush=True)
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


# **Two at a time, not eight, and tunable without a restart.**
#
# Enough to halve a 15-hour queue, shallow enough that the two failure modes
# that have actually hurt this project stay bounded: the quota guard samples
# every 600 s and stops at 0.98 on a weekly window whose exhaustion costs days,
# so N games multiply the overshoot between samples by N; and every container
# event loses whatever is in flight, which is N runs rather than one.
#
# Read from a file on every check rather than frozen at import, for the same
# reason the supervisor re-reads quota every loop: the queue runs for fifteen
# hours and the right answer changes inside that window -- quota recovers, the
# box turns out to be stable, a long game needs the machine to itself. Changing
# it must not cost a driver restart, because a restart discards whatever is
# mid-flight.
#
#     echo 3 > "$SP/concurrency"     # takes effect within 10 s
#
# Raising it lets waiting games start immediately. Lowering it never kills a
# running game -- it just stops new ones starting until the count drops, which
# is the same politeness the supervisor shows at the quota ceiling.
CONCURRENCY_FILE = SP / "concurrency"
CONCURRENCY_MAX = 8


def concurrency() -> int:
    """The live limit: the control file, else the environment, else 2."""
    raw = ""
    try:
        raw = CONCURRENCY_FILE.read_text().strip()
    except OSError:
        raw = os.environ.get("CCARC3_CONCURRENCY", "")
    try:
        n = int(raw)
    except ValueError:
        return 2
    # A typo in a one-line file should not launch fifty solvers.
    return max(1, min(CONCURRENCY_MAX, n))


_slots = threading.Condition()
_running = 0
_last_limit = 0
_waiting: set[int] = set()

# **The abort has to be a flag, because an exception cannot reach a sibling.**
# `_run_one` raises SystemExit when the ARC endpoint is unreachable, on the
# reasoning that every remaining game fails identically so the queue should stop
# rather than burn through. That reasoning is right and the mechanism was not:
# the raise happens inside a worker thread, `one_pass` only learns of it through
# `as_completed`, and on 2026-08-07 seventeen futures failed in milliseconds --
# so thirteen games had already started and failed before the main thread
# processed the first. The abort that exists to stop the queue arrived last.
#
# A flag is checked by whoever is about to act, which is the only thing that
# stops work that has not begun.
_aborted = threading.Event()


def _take_slot(gid: str, rank: int) -> None:
    """Block until the live limit leaves room, and until it is this game's turn.

    **`rank` is what makes the queue order real, and without it the ordering was
    decorative.** Every game gets a thread up front, so all of them arrive here
    at once and park on one condition. `_free_slot` calls `notify_all`, and the
    waiter that happens to be scheduled first takes the slot -- Python guarantees
    no ordering across `Condition.wait`, and the 10-second timed wait below means
    a thread can also wake with no notification at all.

    Caught in production on 2026-08-07. The queue was `cd82, r11l, sc25, su15,
    lp85, tr87, ...` ordered shortest-first; `cd82` and `r11l` took the two
    opening slots, and when `r11l` finished the freed slot went to **`tr87`** --
    past three games ahead of it. `sc25`, `su15` and `lp85` had no workspace at
    all.

    That is precisely the failure the ordering exists to prevent. Shortest-first
    is there so a window that ends early has banked the achievable games instead
    of being spent on a long one; a racy queue can spend the window on the long
    game while the short ones sit unstarted, which is how a fixed order once
    turned a three-game experiment into a one-game one in `rerun_losses.py`.

    So a waiter may claim a free slot only when it is the lowest-ranked waiter.
    Strict FIFO by queue position, and starvation-free: the minimum always
    proceeds, so every rank becomes the minimum eventually.
    """
    global _running, _last_limit
    with _slots:
        # Idempotent: `_register_queue` normally seeded this before any thread
        # started. Adding here too means a thread that somehow arrives outside a
        # registered pass still queues rather than deadlocking on a `min()` of a
        # set it is not in.
        _waiting.add(rank)
        while True:
            if _aborted.is_set():
                _waiting.discard(rank)
                raise _Aborted()
            limit = concurrency()
            if limit != _last_limit:
                print(f"    concurrency = {limit}"
                      f"{' (was ' + str(_last_limit) + ')' if _last_limit else ''}",
                      flush=True)
                _last_limit = limit
            if _running < limit and rank == min(_waiting):
                _waiting.discard(rank)
                _running += 1
                # **Wake the others, because admitting one changes who the
                # minimum is.** Rank ordering introduced a stall the unordered
                # gate could not have: `_free_slot` notifies once, every waiter
                # wakes, and all but the lowest find `rank != min(_waiting)` and
                # go back to sleep. The lowest is then admitted -- and nobody
                # tells the new lowest that it is now eligible, so if the limit
                # has room for it, it waits out the full 10-second timeout for
                # no reason. Caught by the live-retune test: raising the limit
                # from 1 to 3 mid-flight left the third game asleep and peak
                # concurrency at 1.
                _slots.notify_all()
                return
            # Timed wait, so a raised limit is picked up without anyone
            # signalling -- the file is edited by a human, not by this process.
            _slots.wait(timeout=10)


def _free_slot() -> None:
    global _running
    with _slots:
        _running -= 1
        _slots.notify_all()


def _register_queue(n: int) -> None:
    """Declare every rank a waiter *before* any thread starts.

    **Rank ordering is a priority, not an arrival order, and without this it
    degrades to the latter.** `_take_slot` grants to the lowest-ranked waiter,
    which orders games correctly once they are all queued -- that is the case
    that misbehaved in production, when a slot freed and four games were already
    waiting. It does nothing for the opening burst: the pool starts N threads
    that arrive over some microseconds, and whichever arrives first is briefly
    the only waiter and therefore the minimum, so it takes a slot however long
    its game is.

    A test that starts the threads backwards makes that visible immediately --
    the last game in the queue took the first slot. In production the window is
    small, but "small" is not a property to rely on when the whole point of the
    ordering is to protect against a window that ends early.

    Seeding the full set up front makes the first grant as ordered as every
    later one. A rank whose thread has not started yet still holds its place;
    the pool has a worker per game, so it will arrive.
    """
    with _slots:
        _waiting.clear()
        _waiting.update(range(n))
        _slots.notify_all()


def one_pass(games: list[str], infos: dict) -> None:
    # Every game gets a thread; `_take_slot` is what bounds how many are actually
    # playing. A fixed-size pool would freeze the limit for the whole pass, which
    # is the thing being avoided.
    _register_queue(len(games))
    with concurrent.futures.ThreadPoolExecutor(max(1, len(games))) as pool:
        # `rank` is the game's position in the shortest-first queue, and
        # `_take_slot` honours it. Submission order alone does not, and neither
        # does thread start order -- hence the registration above.
        futures = {pool.submit(_guarded, g, infos, i): g
                   for i, g in enumerate(games)}
        for fut in concurrent.futures.as_completed(futures):
            # `_run_one` swallows its own failures; anything arriving here is a
            # driver bug, and losing it silently is how a queue quietly stops
            # making progress.
            exc = fut.exception()
            if isinstance(exc, SystemExit):
                raise exc
            if exc is not None:
                print(f"    {futures[fut]}: driver error {exc!r}", flush=True)


class _Aborted(Exception):
    """This game never started because the pass was aborted. Not a failure."""


def _guarded(gid: str, infos: dict, rank: int) -> None:
    try:
        _take_slot(gid, rank)
    except _Aborted:
        # Never started, so nothing to free and nothing to report. It stays
        # outstanding and the next pass picks it up.
        return
    try:
        if _aborted.is_set():
            return
        _run_one(gid, infos)
    finally:
        _free_slot()


def _run_one(gid: str, infos: dict) -> None:
    banked = OUT / gid / "clean_result.json"
    if banked.exists():
        prior = json.loads(banked.read_text())
        print(f"\n=== {gid}: already has a clean run "
              f"({prior.get('levels_reached')}/{prior.get('levels_total')}), skipping",
              flush=True)
        return

    info = infos.get(gid)
    if info is None:
        print(f"\n=== {gid}: not in list_games() any more, skipping", flush=True)
        return

    rescued = salvage(OUT / gid, gid) if (OUT / gid).exists() else None
    if rescued is not None:
        banked.write_text(json.dumps(rescued, indent=1))
        print(f"\n=== {gid}: salvaged a clean result from an earlier attempt "
              f"({rescued.get('levels_reached')}/{rescued.get('levels_total')})",
              flush=True)
        return

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
    except Exception as exc:                      # noqa: BLE001 -- one game must not end the pass
        traceback.print_exc()
        # **Say what the harness was talking to.** On 2026-08-06 seven games
        # were burned through three passes on
        # `list_games -> [Errno 111] Connection refused`, and the traceback
        # alone could not say whether the driver was pointed at ARC or at a
        # dead loopback proxy -- the two are the same line of code and only
        # the resolved root tells them apart. Reproducing the startup by hand
        # afterwards showed the real API, so the running process differed from
        # the code on disk in a way nothing recorded.
        from athanor.ccarc3 import client as _c  # noqa: PLC0415
        print(f"    root={_c.ROOT_URL} proxy={os.environ.get('CCARC3_PROXY_URL')} "
              f"arc_root={os.environ.get('CCARC3_ARC_ROOT')}", flush=True)
        print(f"    {gid} attempt {n} raised; treating as interrupted", flush=True)
        if isinstance(exc, RuntimeError) and "Connection refused" in str(exc):
            # **A dead endpoint is not this game's fault, and retrying is not
            # a strategy.** Every remaining game fails the same way in
            # milliseconds, so the loop marks the whole queue interrupted,
            # burns its 12 passes and gives up on games it never launched.
            # Stop instead, and let the supervisor restart the driver -- a
            # fresh process rebuilds the proxy that went missing.
            # **Set the flag before raising.** The raise unwinds this thread;
            # the flag is what reaches the other sixteen. Without it they all
            # ran -- thirteen games launched and failed in the milliseconds
            # before the main thread even saw the first SystemExit.
            _aborted.set()
            with _slots:
                _slots.notify_all()          # wake every waiter to see the flag
            raise SystemExit(
                "aborting: the ARC endpoint is unreachable, so every "
                "remaining game would fail identically. Restart the driver."
            )
        return
    finally:
        # One shim per game, so a finished game gives its port back. Games run
        # concurrently now; without this the registry grows a live listener per
        # attempt for the lifetime of the driver.
        ab.release_proxy(gid)

    state, data = verdict(workspace_of(run_dir, gid))
    mins = (time.time() - started) / 60
    # Prefixed, because two games interleave in this log now and an unlabelled
    # "CLEAN in 20 min" belongs to whichever one you assume it does.
    short = gid.split("-")[0]
    if state == "clean":
        (OUT / gid / "clean_result.json").write_text(json.dumps(data, indent=1))
        print(f"    {short} CLEAN in {mins:.0f} min — "
              f"{data['levels_reached']}/{data['levels_total']}, "
              f"won={data['won']}, {data.get('actions_used')} actions", flush=True)
    else:
        # `.get("error", state)` returns None rather than the fallback:
        # `collect_outcome` writes the key with a null value on a clean exit,
        # so the default never applies and a discard printed "— None".
        reason = (data or {}).get("error") or state
        print(f"    {short} discarded after {mins:.0f} min — {reason}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
