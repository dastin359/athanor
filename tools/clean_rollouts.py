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
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

# **The scratchpad path is overridable, because it is not stable.** It encodes a
# session UUID, and the container is recycled every 10-50 minutes -- a new one
# gets a new path, and every tool that hard-coded this string pointed at a
# directory that no longer exists. `quota.sh` already took `CCARC3_SCRATCH`; the
# other four did not, so a fixture run or a fresh container silently read the
# wrong tree. Same defect class as the rest of this file: a value that agrees
# with the truth only in the environment you happen to test it in.
SP = pathlib.Path(
    os.environ.get("CCARC3_SCRATCH")
    or "/tmp/athanor-ccarc3-codex/scratchpad"
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
# The name must still contain `clean_rollouts`, for a narrower reason than this
# comment first claimed. It said `build_trace_audit` "decides whether a run is
# one of ours by looking for it in the working directory, so a rename would
# silently drop the whole sweep out of the audit" -- that is false. Ingestion is
# by explicit path: `main()` reads exactly the directories passed as `--ingest`,
# and neither `ingest()` nor `runs_row()` looks at the name.
#
# What does key on it is the **live-process scan** (`build_trace_audit.py:900`),
# which finds running solvers by matching their cwd to report in-flight
# progress. A differently-named sweep directory is invisible to that -- so the
# audit page would show the sweep as idle while it ran, which is worth avoiding
# and is not the same as losing the runs.
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
from athanor.ccarc3 import scoring           # noqa: E402

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
    # **bp35 first, and it is the only entry here placed by outcome rather than
    # by length.** Every other game in this list is banked at E=1.0000; bp35 is
    # the single shortfall at 0.7252, and that figure predates the win-frame
    # replay prompt, so nothing has ever measured what the current harness scores
    # on it. The submission's headline number is capped by exactly this game.
    #
    # Ordering by length -- "bank what can finish before betting a window on what
    # probably cannot" -- was written when a five-hour window was believed to be
    # the binding constraint. It is not: bp35 has already run 4.86 h to
    # completion once, and the supervisor has never made a ceiling stop. The
    # weekly window is the real limit, and it constrains the sweep's total, not
    # the placement of one game inside it.
    #
    # At rank 18 under concurrency 3, bp35 would have run near the very end --
    # returning the one uncertain answer at the moment nothing could be done with
    # it. The card scores the BEST play (`scoring.py`, select="best"), so a
    # retry raises the card and cannot lower it. That makes an early answer
    # actionable and a late one merely informative.
    "bp35-0a0ad940",   # 9 levels, 651 baseline, 4.86 h — the only sub-1.0 run
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
    # keys on `AGENTS.md`, `DOCTRINE.md`, `session.py` and `meta.json`, and two
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
    "ar25-0c556536",   # 8 levels, 748
    "ls20-9607627b",   # 7 levels, 776
    "cn04-2fe56bfb",   # 6 levels, 789
    "g50t-5849a774",   # 7 levels, 879
    "m0r0-492f87ba",   # 6 levels, 1107
    "dc22-fdcac232",   # 6 levels, 1228
    "re86-8af5384d",   # 8 levels, 1255
]

# **Scoping the sweep to named games, opt-in and loud.** `GAMES` is the full
# 25-environment benchmark and there was no way to run a subset -- so a sweep
# interrupted at game 14 could only be resumed by re-running the whole list and
# leaning on the skip-if-banked guard, and a single-game validation of the
# harness could not go through this driver at all.
#
# `CCARC3_ONLY` takes comma-separated ids or 4-character prefixes. Unset, this
# is a no-op and `GAMES` is byte-identical to the list above -- that property is
# asserted in the tests, because a filter that silently narrows the submission
# sweep is far worse than no filter.
#
# An expression that matches NOTHING raises. Running zero games and printing
# "all 25 have a clean run" is precisely the shape of silent success this
# project keeps finding, and a typo in a prefix is the likely cause.
_ONLY = [t.strip() for t in (os.environ.get("CCARC3_ONLY") or "").split(",") if t.strip()]
if _ONLY:
    _selected = [g for g in GAMES if g in _ONLY or g.split("-")[0] in _ONLY]
    if not _selected:
        raise SystemExit(
            f"CCARC3_ONLY={os.environ['CCARC3_ONLY']!r} matched none of the "
            f"{len(GAMES)} games. Ids look like 'bp35-0a0ad940'; a bare 'bp35' "
            f"also works. Refusing to run an empty sweep."
        )
    print(f"CCARC3_ONLY: {len(_selected)} of {len(GAMES)} games "
          f"({', '.join(g.split('-')[0] for g in _selected)})", flush=True)
    GAMES = _selected
BUDGET_MULTIPLE = 5.0

# **Long enough for the longest game, not the default 2 h.** The driver inherited
# Ccarc3Config's 7200 s, and `lf52` took 3.55 h in the arm -- so it would have hit
# the cap on every attempt, been correctly marked interrupted by the wall-clock
# guard, discarded, and retried forever without ever banking. A game that cannot
# finish should fail because the box died, not because the harness cut it off.
WALL_CLOCK_S = 6 * 3600


def _proc_ppids() -> dict[int, int]:
    """Return pid -> ppid for every readable process."""
    out: dict[int, int] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            out[int(entry.name)] = int(raw[raw.rindex(")") + 2:].split()[1])
        except (OSError, PermissionError, ValueError, IndexError):
            continue
    return out


def _live_driver_pids() -> set[int]:
    """Return every live clean-rollout driver that may own a solver."""
    owners = {os.getpid()}
    suffix = "/" + pathlib.Path(__file__).name
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if any(a.decode("utf-8", "replace").endswith(suffix) for a in argv if a):
            owners.add(int(entry.name))
    return owners


def _has_live_owner(pid: int, ppids: dict[int, int], owners: set[int]) -> bool:
    """Return whether a live rollout driver appears in ``pid``'s ancestry."""
    seen: set[int] = set()
    current = ppids.get(pid)
    while current and current not in seen:
        if current in owners:
            return True
        seen.add(current)
        current = ppids.get(current)
    return False


def kill_orphan_solvers() -> int:
    """Kill solvers left running by a previous driver, before starting work.

    **Stopping this driver does not stop its solver.** `run_game` waits on a
    `codex` subprocess; kill the parent and the child is reparented to a system
    reaper and
    keeps going -- still acting on the game, still spending quota, and now
    uncollectable, because `collect_outcome` runs in the parent that just died.
    Its `result.json` will never be written no matter how the run ends.

    Restarting the driver to pick up a code change leaked two such trees in one
    session, on `ka59` and `wa30`, both still writing to their traces minutes
    later. One of them was duplicating a game the new driver had already
    restarted, so two solvers were exploring the same environment on two
    scorecards for no benefit.

    Identified by working directory rather than by name: a solver's cwd is its
    workspace under `clean_rollouts`, and only trees with no live driver in
    their ancestry qualify. Testing for PPID 1 is not portable: systemd, WSL,
    containers, and the Codex relay may install a different subreaper.
    """
    killed = 0
    root = OUT.resolve()
    ppids = _proc_ppids()
    owners = _live_driver_pids()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cwd = (entry / "cwd").resolve()
            raw = (entry / "stat").read_text()
        except (OSError, PermissionError):
            continue
        # **Containment, not a substring.** This read `str(OUT) not in str(cwd)`,
        # and `clean_rollouts` is a prefix of `clean_rollouts_validate` and of
        # `clean_rollouts_submission`. A driver on the default sweep therefore
        # matched -- and would SIGTERM the process group of -- an orphan belonging
        # to a differently-named sweep running beside it, which is the one
        # arrangement `CCARC3_SWEEP_DIR` exists to make safe.
        if cwd != root and root not in cwd.parents:
            continue
        # **Fields split after the last `)`, not across the line.** `comm` is
        # field 2, it is arbitrary bytes chosen by the process, and node sets it
        # from `process.title`. One space in it shifts every field right, so
        # `stat[3]` reads a fragment of the name instead of the ppid, the `!= "1"`
        # test never matches, and the orphan is silently left running -- the
        # cleanup fails by finding nothing, which looks exactly like a clean box.
        # Same defect as `build_trace_audit._process_started` carried until today,
        # and here the neighbouring field is the pgid handed to `killpg`.
        fields = raw[raw.rindex(")") + 2:].split()
        pgid = fields[2]
        if entry.name == str(os.getpid()):
            continue
        if _has_live_owner(int(entry.name), ppids, owners):
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
    scored from later and hoped over. The comparison itself lives in
    `scoring.card_disagreement`, in one place because this guard has three copies
    (here, `proofread_trace.py`, `restore_clean_rollouts.py`) and they drifted:
    all three read `max(levels_completed)`, which under the sweep's shared card
    silently began including the high-water mark of every discarded earlier
    attempt of the same game.
    """
    card_file = game_dir / "scorecard.json"
    if not card_file.exists():
        return "no scorecard"
    try:
        card = json.loads(card_file.read_text())
    except json.JSONDecodeError:
        return "scorecard unreadable"
    return scoring.card_disagreement(card, data.get("game_id"), data)


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

# **The durable half of the pin, and deliberately only half.**
#
# Everything `sweep_card` uses to decide whether it may mint a card lives in
# `SP` -- and `SP` is the one store that reverts to an image snapshot when the
# container is replaced, which on 2026-08-06 happened three times in 45 minutes.
# When it reverts, the pin and the history vanish *together*, so
# `SHARED_CARD_FILE.exists()` is False, the entire refusal ladder below is
# skipped, and a fresh card is minted in silence. Every rung of that ladder --
# the unreadable-file refusal, the liveness probe, the "N games were scored on
# it" hard stop -- reads one or both of those two files, so a replacement
# disarms all of them at once. Confirmed present on 2026-08-09: both files gone
# after a replacement, and the error text that says "the card_id needed to
# recover is in the history file" pointing at a file that no longer exists.
#
# It is the same defect the `dead` rename below already carries a comment about
# -- "the next `sweep_card()` finds no card file at all, skips this whole branch,
# and silently mints a fresh card" -- arriving through a different door. That one
# was fixed by reordering. This one cannot be, because the trigger is deleted by
# the platform.
#
# **Only the card id and the history go here. Never the cookies.** A pinned card
# carries `GAMESESSION` and four `AWSALBAPP-*` values, and this directory is
# committed and pushed to GitHub -- writing them here would publish live
# credentials. That is not a limitation to work around: the cookies are the only
# route back to a card, so a replaced container genuinely cannot resume the old
# one. What it *can* do is know the card existed and refuse to pretend otherwise,
# which is the whole decision the driver says it will not make for you.
# Derived from this file's location (`<repo>/tools/`), never written: the
# hard-coded path is this container's clone location. Same reason as every other
# derived path in this tree.
_REPO = pathlib.Path(__file__).resolve().parents[1]
DURABLE_CARD_DIR = _REPO / "evidence" / "ccarc3" / "sweep_card"
DURABLE_CARD_HISTORY = DURABLE_CARD_DIR / "history.jsonl"


# An ARC scorecard id is a UUID -- `ebb808ef-0693-4d6e-a7e4-3ef12210e2f2`. Nothing
# else may reach the committed history; see `_remember_card`.
_CARD_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _remember_card(card_id: str, event: str, **extra) -> None:
    """Append one line to both histories. Never raises; never writes a cookie.

    **The durable half takes UUIDs only, and that is not a test accommodation.**
    Within an hour of this history becoming durable it had 27 rows in it reading
    `C1`, `DEAD`, `FRESH`, `card-A`, `card-B` -- the fixtures of five test files
    that exercise `sweep_card` and its retire path, appending to the real
    committed file on every `pytest` run, which `preserve_evidence.sh` then
    pushed. A test suite that writes into the evidence tree is bad on its own,
    but the sharper problem is that this file is the *input* to the guard
    deciding whether a sweep may open a card: garbage here is garbage in that
    decision, and a fixture id colliding with a real one would refuse a sweep
    for no reason.

    The scratchpad copy still takes anything -- it is local, disposable, and the
    thing a human reads when debugging one box.
    """
    row = {"card_id": card_id, "event": event, "at": time.time(), **extra}
    line = json.dumps(row) + "\n"
    targets = [SHARED_CARD_HISTORY]
    if _CARD_ID.match(card_id or ""):
        targets.append(DURABLE_CARD_HISTORY)
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:                          # noqa: PERF203
            print(f"    could not record card history in {path} ({exc})", flush=True)


def _durable_cards() -> list[str]:
    """Card ids this sweep has opened, oldest first, from the committed history."""
    out: list[str] = []
    try:
        text = DURABLE_CARD_HISTORY.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            got = json.loads(line).get("card_id")
        except ValueError:
            continue
        if got and got not in out:
            out.append(got)
    return out


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

    # **A missing pin is not the same as a first run, and this is where a
    # container replacement used to walk straight through.** Every rung of the
    # ladder below reads `SHARED_CARD_FILE`, so when the scratchpad reverts they
    # are all skipped at once and the sweep mints a fresh card without a word.
    # The committed history is what survives, and it answers the only question
    # that matters: has this sweep already opened a card?
    #
    # If it has, and games are banked on it, minting would build a submission
    # that silently omits them -- the same hard stop the reaped-card branch below
    # makes, for the same reason, reached by the door that used to be unguarded.
    # The cookies are not here and cannot be (see DURABLE_CARD_DIR), so resuming
    # the old card is genuinely impossible; the choice between restarting the
    # sweep and accepting a partial artifact belongs to an operator.
    if not SHARED_CARD_FILE.exists() and (prior := _durable_cards()):
        seen = _cards_seen()
        stranded = {cid: seen.get(cid, []) for cid in prior if seen.get(cid)}
        if stranded:
            detail = "; ".join(
                f"{cid} carries {len(games)} game(s) "
                f"({', '.join(g.split('-')[0] for g in games)})"
                for cid, games in stranded.items())
            raise SystemExit(
                f"{SHARED_CARD_FILE} is missing but this sweep has already "
                f"opened a card: {detail}. The pin lives in the scratchpad, "
                f"which reverts when the container is replaced, so its absence "
                f"is not evidence of a fresh sweep. Those games cannot be moved "
                f"to a new card and their session cookies are not recoverable, "
                f"so continuing would build a submission that silently omits "
                f"them. Restart the sweep against a fresh card deliberately "
                f"(delete {DURABLE_CARD_HISTORY.name}, or set CCARC3_SWEEP_DIR "
                f"to a new directory), or accept a partial artifact — this "
                f"driver will not choose for you."
            )
        print(f"shared card pin missing; the committed history names "
              f"{len(prior)} earlier card(s) with no banked games "
              f"({', '.join(c[:8] for c in prior)}), so nothing is stranded — "
              f"opening a new one", flush=True)

    if SHARED_CARD_FILE.exists():
        # **An unreadable card file is an operator decision, not a crash.**
        # `sc.load` raised straight out of here on truncated JSON: the driver
        # died, the supervisor relaunched it ten minutes later into the same
        # crash, and the loop never advanced -- with the card carrying the
        # banked games stranded behind a file nobody was going to look at,
        # because from the outside a crash-looping driver and a busy one both
        # just look like "not finished yet".
        #
        # `save` is atomic now, so this should not arise from our own writes.
        # It is still reachable: a file restored by hand, a truncated copy, a
        # partial rsync. Deliberately NOT auto-reminting -- that is the same
        # decision the `played` branch below refuses to make for you, and the
        # card_id needed to recover is in the history file.
        try:
            card = sc.load(SHARED_CARD_FILE)
        except Exception as exc:                       # noqa: BLE001
            last = ""
            try:
                for line in SHARED_CARD_HISTORY.read_text().splitlines():
                    got = json.loads(line).get("card_id")
                    if got:
                        last = got
            except Exception:                          # noqa: BLE001
                pass
            raise SystemExit(
                f"{SHARED_CARD_FILE} exists but cannot be read ({exc!r}). "
                f"Refusing to mint a fresh card: any games already scored onto "
                f"the old one cannot be moved, so continuing would build a "
                f"submission that silently omits them."
                + (f" The last card opened was {last} (per "
                   f"{SHARED_CARD_HISTORY.name})." if last else "")
                + " Restore the file, or delete it to mint a new card"
                  " deliberately — this driver will not choose for you."
            )
        # **Retry before declaring death.** This caught every exception and
        # treated all of them as "the card is gone" -- so one refused connection
        # or a proxy blip would retire a live card carrying finished games. A
        # 404 is the card; anything else is the network until proven otherwise.
        gone = None
        for attempt in range(3):
            try:
                sc.read_card(card, GAMES[0])
                print(f"shared card {card.card_id} — still reachable", flush=True)
                return card
            except Exception as exc:                   # noqa: BLE001
                gone = exc
                if "not found" not in str(exc) and "404" not in str(exc):
                    time.sleep(2 ** attempt)           # transient: try again
                    continue
                break

        # **Whether this matters depends entirely on whether the card had games.**
        # An empty card reaped before the sweep starts costs nothing -- and that
        # is the common case, because this opens the card at driver start while
        # the first game may be an hour away. Five such remints were recorded on
        # 2026-08-08 in 6.5 hours, every one of them harmless and every one of
        # them logged as though it were a disaster.
        #
        # A card with games on it is the opposite: those games cannot be moved,
        # so reminting silently produces a submission missing everything banked
        # so far. That is a decision for an operator, not a default.
        played = _cards_seen().get(card.card_id, [])
        _remember_card(card.card_id, "retired", games_lost=len(played),
                       reason=str(gone)[-160:])
        # **Refuse BEFORE retiring the file, or the refusal deletes its own
        # trigger.** This retired `shared_card.json` first and raised second, so
        # the hard stop lasted exactly one process: the supervisor relaunches
        # within ten minutes, the next `sweep_card()` finds no card file at all,
        # skips this whole branch, and silently mints a fresh card. An operator
        # decision the driver explicitly declines to make for you was therefore
        # made for you, on a timer, and the log line announcing it scrolled past.
        if played:
            raise SystemExit(
                f"shared card {card.card_id} is gone and {len(played)} game(s) "
                f"were scored on it ({', '.join(g.split('-')[0] for g in played)}). "
                f"They cannot be moved to a new card, so continuing would build a "
                f"submission that silently omits them. Re-run those games against "
                f"a fresh card, or accept a partial artifact deliberately — this "
                f"driver will not choose for you."
            )
        dead = SHARED_CARD_FILE.with_suffix(f".{card.card_id[:8]}.dead.json")
        SHARED_CARD_FILE.replace(dead)
        print(f"shared card {card.card_id} was reaped before any game reached it "
              f"({str(gone)[-60:]}); nothing lost, opening another", flush=True)

    card = sc.open_card(tags=("ccarc3", "clean-rollouts"))
    sc.save(card, SHARED_CARD_FILE)
    _remember_card(card.card_id, "opened")
    print(f"shared card {card.card_id} — opened, all games will score onto it",
          flush=True)
    return card


def _outstanding() -> list[str]:
    """Games with no banked clean run, fewest attempts first, GAMES order to break ties.

    **Fewest attempts first.** A plain GAMES walk starves the tail: the driver dies
    with its container, and on relaunch it starts again at the first outstanding
    game, so a game that can never finish in one window takes every window and
    nothing behind it is ever tried. That is exactly how a fixed order once turned
    a three-game experiment into a one-game one in `rerun_losses.py`. With all
    counts at zero this is identical to GAMES order.

    **The `GAMES.index` tie-break is provably redundant, and kept anyway.** The
    generator yields in `GAMES` order and `sorted` is guaranteed stable, so equal
    attempt counts already retain that order; dropping the second key is an
    equivalent mutant, and it survives the tests below because it must. It stays
    because the rule it states is the load-bearing one -- an input that stopped
    being `GAMES`-ordered would change the answer silently otherwise.
    """
    fresh = sorted(
        (g for g in GAMES if not (OUT / g / "clean_result.json").exists()),
        key=lambda g: (completed_attempts(OUT / g, g), GAMES.index(g)),
    )
    # Behind everything unplayed, on purpose: a complete card is worth more than
    # a polished one, and an environment already banked is already on the card.
    return fresh + sorted(
        (g for g in GAMES if g not in fresh and _wants_replay(g)),
        key=lambda g: (completed_attempts(OUT / g, g), GAMES.index(g)),
    )


def enable_nudging() -> None:
    """Tell `run_game` to resume a solver that quits while it can still act.

    When a solver stops early, the alternative is discarding the attempt and
    replaying the game from level zero. Measured on `bp35` (2026-08-09): the
    discarded give-up cost $23.74 and its from-scratch replacement another
    $26.55, and neither banked.

    **Called from `main()`, not at import, and that is load-bearing.** This is
    process-wide state, so setting it at module scope turned nudging on for
    anything that merely *imported* this driver -- including the test suite,
    where it silently changed the behaviour of two stream-rotation tests that
    have nothing to do with nudging. The file already carries this lesson twice:
    `install_strip` is a function you call for the same reason, and
    `ablate_baselines._install_patch` says outright that "import side effects
    that rewrite another module's globals are exactly the kind of thing that is
    invisible until it produces a wrong number".

    `setdefault`, so an operator who names a value keeps it -- including `0`.
    """
    # Three, raised from two on 2026-08-10 by operator decision, in the same
    # pass that removed the half-allowance restriction on what counts as
    # quitting. The two changes compound: more runs are now recognised as
    # give-ups, and each gets one more chance to continue before the game is
    # replayed from level zero.
    os.environ.setdefault("CCARC3_MAX_NUDGES", "3")


# **One driver at a time, enforced rather than assumed.**
#
# The supervisor starts a driver whenever it sees none running. So any window
# between stopping one and starting the next -- 30 seconds is enough -- lets the
# supervisor's ten-minute cycle fire into the gap, and then a hand-started driver
# makes two. Measured 2026-08-11: two drivers, ten solvers, every one of five
# games being played twice at once.
#
# The card survives that (a duplicate play is just a play, and the environment
# keeps its best), but it doubles the burn, and the cleanup is delicate --
# solvers are `setsid` into their own process groups, so killing a driver's group
# does not reach them and each one has to be matched to its parent by walking
# /proc.
#
# A lock makes the second driver a no-op instead of a hazard. `flock` and not a
# pidfile: a pidfile written by a driver killed with SIGKILL outlives it and
# locks the sweep out forever, while a flock is released by the kernel when the
# holder dies however it dies.
def _only_driver() -> "object | None":
    """Hold the sweep's adjacent lock, or return if another driver has it."""
    import fcntl
    lock = OUT.parent / f"{OUT.name}.driver.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fh = lock.open("w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def main() -> int:
    held = _only_driver()
    if held is None:
        print(f"another driver already holds {OUT.name}.driver.lock — exiting so "
              f"the two do not play every game twice at once", flush=True)
        return 0
    enable_nudging()
    install_strip()
    OUT.mkdir(parents=True, exist_ok=True)

    # **Nothing to do means no card.** `sweep_card()` ran first and opened a
    # scorecard unconditionally, so a driver relaunched onto a finished sweep --
    # which is what the supervisor does every ten minutes, forever, once the
    # queue empties -- minted a fresh card, played nothing onto it, and exited.
    # Six abandoned cards an hour against the ARC account, and since the card
    # history became durable each one also appends a line to a committed file
    # that `preserve_evidence.sh` pushes. A card is an artifact with a URL a
    # submission points at; opening one is not free and is not bookkeeping.
    if not _outstanding():
        print(f"all {len(GAMES)} have a clean run — no card opened, nothing to do",
              flush=True)
        return 0

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
        # Ordering rationale lives in `_outstanding`, which is now the only
        # place that computes this -- it was open-coded here and again in
        # `main`'s preamble, two copies of one rule that can drift apart.
        outstanding = _outstanding()
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


# **Overridable, because a validation run is not a sweep.** A game cut short by
# a container replacement is retried, and the bound on that is the only thing
# between one authorised game and twelve paid-for attempts at it. `bp35`'s prior
# is 3.9 hours; this box was replaced 2.5 hours ago. Twelve is right for a
# 25-game sweep that must finish; it is not right for a single smoke test, and
# the difference belongs to whoever launches it rather than to this file.
#
# Same shape as `CCARC3_CONCURRENCY` below, and read the same way.
def _max_passes() -> int:
    raw = os.environ.get("CCARC3_MAX_PASSES", "")
    try:
        n = int(raw)
    except ValueError:
        if raw:
            print(f"CCARC3_MAX_PASSES={raw!r} is not a number; using 12", flush=True)
        return 12
    if n < 1:
        print(f"CCARC3_MAX_PASSES={n} is below one pass; using 1", flush=True)
        return 1
    return n


MAX_PASSES = _max_passes()


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
    #
    # **0 means hold, and that has to be honoured.** This clamp read
    # `max(1, ...)`, so writing 0 -- the obvious way to say "start nothing", and
    # the way this project has actually used the file during a launch freeze --
    # quietly ran one game anyway. The heartbeat printed `conc=0` beside a
    # running solver for days and the two were never read as contradictory. A
    # brake that reads as engaged and is not is worse than no brake at all.
    #
    # A negative number is still a typo rather than a request, and it clamps to
    # 0 rather than 1: of the two ways to misread a malformed brake, holding is
    # the recoverable one.
    return max(0, min(CONCURRENCY_MAX, n))


_slots = threading.Condition()
_running = 0
# **`None`, not 0, and this cost a launch.** The gate reports the limit only when
# it *changes* from this, so a sentinel of 0 is indistinguishable from a real
# limit of 0 -- and 0 is the value that means "hold, start nothing". A driver
# launched against an engaged brake therefore parked in `_take_slot` and printed
# nothing at all: the one line that says why it is not working is suppressed in
# exactly the case where it is needed.
#
# Measured 2026-08-09. `scratchpad/concurrency` held `0` from a launch freeze on
# Aug 7, the container was replaced and the snapshot restored it, and a `bp35`
# validation run sat silent for fifteen minutes looking like a slow start. The
# file the brake lives in is in the store that reverts, so a freeze set days ago
# comes back from the dead and holds every launch after it.
_last_limit: int | None = None
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
                was = "" if _last_limit is None else f" (was {_last_limit})"
                # 0 is a brake, not a small number, so it says what it means and
                # where to release it. A reader who sees "concurrency = 0" and
                # nothing else has to know this file exists to act on it.
                held = (f" — holding, nothing will start until "
                        f"{CONCURRENCY_FILE} says more than 0" if limit == 0 else "")
                print(f"    concurrency = {limit}{was}{held}", flush=True)
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


def _shared_card_id() -> str:
    """The card this sweep is supposed to be building, or "" if not sharing."""
    try:
        return json.loads(SHARED_CARD_FILE.read_text()).get("card_id", "")
    except (OSError, ValueError):
        return ""


def _cards_seen() -> dict[str, list[str]]:
    """``{card_id: [game, ...]}`` over **every** attempt this sweep has on disk.

    **Two guards read `attempt_1` and meant "any attempt".** A game whose first
    attempt was interrupted and whose second was the banked one leaves
    `attempt_1/.../trace.state.json` missing or holding a different card, so
    `sweep_card`'s refusal to remint over a card with games on it counted zero
    and continued -- failing open, on the one decision the driver explicitly
    refuses to make for an operator. The scan is a few file reads per pass and
    the thing itself.
    """
    seen: dict[str, list[str]] = {}
    # **Only attempts that produced a result count as "scored on" a card.**
    # `trace.state.json` is written by `open()` before a single action, so an
    # attempt killed by a container replacement -- the ordinary case this driver
    # discards and re-runs -- used to count as a game on the card. That inflates
    # `sweep_card`'s refusal (stopping the sweep over work that was never
    # banked) and the split report alike. A result.json beside the state file is
    # the cheapest evidence the attempt got somewhere.
    # **Also read the durable copy, because the live one dies with the pin.**
    #
    # This scanned only OUT, which lives in the scratchpad. A container
    # replacement reverts the scratchpad -- taking the card pin AND every
    # `trace.state.json` with it -- so at the exact moment `sweep_card` asks
    # "does the previous card carry games?", the evidence it reads is gone.
    # It then answers "nothing is stranded, opening a new one" and silently
    # builds a partial artifact.
    #
    # That is the failure the guard exists to prevent, reached through the guard
    # itself: the check named the card's contents and read a proxy for them, and
    # the proxy is destroyed by the same event that triggers the check.
    #
    # Measured 2026-08-11. The replacement stranded `e6e6a61c` carrying seven
    # games; `OUT` was empty and the durable evidence under `evidence/ccarc3/`
    # -- which `preserve_evidence.sh` pushes to origin -- named all seven.
    import gzip
    durable = DURABLE_CARD_DIR.parent / OUT.name
    for state in sorted(durable.rglob("trace.state.json.gz")):
        # **The same "a result, not merely a trace" rule as the live loop below,
        # which is where it was applied and where it stopped.**
        #
        # `trace.state.json` is written by `open()` before a single action, and
        # `preserve_evidence.sh` gzips it into the evidence tree on its next tick
        # -- so an attempt that is interrupted and correctly discarded still
        # leaves a durable state file naming the card. Counting that made
        # `sweep_card` refuse to open a fresh card over work that was never
        # banked, which is the exact inflation the comment below already forbids,
        # surviving in the sibling loop it was never applied to.
        #
        # Measured 2026-08-12. The WSL VM was terminated 13 minutes into a bp35
        # run; the attempt produced no `result.json`, the preserver had already
        # archived its `trace.state.json.gz`, and the card was 404 by the time
        # the driver came back. `sweep_card` then hard-stopped the sweep --
        # "1 game(s) were scored on it (bp35)" -- over a discarded attempt with
        # nothing on the card to lose. On a box where an interruption is the
        # ordinary case, that is a sweep that cannot restart itself.
        #
        # The durable tree holds gzipped copies, so accept either spelling; the
        # preserver compresses, and a hand-placed file may not be.
        if not any((state.parent / n).exists()
                   for n in ("result.json", "result.json.gz")):
            continue
        try:
            cid = json.loads(gzip.decompress(state.read_bytes())).get("card_id", "")
        except (OSError, ValueError, EOFError):
            continue
        if cid:
            gid = state.parent.name
            if gid not in seen.setdefault(cid, []):
                seen[cid].append(gid)

    for state in sorted(OUT.glob("*/attempt_*/*/trace.state.json")):
        if not (state.parent / "result.json").exists():
            continue
        try:
            cid = json.loads(state.read_text()).get("card_id", "")
        except (OSError, ValueError):
            continue
        if cid:
            gid = state.parent.name
            if gid not in seen.setdefault(cid, []):
                seen[cid].append(gid)
    return seen


# **Replay an environment the harness itself judged short, not one an operator
# picked.** ARC scores an environment by its BEST play and renders every play on
# the public card, so replaying is the documented mechanic rather than a
# loophole. What makes a reported number honest is that the rule is fixed in
# advance, applied to every environment alike, and stated with the result -- an
# operator clearing one game's bank because that game disappointed is selection,
# and looks like it.
#
# So: a run that completes below `RETRY_BELOW` is not done. It goes back to the
# queue, behind every environment that has not yet been played at all, until it
# scores or it has had `RETRY_ATTEMPTS` tries.
#
# Bounded because it has to be: `bp35` is 4.9 hours and about $79 an attempt, so
# an unbounded "retry until 1.0" hands one stubborn environment the entire
# budget and banks nothing else.
RETRY_BELOW = float(os.environ.get("CCARC3_RETRY_BELOW", "1.0"))
RETRY_ATTEMPTS = int(os.environ.get("CCARC3_RETRY_ATTEMPTS", "3"))


def _rhae(workspace: pathlib.Path, gid: str) -> float | None:
    """This run's RHAE, or ``None`` when it cannot be computed.

    `None` is deliberately not zero. A scoring failure -- no baselines, an
    unreadable ledger -- must not read as "scored badly" and send a finished game
    back for another $79 attempt. Unknown means leave it banked.
    """
    try:
        from athanor.ccarc3 import client as _client, ledger as _ledger, scoring as _scoring
        base = _client.baselines_for(gid)
        if not base:
            return None
        # **Argument order: transitions first, baselines second.** This read
        # `score_run(base, load(...))` — swapped — so every call raised
        # `'int' object has no attribute 'full_reset'` the moment scoring touched
        # what it thought was a ledger. The `except` below turned that into
        # `None`, `_wants_replay` short-circuits on `None`, and the replay policy
        # this function exists to drive has therefore **never once fired**. A
        # game banked at 7/10 with won=False sat there and was never requeued.
        #
        # **And `.score`, not `.E`.** Fixing the argument order surfaced a second
        # error on the same line — `EnvironmentScore` exposes `raw`, `cap` and
        # `score`, never `E` — which had been sitting behind the same blanket
        # `except` all along. Two independent mistakes in one expression, neither
        # ever observed, because the only thing that ever read the result treated
        # every failure as "this run cannot be scored".
        return float(_scoring.score_run(_ledger.load(workspace / "trace.jsonl"), base).score)
    except (TypeError, AttributeError) as exc:
        # **A programming error is not an unscoreable run, and must not look like
        # one.** The blanket catch below is right for its stated purpose — a run
        # that genuinely cannot be scored must still bank rather than be lost —
        # and that is exactly why it hid a swapped argument for the life of the
        # feature. These two exception types cannot come from bad data; they mean
        # this function is wrong. Still banks, because losing the run helps
        # nobody, but says so in terms nobody will read as routine.
        print(f"    {gid.split('-')[0]}: SCORING IS BROKEN ({type(exc).__name__}: "
              f"{exc}). This is a bug in _rhae, not a property of the run — the "
              f"replay policy is inert until it is fixed.", flush=True)
        return None
    except Exception as exc:  # noqa: BLE001 -- scoring must never fail a bank
        print(f"    {gid.split('-')[0]}: could not score this run ({exc}); "
              f"banking it as-is", flush=True)
        return None


def _wants_replay(gid: str) -> bool:
    """True when a banked environment scored short and has attempts left."""
    banked = OUT / gid / "clean_result.json"
    if not banked.exists():
        return False
    try:
        e = json.loads(banked.read_text()).get("E")
    except (OSError, ValueError):
        return False
    if e is None or float(e) >= RETRY_BELOW:
        return False
    return completed_attempts(OUT / gid, gid) < RETRY_ATTEMPTS


def _run_one(gid: str, infos: dict) -> None:
    banked = OUT / gid / "clean_result.json"
    if banked.exists() and _wants_replay(gid):
        prior = json.loads(banked.read_text())
        print(f"\n=== {gid}: banked at E={prior.get('E')}, below {RETRY_BELOW} — "
              f"replaying (attempt {completed_attempts(OUT / gid, gid) + 1} of "
              f"{RETRY_ATTEMPTS})", flush=True)
        banked.unlink()          # the card keeps the better play either way
    elif banked.exists():
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

    # **Catch a split while it is still recoverable.** `verify_one_card.py` is the
    # end-of-sweep check; by then a card that was reaped mid-sweep has already
    # cost every game after it. One scan per finished game turns that into a line
    # in the log at the moment it happens.
    #
    # **It used to compare this game's state file against the live shared id, and
    # that comparison cannot ever be unequal.** `_run_one` passes `fresh=True`,
    # `build_workspace` unlinks `trace.state.json` when fresh, so `_resume` never
    # runs -- and `_resume` is the only place `card_id` can become anything but
    # the injected sweep card. `open()` then takes its lent-card branch and
    # `_save_state` writes back exactly the id it was handed. `got == want` by
    # construction, for every game, always: a check that reports by not running,
    # and it made `verify_one_card.py`'s `foreign_card` branch dead for this
    # driver too.
    #
    # The split that can actually happen is a succession across driver restarts:
    # the card is reaped, a restart mints another, and half the sweep is on each.
    # That is visible only in the sweep's own history, so compare against that.
    seen = _cards_seen()
    live = _shared_card_id()
    if live and len(set(seen) | {live}) > 1:
        for cid, games in sorted(seen.items()):
            if cid != live:
                print(f"    *** {len(games)} game(s) scored on {cid}, NOT the live "
                      f"sweep card {live}: {', '.join(g.split('-')[0] for g in games)}"
                      f" — they are missing from the submission artifact ***",
                      flush=True)
        # Stop rather than spend the rest of the queue building an artifact that
        # is already incomplete. `sweep_card` refuses the same split at startup;
        # this is the same refusal for a split that opens mid-pass.
        _aborted.set()
        with _slots:
            _slots.notify_all()

    state, data = verdict(workspace_of(run_dir, gid))
    mins = (time.time() - started) / 60
    # Prefixed, because two games interleave in this log now and an unlabelled
    # "CLEAN in 20 min" belongs to whichever one you assume it does.
    short = gid.split("-")[0]
    if state == "clean":
        data["E"] = _rhae(workspace_of(run_dir, gid), gid)
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
