"""Paired regression arm: the same games, with per-level baselines withheld.

**Why.** `baseline_actions` is not in ARC's published `/api/games` schema — the
docs list `game_id` and `title`; the live server also returns `tags` and
`baseline_actions`. This harness leans on that undocumented field hard: 22
doctrine references, all of §6, the 1.0x pace warning inside `client.status()`
(called 256 times across six runs), `client.pace()`, and the workspace
`CLAUDE.md`, which hands the array to the solver as a table row. If the
semi-private set withholds it, every baseline-derived signal goes quiet at once
and the public-set figure would be measuring a solver with information the real
evaluation does not supply.

**Design.** Paired arm over **every scored game**, so each is its own control at
its own recorded RHAE — twelve at 1.000 or near it, plus `tn36` at 0.449, which
makes that one a gain arm rather than a regression arm. A fresh game would
confound "the information mattered" with "that game was hard". Same model, same
effort, same action cap, same doctrine except the parts unusable without the
numbers.

**One variable.** Only the per-level array is withheld:

- `session.py` gets `baseline_actions=()`. No client change is needed —
  `baseline_here` returns None, so `status()` drops its pace line and `pace()`
  returns nothing. The signal disappears exactly as it would if ARC stopped
  sending the field.
- `CLAUDE.md` loses the baseline table row and the paragraphs that read it.
- `DOCTRINE.md` loses §6 and §6a, which instruct the solver to steer by a number
  it can no longer see.

Deliberately **kept**: §0 (how scoring works) and the action cap. Cutting §0
would ablate "does the solver know the objective", a different variable; and an
agent is terminated at a budget on the real benchmark whether or not it knows the
human medians, so hiding the cap would change the task rather than the signal.

**Power, stated up front.** Thirteen paired games, twelve of which the control
won. Under the null, expect 12/12 again; McNemar on the discordant pairs means
3 losses reaches p=0.125 and 5 reaches p=0.031 — so this detects a moderate-to-
large effect and nothing subtler. Report it paired, game by game, never as two
pooled rates: the pairing is the entire design.

**Quota, not wall clock, is the binding constraint.** The window was 0.91 with
~19h to reset when this was written, and the guard stops at 0.95. Expect the arm
to bank two or three games, pause, and resume after the reset. It is idempotent,
so that costs only the relaunch.
"""
import gzip
import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/user/athanor/src")

from athanor.ccarc3 import Ccarc3Config, list_games, run_game
from athanor.ccarc3.client import HIDE_BASELINES_ENV
from athanor.ccarc3 import session as sess

SP = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
RUNS = SP / "ablate_nobaseline"
BUDGET_MULTIPLE = 2.0          # identical to batch 3, so the arms are comparable

# **Every scored game**, per the operator: "rerun all scored games without
# baseline info provided". Read from results.jsonl rather than hardcoded, so
# games finishing after this was written are included automatically.
#
# Ordered cheapest-first. Quota is the binding constraint, not wall clock: the
# guard stops the arm at 0.95 of the seven-day window, so cheap games first
# maximises how many pairs are banked before it does. The arm is idempotent, so
# it resumes across the reset.
def scored_games():
    # **Controls are the non-ablated rows only.** `snapshot_results.py` appends
    # this arm's own results to the same ledger with `batch="ablate_nobaseline"`.
    # Keyed by game_id alone, the arm's row overwrites the control it is supposed
    # to be compared against -- so every pair reads as a tie and the progress line
    # "same games with baselines: N/N" is the arm grading itself. Filtering by
    # batch is what makes the pairing real; it also keeps a game this arm ran but
    # nobody ever ran *with* baselines out of the paired set, where it would
    # contribute a fabricated control.
    ledger = SP / "results.jsonl"
    latest = {}
    if ledger.exists():
        for line in ledger.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("batch") == "ablate_nobaseline":
                continue
            if row.get("game_id") and row.get("rhae") is not None:
                latest[row["game_id"]] = row
    # A game with no recorded cost sorts last rather than first: unknown is not
    # cheap, and putting it early would spend the window on a guess.
    reruns = [
        g for g, _ in sorted(
            latest.items(), key=lambda kv: kv[1].get("cost_usd") or float("inf")
        )
    ]
    # **The untouched environments come after the reruns, deliberately.** A rerun
    # is a paired observation -- it has a control at a known RHAE, so it tests
    # whether withholding baselines costs anything. An untouched game has no
    # control and can only extend coverage. Under a quota ceiling the paired ones
    # are worth strictly more, so they go first; the rest follow cheapest-baseline
    # first once the science is banked.
    played = set(reruns)
    fresh = sorted(
        (g for g in list_games() if g.game_id not in played),
        key=lambda g: g.baseline_total,
    )
    return reruns + [g.game_id for g in fresh], latest


GAMES, CONTROL = scored_games()


def strip_baselines(root: pathlib.Path) -> None:
    """Remove every per-level baseline the solver can see. Idempotent."""
    sp = root / "session.py"
    text = sp.read_text(encoding="utf-8")
    stripped = re.sub(r"baseline_actions=\([^)]*\),", "baseline_actions=(),", text)
    if stripped == text and "baseline_actions=()" not in text:
        raise RuntimeError(f"{sp}: baseline_actions not found; template changed?")
    sp.write_text(stripped, encoding="utf-8")

    cm = root / "CLAUDE.md"
    kept = []
    for line in cm.read_text(encoding="utf-8").splitlines():
        if "baseline" in line.lower():
            continue
        kept.append(line)
    cm.write_text("\n".join(kept) + "\n", encoding="utf-8")

    # §6 and §6a tell the solver to steer by a number it can no longer read.
    doc = root / "DOCTRINE.md"
    lines = doc.read_text(encoding="utf-8").splitlines()
    out, skipping = [], False
    for line in lines:
        if re.match(r"^## 6\. ", line):
            skipping = True
            continue
        if skipping and re.match(r"^## (?!6)", line):
            skipping = False
        if not skipping:
            out.append(line)
    doc.write_text("\n".join(out) + "\n", encoding="utf-8")

    # **meta.json carries the whole GameInfo, including baseline_actions.**
    # `build_workspace` writes it and nothing here used to touch it, so twelve
    # runs of this arm shipped the array in a file the solver reads during
    # orientation. All twelve had it in context; `tu93` went further and called
    # `arc.score_run(ts, bl)` with it. The arm measured nothing it claimed to.
    meta = root / "meta.json"
    if meta.exists():
        blob = json.loads(meta.read_text(encoding="utf-8"))
        blob.pop("baseline_actions", None)
        meta.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    # **Scan the whole workspace, not the files this function edited.**
    # The old check looked at exactly `sp`, `cm` and `doc` -- the three it had
    # just rewritten -- so it could only ever confirm its own edits. A leak check
    # scoped to what you changed proves nothing about what you shipped. Any file
    # the solver can open is in scope, so every file is.
    # Match the *values*, not the identifier: `baseline_actions=()` is the
    # blanked form this function produces and must not trip its own check. So
    # require a container that actually opens onto a digit --
    # `baseline_actions=(22,` and `"baseline_actions": [22,` both match, `=()`
    # does not.
    numbers = re.compile(r"""baseline_actions["'\s:=]*[(\[]\s*\d|baseline actions per level""")
    leaked = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".md", ".json", ".txt", ".yaml", ".yml"}
        and p.name not in {"trace.jsonl", "stream.jsonl"}
        and numbers.search(p.read_text(encoding="utf-8", errors="ignore"))
    )
    if leaked:
        raise RuntimeError(f"baselines still reachable in {leaked}")


_original_build = sess.build_workspace


def build_without_baselines(config, info=None):
    ws = _original_build(config, info)
    strip_baselines(ws.root)
    # **Close the channel no file edit can reach.** The solver holds an API key
    # and imports the package, so `arc.list_games()` would hand back
    # `baseline_actions` for all 25 environments however thoroughly the
    # workspace is sanitised. This is set on the *child's* environment only --
    # this process keeps the real numbers, which it needs to build the queue and
    # to score.
    #
    # Verified unused for the first twelve runs, so it changes no result already
    # banked; it stops the thirteenth from being the one that quietly does it.
    ws.env[HIDE_BASELINES_ENV] = "1"
    return ws


def batch6_running() -> bool:
    for d in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            argv = (d / "cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue
        if any(a.endswith("/batch6.py") for a in argv):
            return True
    return False


# **Runs concurrently with batch 3 by default**, on the operator's instruction:
# "you can launch the first non-baseline rollout once you confirm everything is
# ready. No need to wait for the current s5i5 to finish." That doubles the quota
# burn rate, which is why `quota_guard.sh` was extended to stop *both* runners at
# 0.95 rather than only batch 3. Set ABL_WAIT=1 to serialise instead.
if os.environ.get("ABL_WAIT") == "1":
    waited = 0
    while batch6_running():
        print(f"waiting for batch 3 to stop... ({waited}s)", flush=True)
        time.sleep(60)
        waited += 60
        if waited > 10800:
            print("batch6 still running after 3h; starting anyway", flush=True)
            break

def _install_patch() -> None:
    """Redirect workspace building through the strip. Called from `main()` only.

    **Not at module scope, and that is load-bearing.** `faithful_arm.py` imports
    this module for `strip_baselines`, then installs its *own* builder and keeps
    `_original_build = sess.build_workspace` as the thing to wrap. If importing
    here had already swapped in `build_without_baselines`, that capture would
    grab the patched function and every faithful-arm workspace would be stripped
    twice — once correctly, once against a template that no longer matches.

    Import side effects that rewrite another module's globals are exactly the
    kind of thing that is invisible until it produces a wrong number.
    """
    import athanor.ccarc3 as pkg

    sess.build_workspace = build_without_baselines
    pkg.build_workspace = build_without_baselines


def restore_banked_results() -> int:
    """Put finished results back on disk before anything decides what to re-run.

    **The skip check below reads result.json off disk, and disk is not durable.**
    A container replacement rewinds the scratchpad to an image snapshot: on
    2026-08-06 that left 13 of the 25 banked arm results missing while the games
    were long since scored. The loop would have read those as unstarted and re-run
    all 13 — one of them `sk48`, at a moment when another driver held a live ARC
    card for it, which is two solvers acting on one card and one ledger.

    This is the same ordering bug that was fixed in tools/rerun_losses.py by
    restoring before the skip check rather than after. The durable copy of every
    result lives in the trace-audit store, so that is what is read back.
    """
    store = pathlib.Path("/home/user/athanor/evidence/ccarc3/trace_audit/spans.json.gz")
    if not store.exists():
        print("    banked-result store missing; skip check is running on disk alone",
              flush=True)
        return 0
    restored = 0
    try:
        data = json.loads(gzip.open(store).read())
    except (OSError, ValueError) as exc:
        print(f"    banked-result store unreadable ({exc.__class__.__name__})", flush=True)
        return 0
    for gid, entry in data.items():
        if "@" in gid:                      # a re-run, kept under its own id
            continue
        target = RUNS / gid / "result.json"
        if target.exists():
            continue
        result = (entry or {}).get("result") or {}
        if not result.get("game_id"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1))
        restored += 1
    if restored:
        print(f"    restored {restored} banked result.json from the trace-audit store",
              flush=True)
    return restored


def main() -> None:
    """Run the arm. Importing this module must not do that.

    Without the guard, `import ablate_baselines` — which `faithful_arm.py` does,
    to reuse `strip_baselines()` rather than keep a second copy that can drift —
    plays twenty-five games. It also made a startup crash in this module fail
    every importer, which is how the `CONTROL[g]` KeyError was found: by an
    import, not by the arm.

    The monkey-patching above stays at module scope on purpose. It is what makes
    `strip_baselines` meaningful, and an importer that wants the function but not
    the patch should take the function alone.
    """
    _install_patch()
    infos = {g.game_id: g for g in list_games()}
    results = []
    # `CONTROL.get`, not `CONTROL[...]`. The queue is no longer "scored games": it is
    # the paired ones *plus* every untouched environment, and those have no control
    # by definition. Indexing directly raised KeyError on `bp35` before a single game
    # ran -- and since the runner dies at import, the supervisor would have seen no
    # runner, restarted it, and crash-looped through the whole quota window.
    #
    # Latent until the CONTROL map was correctly narrowed to non-ablation rows; that
    # fix turned a bug that could not fire into one that fired every time.
    _paired = [g for g in GAMES if g in CONTROL]
    print(f"BASELINE-FREE ARM: {len(GAMES)} games — {len(_paired)} paired against a "
          f"control ({sum(1 for g in _paired if CONTROL[g].get('won'))} of those were wins), "
          f"{len(GAMES) - len(_paired)} untouched environments with no control",
          flush=True)

    # Restore BEFORE the skip check reads disk. See restore_banked_results().
    restore_banked_results()

    for i, game_id in enumerate(GAMES, 1):
        done = RUNS / game_id / "result.json"
        if done.exists():
            try:
                prior = json.loads(done.read_text())
            except json.JSONDecodeError:
                prior = None
            if prior and not prior.get("error"):
                print(f"[{i}/{len(GAMES)}] {game_id}: already finished, skipping", flush=True)
                results.append(prior)
                continue

        # The last direct lookup in the play loop, and the queue is half-derived from
        # a ledger on disk: a game_id in results.jsonl that ARC has since dropped from
        # the public set would not be in `infos`. Safe today (all 25 resolve), guarded
        # anyway because the cost is asymmetric — a KeyError here kills the process,
        # the supervisor restarts it, and it dies on the same game forever. Skipping
        # one stale id costs one game.
        info = infos.get(game_id)
        if info is None:
            print(f"[{i}/{len(GAMES)}] {game_id}: not in list_games() any more, skipping",
                  flush=True)
            continue
        print(f"[{i}/{len(GAMES)}] {game_id} levels={info.levels} "
              f"(baselines withheld; cap {info.suggested_budget(BUDGET_MULTIPLE)})", flush=True)
        try:
            out = run_game(
                Ccarc3Config(game_id, out_dir=RUNS, budget_multiple=BUDGET_MULTIPLE,
                             wall_clock_timeout_s=7200.0),
                info,
            )
            print("   ", json.dumps(out), flush=True)
            results.append(out)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not end the arm
            print(f"    FAILED {type(exc).__name__}: {exc}", flush=True)
            results.append({"game_id": game_id, "error": str(exc)})

        done_ids = [r["game_id"] for r in results if r.get("game_id")]
        won = sum(1 for r in results if r.get("won"))
        ctrl = sum(1 for g in done_ids if CONTROL.get(g, {}).get("won"))
        print(f"    ==> baseline-free {won}/{len(results)} won | "
              f"same games with baselines: {ctrl}/{len(done_ids)}", flush=True)

    print("\n=== arm complete ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
