# AUTOPILOT — what to do when woken, user asleep

Read `docs/ccarc3_memory.md` (symlinked as `scratchpad/MEMORY.md`) first if you have no context. This file is the standing agenda.

## Every wake-up, in this order

1. **`bash scratchpad/quota.sh`** — reports EVERY window. There are at least two,
   and they behave differently:

   | window | warns at | reset | cost of exhausting it |
   |---|---|---|---|
   | `five_hour` | utilization 0.90 | ~5h | a ~2h pause |
   | `seven_day` | **utilization 0.75** | up to ~7d | **a blackout of up to days** |

   **The seven-day limit is the one that matters.** It warns earlier (0.75) and
   costs vastly more to hit. A quota check that reports only the five-hour window
   is watching the wrong thing — that mistake ran for hours here before the
   `seven_day` event first appeared and revealed it.

   | overall status | policy |
   |---|---|
   | `allowed` | full speed |
   | `allowed_warning` | **launch nothing new**; let in-flight finish; report and wait |
   | `rejected` | stop; wait for the reset of whichever window rejected |

   Overage is `rejected` / `out_of_credits` on the five-hour: exhaustion is a hard
   kill of in-flight solvers, not a slowdown. Assume the same for the weekly.

   **`status=unknown` usually means no solver has run recently, not that
   something broke.** `quota.sh` reads `rate_limit_event` lines out of *solver*
   `stream.jsonl` files, so it can only see quota while quota is being spent. A
   build-only stretch — a long proofread, a harness repair — goes stale within
   the hour and then reports `unknown`, and this session's own spend never
   appears there at all. Do not treat that as a reason to stall indefinitely, and
   do not treat it as permission either. Starting **one** game is the cheapest
   way to restore the reading; it came back `allowed` within three minutes on
   2026-08-07 after 92 minutes of `unknown`.

   **A window that has reset has *no* reading, which is not the same as a low
   one.** After a rollover the old line reads `[VOID (window reset since this
   reading)]` and nothing replaces it until an event for that window arrives. The
   overall status can therefore say `allowed` on the strength of the five-hour
   window alone while the seven-day window — the one whose exhaustion costs days —
   is entirely unobserved. Raise concurrency in steps in that state, not straight
   to the ceiling: 1 to prove the path, 2 once the five-hour reading is fresh, 3
   only once a `seven_day` event has actually landed.

2. **Is work already running?** Check by *workspace directory*, never by process
   name — this session's own CLI carries `claude-opus-5`, `athanor` and `Monitor`
   in its argv, so a name match reports phantom work (and `pkill` on one killed a
   live monitor):

   ```bash
   for pid in $(pgrep -f claude); do
     c=$(readlink /proc/$pid/cwd 2>/dev/null)
     case "$c" in *scratchpad/runs*) echo "solver: pid=$pid $c";; esac
   done
   ```

   The glob is `*scratchpad/runs*`, **not `*/runs/*`**. Batch directories are
   `runs`, `runs2`, `runs3`, ... and `*/runs/*` matches only the first — it
   reported no solver while `su15` was live in `runs3/`, which is exactly the
   duplicate launch this check exists to prevent. Verified 2026-08-02: the old
   glob returned nothing, the new one returned `pid=464 .../runs3/su15-1944f8ab`.

   `pgrep -fc "athanor cc"` is **not** a substitute: it counts this session's own
   CLI. Trust the cwd check.

   If a solver is running, report and stop. Do not launch duplicates.

3. **If nothing is running and quota is `allowed`**, take the next item below.

## While detached work runs: keep a WAKING heartbeat

A detached subprocess produces no session turns, so the platform may see an idle
session while a batch runs for an hour. A *quiet* monitor — one that only speaks
on change — saves tokens and wakes you never. Use one that fires every ~7
minutes instead, for as long as anything is running. A wake-up costs one cheap
turn; a container recycle costs a run.

Hypothesis, not fact — see MEMORY.md. The policy stands either way, and the
second half of it is the part that always pays: **make every batch idempotent**,
so a recycle costs minutes rather than the run.

## Agenda, in order

> ### ⚠ CURRENT AGENDA — 2026-08-08. Everything below this block is history.
>
> **This file now lives in the repo** (`docs/ccarc3_autopilot.md`); the
> scratchpad path is a symlink. It was scratchpad-only until 2026-08-08, which
> `tools/rehydrate_bootstrap.sh` had already argued was unsafe: "there is no
> on-disk location that reliably survives a replacement… origin is the only
> store that has never lost anything." It survived one replacement by predating
> the snapshot, which is luck, not durability — and the standing agenda is
> exactly the file whose loss or staleness causes wrong action.
>
> The wake-up prompt still recites the ARC-AGI-2 agenda ("finish the 4.8 batch
> with --resume-incomplete, land the staged harness changes, run
> final_analysis.py, collect the 3v3 ablation, then the random-sample batch").
> **Every one of those finished days ago.** They are kept below as the record,
> not as work. Do not launch any of them.
>
> **Track: CCARC3 (ARC-AGI-3). The ARC-AGI-2 queues are parked by operator
> instruction — "Let's don't run the queues 3. Let's spend the tokens on more
> important tasks—arcagi 3."**
>
> **State.** The 25-environment baseline-free arm is complete. **The clean
> rollout is complete: 25 of 25 games banked**, every run one-shot, proofread
> and card-corroborated. The publishable figure is **98.38% over 17
> environments**, 16 at E=1, `bp35` 0.7252 the only shortfall — and it is a
> floor, because every sub-1.0 clean rollout predates the win-frame replay
> prompt. The driver exits immediately with "all 25 have a clean run"; the
> supervisor relaunching it every ten minutes is a no-op, not a fault.
>
> (This block previously read "the 8-game clean rollout is complete" and "17
> environments have no result". Both were true on 2026-08-07 and neither is now.
> A standing agenda that overstates remaining work is one that gets acted on.)
>
> **1. The solver-surface repair is landed.** The exhaustive second pass and the
> twelve deep-proofread findings are closed out — see
> `docs/ccarc3_open_findings.md`, which records all twelve as confirmed and
> fixed, which three came with a wrong proposed fix, and the five further
> defects that checking them surfaced. Nothing there is outstanding.
>
> **2. THE ONE OPEN ITEM: the 25-game sweep onto a single shared scorecard.**
> `CCARC3_SWEEP_DIR=clean_rollouts_submission tools/clean_rollouts.py`, ~$652.
> A submission needs all 25 games on ONE card and **nothing already banked can be
> retro-fitted onto one** — a card is state on a single backend instance, pinned
> by its `AWSALBAPP-*` stickiness cookies. This is the only reason to spend
> quota.
>
> **Blocked on quota, not on code.** The seven-day window has been at
> util=0.83 against a 0.75 warning threshold; it resets ~2026-08-11 21:00 PDT.
> Under `allowed_warning` the policy is *launch nothing new*, and a sweep is the
> largest possible new launch. Do not start it on a `[VOID]` five-hour reading.
>
> Four fixes that each would have broken or wasted that sweep are already in:
> the corroboration guard no longer counts plays from discarded attempts;
> `sweep_card`'s refusal to remint over a card carrying finished games no longer
> reads `attempt_1` only and so no longer fails open; the mid-sweep split check
> now compares against the sweep's history instead of a value equal to itself;
> and `HIDDEN_FIELDS` has a roll-call so `baseline_actions` cannot be dropped.
>
> **3. Re-publish the audit artifact after each finish.** Same URL
> (`.../artifact/7447856a-b587-4d52-9c5c-a839de3eb6ee`), same file path.
> `refresh_audit.sh` is the hourly safety net behind the finish watcher; it
> prints nothing on a no-op and a no-op needs no narration.
>
> **Standing constraints that keep getting rediscovered:** Opus 5 only, high
> effort. No control arm. Never open a PR. Push to
> `claude/athanor-cc-harness-variant-jpqw7t`. The API key lives in
> `scratchpad/arc3/.env` and never enters the repo. Report every timestamp in
> America/Los_Angeles, stamped at the source (`CLAUDE.md`).

**DONE — do not re-run:**
- 4.8 head-to-head, both arms complete. Arm A 23.33/35 = 66.7% (21 solves);
  Arm B 9.00/9.00. Baseline on Arm A is **7.33 points, not zero** — gain is
  **+16.00**, not +23.33. Projection 108.33/120 = 90.3% vs baseline 76.9%.
- Staged harness changes landed: `arc.unreached()`, three doctrine additions,
  `--ablate`, `--resume-incomplete` (+ ledger-based skip), salvage-on-timeout,
  default timeout 5400s. 312 tests.
- **3v3 ablation: NEGATIVE.** doctrine 3/3, ablated 2/3, Fisher p = 0.500. The
  original finding does not reproduce; it was conditional on a harness state
  where the primary hypothesis reliably failed. Written up in the results log.

**FINISHED long ago — the block below said RUNNING for days after it stopped. Kept as the record; see the CURRENT AGENDA block above:**
- `strat_hard/` — 20 Arm A tasks under **Opus 5** (the 21-task eligible pool).
  Pairs against the Opus 4.8 numbers already on disk for the *same* tasks, same
  harness commit. This is a model comparison on identical hard tasks.
- `strat_easy/` — 15 tasks from the 85 the CoT baseline solves, Opus 5.
  Together these give a stratified full-split estimate:
  `(hard_rate x 35 + easy_rate x 85) / 120`.
- `harness_v2/gain/` — 3 tasks never solved under Opus 5 (`9bbf930d`,
  `0934a4d8`, `20270e3b`), baseline 0.00. `harness_v2/regress/` — 4 previously
  solved (`28a6681f`, `e8686506`, `78332cb0`, `7b5033c1`), baseline 1.00. Tests
  whether the harness changes since 08-01 help without breaking anything.

**REDIRECTED 2026-08-02 (user, verbatim): "Let's don't run the queues 3. Let's
spend the tokens on more important tasks—arcagi 3."**

The ARC-AGI-2 agenda below is therefore **parked, not cancelled**. Do not launch
items 0-3 without a fresh instruction. Current priority is CCARC3.

**CCARC3 — the live track.** Design note: `docs/ccarc3_design.md` (in the repo,
pushed). It is settled through §6 and marks its own provenance: [SDK] verified,
[PLAY] operator experience, [DESIGN] proposed.

**NO LONGER BLOCKED — the key is in `scratchpad/arc3/.env`, and the harness
works end to end.**

### State as of 2026-08-02, 23:55

**THE METRIC IS RHAE, NOT WINS.** Implemented in
`src/athanor/ccarc3/scoring.py`, validated against the rubric's published
example. Per environment,
`E = min(C, E_raw)` where `E_raw = sum(l * min(1.15, (h_l/a_l)**2)) / sum(l)`
over 1-indexed levels and `C = sum(1..k)/sum(1..n)` for `k` of `n` levels
finished; the total is the **unweighted mean over all 25 public environments**,
unattempted ones counting as zero. Consequences that keep getting forgotten:

- **The ratio is squared, then capped.** 2x the human baseline scores 0.25.
- **Winning is not automatically 1.0.** It only reaches 1.0 because `C` caps it
  there *and* `E_raw >= 1`. A sloppy win scores well below 1.
- **Losing is not automatically 0.** `tn36` reached 6 of 7 levels and scored
  **0.449**. Partial credit is real; abandoning a game early is not free.
- `h_l` is the **upper-median of humans playing for the first time**
  (docs.arcprize.org/methodology) — a discovery baseline, not a mastery one.

**Opus 5's published figure is 40.676%** (mean of the 25 per-environment scores
in `cli.OPUS5_PER_ENVIRONMENT`). An earlier **30.16%** came from a WebFetch
summariser and is wrong — the user caught it. Do not re-quote it.

**Current standing: 7.396/25 = 29.58% over 8 scored.** Behind Opus 5 by 2.77
environment-units.

**Batch 3 RUNNING** — `scratchpad/batch6.py`, log `scratchpad/batch6.log`, runs
in `scratchpad/runs3/`, the remaining 18 games cheapest-baseline-first.
**Idempotent** — a finished game is skipped. Relaunch verbatim:

```bash
set -a && . scratchpad/arc3/.env && set +a
setsid nohup .venv/bin/python scratchpad/batch6.py \
    > scratchpad/batch6.log 2>&1 < /dev/null &
```

Done so far:

- `tn36` 6/7 levels, RHAE **0.449**, $39.78 / 155 turns.
- `su15` **WON 9/9**, 168 actions vs 361 baseline (0.47x), RHAE **1.000**,
  $30.42 / 163 turns. Eight of nine levels hit the 1.15 cap; level 6 ran to
  47/31 and scored 0.435. It could have taken ~196 actions before `raw` fell
  below 1.0 — the completion cap, not efficiency, is what carries a win.
- `lp85` **WON 8/8**, 94 actions vs 388 baseline (**0.24x**), 0 deaths, RHAE
  **1.000** with `raw` at the theoretical maximum 1.15 — every level capped.
  $8.79 / 68 turns. Its level 8 carried a **159**-action human baseline and fell
  in **9**. Same inversion as `su15`'s 115-baseline level 3 in 10: the levels
  humans find hardest are not the ones that cost this solver anything.

**Standing: 9.396/25 = 37.58%, 9 wins from 10 games. Gap to Opus 5: 0.773
units — one more win passes them.** Total spend $131.66.

**BATCH DELIBERATELY STOPPED AFTER `lp85` — it did not crash.**
`scratchpad/stop_after_lp85.sh` fired correctly and has now exited:

```
lp85-305b61c3 finished; stopping batch6 before it launches another game
  killed driver 1076
  killed orphaned solver 22414 (.../runs3/vc33-5430563c)
```

Reason: the seven-day window hit **0.89** with ~27h to reset and each game costs
~0.02, so four or five more games would exhaust it — and exhausting the weekly
window is a blackout of up to days. This is the `allowed_warning` policy above,
applied. Resuming costs one relaunch; the batch skips finished games.

`runs3/vc33-5430563c/` is a **half-built workspace from the killed game 4** — no
trace, no result. A relaunch treats it as a fresh run (`resumed =
trace.exists()`), so leave it; do not "clean it up".

**`scratchpad/results.jsonl` is the durable ledger — read figures from there.**
A review subagent overwrote `runs/` and `runs2/`, destroying the traces for six
won environments; their figures survive only in that file and in
`docs/ccarc3_results.md`. `scratchpad/snapshot_results.py` appends to it every
heartbeat. **Never point a sub-agent at `runs*/`**, and note that "do not edit
any files" did *not* stop the one that replaced a directory.

**`scratchpad/heartbeat.sh` is the monitor.** Its liveness check splits
`/proc/<pid>/cmdline` on NUL and requires a whole argv element to match
`.*/batch6\.py`. **Do not "simplify" it to `pgrep -f` or a substring `case`** —
the watcher's own command line contains the script name, so it sees itself,
concludes the batch is alive and never reports the end. That bug class has bitten
three times here, once as a two-process deadlock.

**Cost model, measured over 4 runs: a run costs turns x ~$0.10.** Spread
$0.084-0.121/turn, tracking context size, which grows within a run. NOT driven
by tool output — a whole run's tool output is ~25K tokens against 5-18M input
tokens. So budget in **turns**, not actions: cd82 and sb26 took 121 vs 125
actions and differed 3.3x in cost. Expect **$3-10 per game**.

**Landed today** (all pushed, 497 tests):
- `effective_actions()` — `{action: (changed, tried)}` off the ledger, free.
- `client.pace()` / `level_pace()` — per level `(spent, baseline, ratio)`.
- Pace warning in `client.status()`, at **1.0x** (measured, see below).
- **The ls20 resume bug is diagnosed and fixed.** `_last_advanced` was not
  persisted, so the RESET-after-advance refusal stood down across the process
  boundary and a won game was replayed. Also: the server sent
  `full_reset: false` on a 6→0 transition — a level going *down* is now the
  fact, the flag a hint.
- `collect_outcome` reports `playthroughs` / `actions_final_playthrough`;
  `report` re-derives from the trace so harness fixes reach finished runs.
- Tests that cross a process boundary and check every refusal still fires, and
  that every `client.x()` / `arc.x()` in the generated workspace resolves.

**Measured, n=26 level-attempts over 5 games:** 24 of 25 cleared levels
finished at ≤0.92× baseline, median 0.52×. Nothing in the sample between 0.92×
and 3.44×, so no threshold in that range fits better than another — 1.0× was
chosen on asymmetric cost, not fit. The catch side is **n=1**; do not claim it.

### Open decisions left with the user — do not resolve these unilaterally

1. **The action budget is 40% of the official one.** ARC terminates an agent
   after **5n actions per level** (technical report); batches run at
   `BUDGET_MULTIPLE = 2.0` of the whole-game baseline. `level_budget_multiple`
   = 5.0 now implements the per-level rule in `client`, but the *total* cap is
   still 2.0x. Restarting batch 3 at the official budget would make results
   comparable to the leaderboard — and would cost far more. The batch is
   idempotent, so the option stays open. **User has not answered.**

2. **Play selection: does the scorecard score the first winning play or the
   best one?** If the *best* play scores, explore-then-replay raises the score
   and doctrine §0a ("DO NOT DO THIS") is wrong. If the *first winning* play
   scores, replay is pure waste. Gates both the replay question and whether
   terminating at first WIN is safe.

   **Measured 2026-08-03: scorecards are ephemeral.** Two `card_id`s from runs
   finished the same day both returned `404 card_id not found` while the same
   key still listed all 25 games. **No finished run can ever answer this** —
   including `ls20`, the only one that ever produced two plays.

   `collect_outcome` now snapshots `scorecard.json` beside the trace
   (`29280b0`), so every future run keeps it. **The first attempt at this
   (`9de0e1b`, in `ArcClient.close()`) was inert** — `close()` is reachable only
   from `__exit__`, the generated `session.py` never closes, and solvers do
   `from session import client`. Check the trigger fires on the real path, not
   just that the function is correct. **The experiment, cheap, run it on the smallest
   game:** clear a level inefficiently → RESET at action-counter-zero to open a
   second play → clear the same level efficiently → finish → read the saved
   scorecard and see which play the reported figure follows.

   That file also gives the first real check on `scoring.py`, which derives
   `actions_by_level` from the trace and has never been compared against the
   server's own numbers.

### >>> `ablate.log` MISCOUNTS: it says 8/9, the truth is 8/8 <<<

A session-worker restart on 2026-08-03 SIGTERMed the in-flight `sc25` solver.
`collect_outcome` still wrote a `result.json` — `won: false`, `levels_reached: 0`,
**`actions_used: 5`, `duration_s: 137`, `exit_code: 143`** — and the arm counted
it as a loss before anything could intervene. 143 is 128+15, a kill, and five
actions in two minutes is not a run.

The workspace is quarantined at
`ablate_nobaseline/.sc25-KILLED-by-worker-restart` rather than deleted, so the
arm re-runs `sc25` on the next relaunch: the idempotent skip keys on
`result.json`, which no longer exists at the live path.

**`ablate.log`'s running tally is wrong from that point on and cannot be
repaired** — it is append-only and the line is already written. `results.jsonl`
and `docs/ccarc3_results.md` are authoritative; both show **8/8**, and the killed
row never entered the ledger.

Generalisation worth keeping: **a `result.json` is not proof a game was played.**
Check `exit_code` (>128 means a signal), and treat a run with a handful of
actions and a short duration as infrastructure, not evidence. AUTOPILOT's own
rule already said an error is only a real loss if it was not accepted; this is
the same rule applied to a signal rather than an exception.

### >>> STANDING CONSTRAINT (operator, 2026-08-03) — BASELINE-FREE ONLY <<<

**"All runs after the 13th scored game must be baseline free."**
**"No control arm from now on!"**

So: **do NOT relaunch `batch6.py`.** It builds workspaces that hand the solver
`baseline_actions` in `session.py` and `CLAUDE.md`, and it was stopped mid-game
for exactly that reason — `bp35` was killed 18 actions in and its workspace moved
to `runs3/.bp35-abandoned-with-baselines`.

The only sanctioned runner is **`scratchpad/ablate_baselines.py`**, which strips
`baseline_actions`, the CLAUDE.md rows and doctrine §6/§6a from every workspace
it builds, and verifies the strip before launching. Relaunch verbatim:

```bash
set -a && . scratchpad/arc3/.env && set +a
setsid nohup bash tools/supervisor.sh >> scratchpad/supervisor.log 2>&1 < /dev/null &

**`tools/supervisor.sh`, never `scratchpad/supervisor.sh`.** This block said the
scratchpad path until 2026-08-07, and following it started the Aug-3 copy: no
proxy validation at all, and it starts `ablate_baselines.py` rather than
`clean_rollouts.py`, so it launches the wrong experiment into a proxy it never
checks. That is the configuration behind the never-explained incident where seven
games churned on `Connection refused`. The stale copy is renamed
`supervisor.sh.STALE-DO-NOT-RUN`; the maintained one is in the repo, where it is
version-controlled and gets the fixes.
```

**Launch `supervisor.sh`, not the runner directly, and not `quota_guard.sh`.**
The supervisor starts `ablate_baselines.py` itself and is bidirectional: it stops
the arm politely at util ≥ 0.98 (waiting for every started game to have a
`result.json` first) and restarts it when the window recovers below 0.90, or
immediately if the runner exited on its own with util under 0.98. `quota_guard.sh`
is the retired one-directional version — it stops and exits, so resuming needed a
human. It also still watches `batch6.py`, which is retired.

**The queue is 25, not 13.** `scored_games()` derives it live: every game with a
non-ablation row in `results.jsonl` first (paired observations are worth more
under a quota ceiling), then the untouched environments cheapest-baseline first.
Controls are read only from rows whose `batch` is **not** `ablate_nobaseline` —
without that filter the arm's own results overwrite the controls they are meant
to be compared against and every pair reads as a tie. Anything that plays a game
must go through `strip_baselines()`.

The 14 baseline-ful results already on disk stay as historical record — they are
what the 53.58% figure is built on, and `docs/ccarc3_results.md` now needs to say
that no further runs will be comparable to them.

### >>> TWO FLAGS SHIPPED OFF, TO FLIP WHEN THE ARM COMPLETES (2026-08-03) <<<

Both landed in `38cedb2`, both default `False`, both need one line added to the
workspace template in `src/athanor/ccarc3/session.py` to take effect.

1. **`ArcClient.hide_baselines=True`** — the *correct* way to run the
   baseline-free arm. The current arm blanks `GameInfo.baseline_actions`, which
   is the same array `level_budget` derives ARC's official 5n per-level
   termination rule from, so hiding the numbers also **switched off the rule**.
   `su15` ran a 31-baseline level to 268 actions (8.65×) and an 8-baseline level
   to 182 (22.75×); ARC would have terminated it at 155 and 40. The arm is
   therefore two-variable and more permissive than the real benchmark.
   `hide_baselines` silences every solver-facing surface through one gated
   property while `level_budget` keeps reading the real value.

2. **`ArcClient.show_score=True`** — `status()` gains `score_now`,
   `score_ceiling` and `completion_cap`, and names `restart_for_replay()` when
   the ceiling drops below 1.000. `completion_cap` needs no baselines, so it
   works in either arm.

3. **A doctrine paragraph on replaying after a full clear.** Staged, not applied
   — the operator has an open question on it (asked 2026-08-03, unanswered).

   §0a covers replaying *mid-run* ("use it when you were slow") and says nothing
   about the case where every level is cleared. `E = min(cap, raw)`; clear
   everything and `cap` is 1.0, so `E` **is** `raw`, and a run that finished at
   `raw` 0.98 is sitting on score it could still collect. `sp80` did exactly
   that: 6/6 cleared, `raw` 0.9785, **71% of its budget unspent**, stopped. That
   0.0215 is its entire deficit against its control.

   Worse for this arm, the one §0a line that *does* address a full clear is dead
   here: *"If your levels came in under baseline ... a replay gains you exactly
   nothing"* cannot be evaluated without baselines, and the check it names —
   `arc.score_run(client.transitions(), baselines)` — takes a variable the
   solver has not got. The prompt reinforces the stop: *"win as many of its N
   levels as you can"*, goal met at N.

   Verified before staging (`622f079`): the replay **is** available after a win.
   17 of 17 winning traces sent the final advance and WIN in one frame, so
   `_last_advanced` is true and `restart_for_replay()` is permitted; no run has
   ever used it. One dead end exists — an action *after* the winning one clears
   the flag and locks the score in — but it is unreachable against the live
   server.

   Text to append to §0a when applying:

   > **Clearing every level is not the end of the scoring.** `E = min(cap,
   > raw)`. Once you have cleared them all, `cap` is 1.0 and your score *is*
   > `raw` — so any level you finished above its baseline is still costing you,
   > and a replay is worth `1.0 - raw`. The replay is available at exactly that
   > moment: the winning frame is a level advance, so the action counter is
   > zero and `restart_for_replay()` is legal. One action later it is not.
   >
   > **If you cannot compute `raw`, replay anyway when you have the budget.**
   > Without baselines you cannot tell whether you finished at 1.15 or 0.90, so
   > "I won" and "I scored what this game was worth" are different claims and
   > you can only verify the first. A replay that executes the route you now
   > know costs actions you were not going to spend and cannot lower your score.

4. **Two stale statistics in the shipped doctrine, §0a.** Factual corrections,
   but behaviourally loaded, so staged with the rest.

   - *"of ten runs on record, eight were already at the cap"* → recomputed over
     every trace: **19 runs, 17 cap-binding, 2 raw-binding.** The claim's shape
     holds and gets stronger (89% vs 80%).
   - *"On the one run this project has lost, that is 0.750 − 0.449 = +0.301"* →
     `tn36` is no longer a loss. The baseline-free arm scored **1.000** on it,
     7/7 in 220 actions. Leaving this is worse than a stale number: it presents
     as an unrecoverable loss the one environment this harness has since
     maxed, and the true story argues *harder* for replaying than the text does.

   Suggested replacement for the second bullet:

   > - If overruns dragged your raw score below the cap, a replay is worth
   >   `cap − raw`. `tn36` is the worked case: one run finished at `raw` 0.449
   >   against a `cap` of 0.750, having cleared six of seven levels — and a
   >   later run of the same environment cleared all seven in **220 actions
   >   against that run's 631**, scoring 1.000. The routes were the same. What
   >   differed was spending them instead of finding them.

**Do NOT flip any of the four mid-arm.** Twelve of the arm's twenty-five games
are banked without them; a variable that changes halfway makes two
half-experiments. The faithful configuration is a **separate arm**, run after
this one finishes.

Counter-argument, recorded so it can be weighed rather than rediscovered: of the
13 games left, only 2 (`sc25`, `s5i5`) are paired observations — the other 11 are
untouched environments with no control, so a mid-arm change would contaminate
far less than it first appears. It would still split the arm's internal
consistency, which is why it is staged rather than done.

Written up in `docs/ccarc3_results.md` (§ `su15` — the arm's first lost game) and
`docs/ccarc3_design.md` (the confound, plus the general principle: *an ablation
must remove the information, not the machinery that happens to read it from the
same place*).

### >>> FIXED 2026-08-03: the parked arm could never have restarted itself <<<

**Symptom to recognise if it comes back:** `quota.sh` reports a `seven_day` line
whose `age=` keeps climbing while `util=` never moves.

`quota.sh` derives its readings from `rate_limit_event` lines in solver
`stream.jsonl` files, so **it only refreshes while a solver is running.** With
the arm parked that is a closed loop:

```
no solver -> no fresh reading -> util stays 0.98 -> supervisor never starts
          -> no solver
```

And the weekly reset does not break it. When `resetsAt` passes, `quota.sh` marks
the window `VOID` in its note but still **prints the stale `utilization` field**
— the verdict changes, the number does not. So `util_now()` would have kept
extracting 0.98 for ever, and the arm would have sat out its remaining 13 games
with a completely free quota window.

Fixed in both `supervisor.sh` and `parked_watch.sh`: a line containing `VOID`
reads as `util=0.00`, because VOID means `resetsAt` is in the past, which means
the window rolled over. Parser checked against all five states quota.sh can
emit (fresh/at-ceiling, stale-but-binding, VOID-with-a-number,
VOID-with-`<threshold`, fresh-and-healthy).

The old supervisor was killed and relaunched, because it held the old function
in memory and editing a running bash script is unsafe anyway — bash reads by
byte offset. Safe to do only because no runner was in flight; **never restart
the supervisor while a game is running** (see `stop_politely`).

**The general lesson, worth more than the fix:** the guard and the thing it
guards shared a data source that only one of them could refill. Any watchdog
whose input is produced by the work it gates is a deadlock waiting for the
quiet period.

### >>> The unattended restart chain is now tested end to end (2026-08-03) <<<

This note used to say *"the supervisor's START branch has never fired live"*.
It has now been exercised, link by link, without spending any ARC quota:

| link | how it was checked | result |
|---|---|---|
| `util_now()` reads a VOID window as 0.00 | all five states quota.sh can emit | correct |
| decision loop picks START at the reset | loop body replayed with stubs, 7 states | correct |
| `start()` actually launches | `RUNNER` swapped for a stub | launches |
| ...detached, keyed, right cwd | `sid == pid`, `ARC_API_KEY` set, cwd `/home/user/athanor` | correct |
| queue derives to 25 with real controls | `scored_games()` run directly | correct |
| `strip_baselines()` on a fresh workspace | built one, stripped it, checked for leaks | correct |

The decision loop was checked against the sequence it will actually walk
tonight — *hold at ceiling → START at 04:00Z → leave the running arm alone* —
and against the regressions: ceiling hit mid-queue stops politely, a post-ceiling
stop honours the 0.90 hysteresis, and a non-numeric reading does nothing.

**The one link still untested is the real reset**, because it cannot be
simulated: everything above used synthetic quota output. A `send_later` check is
scheduled for **04:12Z** (trigger `trig_01Wj9bdMxc3ArX9bQRwujjN7`) to confirm the
VOID path fires against a genuine rollover rather than assuming it does.

### >>> FIXED 00:55Z: the runner would have crash-looped at 04:00Z <<<

`ablate_baselines.py` **died at import** with `KeyError: 'bp35-0a0ad940'`, before
a single game. The supervisor would have started it at the reset, watched it
exit, seen no runner with quota under the ceiling, started it again — a tight
crash loop across the entire window, zero games played, and no obvious symptom
beyond a growing `ablate.log`.

Line 199 read `CONTROL[g]` while `GAMES` is no longer "scored games": it is the
14 paired ones **plus 11 untouched environments that have no control by
definition**. Now `CONTROL.get`, and the header reports the split honestly.

**This was self-inflicted and worth remembering as a shape.** The earlier fix
that narrowed `CONTROL` to non-ablation rows was correct — without it the arm
graded itself and every pair read as a tie — but narrowing the map turned a
latent `KeyError` into a certain one. *A fix that makes a collection smaller
turns every unchecked lookup into a live bug.* Grep for `[` indexing on anything
that fix touched.

Verified after the change: startup clean, queue 25 (14 paired / 13 of those
wins, 11 untouched), resumes at `[9/25] sc25-635fd71a` skipping the 8 finished.

Caught only because `faithful_arm.py` tried to `import` the module and inherited
its crash — the runner has no `if __name__ == "__main__"` guard, so importing it
runs it. Nothing else would have looked at it before 04:00Z.

### >>> What survives a restart — and which restart (measured 2026-08-03 and 2026-08-07) <<<

**There are two different events and this file used to call both "a container
restart".** They have opposite consequences, and the advice below was written
from the mild one.

| | session-worker restart | VM replacement |
|---|---|---|
| what changes | the `claude` worker process | the whole Firecracker microVM, new kernel |
| `CLAUDE_CODE_WORKER_EPOCH` | increments | increments |
| `pid 1` start time | unchanged | **new** |
| `setsid nohup` work | **survives**, same pid | **dies**, nothing survives |
| Monitor tasks | die | die |
| files on disk | survive | **survive** |

**Telling them apart takes one command.** A worker restart cannot change pid 1;
only a new kernel can.

```bash
ps -o lstart= -p 1          # VM boot time. New value => VM was replaced.
cat /proc/uptime            # same answer, in seconds
echo $CLAUDE_CODE_WORKER_EPOCH   # increments on BOTH, so it cannot distinguish
```

**2026-08-03, 21:48Z — a worker restart.** `supervisor.sh` came through with
**pid 1794 unchanged**, having run two hours. That is what produced the standing
advice "put durability in a detached process, not in a monitor". A preserved pid
is proof the kernel did not change, so this was a worker restart mislabelled as
a container one.

**2026-08-07, 18:55:04Z — a VM replacement, and nothing local survived.** `pid 1`
is `/process_api --firecracker-init` and it started at 18:55:04, in the same
second as `kthreadd`; the oldest non-kernel process on the box afterwards is pid
1 itself. The supervisor was **dead**, and so was a running Workflow that had
completed 4 of 7 agents. Worker epoch went 65 to 66.

  * previous container lifetime: worker spawned **15:16:05Z**, VM booted
    **18:55:05Z** — **3h39m**.
  * no game was running. An audit Workflow was, with six concurrent agents;
    its last write was 18:54:23, 41 seconds before the boot. Timing is
    established, causation is not — do not write "load caused it" into this file
    without a measurement.

**Restarts do not track game activity.** The eight banked rollouts once ran
**10.78 hours continuously on one `boot_id`** with solvers going; this one
happened with the game queue empty. Lifetime here is variable and not something
to plan around in either direction.

**The disk outlives the VM.** Both the scratchpad and the repo checkout are on
`/dev/vda` mounted at `/`, so files persist across a replacement even though no
process does — `supervisor.log` holds lines from 00:15Z on a box that booted at
18:55Z. That is why this file is still here to be read after one.

So, in order of what actually survives:

| durability | survives worker restart | survives VM replacement |
|---|---|---|
| commit **and push** to git | yes | yes |
| files in the scratchpad / repo | yes | yes |
| server-side `send_later` triggers | yes | yes |
| `setsid nohup` detached work | yes | **no** |
| Monitor tasks | **no** | **no** |

Three consequences:

- **A detached process is durable against the common event and not the severe
  one.** The supervisor is still the right home for work that must continue, but
  every wake-up must check it is alive rather than assume it — see the launch
  block above, and use the argv-exact form.
- **Anything that must cross a VM replacement has to be on disk or server-side.**
  A Workflow is neither: it died mid-run here. It is resumable —
  `Workflow({scriptPath, resumeFromRunId})` returns completed agents from cache —
  but only if a session notices and resumes it.
- The `send_later` checks are the only local-state-free guard. Keep them.

### >>> Scheduled checks covering tonight (2026-08-03) <<<

Monitors die every ~30 minutes here. Detached work is longer-lived than this
line used to claim — "reaped every 10-50 minutes" is contradicted by a 10.78-hour
continuous stretch on one `boot_id` and by the 3h39m container measured
2026-08-07 — but it dies outright when the VM is replaced, so neither is a safe
guard for an 8-hour park. `send_later` is server-side and
survives container restarts, so the night is covered by three fixed points:

| when (UTC) | trigger | job |
|---|---|---|
| 00:30 | `trig_01G8maziCXEfbDg9FWVNLhCk` | is `supervisor.sh` alive? relaunch if not |
| 02:40 | `trig_013Sn4SqZ3uNsQ6ZCunCrhPY` | same, last look before the reset |
| 04:12 | `trig_01Wj9bdMxc3ArX9bQRwujjN7` | did the VOID path fire and the arm restart? |

**The supervisor is the single point of failure.** If it is reaped, nothing
restarts the arm at 04:00Z and the whole window is wasted — which is the only
reason the first two checks exist. They deliberately do nothing else: the
restart chain is already tested end to end (table above), so re-verifying it
each time is churn.

None of them may launch the runner by hand before 04:00Z; quota is 0.98 until
then and the supervisor is the thing that decides.

### >>> NEXT WAKE-UP: THE ONE THING TO CHECK <<<

**Operator raised the pause threshold twice on 2026-08-03: first to 0.95, then
to 0.98.** Run until the seven-day window exceeds **0.98**, not 0.75. *"Don't worry about the weekly limit. You don't
have to pause until that becomes >0.95"* and *"If one game is still running when
it crosses 0.95, please just let it finish"*. `scratchpad/quota_guard.sh` implements
exactly that — it waits for every started game to have a `result.json` before
killing the runners, and it watches **both** batch 3 and the baseline-free arm.

**At 0.98 the margin is 0.02, about one game.** Exhausting the weekly window is
a hard kill of in-flight solvers, not a slowdown, so a game that starts near the
line may be lost outright. That is the operator's call and is recorded here as
the accepted risk, not as an objection.

**Detached work gets reaped, roughly every 10-50 minutes.** On 2026-08-03 the
batch driver, the quota guard and the heartbeat all died together mid-game while
this session survived; disk was at 24% and memory nearly free, so it was not
resource pressure. **So the standing wake-up action is: if nothing is running
and quota is under 0.95, relaunch — the batch is idempotent and a game with a
trace resumes rather than restarting.** Relaunch all three:

```bash
set -a && . scratchpad/arc3/.env && set +a
setsid nohup .venv/bin/python scratchpad/batch6.py >> scratchpad/batch6.log 2>&1 < /dev/null &
setsid nohup bash scratchpad/quota_guard.sh >> scratchpad/quota_guard.log 2>&1 < /dev/null &
```
then re-arm the heartbeat (`Monitor` over `scratchpad/heartbeat.sh`).

Historical note — the old rule, kept because the reasoning still applies below
0.95: if `quota.sh` reports the seven-day window back to **`allowed`**, relaunch
batch 3 — that is the whole action:

```bash
set -a && . scratchpad/arc3/.env && set +a
setsid nohup .venv/bin/python scratchpad/batch6.py \
    >> scratchpad/batch6.log 2>&1 < /dev/null &
```

then re-arm the heartbeat (`Monitor` over `scratchpad/heartbeat.sh`). It skips
the four finished games and starts at `vc33`. If still `allowed_warning`, do
nothing and say so — the window was 0.89 with ~27h to reset as of 01:05.

Do **not** arm a long-running monitor to wait for the window: `Monitor` caps at
30 minutes regardless of `persistent: true` (observed repeatedly), so a 27-hour
wait would cost a re-arm turn every half hour and buy nothing. The external
autopilot wake-ups already poll often enough.

**Next, in order, once quota allows:**

1. Continue batch 3 — 14 games remain after `vc33`.
2. **An ablation, not more games, is the higher-value experiment.** §9.8a says
   solvers ignore the rule engine, the forward model and the planner entirely —
   seven runs, zero calls — while six of seven won. Run the same game twice:
   full workspace vs one carrying only what is demonstrably used (`client`, the
   gate, the trace, `render`/`diff`/`png`/`objects`), doctrine sections cut to
   match. If results hold, ~40% of the doctrine is free context.
   **Caveat that killed the last ablation:** the 3v3 doctrine ablation came back
   p = 0.500. These are underpowered at n=3. Pair on the same games and expect
   to need more runs than feels necessary.
3. **Test the revisit signal** (shipped 2026-08-03, `b416582`). `status()` now
   reports how many actions returned the board to a state already seen on this
   level. It is the first thing found that separates the one loss from the wins,
   and it is not the no-op tally (`tn36` L5: 8 no-ops in 309 actions, 95
   revisits) nor pace (`su15` L5 and `tn36` L3 both 1.52x, both cleared, 2% vs
   8%; `tn36` L5 was 5.62x and 31% and never fell). n=22 level-attempts, one
   loss, and level length is a confound. **Falsifiers to watch for in the
   remaining games:** a level lost with near-zero revisits, or a long level
   cleared at 30%+. Recorded as provisional in `docs/ccarc3_design.md`.

4. **CORRECTION — the 5n per-level cap is not a cost lever.** Stated twice that
   it gives "early abandonment of hopeless levels". It does not. ARC-AGI-3
   levels are **sequential**: being cut off on level L ends the environment and
   forfeits every level after it, which is what the client's own refusal says.
   Counterfactual on `tn36`: level 5 needed 309 actions and 5n allows 275, so
   under the official rule it would have been terminated there and scored
   **0.4419 instead of 0.4487**. Matching the rule is about comparability with
   the leaderboard, not about spending less or scoring more.

   The real lever `tn36` points at is recognising overspend *on the level you
   are on*: 185 actions on a 72-baseline level and 309 on a 55-baseline one left
   **29** actions for a final level with a baseline of 62. It reached the last
   level and could not afford it.

Historical note follows.

Previously blocked on: **no `ARC_API_KEY`.** The live API 401s without it;
`arcengine` authors games locally with no key (`scratchpad/arc3/toy_game.py`
drives RESET -> actions -> WIN). So until a key exists:

**READY TO FIRE — resume ls20 to finish levels 5-6.** The first run stopped on a
deliberately halved cap at 5/7 levels. `scratchpad/resume_ls20.py` raises the cap
to 2.0x and keeps the trace, scorecard and rule book, so the 275 actions already
paid for are not repaid. **Gated on quota only** — it was `allowed_warning` at
0.80 when written. Fire it when the seven-day window recovers, or if the user
says to spend into the warning:

```bash
set -a && . scratchpad/arc3/.env && set +a
setsid nohup .venv/bin/python scratchpad/resume_ls20.py \
    > scratchpad/ls20_resume.log 2>&1 < /dev/null &
```

**Launch a run (detached — a plain nohup dies with the shell that spawned it):**

```bash
set -a && . scratchpad/arc3/.env && set +a
setsid nohup .venv/bin/python scratchpad/launch_ls20.py > scratchpad/ls20_run.log 2>&1 </dev/null &
```

Check with `athanor ccarc3 trace --run <dir>`. Never `pkill -f claude-opus-5` —
it matches this session's own process. Target solvers by their workspace cwd.

- Buildable now, no key needed: `arc3.logical()`, `arc3.render()`, `arc3.diff()`,
  `arc3.objects()`, `arc3.trace()`, the `trace.jsonl` writer, three-valued
  predicates, `verify()` / `survey()`. All testable against `arcengine` games.
- Needs a key: everything in §7 of the design note. §7.1 (does RESET after
  GAME_OVER restart the game or the level?) gates the whole over-explore-L1
  doctrine and should be the first thing measured.

**PARKED ARC-AGI-2 agenda — do not launch without a fresh instruction:**
0. **EFFORT ESCALATION (user-requested).** Everything so far is `--effort high`.
   Rerun the tasks that FAIL at high with `--effort max`, paired: same task, same
   model (`opus`), same commit, only effort differs. Two sources of failures:
   - `strat_hard/` results scoring < 1.00 -> rerun at max into `effort_max/hard/`
   - `harness_v2/gain/` results still failing -> rerun at max into
     `effort_max/gain/`, giving a three-rung ladder on the same 3 tasks
     (old harness+high -> new harness+high -> new harness+max), which separates
     harness change from effort change instead of moving both at once.
   Also run 3-4 tasks that SOLVED at high under max, into `effort_max/control/`:
   more thinking is not automatically better and a regression arm is the only
   way to see it. Report paired (task-by-task), not as two pooled rates — n is
   small and the pairing is the whole point.
   Cost note: max is expected to be materially dearer per run than high; check
   `quota.sh` before launching and prefer the gain set if budget is tight.

1. Compute the stratified estimate and write it up. Report the two strata
   separately with their weights; never a bare pooled number.
2. Task #21 — hedge pattern across all runs, from data already on disk, no new
   runs. Does "selection-rule rivals win, edge-case rivals lose" hold? Currently
   5 instances across 2 models.
3. If quota allows, extend the easy stratum — 68 eligible tasks remain, and the
   90.3% projection's weakest term is Arm B's zero-regression measured on 9.

## Hard constraints (user-issued, verbatim)

- **CCARC3 solver runs: Opus 5, effort `high`. ONLY.** — *"only use opus 5. Don't
  use opus 4.8 or fable 5"* + *"Use high effort when running opus 5 in ccarc3
  harness."* No mixed-model, no mixed-effort arms on ARC-AGI-3.
- **"Don't touch b6f77b65. It's buggy"** — exclude from every batch and count.
- **"please don't write this skill into the athanor repo. just treat it as a
  command"** — handoff/session files stay in the scratchpad.
- **"we should give the CCARC agent whatever CC provides except for internet
  access"** — full tool surface; deny only research / network-by-another-door /
  escaping-the-run. Do not narrow it on the sub-agent evidence: at n=2 that is
  confounded with run length, and long runs fail regardless.
- No PR unless explicitly asked. Push only to
  `claude/athanor-cc-harness-variant-jpqw7t`.

## Reporting rules that matter

- **Always restate both arm definitions.** Arm A = the 35 tasks CoT-4.8-high
  **fails**, baseline 0.00/task, measures **gain**. Arm B = 10 of the 85 it
  **solves**, baseline 1.00/task, measures **regression**. Model
  `claude-opus-4-8`, effort `high`, both arms.
- **Never sum the arms.** Tasks were selected by their baseline outcome.
- A run is an infrastructure loss only if `error` AND NOT `accepted`. A resumed
  run can carry a dead launch's error and still be a real result — `446ef5d2`
  scored 1.00 that way.
- **Do not report a threshold.** Four have dissolved tonight: the $3.08 cost
  cutpoint, the 5-minute idle window, the coverage gate, and the confidence
  split. Report rank statistics, or wait for n.

## Report times in Pacific

Operator instruction, 2026-08-07: **"Please use pacific time from now on."**

The box runs UTC and every artifact on disk is UTC — trace mtimes, `date -u` in
the logs, git committer dates, the audit page's build stamp. Do not change any of
that: rewriting stored timestamps would make old evidence disagree with new, and
the whole point of the ledger is that two readings of the same run agree.

Convert at the point of *reporting* instead:

    TZ=America/Los_Angeles date '+%H:%M %Z'
    TZ=America/Los_Angeles date -d @<epoch> '+%H:%M %Z'
    TZ=America/Los_Angeles date -u -r <file> '+%H:%M %Z'   # file mtime, in PT

PDT is UTC-7 (Mar–Nov); PST is UTC-8. Print the zone abbreviation so a reader can
tell which, rather than leaving a bare number that is wrong half the year.

### Reconciling the 0.75 warning with the operator's 0.98 ceiling

Hit for real at 08:22 PDT 2026-08-07, with the first fresh `seven_day` reading of
the night: `util=0.75, reset_in=84h37m`.

The status table above says `allowed_warning` means *launch nothing new*.
`supervisor.sh` carries `LIMIT=0.98`, the operator's ceiling, raised from 0.95 on
2026-08-03. Both are real instructions and they disagree at 0.75.

**The 0.98 ceiling governs the automated stop; the 0.75 warning governs how fast
to spend.** Stopping dead at 0.75 would idle 23% of a weekly window for three and
a half days, which is not what raising the ceiling to 0.98 was for. But the
warning is the only advance notice there is, and the seven-day window is the one
whose exhaustion costs days rather than hours.

So at `seven_day` warning: **do not kill in-flight work, do not raise
concurrency, and lower it one step.** Lowering only gates new starts — it never
touches a running game — so it throttles the burn without discarding anything
already paid for. Re-check at each wake-up; if util climbs toward 0.90, drop
again. The supervisor's 0.98 stop stays as the backstop it was always meant to be.
