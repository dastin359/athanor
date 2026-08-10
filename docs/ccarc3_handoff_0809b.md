# CCARC3 handoff — 2026-08-09, session B

**Written for a session that has only this repository.** Everything needed to
continue is here or reachable from here. The scratchpad does not survive; assume
it is empty.

Branch: `claude/athanor-cc-harness-variant-jpqw7t`. **Never open a PR.**

---

## 0. The sixty-second version

- The harness has had a full mutation audit. **1366 tests pass.** Every on-path
  module has now been audited, including the three solver-facing ones and the
  card-facing half of `scoring.py` (§3) — 120 mutants. The battery is
  re-runnable: `tools/mutation_battery_ccarc3.py`.
- The `bp35` validation run **finished** and banked nothing — both attempts were
  discarded by two different guards, both correctly (§1). No run is in flight.
- Weekly quota is at **0.98** and resets **2026-08-10 21:00 PDT**. Until then
  there is almost nothing to spend; the supervisor is braked at its 0.98 ceiling.
- The big outstanding job is **the 25-game sweep onto one shared
  scorecard** (§8). It is blocked on quota, not on the code.
- The single most important habit: **when you think you have finished auditing,
  you have not.** See §4.

---

## 1. State at handoff

| | |
|---|---|
| HEAD | `4fa7811` (plus preserver commits after it) |
| Tests | 1366 passed, 5 skipped, 1 xfailed |
| Weekly quota | `util=0.98`, resets **2026-08-10 21:00 PDT** |
| Daemons | `supervisor.sh`, `heartbeat.sh`, `preserve_evidence.sh` and `daemon_watchdog.sh`, all ppid 1 |
| Banked | 25/25 games in `clean_rollouts`, one card per game |

### The validation run — finished, nothing banked

`bp35` was run twice in `clean_rollouts_validate` on shared card
`ac0e1272-4d0e-4d7e-9f9a-e108044123d5`, and the sweep ended
`NO clean run after 2 passes`. **$50.29 spent, 4h53m of solver time, nothing
banked locally.** Both attempts were discarded, by two *different* guards, and
both discards were correct:

| | levels | actions | duration | cost | why discarded |
|---|---|---|---|---|---|
| attempt 1 | 6 / 9 | 641 of 3255 | 154 min | $23.74 | **gave up** with 80% of budget unspent — "not a result" |
| attempt 2 | 7 / 9 | 639 of 3255 | 139 min | $26.55 | **signal 15**, exit 143 — interrupted |

What it established, which is worth more than the $50:

* **The give-up guard works on a live run.** `c25b93d` had never fired outside a
  test. It fired, refused to bank a non-result, and re-ran the game.
* **Level 6 is not a capability wall.** Attempt 1 stopped there; attempt 2
  cleared it and reached level 7 with 81% of its action budget still unspent. So
  attempt 1's stop was a genuine §0b failure, cleanly separated from "the game is
  too hard" — which is exactly the separation `clean_rollouts` exists to make.
* **A session-worker restart kills a `claude` solver but not a shell daemon.**
  This is the important one. `preserve_evidence.sh`, `supervisor.sh` and
  `heartbeat.sh` — all `setsid`, all ppid 1 — survived the restart that killed
  the solver. The driver survived too, long enough to observe its child die and
  finish its passes. **`setsid` + ppid 1 is not protection for a solver.** The
  supervisor was not responsible: its log shows no polite stop, and the lifted
  ceiling held.
* **`MAX_PASSES=2` was too tight** to absorb one infrastructure event. For a
  sweep whose games take hours, the retry bound has to exceed the expected number
  of worker restarts over the sweep's wall clock, or a single restart ends the
  game.

**On the card, `bp35` stands at 7 of 9** — `levels_completed [4, 6, 4, 7]`, and
ARC scores the max. The best play is the one we locally refused to bank. See the
card-versus-discard trap below; this is that divergence, live.

### The in-flight run

A single-game validation of the repaired harness, launched on explicit operator
instruction at 05:35 PDT:

```
CCARC3_ONLY=bp35 CCARC3_SWEEP_DIR=clean_rollouts_validate CCARC3_MAX_PASSES=2
```

- Shared scorecard: **`ac0e1272-4d0e-4d7e-9f9a-e108044123d5`** (recorded in
  `evidence/ccarc3/sweep_card/history.jsonl`).
- `bp35-0a0ad940`, 9 levels, cap 3255 actions. Prior from the bank: 990 actions,
  4.86 h, $79, 9/9.
- The supervisor was restarted with **`CCARC3_QUOTA_LIMIT=0.999`**, on operator
  instruction, so the run is not stopped at the 0.98 ceiling. **Above ~0.98 the
  brake is the API's own rate limiting, which does not stop politely.**

**The ceiling has been put back.** `CCARC3_QUOTA_LIMIT=0.999` was a one-run
decision; the supervisor was relaunched at 10:53 PDT with no override and is at
the default 0.98 / 0.90 again (pid 28063, ppid 1). Nothing to do here — recorded
so nobody goes looking for a lifted ceiling that is no longer set.

---

## 2. Standing constraints — do not relitigate these

1. **Never open a pull request.** Push to
   `claude/athanor-cc-harness-variant-jpqw7t` only.
2. **No control arm.** Every run after the 13th scored game is baseline-free.
3. **Opus 5 only**, high reasoning effort, for solvers. Not Opus 4.8, not Fable 5.
4. **The solver gets everything Claude Code provides except internet access.**
   The proxy allowlist is the mechanism; `/api/games` is excluded from it
   deliberately because it returns `baseline_actions` for all 25 environments.
5. **The API key lives in `scratchpad/arc3/.env`, mode 600, never in the repo.**
6. **Report every timestamp in America/Los_Angeles, stamped at the source.**
   Storage stays UTC; only reporting converts. See `CLAUDE.md`.
7. **Do not touch `b6f77b65`.** It is buggy.
8. Sub-agents are permitted. An earlier "do not delegate" instruction was
   removed and must not be reinstated.

---

## 3. What this session did

Twelve substantive commits, `118d0ac..4870465`. Every fix is mutation-verified:
the mutant that reintroduces the bug is confirmed to fail the new test.

### Reporting and the audit page
- **`118d0ac`** — `live_games()` aged a running solver from a procfs inode's
  mtime, understating every process on the box (29 minutes on one alive 4h42m).
  Now `btime` + field 22 of `/proc/<pid>/stat`. Same commit: a sort key in
  `backfill_start_times` that compared a 6-character string against an
  8-character slice and could never fire.
- **`83837a2`** — `fit()` counted spans it does not render, so under
  `--all-generations` the headline figures absorbed the void runs.
- **`ae826de`** — an unreachable ARC API disabled three checks that need no
  network: the provenance void, `mark_generation` and `mark_clean`.

### The contamination gate
- **`424dd79`** — four holes in `proofread_trace`, two of them live: the leak
  scan read only the *first* window of a transcript, and `strayed`'s `/tmp`
  exemption could be dropped without any test noticing, which would exempt the
  whole evidence store. Two coefficients settled against the corpus rather than
  argued (table in `docs/ccarc3_open_findings.md`).
- **`f6f3553`** — the proxy's two untested invariants: an adopted card pinning
  must survive `set_budget`, and the exhaustion refusal must not name the cap
  (the cap ÷ `budget_multiple` is the withheld total).

### The driver
- **`5abca90`** — `verdict()`, which decides whether a run is banked or
  discarded, had *zero* coverage. Nine tests, nine mutants.
- **`f949533`** — the orphan sweep matched sweeps by substring (so
  `clean_rollouts` matched `clean_rollouts_validate`) and split
  `/proc/<pid>/stat` across the whole line.
- **`b5bf415`** — a replaced container silently started a second scorecard.
- **`75fb2d4`** — the driver opened a scorecard before checking it had anything
  to play on it.
- **`653a945`** — `CCARC3_MAX_PASSES`.
- **`4870465`** — the launch brake (§5), `CCARC3_QUOTA_LIMIT`.

### The solver-facing modules — mutation-audited

`grids.py`, `rules.py` and `ledger.py` were the last on-path modules with line
coverage and no mutation audit; the card-facing half of `scoring.py` — the code
the sweep decides on — turned out to be a fourth. **120 mutants, 4 real defects,
38 test holes, 4 argued equivalences.** The battery is
`tools/mutation_battery_ccarc3.py`; run it rather than trusting this paragraph.

The four defects, shortest form:

1. `grids.logical()` returned the caller's own array when it declined to
   reduce, and a copy when it did. Editing the result edited the ledger's
   frame. The aliasing branch is the one real 64×64 frames take; the copying
   branch is the one an `np.kron` fixture takes. `collapse()` had the same
   split on its empty branch. Both now always copy.
2. `grids.cell_boundaries()` pooled frames of different shapes silently and
   bounded the answer by whichever came last — and it is documented to take a
   whole trace, across which ARC-AGI-3 changes board shape. Now refuses, for
   the reason `diff()` already refuses a shape change.
3. `ledger.TraceWriter.append()` recorded score 0 for a frame carrying
   `score: None` beside a real `levels_completed`, because the two-name
   fallback keyed on key-absence. `infer_levels` reads score increments as
   level boundaries, so that collapses a whole trace into level 0.
4. `counts` and `ledger_facts` were missing from their modules' `__all__` —
   the fifth stale enumerated list here. Replaced by a derived rule, not a
   sixth entry.

`scoring.py`'s card-facing half had no code defects either, and eleven of its
25 mutants survived. The two that matter: `score_run` could be made to score the
**first** play with nothing failing, even though scoring the *last* play was
caught — every fixture had its best play first or last, so "best" was only
pinned from one side — and `card_disagreement` could accept a card one level
behind the result and stay silent, in the one check that does not read our own
trace. Two source changes came with it: a non-positive `playthroughs` is now
refused instead of being used as a slice length, and an unreachable guard's
comment no longer claims to handle a case it cannot reach.

`rules.py` had **no** code defects, and that is worth reading rather than
skipping: four of its seven survivors were the exact failure the module exists
to prevent (`vacuous` able to relabel a refuted rule "never applicable",
`perfect` able to crown a model that declined every transition, `regressions`
able to fire on a mechanic that is *working*). A correct module with nothing
checking it is one edit from an incorrect one.

**Three of my own new tests survived the mutants they were written to kill.**
The instructive one: `monotone_rows` defaults to a 0.9 threshold, and a
nine-of-ten fixture sits exactly on it, so the test passed whether or not the
skipped pair was counted. The fixture agreed with the mutant. Run the battery
against new tests too, not only against old code.

### Container replacement recovery
- **`fceee6f`** — `rehydrate_box.sh` re-pointed three of five shadowing
  scratchpad copies and reported success, leaving a **pre-fix baseline strip**
  and a pre-fix supervisor. Now derived from `tools/` rather than enumerated.
- **`cce4ad5`** — the agent proxy's port is refreshed during rehydration.
- **`b72025c`** — the preserver and the restore had never been run against each
  other. `--once` makes a single cycle exercisable.
- **`90dbd60`** — portability asserted over every tool, verified live against a
  second clone on a different branch with its own scratchpad and origin.
- **`2c71bae`** — the test suite was writing fixture card ids into the committed
  evidence tree.

---

## 4. The method, and the one habit that matters

**Mutation auditing.** Break the code on purpose, one small change at a time, and
watch whether a test fails. CAUGHT means something holds the behaviour; UNCAUGHT
means the code could be broken that way and nobody would find out.

`tools/mutation_check.py`:

```python
import sys; sys.path.insert(0, "tools")
import mutation_check as mc
mc.run("tools/clean_rollouts.py", ["tests/test_x.py"], [
    ("name", "what this mutant does", "old text", "new text"),
])
```

It carries two scars. It disables bytecode caching — a same-second,
length-preserving restore once left CPython serving the *mutant's* `.pyc` to
later runs, corrupting verdicts in both directions. And it restores from disk via
a signal handler — a mutant that removed a `ppid == 1` guard made the code under
test SIGTERM the test runner, leaving the mutation in the working tree.

**The recurring defect is a check that passes by not running.** An empty loop, an
unreachable branch, a fixture that cannot express the failure it is named for, a
list that has gone stale, a sentinel indistinguishable from a real value. Reading
the test never reveals it. Mutating what it claims to protect always does.

**The habit:** earlier in this same session I reported the run-path audit as
complete. It was not — re-auditing those same modules produced roughly fifteen
more defects, including two live safety holes. **The find-rate has never
declined.** Treat "no known defects" as "nobody has looked recently".

Two other things worth internalising:

- **Line coverage is not testing.** `verdict()` had 0% and nobody noticed;
  `_outstanding()`'s ordering had full line coverage and no assertion on the
  order.
- **An always-on report is one people stop reading.** Several fixes here are
  about a message firing *only* when it carries information.
- **Some survivors are correct.** Four equivalent mutants are recorded with the
  argument for each in `docs/ccarc3_open_findings.md`. Conflating them with gaps
  is how a suite acquires tests that assert coincidences.

---

## 5. Traps — things that look fine and are not

**The scratchpad reverts to an image snapshot when the container is replaced.**
This happened at 02:44 PDT on 2026-08-09 and three times in 45 minutes on
2026-08-06. Anything kept only there expires. Consequences seen for real:

- A **launch brake** — `scratchpad/concurrency` holding `0`, meaning "start
  nothing" — written during a freeze on Aug 7 came back from the dead and
  silently held a launch. Symptom: driver alive at ppid 1, "pass 1/2", no
  workspace, no solver, no error, indefinitely. **Check this file first when a
  driver appears to do nothing.** `rehydrate_box.sh` now reports it.
- Pre-fix copies of `supervisor.sh` and **`ablate_baselines.py` — the baseline
  strip** — were restored over the repo versions. `rehydrate_box.sh` repairs
  every shadowing copy now, but run it *first thing* on any wake-up.
- The **agent proxy's loopback port** changes when the session worker restarts,
  so `proxy_env` is always stale on a replaced box. The supervisor refuses to
  launch against a dead proxy; `rehydrate_box.sh --links-only` refreshes it, and
  only a live session can.
- **Quota readings go stale silently.** The reading reported all night said
  `util=0.77`; it was 43 hours old and the true figure was 0.96. `quota.sh`
  labels staleness — read the label, and note that utilisation only rises, so a
  stale reading is a *lower bound*.

**Discarding an attempt is local bookkeeping. ARC's card does not forget it.**
`clean_rollouts` throws away an interrupted or gave-up attempt and re-runs the
game, which is the right basis for a claim about *single-attempt* performance.
The scorecard is the opposite: every play ever made on it stays, and **ARC scores
the best play**.

Measured on 2026-08-09, `bp35` on card `ac0e1272-…`:

| | plays | guids | levels | actions |
|---|---|---|---|---|
| after attempt 1 (discarded) | 2 | `4be429b4` ×2 | 4, 6 | 241, 357 |
| during attempt 2 | 3 | `4be429b4` ×2, `7fa4724f` | 4, 6, 3 | 241, 400, 201 |

The guid is what marks a run: attempt 1's two plays share one, because the solver
full-reset once inside a single session; attempt 2 opened a fresh guid. And
241 + 400 = 641, exactly the `actions_used` on the discarded `result.json`.

So a retried game's card number is a best-of-N across attempts including the ones
we refused to bank, while its `clean_result.json` describes one attempt. Both are
correct; they are answers to different questions. For a leaderboard submission the
best-of is fine and probably intended. **Do not put a card figure and a banked
figure side by side and call them the same measurement.** `uncorroborated()`
compares the two and will flag a genuine disagreement — see `df5aec6`, where the
corroboration check had been reading a discarded attempt's rows.

**A watcher reads `CCARC3_SWEEP_DIR` from its own environment, at launch.**
`heartbeat.sh` and `refresh_audit.sh` both default to `clean_rollouts`. A daemon
started before a differently-named sweep exists therefore watches the wrong
directory and reports *that* one's state — on 2026-08-09 the heartbeat reported
"25/25 done" every cycle while a `bp35` run in `clean_rollouts_validate` was
mid-game, and the hourly artifact refresh would not have noticed it finishing.

So: **launch the watchers with the same `CCARC3_SWEEP_DIR` as the sweep**, and
pass it to `refresh_audit.sh` too. The real fix is for them to *discover* live
sweeps rather than take one from the environment — the same
derive-don't-enumerate change made to `rehydrate_box.sh` and `snapshot_results.py`
— and it is the best-value follow-up after §6's three modules.

**Other traps:**

- `git` in a replaced container may be a clone from an older point. The local
  branch can be far behind origin with `origin/…` refs stale too. **Fetch before
  believing anything**; origin is the source of truth and has never lost work.
- `2>/dev/null` on a command does not silence a failed *redirection* — the shell
  opens `< file` before the command exists. Use `{ cmd < file; } 2>/dev/null`.
- Under `set -u`, a variable referenced inside a block that a test extracts and
  runs standalone must be written `${VAR:-default}`. Two tests here slice
  `rehydrate_box.sh` and `preserve_evidence.sh` that way.
- `[ cond ] && exit 0` as the last statement of a block sets the block's exit
  status when the condition is false. Use `if`.
- A shared card is reachable **only** from a session carrying its four
  `AWSALBAPP-*` stickiness cookies. Those are credentials and are never
  committed, so a replaced container genuinely cannot resume a card — it can only
  refuse to pretend the card never existed.

---

## 6. Known gaps — recorded, not fixed

1. **The leak scan identifies the median array by order.** A solver that *sorts*
   it before printing evades the scan. Measured: order-insensitive matching fires
   on 10 preserved streams the ordered scan calls clean, four of them banked
   clean rollouts — so the narrower hole beats voiding four good runs. Table in
   `docs/ccarc3_open_findings.md`.
2. **`surface_digest` excludes `client.py` and the child environment**, because
   stored digests are historical and cannot be recomputed. "Same digest" does not
   mean "identical solver surface".
3. ~~**Three solver-facing modules have never been mutation-audited.**~~
   **Done 2026-08-09** — `grids.py`, `rules.py` and `ledger.py` are audited:
   95 mutants, 4 code defects fixed, 27 test holes closed, 2 equivalent mutants
   argued. The battery is committed as `tools/mutation_battery_ccarc3.py` and
   re-runs in about five minutes. See §3 "The solver-facing modules" and the
   commits `89f31a4`, `55c6990`, `5cc5be0`. Nothing solver-facing in `ccarc3/` is now
   unaudited. The remaining unaudited surface is `client.py` and `scoring.py`,
   which were audited earlier by a different route (the network gate and the
   card-scoring probes) but never with a full mutant battery — a reasonable
   next target, and it also costs no quota.
4. `coverage` is not installed in every container's venv. Do not assume it.

---

## 7. Resuming — exact commands

Paths below assume the scratchpad root is `$SP`. Every tool takes
`CCARC3_SCRATCH`; nothing is pinned to one container (asserted by
`tests/test_no_tool_is_pinned_to_this_container.py`).

**First thing, always:**

```bash
bash tools/rehydrate_box.sh          # ff from origin, repair symlinks, refresh
                                     # proxy_env, report an engaged launch brake
bash tools/quota.sh                  # read the label, not just the number
```

**Check what is alive** (argv-element-exact, never `pgrep -f`, which self-matches):

```bash
for d in /proc/[0-9]*; do [ -r "$d/cmdline" ] || continue
  { tr '\0' '\n' < "$d/cmdline"; } 2>/dev/null | grep -qx '.*/supervisor\.sh' && echo alive
done
.venv/bin/python -c "import sys;sys.path.insert(0,'tools');import build_trace_audit as b;print(b.live_games())"
```

**Relaunch the daemons** (each double-forked so it reparents to init and survives
this session):

```bash
set -a && . "$SP/arc3/.env" && set +a
. "$SP/proxy_env"
( setsid nohup bash tools/preserve_evidence.sh >> "$SP/preserve_evidence.log" 2>&1 </dev/null & )
( setsid nohup bash tools/supervisor.sh        >> "$SP/supervisor.log"        2>&1 </dev/null & )
# heartbeat: run tools/heartbeat_watch.sh under a Monitor; it double-forks the
# heartbeat itself, so a Monitor timeout cannot take it down.
```

**Launch one game:**

```bash
CCARC3_ONLY=bp35 CCARC3_SWEEP_DIR=clean_rollouts_validate CCARC3_MAX_PASSES=2 \
  setsid nohup .venv/bin/python tools/clean_rollouts.py >> "$SP/validate.log" 2>&1 </dev/null &
```

**Score a finished game and rebuild the page:**

```bash
bash "$SP/refresh_audit.sh"                     # exits 0 and silent if nothing new
python3 tools/build_trace_audit.py --ingest <run_dir>
# then write it up in docs/ccarc3_results.md, commit, push
```

**Environment knobs** (all optional, all defaulted):

| variable | effect |
|---|---|
| `CCARC3_SCRATCH` | scratchpad root |
| `CCARC3_BRANCH` | push target; a different account works on a different branch |
| `CCARC3_SWEEP_DIR` | sweep directory; **must contain `clean_rollouts`** |
| `CCARC3_ONLY` | comma-separated game ids or 4-char prefixes |
| `CCARC3_MAX_PASSES` | retry bound (default 12) |
| `CCARC3_CONCURRENCY` | games at once (default 2; the `concurrency` file wins) |
| `CCARC3_QUOTA_LIMIT` / `CCARC3_QUOTA_RESUME` | supervisor ceiling (0.98 / 0.90) |
| `CCARC3_NO_SHARED_CARD` | one card per game instead of a shared one |

---

## 8. The 25-game sweep onto one shared scorecard

The remaining substantive experiment. A leaderboard submission takes one
`scorecard_url`, so all 25 games must land on one card.

- **Cost:** ~$652 at the observed per-game rate.
- **Blocked on quota only.** Reset is 2026-08-10 21:00 PDT.
- **Give it its own directory.** `_run_one` skips any game that already has a
  `clean_result.json`, so pointing a submission sweep at `clean_rollouts` would
  skip all 25 and put nothing on the card:
  `CCARC3_SWEEP_DIR=clean_rollouts_submission`.
- **The card is pinned in `$SP/shared_card.json`, which does not survive a
  replacement.** The card *id* is mirrored to
  `evidence/ccarc3/sweep_card/history.jsonl`, which does. If a replacement
  happens mid-sweep the driver will refuse to mint a second card and will name
  the stranded one; the decision to restart or accept a partial artifact is the
  operator's.
- Everything the sweep depends on — `sweep_card`, the corroboration check, the
  proofread gate, the orphan sweep, the queue order — was audited this session.

---

## 9. Numbers worth not re-deriving

- 25/25 games banked in `clean_rollouts`, on **30 distinct cards** (one per game;
  this was never a shared-card sweep).
- `bp35-0a0ad940` prior: 990 actions, 4.86 h, $79, 9/9.
- Per-game cost across the bank: roughly $26 median, $79 at the top.
- 77 preserved streams, 56 scorecards, 67 result files under `evidence/ccarc3/`.
- Largest per-level human median anywhere in the corpus: **442** (three digits —
  the leak scan's regex allows four on purpose).
- Leak-scan calibration, 77 streams × 16 arrays: `window=3n` → 43 fires;
  `10n` → 46; `30n` → 99; order-insensitive → 53. Widening voids good runs.
- Card stickiness, measured live: 8/8 card reads succeed direct, **1/8** through
  a shim without the `AWSALBAPP-*` cookies.

---

## 10. Retractions from this session

Recorded because the mechanism repeats, not for penance.

1. **"The audit is complete on the run path."** It was not. Re-auditing the same
   modules found ~15 more defects. The mechanism: I treated "I have looked" as
   "there is nothing there".
2. **"Quota is at 0.77."** Reported repeatedly. The reading was 43 hours stale
   and the true figure was 0.96. `quota.sh` labelled it `STALE`; I passed the
   label along without acting on it.
3. **A test that asserted an absence its own fixture guaranteed.** The
   game-level runtime-copy test had no `client.py` in its sandbox to copy, so the
   mutant removing the guard survived. Twice more the same shape: a fixture
   forcing `_last_limit = None`, and a filler string welded into a five-digit
   token the scan discards by design.
4. **The same `set -u` unbound-variable trap, twice in one night** — in
   `rehydrate_box.sh` and then in `preserve_evidence.sh`, two hours apart.

---

*Written 2026-08-09 06:15 PDT. Index: `docs/ccarc3_memory.md`. Standing agenda:
`docs/ccarc3_autopilot.md`. Findings ledger: `docs/ccarc3_open_findings.md`.
Prior handoff: `docs/ccarc3_handoff_0809.md`.*
