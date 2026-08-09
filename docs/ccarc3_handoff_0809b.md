# CCARC3 handoff — 2026-08-09, session B

**Written for a session that has only this repository.** Everything needed to
continue is here or reachable from here. The scratchpad does not survive; assume
it is empty.

Branch: `claude/athanor-cc-harness-variant-jpqw7t`. **Never open a PR.**

---

## 0. The sixty-second version

- The harness has had a full mutation audit. 1255 tests pass. Every on-path
  module has been audited; three solver-facing ones have not (§6).
- A `bp35` validation run is **in flight** as of 06:07 PDT (§1). It may or may
  not be alive when you read this.
- Weekly quota is at **0.96** and resets **2026-08-10 21:00 PDT**. Until then
  there is almost nothing to spend.
- The big outstanding job is **task #35: the 25-game sweep onto one shared
  scorecard** (§8). It is blocked on quota, not on the code.
- The single most important habit: **when you think you have finished auditing,
  you have not.** See §4.

---

## 1. State at handoff

| | |
|---|---|
| HEAD | `4870465` (plus preserver commits after it) |
| Tests | 1255 passed, 5 skipped, 1 xfailed |
| Weekly quota | `util=0.96`, resets **2026-08-10 21:00 PDT** |
| Daemons | `supervisor.sh`, `heartbeat.sh`, `preserve_evidence.sh`, all ppid 1 |
| Banked | 25/25 games in `clean_rollouts`, one card per game |

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

**If it finished:** score it, write it up in `docs/ccarc3_results.md`, rebuild the
audit page (§7).

**If it was interrupted:** `clean_rollouts` discards interrupted attempts rather
than scoring them, so there is nothing to salvage and nothing was corrupted. The
card above will be named in the history; a fresh sweep in a *new*
`CCARC3_SWEEP_DIR` is the clean restart.

**Put the ceiling back.** `CCARC3_QUOTA_LIMIT` was a one-run decision. Relaunch
the supervisor with no override to restore 0.98.

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
3. **Three solver-facing modules have never been mutation-audited** — only line
   covered: `src/athanor/ccarc3/grids.py` (what the solver sees of the board),
   `rules.py`, and `ledger.py` (which writes `trace.jsonl`, from which every
   downstream number is derived). **This is the highest-value remaining audit
   work and it costs no quota.**
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

## 8. Task #35 — the 25-game sweep onto one shared scorecard

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
