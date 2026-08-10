# CCARC3 handoff — 2026-08-08, written at 708k context

> **SUPERSEDED — do not act on this file.** The current handoff is
> `docs/ccarc3_handoff_0809b.md`; `docs/ccarc3_handoff_0809.md` sits between
> them. Numbers here were true on 2026-08-08 and several were later retracted —
> see §A of `_0809.md` and §10 of `_0809b.md`. Kept because its defect write-ups
> are still the fullest record of that day, not because its state is current.


Read this in full before acting. It is the state of the project at the end of a
day in which **30 defects were found and fixed across three passes**, and the
single open task is a $652 run that is blocked on quota until ~08-11.

Everything durable is in this repo. The scratchpad reverts to an image snapshot
when the container is replaced — `tools/rehydrate_bootstrap.sh` records that as
a retraction of its own earlier claim, and the conclusion it reaches is the one
to trust: **origin is the only store that has never lost anything.**

---

## §A. RETRACTIONS — read these first

A handoff that drops a retracted claim is worse than one that omits the topic:
the next session inherits the confident wrong version and re-derives nothing,
because nothing looks broken. Every item here was stated confidently by me and
later disproved.

1. **"The heartbeat is healthy."** It had been dead for days. `runner_alive`
   matched `ablate_baselines.py`, retired long before, so it was permanently
   false — and the loop read false as "the arm ended", printing a log tail from
   2026-08-07 and exiting on its FIRST poll. From outside it looked identical to
   a working heartbeat. Fixed in `3d8c762`.

2. **"That fix is complete."** It was half a fix. I repointed `runner_alive` and
   left the entire reporting block below it reading the retired arm's log and
   directory, so every emitted field described a sweep that had already ended.
   Found by the shell audit, fixed in `71837a4`.

3. **"Quota is `allowed`, util 0.13, we can run the sweep."** Fabricated. A
   mutation-testing agent had left fixture trees in the live scratchpad and
   `quota.sh` scans that directory for a living; one planted file said
   `seven_day allowed 0.13`, another `rejected 0.91`. The real number was 0.84.
   I did not launch. Fixed in `93d27b1`, refined in `abafd1a`.

4. **"quota.sh now ignores only debris."** It was also discarding three REAL
   `cc_harness` runs, because I derived the "is this a real workspace" marker
   list from ccarc3 layouts only. Harmless by luck — those readings were 122h
   stale — but a fresh cc_harness run would have been dropped while the tool
   printed a confident status line. Fixed in `abafd1a`.

5. **"46 of 157 clicks is 20% of actions."** It is 29%. 20% is 67 of 336, the
   whole-run rate. Two denominators welded together. (`fbf8887`)

6. **"`tn36` L5 is the level the lost run never cleared."** It cleared, at 5.62×.
   `cap` 0.750 = 21/28 is reachable at six cleared levels and no other count.
   Three copies of the project disagreed about this one fact. (`fbf8887`)

7. **"The three losses were efficient on the levels they cleared, `sk48` at
   0.18× and `sp80` at 0.18×."** Neither is an efficiency ratio: 0.18 is
   `tn36`'s budget-utilisation fraction, borrowed and attributed to two other
   games. Real figures on cleared levels: `sp80` 0.37×, `sk48` 0.94×, `tn36`
   **1.45×** — over the human count. (`961af24`)

8. **`wa30` "+0.5833"** is `re86`'s delta. At 5 of 9 `wa30`'s ceiling is 0.3333,
   so its fix is worth ≥0.6667. (`fbf8887`)

**The mechanism behind all of them, and behind the 30 defects fixed today, is
one thing: a check, test, claim or figure that names something and reads a PROXY
for it.** The signature is that it *passes by not running* — an empty loop, an
unreachable branch, a fixture that cannot express the failure, a comparison
equal by construction, a denominator from a different measurement.

**The counter-practice that works, every time: mutate the thing the check claims
to protect and confirm the check fails.** If it stays green, the check is
decorative. For a figure, the equivalent is re-deriving it from a source that
could not have been copied from the same place.

---

## §B. Where the project actually stands

**Publishable figure: 98.38% over 17 environments** — 16 at E=1.0, `bp35` 0.7252
the only shortfall. It is a floor, not a ceiling: every sub-1.0 clean rollout
predates the win-frame replay prompt (`fd18b89`).

- 25 of 25 games have a banked clean run in `scratchpad/clean_rollouts/`.
- The driver exits immediately with "all 25 have a clean run". The supervisor
  relaunching it every ten minutes is a **no-op, not a fault** — this reads like
  a failure in the log and is not one.
- Evidence is preserved under `evidence/ccarc3/` and pushed continuously.
- The canonical per-run store is `evidence/ccarc3/trace_audit/runs.json.gz`.
  Prefer it over re-deriving from `result.json`; it carries E/raw/cap/levels/
  actions/plays/cost per run.

### Scoring, so it is never re-derived wrong
Per completed level `S = min(1.15, (h/a)²)`. `raw` = mean weighted by level
index. `cap = sum(1..completed)/sum(1..n)`. **`E = min(cap, raw)`.** ARC scores
the **best** play, not the last.

---

## §C. THE ONE OPEN TASK — the 25-game sweep (task #35)

    CCARC3_SWEEP_DIR=clean_rollouts_submission tools/clean_rollouts.py

~$652. **Blocked on quota, not on code**: seven-day window at util=0.84 against a
0.75 warning line, resetting ~2026-08-11 21:00 PDT. Under `allowed_warning` the
standing policy is *launch nothing new*, and a sweep is the largest possible new
launch.

**Why it must be re-run at all:** a submission takes one `scorecard_url`, so all
25 games must land on ONE card, and **nothing already banked can be retro-fitted
onto one**. A card is state on a single backend instance, reachable only from a
session carrying its `AWSALBAPP-*` stickiness cookies. Injecting `card_id` alone
does not work — that was retracted on 08-08 and the shim adopts cookies instead.

**Before spending, verify these still hold** (all were fixed today, each with a
mutation-verified test):

| what | where | why it matters |
|---|---|---|
| corroboration scopes to this attempt's plays | `scoring.card_disagreement` | a shared card accumulates rows across attempts; `max()` banked a frozen card |
| …and to rows this attempt added | `client.card_plays_at_open` | on a RETRY the predecessor's rows corroborated the banked run |
| resume refuses a reaped game | `client._assert_server_agrees` | card and game are on different clocks; 25 games keep the card warm |
| resume refuses a card with no entry for this game | same | the `or card` fallback read the other games' aggregate |
| the remint refusal survives a restart | `clean_rollouts.sweep_card` | it deleted its own trigger; supervisor restarts in 10 min |
| a solver cannot close the lent card | `arc_proxy` | one forwarded endpoint could finalize the whole artifact |
| a corrupt `rules.json` cannot lose a won run | `session.collect_outcome` | it raised before `result.json` was written |
| the timeout reaches the solver's children | `session.run_game` | grandchildren kept spending after the kill |

**Three more found 2026-08-09, all aimed at THIS sweep, all the same shape** —
the sweep-directory name written as a literal in one place and derived from
`CCARC3_SWEEP_DIR` in another. Under
`CCARC3_SWEEP_DIR=clean_rollouts_submission` each failed silently:

| what | where | what it would have cost |
|---|---|---|
| the restore writes where the driver reads | `restore_clean_rollouts.py` | after a container replacement it restored banked games to `clean_rollouts/` while the driver read `clean_rollouts_submission/`, found nothing, and **re-ran games already on the shared card** — money twice, and a card cannot un-play a game |
| the rebuild command points at the workspace | `refresh_audit.sh` | the `case` matched the literal, so a renamed sweep fell through to the flat arm and emitted `--ingest <game dir>`, which holds only `clean_result.json` — the audit announces new results and the artifact never changes |
| the ledger records this arm at all | `snapshot_results.py` | its batch list was written for the arms of the day; **25 banked games, zero rows**. Recovering them added 30 |

**Also new: the supervisor now refuses to launch without a usable key.** It
sourced `$SP/arc3/.env` unchecked in a file with no `set -e`, so a fresh
container — where the scratchpad has reverted and the key is deliberately not in
the repo — would launch a runner that 401s every action, and relaunch it every
ten minutes. `key_ok` checks the VALUE, not the file.

**Watch during the sweep:** `lf52` (7/10, E=0.4537 — the biggest recoverable
loss) and `bp35` (0.7252, the only sub-1.0 in the publishable set).

---

## §D. What was done on 2026-08-08

Three passes, ~60 commits, all pushed to
`claude/athanor-cc-harness-variant-jpqw7t`. **Never open a PR.**

1. **12 deep-proofread findings** (`d5746c9`…`fbf8887`) — all confirmed, all
   fixed. 3 came with a proposed fix that was wrong; checking them surfaced 5
   further defects nobody had reported. Recorded in
   `docs/ccarc3_open_findings.md`.
2. **15 shell-layer findings** (`71837a4`…`ad623d0`) from a 24-agent audit of
   the autopilot scripts, which had never been checked for this defect class.
   Plus 111 mutation-verified regression tests where there had been none.
3. **15 findings from a whole-surface hunt** (`d350391`…`961af24`) — 7
   sweep-corrupting, 8 misreporting. All fixed.

**970 tests pass**, and they no longer depend on an ambient `ARC_API_KEY`
(791/5-skipped without a key, 970 with one).

---

## §E. Operating constraints (these keep getting rediscovered)

- **Opus 5 only, high effort.** Not 4.8, not Fable.
- **No control arm.** All runs baseline-free.
- **Never open a PR.** Push to `claude/athanor-cc-harness-variant-jpqw7t`.
- **The API key lives in `scratchpad/arc3/.env` (mode 600) and never enters the
  repo.** `preserve_evidence.sh` refuses a commit if it appears under evidence/.
- **Do not touch `b6f77b65`** — it is buggy, by operator instruction.
- **Report every timestamp in America/Los_Angeles, stamped at the source.** See
  `CLAUDE.md`; the box runs UTC and hand-conversion gets skipped.
- Solvers run as uid 0, the same user that owns the parent transcript.

---

## §F. Environment and daemons

**Five** long-lived processes, all checked argv-element-exact (**never
`pgrep -f`** — it matches this session's own CLI, and once killed a live
monitor; on 2026-08-09 it was still in `supervisor.sh`'s orphan sweep, matching
6 live pids against 0 for the argv-exact form, because the scratchpad path
itself contains `claude-0`):

| process | role | if down |
|---|---|---|
| `tools/supervisor.sh` | relaunches the runner every 10 min under the quota ceiling | `setsid nohup bash tools/supervisor.sh >> $SP/supervisor.log 2>&1 < /dev/null &` |
| `tools/preserve_evidence.sh` | pushes evidence to origin every 300s | same shape |
| `context_watch.py` | context threshold alarm | see `~/.claude/skills/context-handoff` |
| `tools/heartbeat.sh` | **detached** (`ppid 1`); reports daemon loss and new work | the watchdog relaunches it within 60s |
| `tools/heartbeat_watch.sh` | Monitor-hosted watchdog over the heartbeat | re-arm the Monitor over **this**, never over `heartbeat.sh` |

**CORRECTED 2026-08-09 — this table used to say `heartbeat.sh` was
"Monitor-hosted" and to "re-arm the Monitor" on it.** That coupling is what made
the Monitor's timeout the heartbeat's lifetime: the Monitor expired, took the
heartbeat with it, and nothing reported the loss, because `daemon_check` lists
the other three and cannot list itself. Arm the Monitor over
`heartbeat_watch.sh`; it launches the heartbeat **double-forked** so it
reparents to init and outlives both the Monitor and a session-worker restart.

`setsid` alone is not enough and looked like it was — it gives a child its own
SESSION, not a new PARENT, so a descendant sweep still reached it. The field to
check is `ppid`: the surviving daemons all read **1**. Verified across three
Monitor timeouts and one worker restart.

**The agent proxy port changes when the session worker restarts.** Long-lived
daemons keep the value they launched with, so `$SP/proxy_env` must be refreshed
from a live session; both daemons re-read it each cycle. The supervisor refuses
to launch into a dead proxy and says so — that guard is correct and has fired.

`tools/rehydrate_box.sh` symlinks the scratchpad paths the standing prompts use
(`AUTOPILOT.md`, `MEMORY.md`, `quota.sh`, `heartbeat.sh`, `refresh_audit.sh`)
back at the repo, so they cannot drift or be lost.

---

## §G. Pending work, in order

1. **Task #35, the sweep.** Blocked on quota until ~08-11 21:00 PDT. Everything
   else is ready.
2. **Re-publish the audit artifact** after any new game finishes — same URL
   (`.../artifact/7447856a-b587-4d52-9c5c-a839de3eb6ee`), same file path.
   `refresh_audit.sh` is the hourly safety net; silence means no-op and needs no
   narration.
3. **Deliberately left open, all cosmetic:** three retry counts, one redundant
   regex alternative, one equivalent mutant. Named in `155cff0`.
4. **`lf52`'s "6 of 10 at E=0.4909" is internally inconsistent** — `cap(6,10)` is
   0.3818 — and the run is not on disk to settle which half is wrong. Both
   readings are stated in `ccarc3_results.md`; the bound holds under either.

---

## §H. Things that will mislead you

- `pgrep -fc "athanor cc"` counts this session. Check by **workspace cwd**.
- A driver exiting instantly is the finished sweep, not a crash.
- `quota.sh` reporting `unknown` usually means no solver has run recently, not
  that something broke. It reads rate-limit events out of *solver* streams.
- A window that has RESET has no reading, which is not the same as a low one.
- Agents told "report only" have edited the tree twice. **Read every diff before
  committing** — this project once committed an agent's sabotage via `git add -A`
  unread, in the commit whose message was about agents editing the tree.
- `git checkout --` on a file with uncommitted work discards it. Keep a byte copy.
