# CCARC3 handoff — 2026-08-09 (supersedes ccarc3_handoff_0808.md)

> **SUPERSEDED — do not act on this file.** The current handoff is
> `docs/ccarc3_handoff_0809b.md`. Read this one only for its §A retractions,
> which `_0809b.md` §10 continues rather than repeats.


Written at 22:34 PDT with context at 733,461 of a 784,134 auto-compact trigger.
Read §A before trusting any number anywhere, including in the older handoff.

**What this session was.** One long bug-clearing pass under a standing operator
instruction: *"use all your remaining quota towards bug fixes so that i'll have
an absolutely clean and bug free harness to run large scale rollout with the
other CC account."* 24 commits, `155cff0..5480fe6`. Tests 971 → **1112
collected**. No games were played and no ARC quota was spent on solving.

**The commit messages are the record of WHAT changed** — they are long and
specific on purpose. `git log 155cff0..HEAD` is the index. This file carries what
git does not: the retractions, the open decision, and the traps.

---

## §A. RETRACTIONS — read these first

Four confident claims I made this session and later disproved. Each was wrong in
the same way: **I reached for a mechanism that would explain the observation and
stopped, instead of asking what record would distinguish it from the
alternative.** In all four the deciding artifact was already on disk.

### A1. "Claude Code's reported cost treats cache as free." WRONG.

I fitted four price vectors to 14 runs and reported the best fit as fact. The
CLI bundle settles it (`/opt/node22/lib/node_modules/@anthropic-ai/claude-code/cli.js`):

```js
function h15(A,q){return q.input_tokens/1e6*A.inputTokens
  + q.output_tokens/1e6*A.outputTokens
  + (q.cache_read_input_tokens??0)/1e6*A.promptCacheReadTokens
  + (q.cache_creation_input_tokens??0)/1e6*A.promptCacheWriteTokens + ...}
```

Cache **is** charged. The real explanation of the 3.10× gap: the bundle's rate
table contains a tier at exactly **one third of Opus in every field** —
`5/25/6.25/0.5` against `15/75/18.75/1.5` — which reproduces the reported figure
at **1.03×**, and `x15(rates, discountPercent)` scales every field by
`(100-q)/100` from a gate named `tengu_penguin_mode_promo`. Solving for a uniform
discount on Opus gives **K = 0.323**, one third to three digits.

Corrected totals for this VM (Aug 1 02:16 PDT → Aug 8, 7.8 days, 23 distinct
container boots):

| basis | total |
|---|---|
| Opus list, cache included | **≈ $13,450** |
| at the rates actually applied (≈ Opus/3) | **≈ $4,480** |
| ~~"cache free"~~ (my error) | ~~$3,011~~ |

My $3,011 was wrong in **both directions at once** — dropped cache costs it
shouldn't have, missed a ~3× discount it should have applied. Two errors partly
cancelling is why the fit looked convincing. **A best fit among the options you
happened to try is not a measurement.**

### A2. "The seven-day quota window is rolling, so idle time decays it." WRONG.

I told the operator the sweep might start early. It cannot. Every
`rate_limit_event` in the current window carries the identical
`resetsAt: 1786420800` = **Aug 10, 21:00 PDT**, and only two distinct `resetsAt`
values exist across all history. Utilization climbs monotonically inside it:
0.75 → 0.76 → 0.77 → 0.83 → 0.84.

So `quota.sh`'s doctrine — *"stale + allowed_warning → still binding,
utilization only rises"* — is **correct and I was wrong to doubt it**. Worse for
my suggestion: this session's own token use draws on the same account, so the
true figure is **≥ 0.84**, not lower. Running a game to "refresh the reading"
would spend ~$60 to confirm a constraint that can only have tightened.

### A3. "The heartbeat now survives a Monitor timeout." WRONG at the time.

`setsid` gives a child its own **SESSION, not a new PARENT**, so a descendant
sweep still reaches it. `scratchpad/heartbeat.log` shows relaunches at 16:26,
16:58 and 17:28 PDT — one per 30-minute Monitor cycle — while `supervisor.sh` and
`preserve_evidence.sh` sailed through all three. The instance I had measured was
orphaned by accident (started from a shell that exited), and I credited the
decoupling for a property it did not have.

**The field that distinguishes a survivor is `ppid`, which reads 1.** Fixed with
a double fork — `( setsid bash "$X" … & )`, subshell exits at once. Since
verified across three Monitor timeouts **and one session-worker restart**.
A launch line containing `setsid` is not evidence of anything.

### A4. "The results doc's `lf52` claim needs correcting." WRONG — I nearly broke it.

`docs/ccarc3_results.md` says *"`lf52`'s two figures still disagree and the run
is not on disk to settle it."* I found
`scratchpad/ablate_nobaseline/lf52-271a04aa/result.json` (won, 10/10, 941
actions) and almost "corrected" it. That file is **attempt 2**, the retry that
won. The disputed *"6 of 10 at E=0.4909"* is **attempt 1**, and no trace or
stream for it survives anywhere — I checked scratchpad and evidence. **The doc is
right. No edit was made.**

---

## §B. State at handoff

- **Branch** `claude/athanor-cc-harness-variant-jpqw7t`, HEAD **`5480fe6`**, tree
  clean, 0 unpushed. **Never open a PR.**
- **Tests** 1112 collected / 1107 passing, 5 skipped, 1 xfailed. Suite takes ~96s.
- **32/32** audit findings closed (`tools/verify_audit_findings.py`, now run by
  `tests/test_audit_findings_stay_closed.py`).
- **77 preserved streams, 0 leak exposures** (`tools/leak_exposure.py`).
- **Quota** `allowed_warning`, seven_day util **0.84**, resets **Aug 10 21:00 PDT**.
  Reading is ~26h stale and *conservatively* so (see §A2).
- **Daemons** (all argv-element-exact; **never `pgrep -f`**):
  `supervisor.sh` 15481 ppid 1 · `preserve_evidence.sh` 15482 ppid 1 ·
  `heartbeat.sh` 28367 ppid 1 · `heartbeat_watch.sh` under the Monitor ·
  `context_watch.py` **re-armed 22:32 PDT pinned with `--transcript`**.
- **25/25 games banked** in `scratchpad/clean_rollouts/`. Driver exits at once.
- **Disk** 19 GB free; a 25-game sweep needs ~0.6 GB (635 MB measured).
- **Results ledger** `scratchpad/results.jsonl`, **107 rows across 12 batches**.

---

## §C. THE OPEN DECISION — bp35 validation run (blocks nothing else)

The operator asked to run one bp35 game "using the latest infra", then said:
*"Don't launch bp35 until you confirm that you've done all you could on the
auditing."* **Nothing has been launched.** I started it once at 20:59 PDT,
killed it two minutes later when told to audit first, and removed its directory.

**Do not launch without an explicit go.** Two shapes, same cost (~$60–80 —
bp35's two prior runs cost $57 and $79, *not* the ~$26 sweep average):

1. `run_game` directly into an isolated dir with `CCARC3_NO_SHARED_CARD=1`.
   Validates the solver + analysis path. Opens no scorecard. Launcher is written:
   `scratchpad/run_validate_bp35.sh`.
2. `CCARC3_ONLY=bp35 CCARC3_SWEEP_DIR=clean_rollouts_validate tools/clean_rollouts.py`
   — the **real driver**, so it also exercises `sweep_card`, cookie binding, the
   skip-if-banked guard, corroboration and the bank path. Opens a throwaway card.
   **This is the one I recommend**: all three of today's sweep-aimed defects lived
   in that machinery, and (1) validates the part I am least worried about.

Caveat for (2): `MAX_PASSES = 12`, so an interrupted game can be retried up to 12
times. For a smoke test, cap it lower.

---

## §D. Task #35 — the 25-game sweep (the actual remaining work)

    CCARC3_SWEEP_DIR=clean_rollouts_submission tools/clean_rollouts.py

~$652. **Blocked on the quota clock, not on code**, until Aug 10 21:00 PDT.
A submission is one `scorecard_url`; nothing already banked can be retro-fitted
onto one card, because a card is state on a single backend instance reachable
only via its `AWSALBAPP-*` stickiness cookies.

**Three defects found this session were aimed squarely at this run**, all the same
shape — the sweep-directory name written as a literal in one place and derived
from `CCARC3_SWEEP_DIR` in another, each silent under a renamed sweep:

| what | commit | what it would have cost |
|---|---|---|
| restore wrote where the driver does not read | `236d0e5` | after a container replacement it restored banked games to `clean_rollouts/` while the driver read `clean_rollouts_submission/`, found nothing, and **re-ran games already on the shared card** — money twice, and a card cannot un-play a game |
| `refresh_audit`'s rebuild dispatch matched the literal | `1eaf3c0` | emitted `--ingest <game dir>`, which holds only `clean_result.json`; the audit announces new results and the artifact never changes |
| the ledger recorded this arm at all | `9595352` | 25 banked games, **zero rows**; recovering them added 30 |

Verify before spending: `tools/verify_one_card.py <sweep>` — and note it now
refuses an empty sweep and any game recording no `card_id` (`5480fe6`).

---

## §E. What was fixed, by class

Full detail is in the commit messages. The classes, because they recur:

**1. A guard that passes by not running** (the dominant class, ~10 instances)
- supervisor launched a runner with **no API key** — `.env` sourced unchecked in
  a file with no `set -e`; the guaranteed state of a fresh container (`a04c0f5`)
- the API-key leak guard resolved paths against the **caller's cwd**, so from the
  wrong directory it returned CLEAN with a key in a staged file (`7b1f0dc`)
- the reap clock was never asserted as **wound**; without the stamp
  `last_touched` stays 0.0 and `if self.last_touched:` skips the check entirely
  on every resume (`648975a`)
- `ARCPRIZE_API_KEY` strip: the assertion existed and **could not fail**, because
  the test never set the variable (`9376cff`)
- restore banked runs carrying an `error`, which the driver then skips forever
  (`5480fe6`)
- `verify_one_card` passed an **empty sweep** as submittable (`5480fe6`)

**2. Stale enumerated lists** (5 instances) — a list of "the arms of the day"
that goes out of date with no symptom: the snapshot ledger's batches (`9595352`),
the live panel's cwd filter (`4637095`), the box fingerprint's work count
(`b47f8c9`), and the UNRECORDED warning's own name list, which **could not stop
growing** until replaced with a structural game-id check (`1478dd1`).

**3. Substring process matching** — `pgrep -f claude` in `supervisor.sh`'s orphan
sweep matched **6 live pids** (this session's CLI, `context_watch.py`, the
environment manager, a shell) against **0** for the argv-exact form, because the
scratchpad path is `/tmp/claude-0/…` (`6bbe55d`).

**4. Prose-anchored checks** — two audit findings permanently "REOPENED", one of
which **required a leaked figure (`190/h`, tn36's) to still be present**: closing
it as written would have put a real median back into the doctrine every solver
reads (`aee8e7b`).

**5. Durability** — the shared card was the one critical file not written
atomically (`321ae83`); `snapshot_results.py`, whose entire purpose is surviving
data loss, lived **only in the scratchpad**, the one store that reverts
(`9595352`).

---

## §F. The audit method, and what it covered

Mutation testing: break the guard, confirm a test fails. **A guard nothing kills
is untested, whatever its coverage says.**

| module | mutants | real defects |
|---|---|---|
| `scoring.py` | 9 | 1 (dead branch awarding the cap for a zero-action level) |
| `client.py` | 7 | 1 (reap clock) |
| `session.py` | 6 | 1 (`ARCPRIZE_API_KEY`) |
| `arc_proxy.py` | 7 | 0 |
| `rules.py`, `grids.py`, `gate.py`, `ledger.py` | 11 | 0 |
| `clean_rollouts.py` | 4 | 3 untested guards |
| `proofread_trace.py` | 5 | 0 |
| `verify_one_card.py` | 7 | 2 |
| `restore_clean_rollouts.py` | 3 | 1 |
| `cli.py` | 4 | 4 — **but nothing automated calls it** |

**Two documented equivalent mutants** (unreachable, not untested; both recorded
in source): `capped()`'s `>` → `>=` — no IEEE double squares to exactly 1.15;
and the reap deadline's `>` → `>=` — `idle` is a difference of two `time.time()`
floats.

**Unaudited and on the path:** `build_trace_audit.py` beyond `live_games`, and
`rerun_losses.py`.

---

## §G. Traps — things that will mislead a fresh session

- **A background mutation pass makes `git add -A` unsafe for its whole duration.**
  A stop hook saying "commit your changes" points straight at a mutant. Caught by
  committing with an explicit pathspec; the mutant in the tree at that moment
  removed `env.pop("ARCPRIZE_API_KEY")`. Run mutation passes in the foreground.
- **`daemon_check` reports `context_watch.py` DOWN when it exits normally.** That
  watcher **exits on its first threshold crossing by design** — the exit *is* the
  alarm. Re-arm it; do not treat it as a crash.
- **Any inline process check you write is subject to the self-match trap**, not
  just the ones in the repo. I wrote `grep 'mut_cli\|mutate_'` as a safety probe
  and it matched **its own shell**, reporting a mutation running that had already
  exited. Check the thing you care about (does the tracked source differ from
  HEAD) rather than proxying it through "is a tool running".
- **`levels_completed` on a scorecard is ONE ENTRY PER PLAY holding that play's
  level count** — not a list of level indices. A fixture using `range(levels)`
  reads as N plays whose best is N-1 and refuses every restore.
- **A test suite where every case asserts a refusal can be entirely vacuous.**
  My restore fixtures lacked a preserved scorecard, so nothing was restored at
  all and the guard test passed; only a clean-run **control** failing beside it
  revealed that. Always include the positive control.
- **Monitors clamp to 30 minutes** whatever you request (`persistent: true` and
  `timeout_ms: 3600000` both return `timeout 1800000ms`). Re-arm on the timeout
  notification; that is routine, not a fault.
- **`docs/ccarc3_handoff_0808.md` §F was corrected in `51e876c`** — it used to
  tell a fresh session the heartbeat was "Monitor-hosted" and to re-arm the
  Monitor on it, which is exactly the coupling that killed it.

---

## §H. Operator constraints still in force

- **Never open a PR.** Push only to `claude/athanor-cc-harness-variant-jpqw7t`.
- **Opus 5 only, high effort.** Not Opus 4.8, not Fable 5.
- **No control arm.** All runs after the 13th scored game are baseline-free.
- **Don't touch `b6f77b65`** — it is buggy.
- API key lives in `scratchpad/arc3/.env` (mode 600), **never in the repo**.
- Every timestamp in a report or commit is **America/Los_Angeles**, stamped at
  the source (`TZ=America/Los_Angeles date`), not converted by hand — see
  `CLAUDE.md`.
- The operator is weighing a **second Max 20x account** for the rollout. Auth
  here is host-managed (`CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST`, token on an
  inherited fd), so the supported path is to **launch from a session started by
  that account**, not to swap credentials into this container. Portability work
  landed today makes that viable: `CCARC3_SCRATCH`, `CCARC3_REPO`,
  `CCARC3_BRANCH`, `CCARC3_SWEEP_DIR`, `CCARC3_ONLY`, and repo paths derived from
  `${BASH_SOURCE[0]}` / `__file__` throughout.

---

## §I. Numbers worth not re-deriving

- **VM lifetime** Aug 1 02:16 PDT → Aug 8 22:34 PDT = **7.8 days**, 23 distinct
  container boots (`evidence/box_fingerprint.tsv`, 33 rows).
- **Tokens, all 566 transcripts**: input 0.35 M · output **31.52 M** · cache write
  97.09 M · cache read **6,281.7 M**. Cache reads are **97.99%** of 6.41 B total.
- **Solver runs** 191 with a recorded `cost_usd`, **$2,471** by Claude Code's own
  accounting.
- **Per-game cost** ~$26 average; bp35 specifically $57 and $79.
- **A 25-game sweep** ≈ $652, ~635 MB.
- **Cost per category, Opus list / applied**: interactive $6,390/$2,130 · solver
  mains $5,414/$1,805 · session subagents $931/$310 · workflow agents $670/$223 ·
  solver-spawned subagents $45/$15.


---

## §J. Artifacts that exist nowhere else

These are here because they are **not files that will survive**: the mutation
harnesses live in `/tmp`, and `scratchpad/` reverts to an image snapshot when the
container is replaced. Everything else in this session is in git.

### J1. The mutation harness pattern

Every audit in §F used this shape. Run it in the **foreground** (§G: a background
pass makes `git add -A` unsafe). Scope `FAST` to the module's own tests, and
escalate an apparent survivor to the whole suite before believing it.

```python
import pathlib, subprocess
FAST = ["tests/test_x.py", "tests/test_y.py"]          # module's own tests
def run(sel):
    return subprocess.run([".venv/bin/python","-m","pytest","-q",*sel],
                          capture_output=True, text=True, timeout=900).returncode

P = pathlib.Path("src/athanor/ccarc3/target.py"); BAK = P.read_text()
MUTS = [("human-readable name", "exact old text", "mutated text")]

killed, survived, skipped = [], [], []
for name, old, new in MUTS:
    if BAK.count(old) != 1:                 # anchor must be unique or the
        skipped.append(f"{name} (anchor {BAK.count(old)}x)"); continue   # mutation is a lie
    P.write_text(BAK.replace(old, new))
    rc = run(FAST) or run([])               # escalate before calling it a survivor
    (killed if rc else survived).append(name)
    P.write_text(BAK)                       # restore after EVERY mutant
P.write_text(BAK)
```

Three failure modes I hit, all of which produce a false "SURVIVED":

- **A mutation that does not mutate.** `HIDDEN_FIELDS = frozenset(set()) or frozenset({...})`
  evaluates to the *original* set, because an empty frozenset is falsy. Always
  print the mutated line and confirm it changed.
- **A mutation that does not parse.** Splitting `ws.env.pop(...)` on `env` gave
  `ws.pass` — the "failure" was a collection error, not a killed mutant.
- **An unrelated failing test.** A broken test of my own failed under *every*
  mutant, so the harness credited its death to whichever one was running. Confirm
  the suite is green before starting.

### J2. Exact re-run commands

```bash
cd /home/user/athanor
.venv/bin/python -m pytest -q                          # 1107 pass, ~96 s
timeout 300 .venv/bin/python tools/verify_audit_findings.py    # 32/32 closed
timeout 600 .venv/bin/python tools/leak_exposure.py            # 77 streams, 0 exposures
ARC_API_KEY=dummy .venv/bin/python tools/snapshot_results.py   # silent = no new arm
bash "$SP/quota.sh"                                    # five_hour + seven_day
bash "$SP/refresh_audit.sh"; echo $?                   # 0 nothing new, 10 new results
tools/verify_one_card.py "$SP/clean_rollouts"          # expect SPLIT: pre-shared-card
```

`$SP` = `/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad`
on this box; every tool now honours `CCARC3_SCRATCH` instead.

### J3. The bp35 validation launcher (scratchpad copy will not survive)

```bash
#!/bin/bash
set -uo pipefail
REPO=/home/user/athanor
S=<scratchpad>
set -a; . "$S/arc3/.env"; set +a
[ -f "$S/proxy_env" ] && . "$S/proxy_env"
export CCARC3_NO_SHARED_CARD=1          # option 1 only; drop this for option 2
cd "$REPO" || exit 1
exec .venv/bin/python - <<'PY'
import pathlib, sys, os, time
sys.path.insert(0, "/home/user/athanor/src")
from athanor.ccarc3 import Ccarc3Config
from athanor.ccarc3.session import run_game
out = pathlib.Path(os.environ["CCARC3_SCRATCH"]) / "clean_rollouts_validate" \
      / "bp35-0a0ad940" / "attempt_1"
run_game(Ccarc3Config("bp35-0a0ad940", out_dir=out, budget_multiple=5.0,
                      wall_clock_timeout_s=6*3600, fresh=True))
PY
```

Launch it double-forked so it reparents to init (§A3):
`( setsid bash "$S/run_validate_bp35.sh" >> "$S/validate_bp35.log" 2>&1 < /dev/null & )`

**`exec` replaces the shell's argv**, so afterwards it is `.venv/bin/python -`,
not the script name — do not look for the launcher's name in `/proc` to check it
started. Look for the `claude` solver and its cwd.

### J4. Relaunch block for the daemons

```bash
cd /home/user/athanor
( setsid bash tools/supervisor.sh        >> "$SP/supervisor.log" 2>&1 < /dev/null & )
( setsid bash tools/preserve_evidence.sh >> "$SP/preserve.log"   2>&1 < /dev/null & )
( setsid .venv/bin/python /root/.claude/skills/context-handoff/scripts/context_watch.py \
    --transcript /root/.claude/projects/-home-user-athanor/$CLAUDE_CODE_SESSION_ID.jsonl \
    >> "$SP/context_watch.log" 2>&1 < /dev/null & )
```
Then arm a Monitor over **`tools/heartbeat_watch.sh`** — never over
`heartbeat.sh` itself. Confirm each shows **ppid 1**.

Refresh `$SP/proxy_env` from the live session whenever the worker restarts; the
supervisor refuses to launch into a dead proxy and says so. That fired for real
at 20:09 PDT today (`:35889` → `:36327`) and the refusal was correct.
