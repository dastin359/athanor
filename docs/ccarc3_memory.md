# START HERE — read the handoff in full before acting

**`/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad/project_ccarc3_session_state_0808.md`** — ~25 KB, ~6.3k tokens. Open it
NOW, in full, before acting on anything below. Its **section K (RETRACTIONS)**
lists claims made confidently in that session and later disproved, including a
**published score that was withdrawn**. Its **APPENDIX** carries the finding
that our **world-model API went unused** — 1 of 43 runs called `arc.predict`,
0 called `shortest_path` — though 14 of 43 wrote their OWN forward model, so the
stronger "our solvers don't build world models" claim is RETRACTED in L — and **section P** the shared scorecard: built and
verified live, with **two more retractions**, one of which killed section M's
own plan (injecting `card_id` alone cannot work; a card is pinned to one
backend by `AWSALBAPP-*` stickiness cookies) and one of which corrected a false
claim that had been sitting in `client.py` (**a closed card DOES survive** — a
404 means you asked the wrong instance). A fresh session that skips it inherits
the confident wrong version and re-derives nothing, because nothing looks broken.

**Next action, when quota allows** (7-day window was at util=0.83): the
25-game sweep onto one card — `CCARC3_SWEEP_DIR=clean_rollouts_submission
tools/clean_rollouts.py`. Nothing banked can be retro-fitted onto a shared
card. See P.5 for the two risks worth knowing before spending ~$551.

**ALL TIMES IN REPORTS ARE PACIFIC.** This was already recorded at line ~125
and was still violated on 2026-08-08, because probe logs and `date` return UTC
and get pasted through unconverted. The rule is not the fix — the fix is that
anything which writes a timestamp writes it in Pacific. Never report a raw
`date`/`%H:%M` from this box without converting.

**START WITH THE REPO, NOT THIS FILE.** `docs/ccarc3_open_findings.md` lists 12
unadjudicated proofread findings, several sweep-blocking, and it survives a
container replacement — the scratchpad does not. Then §Q of the handoff for the
night of 08-08→09: ~30 defects fixed, three live median leaks closed (no banked
result moved), and five errors of mine worth knowing before trusting my summaries.

**This file is a routing table, not the handoff.** A hook is a reason to open a
file, never a substitute for having opened it.

Base directory for unqualified paths below: `/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad`.

---

# CCARC3 — what a fresh session needs to know

Written 2026-08-08. The previous version was from 08-02 and six days stale; if
this one is more than a day old, trust the repo over it. **`docs/ccarc3_results.md`
and the git log are the record; this file is orientation.**

## The experiment

Claude Code plays ARC-AGI-3 through a harness that withholds the per-level human
medians (`baseline_actions`), so the solver cannot steer by the number it is
scored against. RHAE: per completed level `S = min(1.15, (h/a)^2)`; `raw` is the
weighted mean by level index; `cap = sum(1..completed)/sum(1..n)`;
**`E = min(cap, raw)`**; the server scores the **best play** across replays.

### Standing result — quote this, not anything else

| | environments | score |
|---|---|---|
| **baseline-free, repaired surface** | **17 of 25** | **16.7252 / 17 = 98.38%** |
| contaminated (a median was in the doctrine they read) | 8 | **not poolable** |

**A 25-environment figure of 96.53% was published and is withdrawn.** Its eight
non-repaired contributors ran while `DOCTRINE.md` §0a spelled out
`65.533 = (17/21)^2 x 100` — and 17 is a real per-level median. All eight started
before the 05:47:33Z 2026-08-07 fix; their preserved workspaces still contain the
string. None of the 17 is affected.

16 of the 17 scored exactly 1.0000. The one below is `bp35` at 0.7252, the only
run in the cohort with a single play: it cleared 9 of 9, read `[cap 1.000 = 9/9
levels]` as a perfect score, and stopped one action before `restart_for_replay()`
became illegal.

## What is running

Check, do not assume — `AUTOPILOT.md` has the argv-exact liveness idiom.

* `tools/supervisor.sh` — **the repo copy, never `scratchpad/supervisor.sh`**,
  which is an Aug-3 fossil with no proxy validation that starts the wrong runner.
* `tools/preserve_evidence.sh` — copies run artefacts into `evidence/` every 300s
  and pushes. Sources the key from `arc3/.env` itself.
* `$SP/concurrency` — live-tunable; **0 means the driver launches nothing**.
* A `bp35` rerun under the repaired surface may be in flight in
  `$SP/rerun_bp35_fixed`. It is a demonstration, not a measurement: `bp35`
  already scored 1.0000 under the *old* harness, and all 25 games have scored
  1.0000 at some point, so a 1.0000 here proves nothing. What is worth reading
  from its trace is whether the solver **called `restart_for_replay()` at the win
  frame** — the mechanism, not the score.

## How to re-answer "is everything fixed?"

    python tools/verify_audit_findings.py     # 32 findings, exits 1 if any reopened
    python tools/leak_exposure.py             # preserved streams vs known leak strings
    python -m pytest tests/ -q                # ~713

**Run them. Do not answer from memory** — that produced two wrong answers in one
session, both of the form "I edited that file" standing in for "I fixed that
finding".

## The failure mode this project keeps producing

**A guard that checks a proxy for the thing rather than the thing.** Nine
instances so far: a plaintext grep over a gzipped tree; an exit code read as a
verdict; a digest presented as a date filter; depth-and-cost used for an RHAE
maximum; a leak scan over only the files it had just edited; a liveness check
whose pattern matched its own command line; a median test that read comments and
strings while the array sat in code; a docstring guard that matched prose near
the code instead of the code; `session.ASSETS` exporting the unstripped doctrine
while every check read the stripped workspace copy.

When adding a check, **mutation-test it**: break the thing, confirm it fires.

Second recurring shape: **value-based leak detection does not work.** ~200
medians over 1..400 make the space dense — `97.1` is `100*(67/68)^2`. Ban the
shape, or use provenance. Both attempts to match values voided most of a clean
corpus.

## Operational traps, all measured

* **A VM replacement kills everything local**; a worker restart does not. Tell
  them apart with `ps -o lstart= -p 1` — a new kernel means a new VM. The disk
  survives both. Previous container lived 3h39m; another once ran 10.78h.
* **A daemon keeps the code it started with.** Editing a script changes nothing
  until the process is replaced. This produced the phantom-directory regression
  and the stale-proxy incident.
* **The agent proxy port changes on restart.** Refresh `$SP/proxy_env` from the
  live session's `HTTPS_PROXY` on every check.
* **Monitors cap at 30 minutes** whatever timeout you request, and die with the
  session. Re-arm them; put durability in a detached process or in git.
* `git add -A` can sweep in files the preserver wrote mid-commit.

## Constraints from the operator

* No PR, ever. Push to `claude/athanor-cc-harness-variant-jpqw7t`.
* API key lives in `$SP/arc3/.env`, mode 600. **Never** into the repo.
* Opus 5 only, high effort. No control arm. All runs baseline-free.
* Do not touch `b6f77b65`.
* Pacific time in reports.
