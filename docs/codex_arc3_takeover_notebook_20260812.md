# CodexARC3 takeover notebook — 2026-08-12

This is the self-contained handoff for the next coding agent maintaining the
Codex ARC-AGI-3 path in Athanor. It supersedes chat history and is the first file
to read. Older CCARC3 documents remain valuable design history, but their
Claude-specific runtime, paths, process names, and operational state are not
instructions for this branch.

No live ARC operation is required to verify this handoff. Do not open a game,
create a scorecard, or spend ARC actions merely to reproduce an existing proof.

## 1. Snapshot at handoff

| Item | Verified value |
|---|---|
| Repository | `https://github.com/dastin359/athanor.git` |
| Checkout | `/home/dastin/dev/codex_workspace/athanor-codex` |
| Branch | `codex/port-arc-agi-3-harness-logic` |
| Local and remote HEAD | `45eaab2b95be1602447f873889de2749e4ec2efc` |
| Main operations port | `fc20ece7cc58501b4b2d6ee8e567312ed66b4e17` |
| Original handoff commit | `b0e21a18798365d592285fc2e869f9bb165b5e81` |
| Latest checked CC branch | `claude/athanor-cc-harness-variant-jpqw7t` |
| Latest checked CC tip | `5d23563ddbd32c43dc0d486fff4945e709fd261c` |
| Python in `.venv` | 3.12.7 |
| Codex CLI | 0.147.0 |
| GitHub CLI | 2.94.0, workspace-local |
| Final full suite | `1613 passed, 9 skipped, 1 xfailed, 1 warning` |
| Working tree | Clean and synchronized when handed off |
| Live ARC processes | None launched by this handoff |
| Evidence archive | Not present under `/home/dastin/dev/codex_workspace` |

The one warning is Python 3.12's multiprocessing/fork deprecation in
`test_the_lock_is_released_when_the_holder_dies`; it is not a harness failure.

Verify before doing anything:

```bash
cd /home/dastin/dev/codex_workspace/athanor-codex
git fetch origin
git status --short --branch
git rev-parse HEAD
git rev-parse origin/codex/port-arc-agi-3-harness-logic
git rev-list --left-right --count \
  HEAD...origin/codex/port-arc-agi-3-harness-logic
```

Expected: branch name as above, both revisions equal `45eaab2b...`, divergence
`0 0`, and no worktree entries.

## 2. Non-interference boundary

The operator limited this work to:

```text
/home/dastin/dev/codex_workspace
```

Honor that boundary unless the operator explicitly changes it. In particular:

- Never open, edit, or run commands in any checkout named
  `Agentic-ARC-Solver`; that belongs to a Cursor agent.
- Other agents may be working on Claude/Cursor-native ARC harnesses on this
  machine. Do not inspect their workspaces, processes, secrets, or artifacts.
- Do not run two coding agents concurrently in `athanor-codex`. If parallel work
  is needed, make a distinctly named fresh checkout inside
  `/home/dastin/dev/codex_workspace` and coordinate branches explicitly.
- Never kill a process merely because its executable or script basename looks
  familiar. Codex supervisor and resume logic deliberately match the exact
  resolved driver path to avoid treating another checkout as this one.
- Use `apply_patch` for source edits and preserve unrelated worktree changes.

## 3. Credentials and publication

No credential is stored in Git, this notebook, the solver workspace, or the
evidence archive.

The operator reported that the ARC credential is normally stored at:

```text
~/ccarc3-scratch/arc3/.env
```

That location is outside the allowed Codex workspace. It was deliberately not
read during the operations port. Do not access it unless the operator explicitly
authorizes a live ARC operation. The credential was also pasted into chat during
the prior session, so rotation is recommended before future scored work.

Rules that are not negotiable:

- Never print an ARC key.
- Never commit it.
- Never copy it into a solver workspace.
- Never pass `ARC_API_KEY`, `ARCPRIZE_API_KEY`, or `CCARC3_MAX_ACTIONS` to the
  solver child.
- Never use a credential simply to make tests stop skipping. Offline structural
  tests are the default.
- Any live command must be explicitly authorized for that specific operation.

GitHub CLI is installed here:

```text
/home/dastin/dev/codex_workspace/.tools/bin/gh
```

Its workspace-scoped configuration directory is:

```text
/home/dastin/dev/codex_workspace/.tools/gh-config
```

No persistent authenticated session was left behind. A GitHub PAT was pasted
into chat and used only through an ephemeral shell to push the final commits.
Treat it as exposed and rotate it. Authenticate independently; do not recover or
reuse a token from transcripts.

Publishing policy:

- Commit intentionally.
- Push only `codex/port-arc-agi-3-harness-logic` unless the operator directs a
  new branch.
- Never force-push.
- Do not open a pull request unless the operator asks.

## 4. What was built

The work began as a runtime port of the latest CC ARC-AGI-3 harness to Codex and
now includes the post-port operational safeguards from the CC branch.

Important commit chain, oldest to newest:

| Commit | Purpose |
|---|---|
| `e3ffcf5f` | Initial ARC-AGI-3 harness port to Codex |
| `4b38ed97` | Port latest CC domain/harness safeguards |
| `bfcfdcfc` | Isolate Codex from host config, rules, and memory |
| `b0e21a18` | Original continuation handoff |
| `b33a05fb` | Fail-closed clean single-game command |
| `53b878a4` | Update continuation handoff |
| `fc20ece7` | Port all eight newer CC operations safeguards |
| `45eaab2b` | Update operational handoff |

The eight substantive CC commits incorporated by `fc20ece7` were:

| CC commit | Ported behavior |
|---|---|
| `d54a5570` | Cwd-independent tests/tools and desktop bootstrap |
| `0ac852e9` | Card verification refuses when completeness is unknowable |
| `ba45349a` | Supervisor revives the daemon watchdog |
| `51d59552` | Single-instance supervisor lock with correct fd lifetime |
| `c78f29da` | Single-instance preserver lock with correct fd lifetime |
| `09a8457a` | Portable orphan ownership, real outbound preflight, portable tests |
| `0940d978` | Discarded attempts do not count as card games; external resume guard |
| `3268b9a3` | Python 3.12 f-string leak audit and honest test gating |

At the final comparison, the 38 CC commits after `3268b9a3` were evidence-only.
No newer change remained in `src/athanor/ccarc3` or
`tools/ablate_baselines.py`. Recheck this rather than assuming it remains true:

```bash
git fetch origin claude/athanor-cc-harness-variant-jpqw7t
git log --format='%h %aI %s' \
  3268b9a3..origin/claude/athanor-cc-harness-variant-jpqw7t \
  | grep -v ' evidence:'
git diff --name-status 3268b9a3..origin/claude/athanor-cc-harness-variant-jpqw7t \
  -- src tools/ablate_baselines.py
```

Review new CC changes semantically. Do not cherry-pick Claude-specific literals,
CLI arguments, process names, secret lookup, or scratch paths into Codex.

## 5. Architecture map

### ARC domain and protocol

- `src/athanor/ccarc3/client.py` — HTTP client, game metadata, baseline hiding,
  ARC root selection, session/card operations.
- `src/athanor/ccarc3/arc_proxy.py` — loopback allowlisting proxy. The parent
  holds the ARC credential; the solver receives only a per-game proxy URL.
- `src/athanor/ccarc3/ledger.py` — accepted-command ledger and action accounting.
- `src/athanor/ccarc3/scoring.py` — RHAE/best-play scoring.
- `src/athanor/ccarc3/rules.py`, `gate.py`, `planning.py`, `grids.py` — gameplay
  mechanics and solver-facing helpers.

### Codex runtime seam

- `src/athanor/ccarc3/session.py` — workspace construction, `codex exec --json`,
  JSONL event parsing, process-group lifetime, outcome collection, and bounded
  `codex exec resume` nudges.
- `src/athanor/ccarc3/assets/` — solver-facing `AGENTS.md`, doctrine, and helper
  assets.
- `src/athanor/ccarc3/cli.py` — `games`, `run`, `clean-run`, `batch`, `report`,
  `trace`, and `workspace` commands.

Current defaults in `Ccarc3Config`:

```text
model                  gpt-5.6-sol
effort                 high
budget_multiple        5.0
level_budget_multiple  0.0 (disabled)
wall_clock_timeout_s   7200
sandbox                workspace-write
network_access         true
fresh                  false (resume by default)
```

Codex supplies the thread id in `thread.started`; that id is used for resume
nudges. Token usage comes from `turn.completed.usage`. Do not invent Claude-style
dollar-cost or duration fields that Codex JSONL does not provide.

### Clean boundary

- `src/athanor/ccarc3/clean.py` — first-class, fail-closed single-game clean
  boundary.
- `tools/ablate_baselines.py` — audited workspace strip plus per-game proxy and
  action cap. Importing it does **not** install it; callers must execute
  `install()` and `assert_installed()`.
- `tools/clean_rollouts.py` — benchmark-clean multi-game/shared-card driver.
- `tools/verify_one_card.py` — fail-closed card completeness/split verifier.
- `tools/proofread_trace.py` — post-run contamination and trace proofreader.

### Unattended operations

- `tools/supervisor.sh` — quota duty cycle, exact-path runner ownership,
  read-only ARC reachability preflight, singleton lock, polite stop, watchdog
  repair.
- `tools/daemon_watchdog.sh` and `tools/heartbeat.sh` — daemon repair/reporting.
- `tools/preserve_evidence.sh` — evidence-only preservation and publication,
  protected by a repository-keyed singleton lock.
- `tools/desktop_bootstrap.sh` — Codex-aware desktop verification. Its default
  mode includes live CLI/API probes; use `--no-live` for offline verification.
- `tools/ccarc3_resume.sh` — external/WSL recovery entrypoint. It requires a
  positive brake, persisted sweep identity, exact-path no-duplicate check, and a
  credential before it launches anything.

## 6. Benchmark-clean invariants

The plain command is not benchmark-clean:

```bash
athanor ccarc3 run --game GAME_ID
```

It intentionally preserves historical callers and does not guarantee that the
baseline strip and credential-withholding proxy were installed. Never use it for
a scored clean run.

The explicit clean command is:

```bash
athanor ccarc3 clean-run --game GAME_ID
```

Do not execute it without explicit permission to spend ARC actions. Before every
initial launch and every resumed/nudged launch, `assert_clean_launch()` refuses
unless all of these are true:

1. The workspace is outside every Git repository.
2. The workspace contains no symlinks.
3. `ARC_API_KEY`, `ARCPRIZE_API_KEY`, and `CCARC3_MAX_ACTIONS` are absent from
   the child environment.
4. `CCARC3_HIDE_BASELINES=true` is present.
5. `CCARC3_ARC_ROOT` is an HTTP loopback URL for this game's active proxy.
6. `CCARC3_PROXY_URL` is not disclosed to the child.
7. `meta.json` contains neither `baseline_actions` nor `action_budget`.
8. `session.py` contains `baseline_actions=()`.
9. No workspace file contains the parent credential; the scan does not follow
   symlinks and reports filenames, never secret values.
10. The proxy is installed and enforces the configured game-wide cap.
11. Every Codex command carries:

```text
--ignore-user-config
--ignore-rules
--disable memories
--skip-git-repo-check
```

12. The command is an expected initial `codex exec --json` or a bounded
    `codex exec resume` launch.

The default clean workspace is:

```text
${TMPDIR:-/tmp}/athanor-ccarc3-codex/clean-single
```

For this machine's strict workspace boundary, set `TMPDIR` or `--out-dir` to a
dedicated external solver directory under `/home/dastin/dev/codex_workspace`
that is not inside the repository. Do not point a solver at the source checkout.

`network_access=true` is needed for loopback access to the ARC proxy, but Codex's
workspace-write sandbox exposes network access at sandbox scope rather than by
destination. The proxy protects the ARC credential and ARC route; it is not a
general host-network firewall. Keep the rollout machine free of unrelated
credentials and services, and do not weaken solver-facing integrity rules.

## 7. Action and score accounting

Do not conflate accepted commands with billed actions.

- The proxy reserves one command slot for every accepted ARC command, including
  a play-opening RESET.
- ARC's billed `actions_used` excludes one opening RESET per play.
- Resume seeds the proxy counter from nonempty `trace.jsonl` rows where possible,
  with old state as a compatibility fallback.
- Scoring selects the best play, attributes cumulative action differences to the
  correct level, separates replays, handles RESETs, caps zero-cost scores, and
  respects sequential completion. The eight detailed scoring invariants are in
  `docs/ccarc3_port_guide.md` section 3.

The action budget is intentionally withheld from the solver because it is an
invertible function of the published human baseline. It remains enforced in the
parent proxy.

## 8. Authoritative live proof

Do not rerun this merely to reproduce proof.

```text
Game                 cd82-fb555c5d
Result               6/6 levels won
Configured ceiling   855 command slots (5 x published total)
Winning plays        124, 85, 85 billed actions
Total billed         294
Accepted commands    297
Exit                 0
Timeouts/deaths      none
```

The three-command difference is the three play-opening RESETs. The strict child
received neither the ARC key nor the action cap. Baselines and host Codex
configuration/memory were hidden.

An earlier winning run was contaminated because Codex inherited host config and
read global Codex memory. Commit `bfcfdcfc` fixed both initial and resumed
launches. The strict isolated replacement run above is the authoritative result.

## 9. Setup and offline validation

Fresh clone, if the existing checkout is unavailable:

```bash
cd /home/dastin/dev/codex_workspace
git clone --branch codex/port-arc-agi-3-harness-logic --single-branch \
  https://github.com/dastin359/athanor.git athanor-codex-next
cd athanor-codex-next

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Existing checkout:

```bash
cd /home/dastin/dev/codex_workspace/athanor-codex
source .venv/bin/activate
python --version
codex --version
codex exec --help | grep -E -- \
  '--ignore-user-config|--ignore-rules|--disable|--skip-git-repo-check'
```

Run tests with temporary files inside the allowed workspace:

```bash
mkdir -p /home/dastin/dev/codex_workspace/.test-tmp
TMPDIR=/home/dastin/dev/codex_workspace/.test-tmp \
  python -m pytest -q
```

Last exact result:

```text
1613 passed, 9 skipped, 1 xfailed, 1 warning in 241.76s
```

The full suite is offline by default. Some value-based median checks skip unless
a credential is explicitly provided through the environment or an explicitly
configured `CCARC3_SCRATCH`; they do not search the home directory implicitly.
Do not turn an offline verification into a live API operation just to reduce the
skip count.

Useful focused sets:

```bash
python -m pytest -q tests/test_ccarc3*.py tests/test_ablate_baselines.py \
  tests/test_clean_rollout_queue_order.py

python -m pytest -q tests/test_only_one_supervisor_runs.py \
  tests/test_only_one_preserver_runs.py \
  tests/test_the_supervisor_revives_the_watchdog.py \
  tests/test_ccarc3_desktop_ops.py
```

The second group starts only controlled fixture processes. Its guards explicitly
disable real watchdog revival so a unit test cannot launch a paid sweep or an
evidence-publishing daemon.

## 10. Live operations — authorization required

Merely having the ARC key is not authorization to spend it.

For an explicitly authorized clean single game, prefer the first-class command:

```bash
export TMPDIR=/home/dastin/dev/codex_workspace/arc3-runtime
# Load ARC_API_KEY without printing it and only after explicit authorization.
athanor ccarc3 clean-run --game GAME_ID --fresh
```

For an explicitly authorized clean sweep, use `tools/clean_rollouts.py`, never
plain `athanor ccarc3 run`:

```bash
export CCARC3_SCRATCH=/home/dastin/dev/codex_workspace/arc3-runtime
export CCARC3_ONLY=GAME_PREFIX_OR_ID
export CCARC3_SWEEP_DIR=clean_rollouts_codex_NAME
export CCARC3_MAX_PASSES=1
export CCARC3_MAX_NUDGES=3
.venv/bin/python tools/clean_rollouts.py
```

Before allowing a real launch:

- Confirm `ablate_baselines.install()` and `assert_installed()` succeed.
- Confirm the output directory is outside the repository.
- Confirm `CCARC3_ONLY` selects exactly the authorized game(s); an empty match
  must abort.
- Confirm the brake/concurrency and retry bounds match the authorization.
- Inspect child environment and generated `meta.json` without printing secrets.
- Confirm initial and resume paths both retain every Codex isolation flag.
- Ensure no driver or supervisor for the exact resolved path is already running.
- State that a scorecard/action spend is about to occur before executing it.

The supervisor performs a live, read-only `list_games()` preflight before it
starts a driver. This is better than checking an agent-proxy health endpoint, but
it still uses the ARC credential and network; do not start the supervisor during
offline work.

## 11. Evidence

The prior machine's expected archive name and checksum are:

```text
athanor-codex-full-isolated-evidence-20260812.tar.gz
f6e02555dc3449059a7a1fde0f91893f3105a5278e60bdd01c334298cbf2e13d
```

It was not present under `/home/dastin/dev/codex_workspace` at handoff. If it is
later transferred, verify it before extraction:

```bash
sha256sum athanor-codex-full-isolated-evidence-20260812.tar.gz
```

The archive should contain workspace/run evidence but no ARC key. Do not commit
the archive or unpack it into the repository without an explicit reason and a
secret scan.

`tools/preserve_evidence.sh` is powerful: it can commit evidence and push to a
public remote. Do not start it casually. It refuses the wrong branch and uses a
repository-keyed singleton lock whose children close the lock descriptor. Tests
run it only against throwaway repositories.

## 12. Known limitations and next work

There is no known failing test or incomplete upstream CC mechanism as of the
snapshot above. Sensible next work, in descending order:

1. Fetch and audit any new non-evidence commits on the CC branch. Port behavior,
   not Claude-specific syntax.
2. Avoid attempting replay-gate acknowledgement when no gate is pending, while
   preserving the current no-action-spent behavior.
3. Improve defense-in-depth around Codex's sandbox-wide network access without
   breaking required loopback proxy connectivity.
4. If extending clean commands to new modes, reuse `launch_validator` so both
   initial and `exec resume` paths fail closed. Never create a second, drifting
   copy of the isolation checks.
5. Keep desktop/bootstrap and external-resume tests offline and exact-path
   scoped. A test must never be able to spend ARC actions, publish evidence, or
   detect another checkout as its own runner.

When changing any guard, add a positive control. This codebase repeatedly found
checks that passed because they looked at nothing: missing env variables,
cwd-relative imports, absent arrays, empty process scans, or exception handlers
that converted “cannot tell” into success. A test that only asserts silence or a
zero exit code is usually insufficient.

## 13. Working protocol

For every implementation change:

1. Verify branch, remote divergence, and clean worktree.
2. Read the relevant implementation and tests before editing.
3. Keep all writes inside the authorized workspace.
4. Make the smallest semantic port; preserve Codex runtime differences.
5. Run focused tests.
6. Run the full suite.
7. Inspect `git diff --check`, executable modes, untracked files, evidence paths,
   and credential-shaped strings.
8. Update this notebook when operational facts change.
9. Commit intentionally and push only the designated branch.
10. Do not open a PR or launch ARC unless explicitly requested.

Suggested secret scan before a commit:

```bash
rg -n 'github_pat_|ARC_API_KEY=[[:alnum:]_-]{16,}|ARCPRIZE_API_KEY=[[:alnum:]_-]{16,}' \
  --glob '!*.gz' --glob '!docs/codex_arc3_takeover_notebook_20260812.md' .
```

Expected hits are fixture-only strings in tests. Inspect every hit; never assume
a match is benign.

## 14. Required reading

Read these before modifying runtime or scoring behavior:

1. `docs/codex_arc3_takeover_notebook_20260812.md` — this notebook.
2. `docs/ccarc3_codex_handoff_20260812.md` — live proof and earlier handoff.
3. `docs/ccarc3_codex_port.md` — Codex runtime boundary.
4. `docs/ccarc3_port_guide.md` — scoring and ledger invariants.
5. `src/athanor/ccarc3/session.py` and `src/athanor/ccarc3/clean.py`.
6. `tools/ablate_baselines.py` and `tools/clean_rollouts.py`.
7. Tests covering the code you plan to change.

If this notebook conflicts with a newer explicit operator instruction, follow
the operator. If it conflicts with an older Claude/CC handoff, use this notebook
for Codex paths and runtime behavior.

## 15. First-turn checklist for the next agent

```text
[ ] Work only inside /home/dastin/dev/codex_workspace.
[ ] Do not touch Agentic-ARC-Solver or another agent's checkout.
[ ] Read this notebook and the three linked design/handoff documents.
[ ] Fetch both Codex and CC branches read-only.
[ ] Confirm local/remote Codex HEAD and a clean worktree.
[ ] Confirm Codex exposes all four isolation flags.
[ ] Run the offline full suite with workspace-local TMPDIR.
[ ] Report discrepancies before making speculative fixes.
[ ] Do not read the ARC key or launch ARC without explicit authorization.
[ ] Do not open a PR unless asked.
```
