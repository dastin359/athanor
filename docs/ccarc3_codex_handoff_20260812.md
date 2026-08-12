# Codex ARC-AGI-3 handoff — 2026-08-12

This is the current continuation document for the Codex port. Read it before
the older CCARC3 handoffs: those documents preserve useful design and audit
history, but their branch, runtime, process, quota, and "what is running" notes
describe the Claude Code deployment and are not current instructions for this
branch.

## Sixty-second status

| Item | Current state |
|---|---|
| Repository | `https://github.com/dastin359/athanor.git` |
| Branch | `codex/port-arc-agi-3-harness-logic` |
| Last verified code commit | `bfcfdcfcfb5d95ed73248aa68c31591c3cba6e61` |
| Port commit | `4b38ed97f0d708d548a7d79ac5a0ab502898c0d6` |
| Tests | `1522 passed, 40 skipped, 1 xfailed` |
| Live verification | `cd82-fb555c5d`, won `6/6`, full 5x budget available |
| Live usage | 294 billed actions over three winning plays: 124, 85, 85 |
| Running processes | None belonging to this handoff |

The requested work is complete and pushed: the newer CC harness safeguards are
ported to the Codex runtime, the unit/smoke suite is green, and a strict clean
live run won the game. Do not spend another live run merely to recreate that
proof.

The old checkout `/Users/dastin/dev/Agentic-ARC-Solver` belongs to the Cursor
agent. It was not modified. On another machine, use a fresh clone with a
different directory name and do not point Codex at a checkout used by another
agent.

## Start on the new machine

Python 3.11 or newer and a Codex CLI that supports the isolation flags are
required.

```bash
git clone https://github.com/dastin359/athanor.git athanor-codex
cd athanor-codex
git fetch origin
git switch --track origin/codex/port-arc-agi-3-harness-logic

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

git status --short --branch
git log -3 --oneline
codex --version
codex exec --help | grep -E -- '--ignore-user-config|--ignore-rules|--disable|--skip-git-repo-check'
python -m pytest -q
```

Expected history includes `bfcfdcf Isolate Codex ARC runs from host context`
and `4b38ed9 Port latest CC ARC-AGI-3 harness safeguards to Codex`. Use the
virtual environment's `python`; a random system interpreter may not have
`athanor` or NumPy installed.

The API credential is intentionally absent from Git, this document, and the
evidence archive. If a later live run is explicitly wanted, set `ARC_API_KEY`
only in that terminal or a mode-600 file outside the repository. Never commit
it, print it into a transcript, or put it in a solver workspace.

## What changed in the port

The five ARC domain modules (`grids.py`, `ledger.py`, `rules.py`, `scoring.py`,
and the HTTP behaviour in `client.py`) remain the audited CC implementation.
The runtime seam in `session.py` launches `codex exec --json`, extracts the
thread id from `thread.started`, accounts tokens from `turn.completed`, and
uses `codex exec resume` for bounded give-up nudges. See
`docs/ccarc3_codex_port.md` and `docs/ccarc3_port_guide.md` for the detailed
boundary and scoring invariants.

Two Codex-specific bugs were found during live verification and fixed:

1. A run directory outside a Git checkout failed because Codex was launched
   without `--skip-git-repo-check`.
2. A winning run inherited host Codex configuration and read global Codex
   memory. That made the solver transcript inadmissible even though the game
   result itself was real. Initial launches and `resume` nudges now both pass
   `--ignore-user-config`, `--ignore-rules`, `--disable memories`, and
   `--skip-git-repo-check`. Tests cover both paths.

The first inherited-memory win is not the final proof. It was replaced by the
strict isolated run below, which also won. This is what references to the
"replacement run" mean.

## Strict live-run result

Game `cd82-fb555c5d` was run from a clean workspace with the full configured
budget ceiling of 855 command slots (`5 x` its published baseline total of
171). The solver completed all six levels in each of three plays:

| Play | Levels | State | Billed actions |
|---:|---:|---|---:|
| 1 | 6/6 | WIN | 124 |
| 2 | 6/6 | WIN | 85 |
| 3 | 6/6 | WIN | 85 |
| **Total** | | | **294** |

There were 297 accepted trace commands: the three play-opening RESETs are
accepted but not billed by ARC, hence the difference from 294. There were no
deaths or wasted actions. The process exited 0 and did not time out. The final
scorecard reported 6/6, three plays, and 294 total actions.

The clean-run proofreader found no budget, median, API listing, key, or
out-of-workspace exposure. The recorded child environment was:

```text
ARC_API_KEY=false
ARCPRIZE_API_KEY=false
CCARC3_MAX_ACTIONS=false
CCARC3_HIDE_BASELINES=true
CCARC3_ARC_ROOT=true
```

A scorecard request transiently returned HTTP 502 once and succeeded on the
immediate retry. A replay acknowledgement also produced a harmless
`GateRefusal` because a replayed boundary had no pending gate; neither event
spent an action or affected the result.

## Evidence and transfer

The run artifacts are intentionally outside Git:

```text
/Users/dastin/dev/athanor-codex-live-evidence-20260812/full-isolated/cd82-fb555c5d
```

A transfer-ready archive was prepared on the original machine:

```text
/Users/dastin/dev/athanor-codex-full-isolated-evidence-20260812.tar.gz
SHA-256 f6e02555dc3449059a7a1fde0f91893f3105a5278e60bdd01c334298cbf2e13d
```

Copy it with AirDrop, `scp`, or another trusted channel if the new machine
needs the raw transcript. After copying:

```bash
shasum -a 256 athanor-codex-full-isolated-evidence-20260812.tar.gz
tar -xzf athanor-codex-full-isolated-evidence-20260812.tar.gz
```

The archive contains the workspace, `result.json`, `scorecard.json`,
`trace.jsonl`, and `stream.jsonl`; it does not contain the ARC API key.

## Live-run safety boundary

Do not use plain `athanor ccarc3 run` as a benchmark-clean entrypoint yet. It
runs the game, but a diagnostic attempt showed that this direct path does not
install the baseline-stripping and credential-withholding proxy. The attempt
was aborted after its opening RESET and performed no gameplay.

The supported clean driver calls `ablate_baselines.install()` and
`assert_installed()` before opening a game. A narrowly scoped live validation
through that driver looks like this; run it only with explicit authorization
to spend ARC actions:

```bash
export ARC_API_KEY='set-this-locally; never paste it into Git'
export CCARC3_SCRATCH="${TMPDIR:-/tmp}/athanor-ccarc3-codex"
export CCARC3_ONLY=cd82
export CCARC3_SWEEP_DIR=clean_rollouts_codex_validation
export CCARC3_MAX_PASSES=1
export CCARC3_MAX_NUDGES=3
.venv/bin/python tools/clean_rollouts.py
```

Before allowing a new live run, inspect the generated `meta.json` and process
environment, and require the same five child-environment values shown above.
Keep the output outside the repository so benchmark workspaces cannot inherit
repository or user instructions.

## Best next work

No code change is required to satisfy the original port request. If continuing
development, the highest-value follow-up is to make the clean boundary harder
to misuse: add a first-class single-game clean command (or a `--clean` mode)
that installs and verifies the proxy, keeps the credential out of the child,
and refuses to launch if those invariants are false. Do not silently change the
existing plain command without tests for callers that rely on it.

Other low-priority cleanup: avoid attempting the replay gate acknowledgement
when no gate is pending, while preserving the current no-action-spent
behaviour.

For a fresh Codex session, use this prompt:

```text
Read docs/ccarc3_codex_handoff_20260812.md completely. Verify the branch and
commits, then run the non-live tests. Do not launch ARC or use ARC_API_KEY until
you have independently confirmed that baseline stripping and credential
withholding are active. Never write to a Cursor agent's checkout.
```
