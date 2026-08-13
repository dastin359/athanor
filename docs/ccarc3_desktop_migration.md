# Running CCARC3 rollouts on hardware you own

*Written 2026-08-12, after the third container replacement of the day.*

## Why

The cloud box is replaced on a ~6 hour cadence. Three on 2026-08-12 alone —
**18:48, 02:48, 08:18 PDT** — evenly enough spaced to look like a schedule
rather than an idleness timeout. That last one landed while the keepalive was
armed and firing, which falsifies the "you went idle" explanation I gave for the
02:48 one.

A replacement is not a pause. The scratchpad reverts to an image snapshot, the
repository rolls back to whatever commit the snapshot held, every daemon dies,
and the proxy port changes. `clean_rollouts.py` **discards** an interrupted
attempt rather than scoring it, deliberately — resuming is the confound that
disqualified the first five games. So a replacement mid-run costs the quota and
banks nothing.

`bp35` needs about 4.9 hours. Fitting that into a 6 hour window is a coin flip.

Roughly everything built for durability in the last two days exists for this one
problem, and only this one:

| built | for |
|---|---|
| `tools/card_vault.py` | the scorecard surviving a rollback of the repo |
| `tools/rehydrate_box.sh` | rebuilding a box from the remote |
| `tools/preserve_evidence.sh` at a 300 s tick | traces surviving a snapshot revert |
| the keepalive re-arm loop | noticing the box went away |

On a machine that stays up, none of it is load-bearing.

## What actually has to move

Less than the amount of scaffolding suggests.

| need | note |
|---|---|
| Python ≥ 3.11 | `pyproject` requires it; the cloud box runs 3.11.15 |
| `pip install -e .` | 9 pinned deps, all pure PyPI, no system packages |
| `codex` on `PATH` | `_codex_binary()` is `$CODEX_BINARY or which("codex") or "codex"` |
| `ARC_API_KEY` | one `.env`, mode 600, **never** in the repo |
| a scratch directory | `CCARC3_SCRATCH` |

Audited while writing this:

* **Nothing under `src/` references `HTTPS_PROXY`.** It appears only in
  `preserve_evidence.sh`, `rehydrate_box.sh` and `supervisor.sh` — all cloud
  plumbing. The ARC key shim binds `127.0.0.1` and is localhost-only anywhere.
* **`quota.sh` has no cloud dependency.** It reads `rate_limit_event` lines out
  of solver `stream.jsonl` files under the scratchpad, so the supervisor's
  duty-cycle works identically against a different Codex account.
* **Every tool that needs it honours `CCARC3_SCRATCH`** (16 files). The only
  places the cloud path is still hardcoded are `__pycache__` and one prose line
  in `tools/trace_audit_template.html`.
* Existing overrides: `CCARC3_SCRATCH`, `CCARC3_REPO`, `CCARC3_BRANCH`,
  `CCARC3_SWEEP_DIR`, `CCARC3_PROXY_URL`, `CCARC3_PRESERVE_TICK`,
  `CCARC3_QUOTA_LIMIT`, `CCARC3_QUOTA_RESUME`, `CODEX_BINARY`.
* Disk: a banked 25-game sweep is **635 MB** of scratch, **121 MB** of
  compressed evidence.

## Do it

```bash
git clone -b codex/port-arc-agi-3-harness-logic <repo> athanor-codex
cd athanor-codex
export ARC_API_KEY=sk-...            # read once, written to the scratch dir at 600
bash tools/desktop_bootstrap.sh --scratch ~/ccarc3-scratch
```

The bootstrap **runs the things it configures** rather than checking that they
are present. That distinction is the whole point, and it is the defect class
this project keeps meeting: a check that names one thing and reads a proxy for
it, whose signature is that *it passes by not running*. So it calls ARC, it
installs the baseline strip and then probes it against live data in both
directions, it asks the CLI to actually answer a prompt, and it reads pytest's
exit code rather than grepping its summary. It is idempotent — re-running it is
the right way to check a box that has been idle.

Then:

```bash
export CCARC3_SCRATCH=~/ccarc3-scratch
echo 1 > "$CCARC3_SCRATCH/concurrency"          # the brake; read every 10 s
nohup bash tools/supervisor.sh >> "$CCARC3_SCRATCH/supervisor.log" 2>&1 &
```

The supervisor is the quota duty-cycle; it starts the driver when utilization
allows and stops it politely at the ceiling — waiting for every started game to
have a `result.json` rather than killing mid-game, because a killed parent
leaves the solver burning quota with nothing left to record it.

For a sweep whose scorecard is the submission, give it its own directory. The
driver skips any game that already has a `clean_result.json`, so pointing a
shared-card sweep at a directory holding banked results produces a sweep that
completes with two thirds of the benchmark missing and no error anywhere:

```bash
CCARC3_SWEEP_DIR=clean_rollouts_submission nohup bash tools/supervisor.sh ...
```

## What NOT to start

```
tools/rehydrate_box.sh      rebuilds a box from the remote
tools/card_vault.py         encrypts the scorecard against a repo rollback
tools/preserve_evidence.sh  pushes to a PUBLIC repo every 300 s
```

Evidence pushing is still worth having — it is how results reach the record —
just not at panic cadence:

```bash
CCARC3_PRESERVE_TICK=1800 nohup bash tools/preserve_evidence.sh >> "$CCARC3_SCRATCH/preserve.log" 2>&1 &
```

## The one that will actually bite you

**Stop the machine sleeping.** A suspended desktop kills an in-flight solver
exactly the way a container replacement does, and the driver discards the
attempt either way — same cost, and much easier to cause by accident.

```
linux:  sudo systemctl mask sleep.target suspend.target hibernate.target
macos:  sudo pmset -a disablesleep 1        # or: caffeinate -dimsu -w $$
```

## One thing that genuinely changes

Solver isolation is enforced at the **harness** level. Every initial and resumed
Codex launch carries `--ignore-user-config`, `--ignore-rules`,
`--disable memories`, and `--skip-git-repo-check`; clean runs additionally
assert those flags immediately before every launch. The solver workspace lives
outside the repository, and `ARC_API_KEY`, `ARCPRIZE_API_KEY`, and
`CCARC3_MAX_ACTIONS` are withheld from the child.

What does not travel is the second wall. The solver's `Bash` has whatever
network the configured Codex sandbox allows. ARC access is routed through the
loopback allowlisting proxy, but a workspace-write sandbox with network enabled
still inherits the host's network boundary. Keep the rollout machine free of
unrelated credentials and services.

## Found while checking this

Standing the repo up somewhere else immediately surfaced a bug that had been
invisible for the life of the project.

Running the suite from `/` rather than the repo root, it did not fail a test —
it **failed to collect**, on `ModuleNotFoundError: No module named
'proofread_trace'`. Eight test modules and one tool reached `tools/` with
`sys.path.insert(0, "tools")`: a path resolved against the process cwd, not the
repository. Then, once collection was fixed, three more failures in a second
spelling — `pathlib.Path("tools/preserve_evidence.sh")`, a relative literal
opened directly.

The worst of them was `test_a_real_multi_play_attempt_still_corroborates`, which
guarded its relative path with `if not exists(): return`. From any working
directory but the repo root it asserted nothing and reported green — the control
test for the entire card-corroboration check, quietly not running.

The mechanism is the one this project keeps meeting: **a value that agrees with
the truth in the only environment anyone exercises it in.** Every by-hand pytest
run happens from the repo root, so a relative path was indistinguishable from a
correct one until something ran from somewhere else. "The tests are green" was,
strictly, a claim about the operator's working directory.

Fixed in `tests/conftest.py` (absolute `tools/` on `sys.path`, once) and at each
site. `tests/test_the_suite_does_not_depend_on_the_working_directory.py` holds
the regression: two AST scans, each with a drill proving the scan can see the
defect it claims to catch, and a collection run from a foreign directory. Note
that the static scan found only the class it was written for and missed the
sibling case three files away — the dynamic drill found both. A scan is a cheap
early warning, never the evidence.
