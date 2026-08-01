# Puzzle workspace — task `__TASK_ID__`

Solve one ARC-AGI-2 task in this directory. Everything you need is here; nothing
outside this directory is relevant, and the test outputs are not in it.

## Environment

__ENVIRONMENT__

`from arc import ...` works from anywhere in the workspace — from a script under
`explore/`, from a `__PYTHON__ -c` one-liner at the root, or via
`__PYTHON__ -m explore.foo`. You never need `sys.path` boilerplate.

## Layout

```
task/task.json        __N_TRAIN__ training pairs + __N_TEST__ test input(s). No test outputs.
task/grids.md         the same grids as text, one row per line
task/images/          every grid rendered as a PNG — open them with the Read tool
arc.py                observation + verification helpers (read it; it is short)
dryrun.py             free scoring of solution/solve.py against the training pairs
gate.py               the verification gate
explore/              your scratch scripts. Unlimited, free, not recorded.
solution/hypothesis.md   the rule, in prose        (you create this)
solution/solve.py        the rule, as code         (you create this)
solution/audit.md        generalization audit      (you create this, at the end)
NOTES.md              your durable research state — keep it current
.athanor/             run ledger: iterations, verified invariants, reports
```

## The loop

1. **Look.** `Read task/images/train_0.png` … and the test input(s). Then dump
   structure with a script under `explore/`. ARC is a vision task; the pictures
   and the numbers tell you different things.
2. **Verify.** Establish facts by executing them:
   `arc.verify("every output is 20x20", ...)`. Every recorded invariant narrows
   the space of rules that can be correct.
3. **Hypothesize.** Write the rule into `solution/hypothesis.md` — complete
   enough that a programmer who has never seen this puzzle could reimplement
   `solve()` from it alone.
4. **Implement and dry-run.** Write `solution/solve.py`, then run
   `__PYTHON__ dryrun.py`. It scores your `solve()` against every training pair
   and costs nothing — no iteration, no record. Iterate here until it passes.
   Every bug you catch with `dryrun.py` is a budgeted submission you keep.
5. **Submit.** `__PYTHON__ gate.py submit`. Budgeted and permanent.
6. **Reflect.** The gate's output ends with what to do next. Follow it, and
   append the reflection to `NOTES.md`.
7. **Accept.** Once training passes, write `solution/audit.md`, then
   `__PYTHON__ gate.py accept`. That ends the run — stop working after it
   succeeds.

## Gate commands

| Command | Cost | What it does |
|---|---|---|
| `__PYTHON__ dryrun.py` | free | Scores `solution/solve.py` against the training pairs. Use it before every submission. |
| `__PYTHON__ gate.py status` | free | Distilled research state: iterations, verified invariants, last hypothesis, notes tail. **Run this first after any context compaction.** |
| `__PYTHON__ gate.py submit` | 1 iteration | Runs `solution/solve.py` against every training pair and test input, records the result, reports failures and what to do next. |
| `__PYTHON__ gate.py accept` | free | Finalizes the run using the last submission and `solution/audit.md`. |

## The invariant ledger

`arc.verify(claim, condition)` appends to `.athanor/invariants.jsonl`. Two
semantics worth knowing:

- **Most recent entry per claim wins.** Re-verifying the same claim string
  supersedes the earlier record rather than duplicating it.
- **Retraction.** If a check was wrong — a condition that was accidentally a
  tautology, say — withdraw it with `arc.verify("the claim", retract=True)`. It
  then disappears from `gate.py status`. Do this rather than leaving it: the
  ledger is only worth reading because everything in it has been executed, and
  one claim that merely looks verified devalues all of them.

## Rules the gate enforces

- `solution/hypothesis.md` must exist and be at least __MIN_HYPOTHESIS_CHARS__
  characters before any submission.
- If `solution/solve.py` changed, `solution/hypothesis.md` must have changed
  too. A code change is a change of rule, implementation, or edge case — say
  which. (A refusal here does not cost an iteration.)
- `solve(grid)` receives **only** its argument. It cannot reach the puzzle data,
  and embedding a training output as a literal will be reported.
- `solve(grid)` returns one grid for training inputs; up to __MAX_CANDIDATES__
  candidate grids are allowed for test inputs, and only for a real ambiguity.
- Budget: **__MAX_ITERATIONS__ submissions.** Over the last __BEST_EFFORT__ of
  them the train-100% requirement is lifted so you can commit to your best
  reading of the puzzle.
- `accept` requires `solution/audit.md` with a `CONFIDENCE: <1-5>` line and a
  `DECISION: ACCEPT` line.

## Working habits

- One question per exploration script, named for the question it answers.
- If you import one `explore/` script from another, guard its report behind
  `if __name__ == "__main__":` — otherwise its output re-prints into every
  downstream run and pollutes your context.
- Print summaries, not grids you have already seen. A boolean, a count, or a
  set of shapes usually carries the finding.

## Out of bounds

- No network access, and no need for it: ARC requires no external knowledge.
- Do not read anything outside this workspace. The test outputs live in the
  benchmark dataset; reaching for them invalidates the run, and the harness
  checks.
- Single agent. Do not delegate to sub-agents.

## When context gets compacted

Exploratory interpreter state and your transcript do not survive it.
`.athanor/invariants.jsonl` and `NOTES.md` do. Keep `NOTES.md` written for a
fresh version of yourself: current hypothesis, confirmed facts, refuted
hypotheses *and why*, and the next experiment you meant to run.
