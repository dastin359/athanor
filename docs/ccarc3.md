# CCARC3 — Claude Code as the agent loop for ARC-AGI-3

Design and findings: `ccarc3_design.md`. This file is how to run it.

ARC-AGI-3 is interactive. There is no puzzle to submit an answer to — there is a
game whose rules nobody tells you, and working them out *is* the task. So the
harness supplies a workspace, a toolkit and a gate, and Claude Code supplies the
agent loop. No reviewer, no orchestration.

## Quick start

```bash
export ARC_API_KEY=...             # the live API 401s without one

athanor ccarc3 games                          # what exists, with baselines
athanor ccarc3 run --game ls20-9607627b       # play one
athanor ccarc3 report --out-dir runs/ccarc3   # what happened
```

Defaults are `claude-opus-5` at effort `high`, deliberately, so runs stay
comparable with one another.

## What a run produces

```
runs/ccarc3/<game_id>/
  CLAUDE.md          how to drive the game
  DOCTRINE.md        the measured findings, several counterintuitive
  session.py         a client and gate pre-wired to this game
  trace.jsonl        every action and its frames — the record
  trace.state.json   session state, so each new process resumes the same game
  rules.json         the rule book the gate requires at level boundaries
  notes/             the solver's own scratch space
  stream.jsonl       the raw Claude Code event stream
  result.json        the outcome, read from the trace and nothing else
```

`result.json` is derived from `trace.jsonl`, never from what the solver says it
achieved. On ARC-AGI-2 the submission gate made a claimed solve and a real one
the same thing; here the trace plays that role.

## The action budget

There is no sensible global cap. Across the 25 public games, a baseline
playthrough — one that *already knows the rules* — runs from 171 to 1843
actions, and a single level of `dc22-fdcac232` has a baseline of 578. The stock
SDK's `MAX_ACTIONS = 80` cannot finish any of them.

So the cap is derived per game:

```bash
athanor ccarc3 run --game ls20-9607627b --budget-multiple 4.0
```

`baseline_total × multiple`, floored at 200. The multiple exists because the
solver must *discover* the rules as well as execute them. It is enforced in the
client, not merely advertised to the solver.

The same number is also a control law worth using: **at a large multiple of a
level's baseline, the working hypothesis is probably wrong** — the solver should
re-explore rather than grind. Without it there is no way to tell "this level is
long" from "I have misunderstood this level".

## The level gate

A level boundary is the one moment the solver has just learned something and is
about to need it, so the first action of a new level is refused until the rule
book records what the finished level established:

```python
gate.acknowledge(
    "what the level established",
    mechanics=["game-scoped beliefs, which carry as priors"],
    refuted=["things ruled out"],
)
```

Structural rather than advisory: knowledge crossing a boundary is the only thing
between a solver and re-deriving the game once per level, and the budget cannot
pay for that. The gate does not judge the content — grading a hypothesis would
make it a reviewer.

## Resuming

Runs are long and containers get recycled. Relaunching against the same
`--out-dir` **resumes** by default: the client persists its scorecard, session
cookies, level and action count, so the solver picks up the same game rather
than paying for the first N actions twice. The resumed session is told to read
`rules.json` first and not to RESET, since the earlier session's mechanics were
paid for.

```bash
athanor ccarc3 run --game ls20-9607627b            # resumes if a trace exists
athanor ccarc3 run --game ls20-9607627b --fresh    # discard and start over
```

Resuming also raises the cap, which is the natural move when a run stops on
budget partway through:

```bash
athanor ccarc3 run --game ls20-9607627b --budget-multiple 4.0
```

The larger cap reaches the client while the trace, session and rule book stay
put — so buying more actions never costs the ones already paid for.

## Reading a run

```bash
athanor ccarc3 trace --run runs/ccarc3/ls20-9607627b
```

Reports actions per level, the action mix, how many actions changed the board,
how many were wasted, and the size of the rule book.

`report` prints a row per game and never leads with a pooled rate — games differ
too much in length for a summed "levels solved" to mean anything. Watch the
flags column: `wasted=` (actions issued while dead), `FULLRESET=` (a run that
discarded its own progress), `TIMEOUT`. None of these show up in a levels
number, and all three were real bugs during development.

## Things that will bite you

These are all measured, and each cost something to learn.

1. **A RESET immediately after a level advance is a full game reset.** Score to
   zero, back to level 0. Any other RESET at that moment is harmless. The client
   refuses it, and the flag arming that refusal is persisted — it was not, once,
   and a won game was replayed from level 0 because the guard stood down across
   a process boundary.
1b. **`full_reset` on the frame can be `false` when a full reset just
   happened.** Observed: level 6 → 0 with the flag clear. The client and the
   ledger both treat *the level going down* as the fact and the flag as a hint.
2. **Actions issued while dead are billed and discarded.** The client refuses
   them and counts them as `wasted`.
3. **The API binds a scorecard to the HTTP session, not the API key.** Lose the
   cookies and the server says `game <id> not found` — blaming the game when it
   means the session.
4. **The live server sends `levels_completed`, not `score`.** The stock
   `arc_agi_3` SDK reads `score` and therefore reads 0 forever. This harness
   speaks HTTP directly for that reason.
5. **Every solver action arrives in a new process.** A Claude Code agent works
   in one-shot `python -c` commands, so the client persists its session and
   resumes; without that, the game restarts on every command.
6. **Tool permissions are not optional.** `acceptEdits` approves file writes but
   not Bash, so a run with no `--allowedTools` reaches the model and then takes
   zero game actions.

## Local bench, no API key

`athanor.ccarc3.bench.lineage` is an `arcengine` game built so that rules and
mechanics come apart across levels: the goal colour changes at level 1, a lethal
colour appears at level 2, walls at level 3. That is what makes the design's
central claim — rules are level-scoped, mechanics are game-scoped — falsifiable
rather than merely asserted, and it needs no key.
