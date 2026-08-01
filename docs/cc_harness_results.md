# CC Harness — experiment log

Running record of the Claude Code harness variant on ARC-AGI-2 public eval.

Each round runs several solver agents on different tasks, then reads the traces
(`athanor cc trace`) and the agents' own friction reports to find gaps in the
*infrastructure*, not the puzzles. Fixes ship before the next round.

Method note: every agent is asked to critique the harness as explicitly as it
reports its solution. That has been the highest-yield part of the loop — a
solver that has just spent twenty minutes inside the workspace knows things
about it that no amount of reading the code reveals.

---

## Round 1 — baseline

Three agents (Opus 5), 8-iteration budget, tasks chosen for size and shape
variety rather than difficulty.

| task | solved | test examples | iterations | confidence | explore scripts | invariants | verified before 1st submit | elapsed |
|---|---|---|---|---|---|---|---|---|
| `28a6681f` | yes | 1/1 | 1/8 | 4 | 12 | 29 | 23 | 737s |
| `b0039139` | yes | 2/2 | 2/8 | 5 | 5 | 17 | 10 | 454s |
| `89565ca0` | yes | 1/1 | 1/8 | 4 | 9 | 14 | 12 | 739s |

**3/3 solved.** All integrity-clean. No gate refusals. Budget never approached.

### What the traces showed

The doctrine was followed rather than merely satisfied. Every agent front-loaded
verification — 10 to 23 invariants established *before* the first submission —
and two of the three passed training on their first formal attempt. That is the
intended shape: exploration is free, submission is scarce, so the work moves
into exploration.

More telling, all three independently reported that the discipline changed what
they actually did, and each named a specific place where it mattered:

- On `28a6681f`, the agent wrote an independent time-stepped simulation to
  cross-check its own rule. It *refuted* — which isolated the exact semantic
  commitment the rule needed (grains settle serially, not concurrently). Its
  own summary: "I would never have articulated that from staring at grids."
- On `b0039139`, a colour-permutation equivariance test found a hard-coded
  separator colour that all four training pairs happened to agree on. The
  predictions did not change, but an incidental constant became a role-based
  one.
- On `89565ca0`, an occlusion-reconstruction check found a lattice bridged
  across another shape's line — a case the agent said hand-counting from ASCII
  "would very likely have missed".

Each of those is the same shape: a check that was cheap to run, that the agent
would not have run unprompted, and that converted a visual impression into an
executed fact.

### What broke

| gap | severity | found by |
|---|---|---|
| `from arc import ...` failed in the invocation the contract documents | high | all three agents, independently |
| `arc.png()` was dead code — no Pillow in the solver runtime | medium | all three agents |
| Nothing stated what the runtime actually provided (no numpy either) | medium | inferred from the above |
| A tautology was recorded as a verified invariant | high | `89565ca0` |
| No way to retract a bad invariant | medium | `28a6681f` |
| Gate refusals were never written to the event ledger | medium | building `cc trace` |
| `cc trace` counted a harness-shipped file as agent exploration | medium | the test suite |

The first one is the instructive failure. `python explore/foo.py` puts the
*script's* directory on `sys.path`, not the working directory — so the toolkit
at the workspace root was invisible to exactly the command CLAUDE.md told the
agent to run. Every agent hit `ModuleNotFoundError` on its first experiment and
carried `sys.path` boilerplate for the rest of the run. The harness put friction
on the single most common action in the entire loop, and it took real agents to
notice, because reading the docs and reading the code both look fine in
isolation.

The tautology is the most interesting. An agent wrote
`verify("noise colour is well separated", True if ... else True)` and the ledger
recorded it as `[VERIFIED]`. It caught the mistake itself and re-ran the claim
properly, but nothing in the tooling distinguished a claim backed by a
measurement from one backed by nothing — in the artifact that survives
compaction and is replayed as established fact. That is the doctrine's own
failure mode occurring inside the tool built to prevent it.

### Shipped after round 1

- Mirror `arc.py` into `explore/` so the documented invocation works.
- `dryrun.py`: one-word free scoring of `solution/solve.py`. The doctrine says
  never to use the gate as a debugger; the free alternative had been a
  90-character one-liner with a manual `sys.path` insert.
- Probe the solver's interpreter at build time and state in `CLAUDE.md` exactly
  what is installed and what is not. Advice about a missing library is worse
  than no advice.
- `arc.verify()` now recovers the condition's **source expression** from the
  caller's AST and stores it with the claim, so the ledger records what was
  executed. Compile-time constants are flagged at the call site and surfaced
  separately by `gate.py status` and `cc trace`.
- `arc.verify(claim, retract=True)` withdraws a bad invariant.
- `arc.png()` falls back to the interpreter that built the workspace.
- Gate refusals are written to the event ledger and reported by `cc trace`.
- numpy installed in the solver runtime, matching the flagship's environment.

### What round 1 could not test

Nothing exercised the harness under stress: no run hit a gate refusal, exhausted
its budget, entered the best-effort window, or went through more than two
reflection cycles. The failure-report path — the place where the harness does
most of its talking — is almost entirely unexercised. Round 2 targets that.

---

## Round 2 — in progress

Three agents on deliberately harder ground:

- `13e47133` — on the zero-solve frontier: no submission in a 117-attempt public
  corpus has ever solved it. Athanor solves it. Should stress the budget,
  the reflection loop, and best-effort acceptance.
- `dd6b8c4b` — two test outputs, moderate difficulty.
- `67e490f4` — 30x30 grids, to exercise the gate's grid elision.

Round 2 agents were deliberately **not** told the `sys.path` workaround, so the
import fix is under test.
