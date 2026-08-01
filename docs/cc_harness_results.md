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

## Standing

**24 completed tasks, 23 fully solved, 33 of 34 test examples.** Mean ~1.2
formal iterations. Zero gate refusals in the entire experiment.

The single miss is `9bbf930d` — two attempts, both 0/1, the second carrying
every toolkit improvement. The flagship solves it.

`faa9f03d`, which no logged submission in the frozen public corpus has ever
solved in 127 attempts, **is solved** — see the section below. It came from the
second candidate, not the first.

**Read that percentage carefully — it is not comparable to a full-benchmark
score, for two reasons pulling in opposite directions.**

*Against it:* these 23 tasks are not a random sample. They were deliberately
picked for difficulty — six of the eight zero-solve frontier puzzles, all five
puzzles the flagship itself fails, and the pairs no CoT-only Opus 4.6
configuration solved in eight attempts. A hard-selected subset says nothing
about the other 97 public-eval tasks, which have never been run here.

*For it:* two tasks were attempted twice, and the table takes the better
attempt. That is the same development-then-select-best pattern `RESULTS.md`
documents for the flagship's 95.7%, so the comparison is at least like-for-like
— but on a first-attempt-only basis this variant would be 20 of 23, not 21.

What the numbers do support, because they are head-to-head on the same puzzles:

| comparison | this variant | flagship |
|---|---|---|
| zero-solve frontier (9 pairs) | **8** | 8 |
| pairs the flagship itself fails (6) | **6** | 0 |
| cost per task (n=4 batch) | $1.86 | $3.12 mean / $1.71 median |

The systems fail on *different* puzzles: each solves exactly one frontier pair
the other cannot — `faa9f03d` here, `9bbf930d` there. Between them the entire
frontier falls; neither does it alone. This variant carries no reviewer, no
second model, and no inter-agent artifact exchange.

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

Token cost, measured as sub-agent tokens per task: 81k, 100k, 107k — mean ~96k.
Treat that as an order-of-magnitude figure, not a comparable one: these runs were
driven as sub-agents inside an existing session rather than through
`athanor cc run`, so they carry that session's prompt-cache behaviour and none
of the batch amortisation the flagship's $3.12/task figure assumes. The
comparable number has to come from the subprocess launcher over a real batch.

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

## Round 2 — harder ground

Round 2 agents were deliberately **not** told the `sys.path` workaround, so the
import fix was under test. All three confirmed it worked with zero boilerplate.

| task | solved | test examples | iterations | confidence | explore scripts | invariants | elapsed |
|---|---|---|---|---|---|---|---|
| `13e47133` | yes | 2/2 | 1/8 | 5 | 7 | 7 | 480s |
| `dd6b8c4b` | yes | 2/2 | 1/8 | 4 | 6 | 17 | 855s |
| `67e490f4` | yes | 1/1 | 1/8 | 4 | 13 | 21 | 834s |

**3/3 solved.** Running total: **6/6 tasks, 9/9 test examples**, mean 1.17
iterations, all integrity-clean, zero gate refusals.

`13e47133` is the notable one: it sits on the zero-solve frontier — in the
frozen public corpus of 117 logged attempts, no submission had ever solved it,
and `RESULTS.md` records it as solved exclusively by the flagship system. The CC
variant solved both its test outputs on the first formal submission, with
confidence 5.

### What the traces showed

The same pattern as round 1, sharper. Every agent named a specific experiment
that changed its answer, and in each case it was an experiment whose *outcome it
could not predict*:

- `13e47133`: a two-line sweep over `(connectivity, metric)` settled Chebyshev
  vs Manhattan erosion. The agent's own note: unprompted it would have written
  the obvious bounding-box formula, "been right for the wrong reason" on convex
  rooms, and lost the cells beside a concave corner.
- `67e490f4`: a script asking "does any *other* simple selector also fit?"
  returned 2/7, 4/7, 5/7 for the alternatives — converting a hypothesis that
  merely fit into the only one of four that fits.
- `dd6b8c4b`: BFS distances printed as a sorted list showed
  `[2,3,7,9,11,11,12,15,19 | 20,20,21,22,22,23]` — a clean gap at the cut-off.
  The agent had been hand-computing Manhattan distances and, in its own words,
  "my prose had in fact been converging on the wrong answer".

Two agents also reported the audit requirement doing real work: it is what
prompted the ablation batteries and equivariance tests that turned arbitrary
implementation choices into *verified non-choices*.

### What broke

| gap | severity | found by |
|---|---|---|
| `solution/` not importable — the doctrine's own step 5 needs it | high | all three agents |
| Three documents said bare `python` while CLAUDE.md named the venv | medium | `dd6b8c4b` |
| Gate dumped full test predictions unconditionally | medium | `b0039139`, `13e47133` |
| `verify()` conflates "established X" with "X is false" | medium | `dd6b8c4b` |
| Supersession keyed on the claim string breaks on rewording | medium | `67e490f4` |
| Numeric script names (`05_x.py`) are not importable | low | `67e490f4` |
| `NOTES.md` needs a Read before the first Write | low | `dd6b8c4b` |

The first is the round's instructive failure, and it rhymes with round 1's. The
doctrine tells the solver to run its verified invariants against its own test
predictions before accepting — the last line of defence in a configuration with
no reviewer. Doing that means importing `solution/solve.py`, and the toolkit
offered no way. All three agents wrote `sys.path.insert(0, 'solution')`, a
*relative* path that breaks if the script runs from anywhere but the workspace
root — the exact boilerplate `arc.py` had gone to the trouble of eliminating
everywhere else. **The harness put friction in the one place its own doctrine
sends you**, twice in two rounds.

The interpreter inconsistency is the cheapest lesson: bare `python` worked right
up until `import numpy`, at which point it failed in a way that looks like a
workspace bug rather than a documentation bug.

### Shipped after round 2

- `arc.load_solution()` returns the shipped `solve`, resolved against the
  workspace.
- All documents name the same interpreter.
- Test predictions over 400 cells are replaced by a colour histogram and a
  pointer to the archived `predictions.json`.
- `arc.refute(claim, condition)` records a dead end as a finding rather than as
  an apparent defect.
- `verify(..., key=)` for explicit supersession across rewording.
- Mechanical generalization signals wired into the gate (see below).
- Guidance on importable script names and on `NOTES.md` needing a Read first.

### The reviewer-shaped hole

With no independent reflector, nothing rejects a train-perfect but wrong rule.
`signals.py` recovers the cheap, model-free half of that job: regularities every
training output obeys that a test prediction breaks — fixed output shape, shape
relation to the input, squareness, a fixed output palette, colours absent from
the prediction's own input, and duplicate candidates that waste ARC-AGI-2's
second attempt.

It only fires when the training set is unanimous and the prediction dissents. On
round 1's three correct solutions it reported nothing; against deliberately
corrupted predictions for the same three tasks it fired on all three.

### Still untested after two rounds

**No submission has ever failed.** Nine of nine tasks passed training on
iteration 1 or 2, so the failure report, the reflection directive, the
best-effort switch, and every gate refusal remain unexercised by a real agent.
The harness does most of its talking on the failure path, and that path has been
validated only by unit tests.

---

## Round 3 — in progress

Three agents on tasks the **flagship system itself fails**, chosen specifically
to produce failures:

- `faa9f03d` — Athanor scores 0/1 even in an extended 120-turn attempt; never
  solved by CoT-only Opus 4.6 at any thinking level.
- `2b83f449` — Athanor scores 0/1; never solved by CoT-only Opus 4.6 in 8
  attempts.
- `88e364bc` — Athanor solves 1 of 2 test outputs; test 0 never solved by
  CoT-only Opus 4.6.

Agents were told the difficulty honestly and asked to prioritise evaluating the
failure reports over solving.

---

## The finding that matters most

`88e364bc` produced the first train-perfect-but-wrong result, and it is worth
more than the six clean solves.

The task's rule is "tokens slide until blocked". One question was genuinely
ambiguous: may a diagonal slide pass the tip of a wall? Both readings reproduce
all three training pairs. The agent killed the strict reading by execution — it
strands a token in open space, "whereas all ten training tokens rest with a wall
directly ahead" — and recorded that as a refutation. It then wrote, in its own
report:

> the ambiguity is gone and the second candidate slot went unused, which is the
> right outcome.

It was wrong. The prediction missed by **2 cells out of 400** — right shape,
right palette, 99.5% pixel accuracy — and the alternative reading it had in hand
was very likely the correct one. The result matched the flagship system exactly
(1 of 2 test examples, the same one missed).

Two things follow, and both are uncomfortable for the thesis.

**Executed verification can produce confident wrong answers.** The invariant was
real: all ten training tokens do rest against a wall. The error was extending it
to the test input, which nothing had established. That is an inductive leap
wearing a proof's clothes, and it is *more* dangerous than prose reasoning
precisely because it comes with a `[REFUTED]` line in the ledger and reads as
settled. Code-as-verification compresses the cost of checking; it does not
convert induction into deduction, and the harness had been silent about the
difference.

**Mechanical signals cannot cover this.** `signals.py` fired nothing, correctly:
shape, palette and structure were all consistent. A 2-cell semantic near-miss is
exactly the class of failure that needs a *reasoning* reviewer, which is what
Athanor's independent reflector is and what this variant drops. The mechanical
substitute bounds the reviewer-shaped hole; it does not close it.

There is also a cheaper lesson. Across every accepted run, **9 of 11 test
examples got a single candidate** — ARC-AGI-2 allows two, so eight free attempts
were forfeited. Athanor has a mechanism for exactly this: the reflector's
EXPAND_CANDIDATES verdict, which fires when a rule is sound but ambiguity
remains between plausible branches. The CC variant has no equivalent, and on
`88e364bc` the cost was measurable.

**Shipped in response:** the generalization-audit directive now asks the solver
to examine *how* it ruled out an alternative — killing it with a training pair
it fails is a proof; killing it by extending a training-only regularity to the
test is a leap — and to spend the second candidate whenever an alternative
reproduced every training pair and died only to a leap.

That is guidance, not a mechanism, and it is weaker than what it replaces. The
honest conclusion is that this is the first measured cost of dropping the
reviewer.

---

## Round 3 — the failure path, and a fix validated in flight

Three agents on tasks the flagship system itself fails, plus one run with a
deliberately starved 3-iteration budget to force the budget-exhaustion machinery
that three rounds of natural running never reached.

| task | result | test examples | iterations | confidence | candidates | flagship reference |
|---|---|---|---|---|---|---|
| `88e364bc` | partial | 1/2 | 1/8 | 4 | 1, 1 | flagship also 1/2, same example missed |
| `faa9f03d` | failed | 0/1 | 1/8 | 4 | 2 | flagship 0/1 at 120 turns; 0/127 in the public corpus |

Running total: **9 tasks, 10/13 test examples.**

### The fix that validated itself in flight

`88e364bc` (analysed above) produced the candidate-budget paragraph. `faa9f03d`
ran after it shipped, and its agent volunteered this without being asked:

> the "candidate budget" paragraph in particular is well-written — the
> distinction between *killing an alternative with a training pair* (proof) and
> *killing it by extending a training-output regularity to the test* (an
> inductive leap) is exactly the distinction I was fudging in my own head, and
> reading it is why candidate 2 shipped instead of being argued away.

Direct causal evidence that the fix changed behaviour on the next run. It did
not rescue the task — both candidates were wrong, and `faa9f03d` remains
unsolved by anything in the public record — but the mechanism did what it was
built to do.

The `UNSPENT SECOND ATTEMPT` check also validated itself live: it was active
during `faa9f03d`'s submission and correctly stayed silent, because that agent
had already spent its second slot.

### Still not exercised

**Four rounds, thirteen test examples, and no submission has ever failed.** Every
accepted run passed training on iteration 1 or 2. The failure report, the
reflection directive, the best-effort switch and every gate refusal remain
validated only by unit tests. The starved-budget run exists to break that; the
honest reading so far is that with a competent solver and free unlimited
exploration, the gate is simply never used as a debugger — which is exactly what
the doctrine asks for, and which makes its failure path structurally hard to
reach.

### Shipped during round 3

- `arc.rival(name, solve_fn)` — register an alternative reading; it is scored
  against training and its test predictions kept. The gate reports a rival that
  fits all training pairs and disagrees with the submission, which converts the
  second-attempt question from a judgement into a measurement.
- `UNSPENT SECOND ATTEMPT` — the gate pairs recorded dead ends with test
  examples still carrying one candidate.
- `arc.check(..., show_diff=True)` — which cells a prediction got wrong.
- A durability nudge in `dryrun.py` when the invariant ledger is empty, after
  two hard-task agents were observed running many scripts and recording nothing.

---

## Consolidated results (rounds 1–3)

| task | solved | iterations | confidence | explore scripts | invariants | candidates | note |
|---|---|---|---|---|---|---|---|
| `13e47133` | 2/2 | 1 | 5 | 10 | 16 | 1, 1 | zero-solve frontier |
| `28a6681f` | 1/1 | 1 | 4 | 12 | 29 | 1 | |
| `67e490f4` | 1/1 | 1 | 4 | 13 | 21 | 2 | 30x30 |
| `88e364bc` | 1/2 | 1 | 4 | 14 | 13 | 1, 1 | flagship also 1/2, same example |
| `89565ca0` | 1/1 | 1 | 4 | 9 | 14 | 1 | |
| `9bbf930d` | 0/1 | 2 | 4 | 13 | 18 | 2 | frontier; starved 3-iteration budget |
| `b0039139` | 2/2 | 2 | 5 | 5 | 17 | 1, 1 | |
| `dd6b8c4b` | 2/2 | 1 | 4 | 6 | 15 | 2, 1 | |
| `faa9f03d` | 0/1 | 1 | 4 | 16 | 16 | 2 | flagship 0/1 at 120 turns; 0/127 public |

**10 of 13 test examples across 9 accepted tasks**, every run integrity-clean,
zero gate refusals, mean 1.2 iterations.

Two of the three misses are tasks the flagship also fails, and one of those
(`faa9f03d`) has never been solved by any submission in the public corpus. The
third (`88e364bc`) matches the flagship exactly, missing the same test example.

### Did the candidate fixes work?

The `88e364bc` analysis produced a sequence of changes aimed at one failure:
a solver killing a live alternative with an inductive leap and discarding the
free second attempt. The runs split cleanly around them.

| | test examples | used the second attempt |
|---|---|---|
| before the fixes | 11 | 2 (18%) |
| after | 2 | 2 (100%) |

Two data points is not a result, and both post-fix runs were hard tasks where
hedging is more obviously attractive. But the mechanism is not merely
correlated with the change — one agent stated the causal path directly, and the
other's candidate count went 1 → 2 in the iteration immediately after the prompt
fired. Round 4 adds three more frontier runs to the post-fix sample.

The honest caveat in the other direction: hedging did not rescue either task.
`faa9f03d` and `9bbf930d` both shipped two candidates and both scored 0/1. A
second attempt helps only when one of the two readings is right.

---

## The loop closing: a failure turned into a mechanism turned into a solve

This is the result the improvement loop exists to produce, and it is worth
stating precisely because the causal chain is fully traceable.

**The failure.** `88e364bc` passed training 3/3 and missed by 2 cells out of
400. Its agent had implemented a rival reading, killed it by extending a
regularity from the training *outputs* to the test *input*, and wrote: "the
ambiguity is gone and the second candidate slot went unused, which is the right
outcome." It was wrong, and the free second attempt was discarded.

**The mechanisms built from it**, in order:

1. a paragraph in the generalization audit distinguishing killing a rival with
   a training pair it fails (a proof) from killing it with an out-of-sample
   extrapolation (a leap);
2. `UNSPENT SECOND ATTEMPT` — the gate pairs recorded dead ends with test
   examples still carrying one candidate;
3. `arc.rival(name, fn)` — register an alternative reading; it is scored against
   training and its test predictions kept;
4. `UNHEDGED RIVAL` — the gate reports a registered rival that fits every
   training pair and disagrees with the submission;
5. the same question raised at dry-run time, where acting on it is free, and
   `accept` re-running `solve.py` so hedging costs no iteration.

**The recovery.** On `abc82100` — a zero-solve frontier task, unsolved by any of
121 logged attempts in the public corpus — the agent registered the rival
*"unmatched dots (a colour no legend names as a source) survive unchanged"*,
which reproduced all 4 training pairs. `UNHEDGED RIVAL` fired. It spent the slot.

| | matches ground truth |
|---|---|
| candidate 1 (its primary reading) | no — off by 7 cells |
| candidate 2 (the registered rival) | **yes, exactly** |

The agent's own best reading was wrong. The mechanism is the entire reason the
task scored.

**What this does and does not show.** It is one conversion, and a single
instance does not establish a rate. But the chain is not inferential: the rival
is named in the ledger, the gate's prompt is in the archived report, the
candidate ordering shows the primary reading losing, and the rival's stored
prediction is byte-identical to the ground truth.

It also sharpens the earlier, more pessimistic finding rather than replacing it.
Hedging still buys coverage, not insight — `faa9f03d` and `9bbf930d` both shipped
two candidates and both scored 0/1, because neither reading was right. What
changed on `abc82100` is that the correct reading *was* among the two, and the
harness is what put it there.

---

## Production path: first end-to-end run and first cost figure

Every result above was driven by a sub-agent inside an existing Claude Code
session. `athanor cc run` — the subprocess launcher users would actually invoke
— had been smoke-tested but never used for a full solve. It has now.

```
28a6681f   SOLVED 1/1   2/6 iterations   46 turns   841s   $3.80
           integrity clean · 472 stream messages · 0 malformed
           tokens: 58k out, 2.73M cache read, 97k cache write
```

This validates the last untested component: argv construction against the real
CLI, working directory, permission mode, the tool allowlist, `stream-json`
capture, event translation, PNG reads by the agent, and the contamination scan
running against a real transcript.

**On the cost number.** $3.80 for one task, against the flagship's reported
$3.12/task average. The comparison does not hold yet, in both directions:

- This is a **cold single run**. The flagship's $3.12 is explicitly a
  batch-deployment figure in which only the first puzzle pays the system-prompt
  cache write; every later task hits the provider cache. This run paid 97k
  tokens of cache *writes* that a batch would amortise.
- It is n=1, on a task of middling difficulty that took 2 iterations. The
  flagship's per-task cost ranges from $0.23 to $20.04 with a median of $1.71,
  so a single sample says very little about a mean.
- Claude Code carries a larger default system prompt and a general-purpose tool
  surface that the flagship does not pay for; against that, this variant runs no
  reviewer context and does not re-send puzzle data every turn.

The honest statement is that the CC variant lands in the same order of magnitude
as the flagship on its first measured task, and that a comparable number
requires `athanor cc batch` over a real task set with a warm cache. The harness
can now produce that figure; this run is the evidence the machinery works, not
the answer.

---

## Round 4 — zero-solve frontier sweep, and the candidate measurement

Three of the remaining frontier pairs, all carrying the mechanisms built from
`88e364bc`.

| task | result | iterations | confidence | candidates | rivals registered |
|---|---|---|---|---|---|
| `269e22fb` | **2/2** | 1 | 5 | 1, 1 | 2 |
| `a32d8b75` | **2/2** | 2 | 4 | 2, 2 | 2 |
| `abc82100` | **1/1** | 1 | 4 | 2 | 1 |
| `2b83f449` | **1/1** | 2 | 4 | 1 | 0 |

`2b83f449` is the one to note alongside the frontier results: `RESULTS.md`
records the flagship failing it 0/1, and no CoT-only Opus 4.6 configuration
solved it in 8 attempts. This variant solved it.

### Full standings

**13 accepted tasks, 16 of 18 test examples, 1.31 iterations mean**, every run
integrity-clean, zero gate refusals across the entire experiment.

Four zero-solve frontier pairs solved — `13e47133`, `269e22fb`, `a32d8b75`,
`abc82100` — each one a pair no submission in the frozen public corpus had ever
solved. Two tasks solved that the flagship fails. The two misses are `faa9f03d`
(flagship also 0/1; unsolved by all 127 logged attempts) and `9bbf930d`, plus
`88e364bc` partial at 1/2, matching the flagship exactly.

### Did the candidate mechanisms work?

| | test examples | used the second attempt | rivals registered |
|---|---|---|---|
| before the changes | 11 | 2 (18%) | 0 |
| after | 8 | 5 (63%) | 5 |

The mechanism is used, and the qualitative evidence is stronger than the counts.
Three agents independently described the *concept* — not the function — as what
changed their behaviour:

- `faa9f03d`: "reading it is why candidate 2 shipped instead of being argued
  away."
- `abc82100`: an invariant "*looks* like it refutes the rival but holds only
  vacuously in training. Without the rival concept I think I'd have talked
  myself into that refutation." Its rival was the correct answer.
- `a32d8b75`: it "made me stop asking 'is my rule right?' and start asking
  'which situations does the test need that training never witnesses?'" — which
  surfaced an 8-cell case it had missed in three passes.

That last reframing is the real product. Eight readings on `a32d8b75` died to a
training pair and became proofs; two survived and shipped as second candidates.

### What the round cost

Four more harness bugs, all mine, all found by agents rather than by tests:

- `rival()` advised spending a slot on a reading that predicted identically — it
  stored the predictions and never compared them.
- The dry-run hedging advice named rivals diverging on a *different* test
  example than the one with the free slot: "trains you to skim it."
- An agent spent an iteration submitting purely to *read* the gate's
  candidate-spread line — information I had put only on the budgeted path two
  commits earlier.
- Making `accept` re-run `solve.py` opened a second door into acceptance that
  skipped the hypothesis/code coupling `submit` enforces.

The pattern across all four: **every convenience added to the budgeted path
creates an incentive to spend budget, and every door added to acceptance needs
the same locks as the front one.**

---

## The cache defect: the harness was paying full price on every task

The cost caveat above assumed the only thing missing was a batch. It wasn't. I
ran one and read the token counts per task:

```
task 1   cache_write 52k
task 2   cache_write 61k     ← should have been near zero
```

The second task in a sequential batch wrote *more* cache than the first. There
was no amortisation at all — the mechanism the flagship's $3.12/task figure
depends on was simply not operating in this harness.

**Cause.** Every task runs in its own workspace directory, and Claude Code's
default system prompt embeds the working directory (along with env info, memory
paths, and git status). The cached prefix therefore differed on every single
task. Prompt caching matches on an exact prefix; one differing path near the
front invalidates everything after it.

**Fix.** `--exclude-dynamic-system-prompt-sections`, which relocates those
volatile sections into the first user message. It applies only alongside the
default system prompt — which is what this harness uses, since it composes via
`--append-system-prompt` rather than replacing the prompt outright. Wired to
`CCRunConfig.stable_system_prompt`, default on.

**Measurement.** Rather than wait on a second batch, a direct A/B: the same
trivial prompt (`Reply with exactly: OK`) run in two different working
directories, with and without the flag.

| | workspace 1 | workspace 2 |
|---|---|---|
| **without the flag** — cache_write | 1739 | 1739 |
| **without the flag** — cache_read | 3289 | 3289 |
| **with the flag** — cache_write | 1675 | **745** |
| **with the flag** — cache_read | 3289 | **4219** |

Without the flag, workspace 2 wrote the identical 1739 tokens workspace 1 did —
byte-for-byte no reuse, exactly the batch's signature. With the flag, workspace
2's write dropped to 745 while its read rose to 4219: the second workspace
consumed the prefix the first one had cached.

That closes the caveat left standing after the production run. The earlier
statement — that a comparable cost figure "requires `athanor cc batch` over a
real task set with a warm cache" — was true but incomplete. The harness could
not have produced a warm cache at all, however many tasks it ran.

### How much this is worth, honestly

Less than the framing above implies, and the distinction matters.

What the flag recovers is the **system-prompt prefix** across tasks — a few
thousand tokens. The bulk of a run's cache traffic is *within* a task, as each
turn writes the growing conversation forward; the completed batch below writes
84k-228k cache tokens per task against 1.0M-2.1M read, and none of that is
touched by this fix. The flagship's own accounting puts batch amortisation at
$5.80 across 119 tasks, roughly **$0.05 a task**.

So: a mechanism that was completely broken now works, and it is worth about
five cents per puzzle. Both halves of that are true and the second one is the
one easy to lose. The reason to fix it is that a harness whose cost model
silently doesn't apply is a harness you cannot reason about — not that the
number moved.

### The completed batch

Four unseen public-eval tasks, run sequentially through `athanor cc batch`
(before the fix — this batch is what exposed it):

| task | result | iterations | cost |
|---|---|---|---|
| `e8686506` | SOLVED 1/1 | 1 | $1.702 |
| `78332cb0` | SOLVED 2/2 | 1 | $1.707 |
| `7b5033c1` | SOLVED 1/1 | 1 | $1.066 |
| `3dc255db` | SOLVED 1/1 | 1 | $2.970 |

**4/4 solved, $1.861 per task, 1.0 iterations mean.** Against the flagship's
$3.12 mean and $1.71 median.

Two caveats worth keeping attached to that number. Every one of the four solved
on the first submission, which is the cheap end of the distribution — the
flagship's per-task cost ranges $0.23 to $20.04, and a set of four
first-iteration solves is not a sample of that spread. And n=4 is n=4. What can
be said is that the variant is not in a different cost regime from the
flagship, and that the earlier single-run $3.80 figure was a cold run rather
than a representative one.

---

## Round 5 — the last two frontier pairs, and four ledger defects

`8b7bacbf` and `da515329`, the two zero-solve frontier pairs left. Both
accepted on the first submission.

| task | result | iterations | confidence | candidates | rivals registered |
|---|---|---|---|---|---|
| `8b7bacbf` | **2/2** | 1 | 4 | 2, 1 | 4 |
| `da515329` | **1/1** | 1 | 4 | 2 | 3 |

That completes the sweep: **six zero-solve frontier pairs attempted, six
solved** — `13e47133`, `269e22fb`, `a32d8b75`, `abc82100`, `8b7bacbf`,
`da515329`. Each is a pair no submission in the frozen public corpus had ever
solved.

Both runs are also clean demonstrations of the candidate mechanism doing the
thing it was built for. `da515329`'s rule has one parameter that training
cannot pin down — the chamfer width, observable only at `a >= 4`, and no
training example reaches it. The agent shipped both readings and got it right.
`8b7bacbf` hedged on a wire-crossing that occurs nowhere in training: "no
training pair contains a crossing at all, so nothing says whether an
interrupted stroke continues through."

### Four defects, all in how the ledger reads itself back

The interesting thing about this round is that none of the four was a wrong
answer. All four were the harness **misdescribing its own state** to the agent.

**1. Every dead end was rendered as a success.** `gate.py status` marked
entries by whether the recorded check *held* — but a `refute()` entry that
holds means the hypothesis is dead. The replay therefore read:

```
[OK  ] contact by 4-adjacency only also explains the training data
```

about a reading the agent had killed. As the solver put it: "On a skim after
compaction that reads as the *opposite* of what was found." The status replay
is the workspace's most durable artifact — it is what a post-compaction agent
reads first — so this was inverting the single record most likely to be read
without its context. Now `[DEAD]` / `[OPEN]`.

**2. `rival()` told a solver to delete a correct hedge.** `da515329` had
already shipped its 3/3-fitting rival as candidate 2. `rival()` then said:

> It predicts exactly what you do on every test input, so it is not a divergent
> reading and needs no slot.

The rival was "not divergent" only because it had been found inside the
candidate list it was already in. The solver caught it: "Had I trusted that
line I would have deleted the hedge and forfeited the slot… the one message a
solver most needs to be right about is the one that inverts when you follow the
advice early."

This is the second bug in the same function, and both came from collapsing
distinct situations into one boolean. There are four, and they now print
differently: redundant with candidate 1; *is* candidate 2 already; diverges
from both; diverges but both slots are spent on other readings. `8b7bacbf`
independently hit that last one — told "That is your second candidate" about a
test example whose second slot was already taken.

**3. A claim recorded from a heredoc captured no evidence, silently.** The
constant-condition check is derived from the caller's AST, so when the call
site cannot be read the check cannot fire. A solver recorded a refutation from
a heredoc with a literal `True` — precisely the anti-pattern the check exists
to catch — and got no warning, because the warning depends on source the parser
never reached. It noticed unaided and re-recorded from a file. "No evidence"
and "good evidence" looked identical in the ledger; entries now carry
`unsourced` and say so.

**4. A corrected claim had nowhere to put the correction.** `da515329`'s ledger
contains this live, `[VERIFIED]` invariant:

> the corner jog has a fixed width of 1 (m = a-1); it is instead always a - 2
> wide (m = 2)

A sentence whose first clause is false, stamped verified. Nothing was wrong
with the agent's reasoning — it had killed the first reading and found the
second, and `key=` supersedes by key, so amending the claim text and re-running
is the natural motion. There was simply no way to say *"that reading died, here
is what replaced it"*. Replacements now carry and print what they displaced,
and a claim whose verdict flips between runs says so loudly, because code
written while it held is now built on sand.

The through-line: **a ledger that survives compaction is read by someone with
no memory of writing it, and every one of these four defects was invisible to
the agent that had the context and misleading to the one that would not.**

---

## The frontier, counted properly

I described round 5 as completing "six zero-solve frontier pairs attempted, six
solved". That count was selective, and the correction matters more than the
headline did.

`RESULTS.md` defines the frontier as 9 test outputs across 8 puzzles: pairs
where no submission in the frozen 2026-04-12 public corpus ever succeeded,
with at least 80 logged attempts each. **All 8 puzzles have now been
attempted.** Two of them were attempted in earlier rounds and *missed* — and
listing only the six that were solved quietly dropped them.

| pair | HF attempts | this variant | flagship |
|---|---:|---|---|
| `13e47133 / 0` | 117 | solved | solved |
| `269e22fb / 0` | 119 | solved | solved |
| `269e22fb / 1` | 119 | solved | solved |
| `8b7bacbf / 0` | 125 | solved | solved |
| `a32d8b75 / 0` | 118 | solved | solved |
| `abc82100 / 0` | 121 | solved | solved |
| `da515329 / 0` | 120 | solved | solved |
| `9bbf930d / 0` | 117 | **missed** | solved |
| `faa9f03d / 0` | 127 | **missed** | **missed** |

**7 of 9, against the flagship's 8 of 9.** One pair the flagship solves and
this variant does not (`9bbf930d`, missed on 2 iterations); one pair neither
solves (`faa9f03d`, which no logged submission has ever solved and which
resisted a 120-turn flagship attempt).

> **Superseded.** `faa9f03d` was solved in round 8 — see "faa9f03d" below. The
> frontier standing is now 8 of 9. This table is kept as the round-5 record.

That is the number to quote. It is a good result — every one of these pairs had
100+ failed attempts behind it — and it is one pair short of the system this
variant is being compared against, on a frontier of nine. Both facts belong in
the same sentence.

---

## Round 6 — aimed at the flagship's own misses

With the frontier exhausted, the sharper comparison is `RESULTS.md`'s other
hard set: the 22 pairs no CoT-only Opus 4.6 configuration solved in 8 attempts.
The flagship solves 16 of those and names the 6 it does not:

`88e364bc/0`, `d35bdbdc/0`, `d35bdbdc/1`, `2b83f449/0`, `dbff022c/0`,
`faa9f03d/0` — five puzzles.

That set is the most informative target left, because on it a difference in
either direction is a capability signal rather than a variance one. Standing
before this round: `2b83f449` **solved** (the flagship fails it), `faa9f03d`
missed by both, `dbff022c` not yet attempted.

Round 6 takes three:

| task | why |
|---|---|
| `d35bdbdc` | flagship 1/3; its two unsolved pairs are both in the set |
| `800d221b` | flagship 0/1 (CoT-only Opus 4.6 solves it 3/8, so not in the 22) |
| `88e364bc` | **controlled retry** |

`88e364bc` is the one to watch. It is where this variant lost 2 cells out of
400 in an early round — a solver killed a valid rival by extending a
training-output regularity to the test input, and finished 1/2, matching the
flagship exactly. The entire rival mechanism was built in response to that
single failure. Re-running the same puzzle with the mechanism in place is the
closest thing this experiment has to a controlled test of whether the machinery
recovers a loss it was designed for, rather than merely being used.

A caveat that applies whatever comes back: n=1 per task, and these are the
puzzles where outcomes are least stable. A solve would be evidence the
mechanism can reach the answer, not that it reliably does.

### Round 6 results

| task | result | iterations | candidates | flagship |
|---|---|---|---|---|
| `800d221b` | **1/1** | 1 | 2 | **0/1** |
| `88e364bc` | **2/2** | 1 | 2, 2 | **1/2** |

Both are puzzles `RESULTS.md` records the flagship failing.

**`88e364bc` is the controlled retry, and it worked.** The earlier run on this
puzzle finished 1/2 — a solver killed a valid rival by extending a
training-output regularity to the test input, and missed by 2 cells out of 400.
The entire rival mechanism was built from that single failure. This run shipped
two candidates on *both* test inputs and took both.

The agent's account is mechanistic rather than lucky. Its uncertainty was
precisely located — whether a diagonal ray may pass a cell with an occupied
shoulder — and it identified that training contains only 4 diagonal rays, none
of which witnesses the case. Test 0 needs one reading, test 1 the other. It
found this by writing a script that enumerated the situation types its rule had
to handle on each test input and diffed them against what training witnesses;
that script found a case it "had already dismissed and would otherwise have
shipped unhedged."

Still worth stating plainly: this is one run against one earlier run. It
establishes that the mechanism *can* recover the loss it was designed for, on
the puzzle it was designed from. It does not establish a rate.

### What round 6 cost: four more defects, one of them a repeat

The repeat is the one that matters. The paragraph that produced that decisive
script — *"enumerate the situation types your rule has to handle on the test
input, and check that each one is actually witnessed in a training pair"* — was
printed **only after a submission**. The solver called it "the single
highest-value string the harness printed" and then noted it had arrived only
once the iteration was spent.

This is the third time a convenience has been found on the budgeted path alone,
after the candidate-spread line in round 4 and the hedging prompt before it. The
rule keeps having to be relearned: **anything that changes what a solver ships
must be reachable for free.** It now prints on every dry-run branch, with a test
that fails if the workspace copy and the gate copy drift apart.

The other three:

- `rival()` announced nothing when a re-registration replaced what a name meant.
  A solver fixed a buggy rival by re-registering the same name; the earlier
  entry — which had claimed divergence on both tests — vanished without trace.
  `verify()` goes to considerable trouble to announce supersession; `rival()`
  did not.
- The opaque-name warning fired even when `note=` carried the measured values,
  telling a solver to do the thing it had just done.
- `[DEAD]` read as though the *check* were broken rather than the hypothesis.
  Now `[KILLED]`, matching the word `refute()` itself prints.

And one gap that three separate solvers had reported without my noticing the
pattern: `arc.sweep()`. Each of them ran a tie-break sweep — 8, 14 and 18
candidate rules scored against training in a single pass — and each recorded
two or three of them and left the rest in prose, because `refute()` is
per-claim. One said it directly: "the ledger under-represents what was actually
ruled out." In every case the sweep was what produced the second candidate: the
highest-yield work in the run, and the least durable. `sweep()` records the
whole thing as one entry, and when more than one reading survives it says so —
that is a hedging obligation discovered before any budget is spent, and it names
the exact readings to put through `rival()`.

### A note on the durability check

Watching this round also showed the empty-ledger reminder was reaching almost
nobody: it hung off `arc.show()`, which agents call in roughly one script out of
five to eight, usually an early one. So it was gated on a call they had largely
stopped making by the time the condition became true. It now runs at interpreter
exit as well.

Honesty about the case that prompted it: `d35bdbdc` spent about forty minutes
and nine scripts before recording anything — and then recorded eight invariants
unprompted. The window of unprotected work was real; the agent recovered from it
without help. The fix is justified by the exposure, not by an observed loss.

### Round 6 complete

| task | result | iterations | candidates | flagship |
|---|---|---|---|---|
| `800d221b` | **1/1** | 1 | 2 | **0/1** |
| `88e364bc` | **2/2** | 1 | 2, 2 | **1/2** |
| `d35bdbdc` | **3/3** | 1 | 2, 2, 1 | **1/3** |

**3/3 tasks, 6/6 test examples, 1.0 iterations mean.** Every one of the three is
a puzzle `RESULTS.md` records the flagship failing or partly failing, and
between them they take five pairs the flagship does not.

`d35bdbdc` is the largest single gain: three test outputs, all three correct,
against the flagship's 1/3. Two of its pairs are in the 22-pair set that no
CoT-only Opus 4.6 configuration solved in 8 attempts.

Its solver hedged on two of three test examples for a reason it could state
exactly: two different selectors — nesting depth, and where the gray snake's
ends dock — agree on every training pair and diverge on tests 0 and 1. It
recorded the refutation that mattered most as a claim about the *evidence*
rather than about the puzzle: "training can separate the snake reading from the
depth reading" — refuted, meaning it cannot.

### Four more defects, and one that argues with the doctrine

The interesting one is (1), because it is the doctrine's own mechanism falling
short of what the doctrine asks for.

1. **`verify()` could only record one-liners.** The expression capture reads a
   single expression at the call site. Every substantive check in this run was a
   multi-line function — object detection, a BFS over the containment forest, a
   graph walk — so the honest options were a bare name (correctly flagged as
   opaque) or one unreadable comprehension, which is what the solver tried next
   and then had to retract: "a genuinely bad ledger entry". The harness insists
   the ledger records what executed, and for anything larger than an expression
   it could not. `verify(..., evidence=<measured value>)` is now a first-class
   field, and it answers both warnings because it supplies exactly what they ask
   for.

2. **Not every `refute()` is a rival reading.** The gate listed that
   evidence-claim under "rival readings you ruled out — check how each died",
   which inverts a refutation whose death was the finding.

3. **When live rivals all agree on the open slot, say so.** This solver's rival
   agreed on test 2, and it was shown the dead-end list for test 2 with nothing
   acknowledging the rival had been considered and found irrelevant there. "The
   slot needs a reading you have not named yet" is a different instruction from
   "reconsider these dead ends".

4. **A partial hedge is usually a bug.** This solver's alternative path returned
   early on one input, so `solve()` emitted two candidates for some tests and one
   for another. The nudge said "you registered no rivals" — true, and pointing
   the wrong way. As the solver put it, the harness "did not distinguish 'you
   chose not to hedge' from 'your hedge silently failed'."

---

## The flagship's own misses, counted

`RESULTS.md` names the 6 pairs — 5 puzzles — that the flagship does not solve
among the 22 that defeat every CoT-only Opus 4.6 configuration. All five puzzles
have now been attempted.

| pair | this variant | flagship |
|---|---|---|
| `2b83f449 / 0` | solved | missed |
| `88e364bc / 0` | solved | missed |
| `d35bdbdc / 0` | solved | missed |
| `d35bdbdc / 1` | solved | missed |
| `dbff022c / 0` | solved | missed |
| `faa9f03d / 0` | **missed** | missed |

**5 of 6** at the time of writing. The one miss was `faa9f03d`, which no logged
submission in the frozen public corpus had ever solved and which resisted a
120-turn flagship attempt.

> **Superseded.** `faa9f03d` was solved in round 8. All six pairs the flagship
> fails are now taken.

Set against the frontier table earlier — where this variant takes 7 of 9 and the
flagship 8 of 9 — the two results say something more specific than either alone.
The systems fail on different puzzles. The flagship takes `9bbf930d`, which this
variant missed; this variant takes five pairs the flagship does not. Neither
dominates, and `faa9f03d` defeats both.

That is worth more than a headline number, because it is evidence the two
architectures are not just differently tuned but differently *capable* — and
this one carries no reviewer, no second model, and no inter-agent artifact
exchange. What it has is the gate, the toolkit, and the doctrine.

Which of those three is doing the work is exactly what the next experiment asks.

---

## Round 7 — one gained, one that stayed lost

| task | result | iterations | candidates | note |
|---|---|---|---|---|
| `dbff022c` | **1/1** | 1 | 2 | last untouched puzzle in the flagship's unsolved set |
| `9bbf930d` | **0/1** | 1 | 2 | second attempt; the flagship solves it |

`9bbf930d` is the honest half. It is the one frontier pair the flagship takes
and this variant does not, and a retry carrying `sweep()`, the situation-types
prompt on the free path, and the rival-standing fixes **still missed it** — this
time with two candidates and a well-argued hedge rather than an unhedged guess.
Two attempts, two misses. The frontier standing was 7 of 9 at this point;
`faa9f03d` later took it to 8 of 9, but `9bbf930d` remains the sole miss.

Worth saying plainly: several rounds of harness improvements did not convert
this pair. The improvements are real and measurable elsewhere; they are not a
general solvent.

That run did, though, give the sharpest confirmation of the free-path fix
shipped one round earlier:

> "It fired at exactly the right moment and it changed what I shipped. It
> appeared on my very first `dryrun.py`, before any budget was spent… Without it
> I would have submitted the over-inclusive launch rule as candidate 1. This is
> the single highest-value string the harness printed."

One round before, the same paragraph existed only behind a spent iteration.

---

## The ablation: is the doctrine load-bearing?

Every solver so far was told to follow a code-as-verification doctrine and then
asked whether it helped. All said yes. That is a leading question, and it does
not separate "the harness works" from "Opus 5 is good at ARC".

So: `88e364bc` again — the puzzle where the doctrine arm went 1/2 → 2/2 — with
the doctrine **stripped** from `CLAUDE.md` and the system prompt withheld
entirely. No mention of verification-by-execution, rivals, hedging, or the
second candidate. The toolkit was left untouched and importable. Both arms were
told the benchmark allows two candidates, since otherwise the experiment would
test knowledge of the rules rather than the discipline.

### Result

| arm | candidates shipped | score |
|---|---|---|
| doctrine | test 0 → 2, test 1 → 2 | **2/2** |
| ablated | test 0 → 2, test 1 → **1** | **1/2** |

1/2 is exactly what the flagship scores on this puzzle, and exactly what this
variant scored on it before the rival mechanism existed.

### What the ablated agent actually did

Not what "no doctrine" might suggest. It read `arc.py` and used **the entire
toolkit unprompted** — `verify`, `sweep`, `rival`, `solution_module`,
`load_solution`, `png`, `diff` — and registered **eight** rival implementations.
It ran a sweep on a genuine ambiguity and got two survivors. It hedged correctly
on test 0. Its own account: "Whenever I was unsure I built the alternative and
executed it rather than reasoning about it."

So the practice transferred without the doctrine. What did not transfer was the
*specific* out-of-sample question. The two arms framed different residual
ambiguities: the ablated agent asked whether markers block each other mid-slide
and measured no divergence; the doctrine agent asked whether a diagonal ray may
pass a cell with an *occupied shoulder* and found that test 1 turns on it. The
ablated agent concluded "on test 1 nothing that fits training diverges" — and
shipped one candidate there.

### What this does and does not establish

It does not establish that the doctrine is worth 1 pair in 2. **n=1 per arm**,
on a puzzle whose three recorded attempts have gone 1/2, 1/2, 2/2 — two of three
landing at 1/2. A difference of one test example between two single runs is
inside the noise of that history, and saying otherwise would be reading a
coin-flip as a trend.

What it does show is narrower and more interesting: the toolkit alone is enough
to induce the *practice* — an agent told nothing about verification wrote eight
rivals and swept a tie — but on this puzzle the practice was not enough to find
the question that mattered. Both agents were rigorous. They differed in which
ambiguity they thought to frame, and only one of those framings reached the
second pair.

Two honest limits on top of that. This ablates the **doctrine, not the
toolkit**: `arc.py`'s own docstrings argue for verification-by-execution, and
the ablated agent plainly absorbed them, so the comparison is "tools plus
doctrine" against "tools alone", not against nothing. And a real answer needs
many runs per arm, not one. The value here is that it is the first evidence
pointing at *which component* does the work — and it points away from the system
prompt and toward the toolkit, which is the opposite of what the doctrine's
prominence in this repo would suggest.

---

## A negative result: verification density does not predict correctness

`cc trace`'s headline metric is verification density — invariants and
exploration scripts per formal iteration — and the module's own docstring called
it "the metric that matters". Measured across every run in the experiment, it
does not discriminate at all.

| group | n | invariants | before 1st submission | scripts | iterations |
|---|---:|---:|---:|---:|---:|
| fully solved | 21 | 14.9 | 11.8 | 12.0 | 1.2 |
| not fully solved | 4 | 13.5 | 11.0 | 13.5 | 1.2 |

The failures verify as much as the successes, and by some measures more. The
individual misses:

| task | score | invariants | before 1st | scripts |
|---|---|---:|---:|---:|
| `88e364bc` | 1/2 | 13 | 11 | 14 |
| `9bbf930d` | 0/1 | 18 | 14 | 13 |
| `9bbf930d` | 0/1 | 7 | 7 | 11 |
| `faa9f03d` | 0/1 | 16 | 12 | 16 |

`9bbf930d`'s better attempt established 18 invariants, 14 of them before its
first submission — *above* the average for a solved task — and missed anyway.

**What this means.** Density measures compliance with the doctrine, not quality
of reasoning. It is the right instrument for the job it can actually do:
catching a run that guessed first and verified afterwards, which is the failure
mode the doctrine targets and which has never occurred here. It is the wrong
thing to optimise, and a high number is not evidence a run went well.

The docstring has been corrected, because a metric that describes itself as "the
metric that matters" invites exactly the mistake of reading it as a quality
signal.

**Caveat in the other direction.** Every run in this experiment complies with the
doctrine — even the ablated ones, which verified heavily without being asked to.
So there are no low-density runs to compare against, and this says nothing about
whether density *below* the 11–14 range would hurt. The finding is that within
the range these agents actually produce, more verification does not mean more
correct.

That is consistent with the sharper thing the failures have in common, which is
not a shortage of executed checks but a *missing question*: on `88e364bc` the
ablated agent measured that markers-blocking made no difference and never asked
the corner-cutting question; on `9bbf930d` both attempts verified their rule
thoroughly and hedged on the wrong axis. Executed verification compresses the
cost of answering a question. It does not tell you which question to ask.

---

## The self-audit has never rejected anything

The reviewer was dropped from this variant; the self-audit in `solution/audit.md`
is what stands in for it. `docs/design.md` predicts that substitution will be
weak — "artifact-only review is the only mechanism that can reject a
train-perfect but overfit solution. The self-audit is a weaker substitute by
construction."

Here is the empirical form of that prediction, across 26 audited runs:

- **`DECISION: RETRY` has fired zero times.** Not once has a solver's own audit
  declined its own work — including on the five runs that were wrong.
- **The confidence scale has collapsed to `{4, 5}`.** Nobody has ever claimed 1,
  2 or 3. The highest confidence in the experiment, a 5, appears on a run that
  scored 1/2.

| claimed confidence | n | fully solved |
|---|---:|---|
| 4 | 20 | 16 (80%) |
| 5 | 6 | 5 (83%) |

Two claims here, of different strength. The **resolution** claim — that
confidence does not separate right from wrong — is underpowered: the base rate
is 84%, so there is little variance to predict at n=26, and 80% vs 83% is noise.
Do not read that table as "confidence is uninformative"; read it as "this
experiment cannot tell".

The **range** claim is solid and does not depend on the base rate. An audit that
only ever emits 4 or 5, and never RETRY, is not functioning as a gate. It is
functioning as a reflection prompt — which agents repeatedly credit with
changing their work, and which is genuinely valuable — but the rejection half of
the mechanism is inert.

**What changed as a result.** An unanchored 1–5 invites exactly this collapse,
so the scale is now anchored to something checkable, using the concept the
solvers themselves found most useful — whether the situations the test input
requires are witnessed in training:

```
5  every situation the test requires is witnessed in a training pair, checked.
4  one such situation is unwitnessed and you have hedged it with candidate 2.
3  a situation the test needs is unwitnessed and you have NOT hedged it.
2  the rule fits training and you cannot say why it should generalise.
1  you are submitting because the budget is running out.
```

Whether that produces a 3 — or a RETRY — is an empirical question the next
rounds answer. It may not: a solver confident enough to submit is, by
construction, a solver that thinks it is at 4 or 5. If the anchors change
nothing, that is worth knowing too, and it would be evidence for the stronger
reading of `design.md`'s claim: that self-review cannot reject its own work and
only an independent reader can.

### Ablation, replicated

A second ablated arm, `dbff022c`, same method — doctrine stripped from
`CLAUDE.md`, system prompt withheld, toolkit untouched.

| puzzle | ablated | doctrine |
|---|---|---|
| `88e364bc` | 1/2 | **2/2** |
| `dbff022c` | **1/1** | **1/1** |

One doctrine win, one tie. That is what "inside the noise" looks like, and it is
the reason the first result was not written up as an effect.

The **qualitative** finding replicated much harder than the score did. Told
nothing about verification, rivals, hedging, or the second candidate, the second
ablated agent used `verify`, `refute`, `sweep`, `rival`, `load_solution`,
`solution_module`, and `verify(..., over=)` — a parameter shipped thirty minutes
before its run. It found the same ambiguity the doctrine arm found (which cell
of a legend pair is the key, unresolvable because every training legend is flush
with the top or left edge while the test legend is flush with the bottom), and
hedged it the same way, on the same axis.

It also caught itself writing a tautology — `... or True` — and retracted it
unprompted, which is the round-1 literal-detection fix doing its job for an
agent that was never told the ledger mattered.

So across two arms: **the practice transfers without the doctrine.** An agent
that reads `arc.py` adopts code-as-verification, registers rivals, sweeps ties
and hedges — because the toolkit's API and its printed advice carry the
discipline, not because a system prompt asked for it.

That is a useful thing to know for anyone who would build on this. The
shippable, portable part of the harness is the toolkit. The doctrine may still
supply the specific out-of-sample question that decides a hard pair — that is
what `88e364bc` hints at — but two arms cannot distinguish that from chance, and
this experiment should not pretend otherwise.

### Ablation, third arm — a big gap with an incoherent mechanism

| puzzle | ablated | doctrine | flagship |
|---|---|---|---|
| `88e364bc` | 1/2 | **2/2** | 1/2 |
| `dbff022c` | **1/1** | **1/1** | 0/1 |
| `d35bdbdc` | 1/3 | **3/3** | 1/3 |
| **test examples** | **3/6** | **6/6** | — |

3 of 6 against 6 of 6 looks like a decisive result, and on both puzzles where
the ablated arm lost pairs it landed on *exactly the flagship's score*. It is
tempting to write that up as "the doctrine is worth three test examples".

**That would be wrong, and the reason is in the candidate counts.**

| puzzle | ablated hedges | doctrine hedges |
|---|---|---|
| `d35bdbdc` | 2, 2, 2 | 2, 2, 1 |

On `d35bdbdc` the ablated agent hedged **more** than the doctrine agent — two
candidates on all three test inputs, against the doctrine arm's two-two-one —
and still scored 1/3 against 3/3. It did not fail to hedge. Its *primary rule
was wrong*: it read the puzzle as pointer chains resolved by absorption, where
the doctrine arm read it as dropping every other level of a containment forest.
Two different rules, both reproducing all three training pairs, only one right.

So the mechanism is not consistent across the three puzzles. On `88e364bc` the
gap is a missing hedge; on `d35bdbdc` it is a wrong reading arrived at despite
more hedging; on `dbff022c` there is no gap. "The doctrine teaches you to hedge"
predicts the first and is contradicted by the second.

An inconsistent mechanism across three puzzles is what run-to-run variance looks
like. The alternative — that the doctrine somehow produces better *rule
discovery*, not just better hedging — is a much stronger claim, has no proposed
mechanism, and three runs cannot support it.

One confound worth stating because it runs *against* the ablated arm, not for
it: the ablated runs used a strictly newer toolkit than the doctrine runs they
are compared against — `sweep()` per-reading details, `verify(over=)`, rival
contention detection, all shipped between the two sets. The ablated arm had the
better tools and did worse. That makes the toolkit-improvements story weaker,
not the doctrine story stronger; both are undercut by the same variance.

**What survives from all three arms** is the qualitative finding, which
replicated cleanly every time: an agent given the toolkit and no doctrine reads
`arc.py` and adopts the practice — `verify`, `refute`, `sweep`, `rival`,
`solution_module`, `over=` — registers rivals, enumerates unwitnessed situation
types, and hedges. The third ablated agent wrote a `situation_types.py` script
unprompted and used it to add two hedges it had not originally framed. Nobody
told it to.

The score difference is not established. The behaviour transfer is.

---

## The cache fix at real-batch scale: invisible, as predicted

The `--exclude-dynamic-system-prompt-sections` fix was measured with a
microbenchmark (two trivial prompts in two working directories) and estimated at
about $0.05 a task. A real batch now runs with it, so here is the same
measurement at workload scale:

| batch | task | turns | cache write | cache read | read/write |
|---|---|---:|---:|---:|---:|
| post-fix | `35ab12c3` | 44 | 278,731 | 2,993,663 | 10.7 |
| post-fix | `58490d8a` | 38 | 79,145 | 1,073,954 | 13.6 |
| pre-fix | `e8686506` | — | 129,702 | 1,725,300 | 13.3 |
| pre-fix | `78332cb0` | — | 160,570 | 1,895,479 | 11.8 |

**The fix is invisible here, and that is the correct outcome.** The cross-task
prefix it recovers is a few thousand tokens; within-task cache traffic is
80k–280k. A real improvement of that size cannot show up against that
denominator, and the ratios pre- and post-fix are indistinguishable. This is the
$0.05/task estimate confirmed at scale rather than contradicted.

**A correction, because the surface reading is wrong.** The batch's first two
tasks cost $3.04 and $1.18, and it is tempting to call that drop the cache
warming up. It is not. Task 1 ran 44 turns and wrote 278k cache tokens; task 2
ran 38 turns and wrote 79k. The difference is how much work each task took, not
where it sat in the batch. Cost per task in this harness is dominated by run
length, and any batch-position effect is far below that noise.

The general lesson, which cost a wrong sentence to learn: **a fix whose
mechanism you have verified in isolation will still not be visible in aggregate
metrics if the thing it improves is a small term.** Verifying the mechanism and
verifying the magnitude are separate jobs, and a plausible-looking number
adjacent to a real fix will happily be misread as evidence for it.

---

## What this experiment taught about building the harness

Separate from the ARC numbers. Each of these was paid for by a specific defect,
and each was found by running agents rather than by reading code.

**1. Anything that changes what a solver ships must be reachable for free.**
Learned three times before it stuck. The candidate-spread line, the hedging
prompt, and the "enumerate the situation types your rule has to handle on the
test input" paragraph each existed only behind a spent iteration. The third one
a solver called "the single highest-value string the harness printed", and then
noted it had arrived after the budget was gone. The corollary, learned the same
way: *every convenience on the budgeted path creates an incentive to spend
budget* — one solver submitted purely to read a line the gate printed and the
dry run did not.

**2. The compaction-durable artifact is read by someone with no memory of
writing it.** Nearly every ledger defect is a variant of this. `[OK]` beside a
refuted hypothesis inverts its meaning to a reader without context. A claim
amended in place — "the jog is 1 wide; it is instead always a-2 wide" — leaves a
verified invariant whose own first clause is false. An entry recorded from a
heredoc carries no evidence and looks identical to one that carries good
evidence. None of these confused the agent that wrote them. All of them would
confuse the agent that read them back.

**3. Collapsing distinct situations into one boolean produces inverted advice.**
`rival()` had one sentence for four states — redundant with candidate 1, *is*
candidate 2 already, diverges into a free slot, diverges with both slots spent.
It told a solver that had hedged correctly that its rival "needs no slot",
because the rival was found inside the candidate list it had just been added
to. Acting on that deletes a correct hedge. The fix each time was to name the
states, not to reword the sentence.

**4. Instrument the thing you can measure, then check whether it measures what
you think.** Verification density was the harness's headline metric and its
docstring called it "the metric that matters". It does not predict correctness
at all — failures verify as much as successes. It is a fine compliance check and
a bad quality signal, and nothing but measuring it would have shown that.

**5. Verifying a mechanism and verifying its magnitude are separate jobs.** The
prompt-cache defect was real, the fix was confirmed by a clean A/B, and the
improvement is worth about five cents a task — invisible against within-task
cache traffic. A plausible number sitting next to a real fix (a cost drop that
was actually run length) was misread as evidence for it, in this very document,
before being caught.

**6. Ask the agent to critique the harness as seriously as it reports its
solution.** This was the highest-yield part of the entire loop. A solver that
has just spent twenty minutes inside the workspace knows things about it that no
amount of reading the code reveals, and roughly two-thirds of the defects fixed
here came from that section of the report rather than from tests, traces, or
inspection.

**7. Fast iteration produces untested branches; audit your own diffs.** Two
findings came from reviewing recent commits rather than from any agent: three
gate diagnostics shipped with no tests at all, and the rival-contention check
re-ran `solve()` once per registered rival — quadratic work for an advisory
print, on a solver that had registered eight.

**8. A negative result is worth shipping.** Density does not predict
correctness. The self-audit has never rejected anything in 26 runs. The cache
fix is invisible at scale. Several rounds of improvements did not convert
`9bbf930d`. None of these are flattering and all of them are load-bearing for
anyone deciding what to build on.

---

## `faa9f03d`

`RESULTS.md` describes this pair as the one remaining frontier task: 127 logged
attempts in the frozen public corpus, `ever_solved = 0`. Never solved by any
CoT-only Opus 4.6 configuration at any thinking level. The flagship fails it
0/1, including a dedicated 120-turn attempt.

**This variant solved it. 1/1, on the first submission, at confidence 3.**

### Verification, because the claim warrants it

| check | result |
|---|---|
| test outputs present in the workspace `task.json` | no |
| answer grid in `task/grids.md`, `CLAUDE.md`, `NOTES.md`, `arc.py`, either prompt | no |
| only file containing the answer | `.athanor/iterations/1/report.txt` — the gate's own report, written *after* submission, echoing the agent's prediction back |
| ground-truth match | exact, at **candidate index 1** |

### The primary reading was wrong

That last row is the whole story. The agent's *first* candidate did not match.
The answer came from the **second** candidate — the hedge.

The ambiguity it hedged was precise and it could state it: at a wire crossing,
which wire is drawn on top? Training cannot distinguish "fewest cells" from
"shortest longest-straight-run", because every training pair happens to agree
under both. The test grid separates them. The agent enumerated 98 candidate
orderings against 9 observed decisions in a single `sweep()`, found exactly two
survivors, registered both through `rival()`, and shipped them as the two
candidates. `rival()` told it — correctly — that the second reading "is already
your second candidate, leave it in place", and told it of a third reading that
both slots were spent and the question was "which two of the three survive the
most evidence".

That is the entire mechanism chain, built over five rounds from a two-cell loss
on `88e364bc`, working end to end on the hardest puzzle in the corpus. Without
the second candidate this run scores 0/1, exactly like every other attempt ever
logged against it.

### The confidence anchors fired on their first outing

Two hours before this run, the confidence scale was found collapsed to `{4, 5}`
across 26 audits, with `RETRY` never once fired, and was anchored to whether the
situations the test requires are witnessed in training. This is the first run
under those anchors, and it produced the **first sub-4 confidence in the
experiment**. The agent:

> "the situation-types paragraph … produced `11_situation_types.py`, which found
> that 5 of the 6 things the test needs are unwitnessed and, crucially,
> separated the four harmless ones from the one real leap. Without it I would
> have written 'confidence 4' and not noticed that the non-terminal re-route is
> a distinct reading. **It directly set my confidence number.**"

A 3 was the right answer: the run was correct, and correct *via the hedge*, with
one genuine unhedged leap remaining. An anchored scale reached down and said so.

### Frontier standing

**8 of 9 pairs — the same count as the flagship, and a different eight.**

| pair | this variant | flagship |
|---|---|---|
| `13e47133`, `269e22fb`×2, `8b7bacbf`, `a32d8b75`, `abc82100`, `da515329` | solved | solved |
| `faa9f03d / 0` | **solved** | missed |
| `9bbf930d / 0` | **missed** | solved |

Each system solves exactly one pair the other cannot. Between them the entire
frontier falls; neither does it alone.

### Caveats

n=1. This variant attempted `faa9f03d` once and took it; the corpus's 127
failures come from other systems and configurations, and nothing here says a
second attempt would succeed. The run cost 260k tokens and 58 tool calls, well
above this experiment's average — appropriate for the hardest task in the set,
but not a figure to generalise from.

And the honest shape of the win: the agent did not find the right rule. It found
that *two* rules survived all available evidence, declined to choose between
them, and spent the free attempt it was owed.

---

## What the second candidate was worth

The rival/sweep/hedging chain is the machinery this harness built over five
rounds, starting from a two-cell loss on `88e364bc`. Now that every run is in,
it can be measured directly: for each scored test example, which candidate
matched?

| | count |
|---|---:|
| test examples scored | 39 |
| solved | 35 |
| — by candidate 1 | 29 |
| — **by candidate 2** | **6** |
| examples shipped with two candidates | 21 (54%) |
| hedges that decided the outcome | 6 of 21 (29%) |

**Without the second candidate this experiment scores 29 of 39 instead of 35.**

The six that the hedge won:

| example | what it is |
|---|---|
| `faa9f03d / 0` | never solved by any logged submission, 127 attempts |
| `abc82100 / 0` | zero-solve frontier pair |
| `88e364bc / 0` | flagship fails it |
| `800d221b / 0` | flagship fails it |
| `d35bdbdc / 0` | flagship fails it |
| `d35bdbdc / 1` | flagship fails it |

**Every one is a hard puzzle, and all six are either frontier pairs or pairs the
flagship fails.** The hedging mechanism is not spread evenly across the
benchmark — it is disproportionately what takes the tasks that defeat other
systems. On the easy tasks candidate 1 was simply right.

### Reading this honestly

The 29-vs-35 comparison is against a hypothetical single-candidate policy that
nobody would adopt: ARC-AGI-2 *allows* two attempts, and using both is following
the rules rather than exploiting them. The number does not show the harness
beating the benchmark; it shows how much of this variant's result depends on
machinery that exists to make the second attempt land somewhere useful rather
than on a throwaway variant.

The better framing is the hit rate: **6 of 21 hedges paid off, 29%.** The other
fifteen were insurance that turned out not to be needed. Because the slot is
free, a 29% hit rate is pure gain — but it also means most hedging is wasted
effort in hindsight, and a solver cannot tell in advance which kind it is doing.
That is exactly the situation the harness's advice exists for, and it is why the
advice is framed as *"you cannot separate these on the evidence you have"*
rather than *"this one is probably right"*.

---

## Testing my own variance claim — and losing

When the third ablation arm came in at 3/6 against the doctrine arms' 6/6, I
argued the gap was run-to-run variance rather than a doctrine effect, on the
grounds that the mechanism differed per puzzle. That was an assertion, so I ran
the cheapest decisive test: a **second doctrine run on `d35bdbdc`**, the puzzle
with the largest gap, on a fresh workspace with the current toolkit.

| `d35bdbdc` | score | iterations |
|---|---|---|
| doctrine, round 6 | **3/3** | 1 |
| doctrine, replicate | **3/3** | 2 |
| ablated | 1/3 | 1 |

**The doctrine arm replicated exactly, and my variance claim is weaker for it.**
Two independent doctrine runs both took all three test examples; the ablated run
took one. A coin-flip story has to explain why the coin landed the same way
twice on one side and not the other.

What survives of the original argument is narrower but still stands: the
*mechanism* differs across puzzles. On `88e364bc` the ablated arm lost a pair it
had not hedged; here it hedged **more** than the doctrine arm (2,2,2 against
2,2,1) and lost anyway, because its primary rule was wrong. "The doctrine
teaches you to hedge" still does not explain both.

What is now missing is the symmetric test. The doctrine arm has n=2 on this
puzzle; the ablated arm has n=1. A second ablated run is the obvious completion
and is underway. If it also lands 1/3, variance is a poor explanation and the
honest conclusion is that something in the doctrine matters on this task — most
plausibly in *rule discovery* rather than hedging, which is a stronger and more
surprising claim than the one I started with.

Recording this while the answer is still unknown, because the value of stating a
prediction is lost if it is written afterwards.

### One report that did not survive checking

The replicate's friction section reported two `sweep()` defects: that the
printed summary shows only a count rather than the survivor names, and that the
`details` field "came out empty even though I passed tuples". Both were checked
directly against its own workspace. Neither is true — `sweep()` prints each
survivor by name with its detail inline, and that workspace's ledger contains a
fully populated `details` map for all seven readings.

Solver friction reports have been the highest-yield source of defects in this
experiment by a wide margin. They are still reports, not measurements, and two
of the three claims in this one were wrong. Checking before fixing cost two
minutes and avoided "repairing" working code.

### Did the confidence anchors actually recalibrate anything?

Eight runs have now completed under the anchored scale. Enough to check the
claim rather than repeat it.

| | n | values used | mean |
|---|---:|---|---:|
| before anchors | 27 | {4, 5} | 4.22 |
| after anchors | 8 | {3, 4, 5} | 4.12 |

**The honest reading is: one 3, on the hardest task in the corpus, and almost no
movement otherwise.** The scale reached below 4 for the first time in the
experiment — which is what it was changed to make possible — but the
distribution is still overwhelmingly {4, 5} and the mean moved by 0.1. Two
solvers reported the anchors changing the number they would have written, and
one of those is the sole 3. That is a real effect and a small one.

**`DECISION: RETRY` has still never fired.** 35 audited runs, zero self-rejections.

**And the anchors cannot catch the failure that matters most.** The ablated
`d35bdbdc` run claimed confidence 4 under the anchored scale and scored 1/3. It
was not wrong about which situations the test needed — it hedged all three test
inputs. It was wrong about the *rule*. The anchors are defined over "is this
situation witnessed in training", which is checkable, and a solver whose primary
reading is simply mistaken can satisfy every anchor honestly and still be wrong.

That is the shape of the remaining gap, and it is the same one `design.md`
predicts. Self-review can be made to ask better questions — the anchors
demonstrably do that — but a solver cannot audit its way out of a misreading it
has no reason to suspect. The one thing that has repeatedly recovered that class
of error here is not the audit at all: it is shipping the second candidate.
