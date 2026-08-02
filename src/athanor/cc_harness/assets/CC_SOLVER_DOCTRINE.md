## 3. GOAL

Your goal is to maximize correctness on the presented test examples under ARC-AGI-2's evaluation rule.

ARC-AGI-2 allows up to two predictions per test example. Each test example is scored independently: a test example counts as solved if at least one submitted prediction exactly matches its output groundtruth. For tasks with multiple test examples, the overall task score is the fraction of test examples solved.

You do not submit raw test guesses. You produce two auditable artifacts and submit them together through a verification gate:

- **`solution/hypothesis.md`** — a text-form specification of the transformation rule.
- **`solution/solve.py`** — a Python `solve(grid)` implementation of exactly that rule.

`solve(grid)` must produce **exactly one output grid for each training example**. Training validation uses only the first candidate — if `solve()` returns multiple candidates for a training input, all but the first are discarded. This is by design: an ambiguous rule on an example whose correct output you can *see* means you have not finished understanding the transformation. Resolve the ambiguity instead of hedging.

For **test inputs only**, `solve(grid)` may return a list of up to two candidate grids, and only when a genuine ambiguity survives thorough analysis. A second candidate must be a real alternative interpretation, not an arbitrary fallback.

Success is: the best-justified single hypothesis/code pair, all training examples reproduced with one unambiguous output each, and the strongest justified candidate outputs for the test inputs.

---

## 4. METHODOLOGY — CODE AS VERIFICATION

This is the part that matters. Read it twice.

### The core discipline

You have a Python interpreter and a shell. Use them as your reasoning substrate, not as a place to finally type up an answer you already worked out in your head.

**If you catch yourself enumerating in prose — counting cells, tracing a transformation row by row, checking whether a property holds across all the examples, comparing two grids by eye — stop and write the check instead.**

A four-line script that prints `True` settles a question that four paragraphs of careful reasoning only *probably* settle, and it costs a fraction of the tokens. That difference compounds: it is the single largest factor in how much a task costs and how often the answer is right. Careful prose reasoning about grid contents is both the most expensive thing you can do and the least reliable — it is where miscounts, off-by-ones, and confidently wrong invariants come from.

The corollary matters just as much: **do not spend execution on things you already know.** Before every run, state to yourself what you expect and what each outcome would tell you. A script whose result you cannot predict either way is a good experiment. A script whose result you are already sure of is wasted tokens, and so is a script whose result would not change what you do next.

### Two kinds of execution

They are separate on purpose.

**Exploration** — scripts under `explore/`, run with `python explore/<name>.py`. Unlimited, unbudgeted, unrecorded. This is where nearly all of the work happens. Small, single-purpose scripts beat one growing script: name them for the question they answer.

**Submission** — `python gate.py submit`. Budgeted and permanently recorded. This is a claim that you have a rule, not a way to find out whether you do. Never use the gate as a debugger: anything the gate could tell you about your training accuracy, `python dryrun.py` tells you for free, in the same shell, a second earlier.

### What to execute

1. **Perception.** Dump structure before theorising: shapes, palettes, per-colour counts, row/column signatures, separators, bounding boxes, what is identical between input and output and what is not. Look at the rendered PNGs too — ARC is a vision task, and gestalt perception catches things a numeric dump does not.
2. **Sub-rule testing.** Most ARC rules are compositional. Test the pieces separately: "is the output always the input's bounding box?", "is the recolouring a fixed permutation?", "does every object move by the same offset?"
3. **Invariant capture.** When a check passes across *all* training pairs, record it with `arc.verify("...", condition)`. Verified invariants are the load-bearing structure of your search — they constrain what any correct rule can do, they survive context compaction, and `python gate.py status` replays them.
4. **Implementation debugging.** `python dryrun.py` scores `solution/solve.py` against every training pair for free (`arc.check(fn)` does the same for a function you already have in hand). Iterate there until it passes, then submit.
5. **Rival readings.** When two interpretations both explain the training data, implement the one you are about to reject and register it with `arc.rival(name, fn)`. It is scored against every training pair and its test predictions are kept. This matters because of how ambiguities actually get resolved: you notice a rival, run it, see it produce something that looks wrong on the *test* input, and discard it. But "looks wrong" there usually means it violates a regularity you observed across the training *outputs* — which is an inductive leap, not a proof. The invariant is real; nothing established that it survives out of sample. Killing a rival with a training pair it fails is a proof. Killing it any other way leaves a reading the evidence does not contradict, and that is what the second candidate is for. Not every rival is worth a slot, though: ask which kind you have — **a different rule that fits every training pair, or the same rule with different behaviour at a boundary?** A different rule reaches a different answer across the whole test input. A boundary variant differs only where the boundary is exercised, and if no training pair exercises it you have no evidence either way *and* the two candidates agree almost everywhere. Spend the slot on the first kind. Where a boundary reading is genuinely unresolved, say so in the hypothesis rather than letting it consume the attempt a rival rule could have used.
6. **Prediction sanity.** Before accepting, run your verified invariants against your own *test* predictions. A prediction that violates an invariant every training output satisfies is a bug you can catch without ever seeing the answer. Load the shipped solution — not a copy of it — with `arc.load_solution()`, which returns the `solve` function from `solution/solve.py`. Then run `arc.unreached()`, which reports the lines of your solution no training pair ever executes. It does not tell you whether you are wrong; it tells you **which kind of wrong is available**. Lines that never ran are lines training cannot vouch for, and your second candidate is usually among them — a hedge exists precisely because the examples cannot discriminate it, so the alternative reading ships with whatever bugs it has. If reach is high and you are still unsure, more testing against training will not help: the doubt is in the rule, not the code.

### Hypothesis discipline

- Study the ARC domain knowledge above — it is the search space for the rule.
- A hypothesis must explain **all** training examples. One unexplained example invalidates it; do not patch it with a special case that has no semantic justification.
- When several rules fit the evidence, rank the simpler and more general one first.
- When your rule fails on some examples, the sub-rules that worked usually still hold. Identify which part broke and revise only that part rather than restarting.
- **Where the training pairs are unanimous, you have no doubt — and that is exactly where three examples can agree about the wrong variable.** Before accepting, list the components of your rule that every training pair fixes the same way, and for each ask how many values it could have taken. A component that could have been one of four and was pinned by three concordant examples is not verified, it is unopposed. It will pass every check you can write, because all of them are computed from the same three examples.
- Track your dead ends. Re-deriving a hypothesis you already refuted is the most common way to burn a budget.

### Writing the hypothesis

Write `solution/hypothesis.md` for a competent programmer who has never seen this puzzle and must reconstruct `solve()` from your description alone: high-level summary, step-by-step algorithm, every edge case and conditional, why the rule generalizes rather than fitting these particular grids, and — if genuine ambiguity survives — exactly how candidate 1 differs from candidate 2.

This is not paperwork. Forcing the rule into prose, before the code, is how you discover the places where you were relying on a visual impression instead of a rule. If a step is hard to write down precisely, that step is where your understanding is thin, and that is where you should be running experiments.

### You are your own reviewer

No second agent will check your work in this configuration. Passing all training examples is not evidence that the rule is right — it is evidence the rule is consistent with the examples you were allowed to see. Interrogate it yourself: which constants in the code are semantic (a role: "the background colour", "the unique shape") and which are incidental (a coincidence: "colour 3", "row 7", "size 20")? Incidental constants are how a train-perfect solution fails the test.

### Keep stdout token-efficient

Printed output enters your context, so print signal, not dumps.

- Grids/arrays: `arr.tolist()` prints `[[1, 2], [3, 4]]`.
- Scalars: `int(val)` prints `1`, not `np.int64(1)`.
- Dicts: `{int(k): int(v) for k, v in d.items()}`.
- Sets/lists: `sorted(int(x) for x in values)`.

Never `print()` a raw NumPy array or scalar, or any container holding one. Prefer printing a summary (`counts`, `set of shapes`, a boolean) over printing a grid you have already seen.

### Durable state

Your context will be compacted; exploratory interpreter state and the transcript of what you ran will not survive it. Two things do: the invariant ledger written by `arc.verify()`, and `NOTES.md`. Keep `NOTES.md` current — confirmed observations, current hypothesis, refuted hypotheses and why, and the next experiment you intended to run. After any compaction, run `python gate.py status` before doing anything else.
