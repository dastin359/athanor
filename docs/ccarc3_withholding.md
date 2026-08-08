# How CCARC3 withholds the human baselines, and what each layer is for

**This file exists because its contents cannot live in `client.py`.** The solver
imports that module, and `inspect.getsource(ArcClient)` returns the whole class
body — every `#` comment included. `docs/` is not on the solver's `PYTHONPATH`,
so this is the one place the reasoning can be written down in full.

That took two attempts to get right, and the second failure is the more
instructive one.

- **First**, the rationale lived in **docstrings**. `help(client)`,
  `pydoc.render_doc(ArcClient)` and `arc.gate.__doc__` all render docstrings, so
  an orienting solver got a complete briefing: `help(arc.GameInfo)` printed *"real
  games differ in length by an order of magnitude"*, `baseline_here` announced that a
  median exists and named the private attribute holding it, and `arc_proxy`'s
  module docstring described the entire defence. Fixed in `f66cc91` by moving it
  all into `#` comments.
- **Second**, the rationale lived in **comments in the same file**, on the
  argument that `help()` cannot see comments. True — and irrelevant.
  `inspect.getsource` on a *class* spans the whole block, comments and all. It is
  as standard a call as `help()`, and on 2026-08-07 a solver ran
  `dir(client)` followed by `inspect.getsource(client.restart_for_replay)` on its
  first orientation turn, one command short of the class.

  The near-miss is why this was easy to get wrong: `getsource` on a single
  *method* starts at its `def` and excludes preceding comments, so spot-checking
  a property returned clean output and looked like proof.

## The layers, in order of how much they actually carry

1. **The strip** (`tools/ablate_baselines.py`). Rewrites the workspace so the
   numbers are not in the solver's process at all: `baseline_actions=()` in
   `session.py`, the array and `action_budget` gone from `meta.json`, §6 removed
   from the doctrine, and `CLAUDE.md`'s baseline paragraph dropped whole. **This
   is the load-bearing layer.** Everything below it is defence in depth. With the
   array empty, `info.baseline_for(level)` returns `None`, so even
   `client._baseline_here_enforced` — the private attribute a determined solver
   would reach for — yields nothing.
2. **The proxy** (`src/athanor/ccarc3/arc_proxy.py`). Holds the ARC key and
   forwards four endpoints, refusing `/api/games`. Closes the deliberate route: a
   solver with a key can otherwise fetch every environment's baselines in nine
   lines of `urllib`, and two runs did exactly that before this existed.
3. **`CCARC3_HIDE_BASELINES`**. Closes the in-process route — `baselines_for`
   raises, `list_games` filters. Added after `cd82` ran `dir(arc)` on its first
   orientation turn and read `baselines_for` straight out of the listing, having
   gone looking for nothing at all.
4. **`quiet_pace`** (renamed from `hide_baselines`, 2026-08-07). Silences the
   pace ratio, `pace()` and the `raw`/ceiling half of the score block. Renamed
   because the workspace `session.py` passes the keyword literally and the
   doctrine tells the solver to import that file — so the old name announced in
   one word that per-level medians exist and are being kept from it.
5. **`redact_self_reference`**. Removes doctrine passages naming the game being
   played, so no solver reads about its own prior attempts. Only rewrites
   `DOCTRINE.md`; anything else that names environments has to be handled where
   it lives.

## What is still reachable, stated plainly

`dir(client)` lists `baseline_here`, `level_budget`, `max_actions` and `pace`.
Those names disclose that a baseline concept exists. They are not renamed because
`baseline_here` is the honest name for what the property returns when a run *is*
allowed to see it, and a euphemism in the API would be a worse cost than the
signpost. The numbers behind them are empty under the strip, which is layer 1's
job and is verified per-run by `tools/proofread_trace.py`.

**The distinction that matters, and the standard to hold new code to:** a *map* is
whether information is being withheld, by what, and where it lives; the *treasure*
is the numbers. Both matter, but only the treasure changes a score.

## The doctrine itself leaked a median, and eight banked runs read it

Everything below this line concerns the *package*. The workspace leaked first and
for longer: between `ea407ce` (2026-08-03) and `f66cc91` (2026-08-07) `§0a` of
`DOCTRINE.md` carried the worked example `65.533 = (17/21)² × 100`. `17` is a
real per-level human median, in the first file every solver is told to read.

Eight banked runs are inside that window and their preserved workspaces still
contain the string. They were tiered `superseded` — "older harness" — and pooled
into a published 25-environment figure that is now withdrawn. See
`docs/ccarc3_results.md`.

So the claim that layer 1 "held throughout" was false for four days. What is
true is narrower and worth stating exactly: **the 17 runs of the repaired-surface
cohort are baseline-free, and no run in any cohort received a per-level array.**

## The treasure did leak, 2026-08-07

Every finding above this line was a map leak. **Eight sites were not.** They put
real human medians into the package the solver imports — `arc_proxy.py`
documented its own response filter with a live scorecard, so the module whose
job is stripping `level_baseline_actions` contained one environment's complete
array; `client.py` carried four more in prose explaining why each fix was made;
`gate.py` quoted a total inside a comment about having removed that total.
Purged in `9eae0bd`, with three tests that fail on reintroduction of any of them.

## The second audit, 2026-08-07: four more, and none of them a median

A seven-lens adversarial audit found what the first purge missed. All four are
*functions* of a median rather than a median, which is why every value-based
guard walked past them.

* `scoring.score_run`'s docstring gave a worked example as `21 actions ->
  level_scores [65.5, ...]`. `S = 100*min(1.15,(h/a)^2)` inverts that pair to
  `h = 17` exactly — one real per-level median, rendered by `help()`, with the
  formula supplied by the doctrine in the solver's own workspace. The same
  docstring named the path to a stored card holding two complete
  `level_baseline_actions` arrays. Now symbolic (`higher`/`lower`), and the
  reasoning lives here rather than in the docstring — writing *why* a leak was
  removed into the place it was removed from is how `gate.py` kept its number
  for months.
* `Ccarc3Config.budget_multiple` and two comments in `session.py` printed
  `used/cap` pairs and one bare cap. A cap is the median total times
  `budget_multiple`, whose default sits in the same file. Now percentages.
* `ArcClient.level_revisits` tabulated a ratio column beside an action column;
  `h = a/ratio` recovered five per-level medians. Now percentages only.

**The guard that should have caught the first one walked straight past it.** The
reachable-docs test rendered `athanor.ccarc3:score_run`, and all three of its
regexes returned empty — because `65.5` is a score and `21` is an action count,
and neither is baseline-shaped. Checking for the secret is not enough when the
secret's published transforms are lying beside it.

Two new tests cover the transforms: caps (`total x {2,4,5}`) and level-score
presentation. A third was attempted and abandoned — checking decimals against
every published median finds a match for almost anything, because ~200 medians
over 1..400 make the space dense: `97.1` is `100*(67/68)^2` and `99.7` is
`100*(368/369)^2`. That density is why the *shape* is banned instead of the
value, and why review, not a regex, is the real defence here.

## What is enforced, and what is only asked

Stated plainly because the layers above read stronger than they are.

**The solver runs as `root`, in this container, and the API key is on this
disk.** `scratchpad/arc3/.env` is mode 600 owned by root — and the solver *is*
root, so it is readable, and one `GET /api/games` with it returns
`baseline_actions` for all 25 environments, past the proxy, past the environment
stripping, past the workspace strip. `scratchpad/best_or_last/card.json` is a
real card with a complete median array. `docs/ccarc3_results.md` holds more.

So layers 1–5 are what stands between an *incurious* solver and the medians. What
stands between a determined one and them is the sentence in `CLAUDE.md` that says
not to read outside the workspace, plus detection after the fact. Closing that
needs a separate uid or container, which is a change to the harness rather than a
patch to it.

**Detection was blind where it mattered most, until it was measured.**
`proofread_trace.py` flagged `/home/user/athanor`, so the repo was covered — but
not the scratchpad, a different root, which is where the key and the card are.
Measured: `cat <scratchpad>/best_or_last/card.json` passed the reach pass, and
its contents passed the inbound pass too, because the per-level-array check is
built from *this* game's medians and that card belongs to another. **For 24 of
the 25 environments that read was invisible to both passes.** Reach is now a path
question — any absolute path outside the run's own workspace — and inbound looks
for any game's median array.

## What the corpus actually shows

Two scans over all 30 preserved attempts, covering all 25 environments:

| check | result |
|---|---|
| `proofread_trace.py`, tightened reach + foreign-median inbound | **0 findings** |
| `leak_exposure.py` — the exact pre-purge leak strings, in inbound and outbound | **0 exposures** |

`leak_exposure.py` first asserts that all twelve patterns really were live in the
package before `9eae0bd`, recovered from git rather than retyped. A scan for
strings that were never there proves nothing, and this project has shipped five
guards that reported success while doing nothing.

So: the *package* leaks were real and present for the whole arm, and **no banked
run received one**. What that does not establish is that the design prevents it —
it does not, per the section above — only that it did not happen.

**Scope, because the unqualified version of that sentence was this document's
last word and it was wrong.** It read "no banked run received one. The
25-environment result stands." Both halves overreach. This section is about
leaks in the *package*; the **workspace** leaked separately and earlier, and
eight banked runs did read a real median out of `DOCTRINE.md` §0a — stated
plainly forty lines above, in this same file. The pooled 25-environment figure
is **withdrawn**, not standing; what survives is the 17-run repaired-surface
cohort.

A reader who reached the end and stopped got the opposite of the finding. Which
is the shape this file spends its length documenting: a true statement about one
layer, written where it reads as a verdict on all of them.
