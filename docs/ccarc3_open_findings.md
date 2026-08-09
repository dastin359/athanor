# Deep-proofread findings — all twelve adjudicated, 2026-08-08

Recovered from `wf_166ed0b5-984`'s journal after a container restart killed the
run mid-verify. Twenty were hunted, three verified by the sweep itself, eight
fixed at the time; these twelve were the remainder, and they were **claims to
check, not defects to fix** — two findings from the same sweep had already been
refuted on inspection and one proposed fix would have broken correct code.

**Outcome: 12 checked, 12 confirmed, 12 fixed.** Every one held up, but three of
the twelve needed a different fix than the one proposed, and checking four of
them surfaced five further defects that nobody had reported.

| # | subject | verdict | commit |
|---|---|---|---|
| 3 | `clean_rollouts.uncorroborated` counts discarded attempts | confirmed | `d5746c9` |
| 5 | `test_proofread_reach.py:206` is a tautology | confirmed | `d5746c9` |
| 6 | the best-play test never imports the selector | confirmed | `ecfcd88` |
| 7 | `HIDDEN_FIELDS` has 6 entries, the fixture names 4 | confirmed | `ecfcd88` |
| 9 | `client.py` credits `session.build_cli_args` with layer 3 | confirmed | `da5501e` |
| 10 | `build_trace_audit` cites an environment check that does not exist | confirmed | `da5501e` |
| 11 | `arc_proxy.ALLOW` says "four call sites"; there are five | confirmed | `da5501e` |
| 15 | the mid-sweep split check cannot ever fire | confirmed | `af50059` |
| 16 | the doctrine's §6a level *was* cleared | confirmed | `fbf8887` |
| 17 | `tn36`'s run tally omits the 6-of-7 run | confirmed | `fbf8887` |
| 18 | 46 of 157 is 29%, not the 20% it is offered as evidence for | confirmed | `fbf8887` |
| 19 | the arm caveat mixes counterfactual `E`s with deltas | confirmed | `fbf8887` |

## Where the proposed fix was wrong

**7.** The finding said parametrising over `arc_proxy.HIDDEN_FIELDS` "would fail
on any entry silently removed". It would not: the parametrisation *shrinks* with
the frozenset, so deleting an entry deletes its test and the suite stays green.
Two tests were needed — a roll-call of the six names, each annotated with the
endpoint that emits it, plus the depth parametrisation. The roll-call earned its
keep within the hour: a mutation run timed out mid-iteration, the next command
snapshotted the already-mutated file as its backup, and `baseline_actions` was
gone from the tree for real. The roll-call named it.

**9.** The finding offered "or, better, set it in `session.build_workspace` so the
claim becomes true". That would have been theatre. Layer 3 closes the in-process
route to the medians; on an unstripped workspace layer 1 has already rendered
`baseline_actions=(...)` into the solver's own `session.py`, so arming the flag
there guards a door standing open. The sentence was corrected instead.

**15.** The finding proposed comparing each finished game against the sweep's
recorded history. Right idea, but the same scan then had to replace `sweep_card`'s
`played` computation too — see below.

## What checking them turned up that nobody had reported

- **`sweep_card`'s refusal read `attempt_1` and meant "any attempt"** (found while
  fixing 15). A game banked on `attempt_2` — because its first was interrupted,
  the ordinary case this driver is built around — counted as zero games on the
  card, so the refusal to remint over a card carrying finished work failed *open*,
  on the one decision the driver explicitly declines to make for an operator.
- **Four fixtures in the new verdict test returned exit 2 from the "no stream to
  proofread" early-out**, never reaching the check each one named. Both survived
  the mutation that deletes their verdict. Same shape as finding 5, in the file
  written to fix finding 5.
- **Seven tests took `ARC_API_KEY` from the ambient environment**, so
  `test_resume_agreement.py` passed or errored depending on whether the launching
  shell had sourced `.env`. The suite is now 791 passed / 5 skipped with no key
  and 821 passed with one.
- **`build_without_baselines` had no test at all** — the function every scored run
  goes through. It has one now, built for real and offline.
- **The §6a rewrite used the word "budget"**, which
  `test_no_solver_facing_file_frames_actions_as_an_allowance` bans in any
  solver-facing file. Caught by that test, reworded.

## The shape, again

Eleven of the twelve are one defect wearing different clothes: **a check, a test
or a claim that names a thing and reads a proxy for it.** The signature is that it
*passes by not running* — an empty loop, an unreachable branch, a fixture that
cannot express the failure, a comparison that is equal by construction, a
denominator borrowed from a different measurement, a delta that is really a
ceiling. Finding 16 is the twelfth and is the same family from the other side: a
rule inferred from an observation that was itself misread.

The counter-practice that worked here, every time: **mutate the code the test
claims to protect and confirm the test fails.** Six of these twelve were
originally found that way and all six fixes were verified that way. Where a
figure was the subject, the equivalent is re-deriving it from a source that could
not have been copied from the same place — `cap` = 21/28 forces six cleared
levels whatever the prose says.

## The leak scan, calibrated against the corpus (2026-08-09)

`proofread_trace.array_arrived` is the check that decides whether a run's solver
was handed its own answer, and it is the last one standing between a rollout and
a published score. A mutation pass over it left four survivors — three of them
live holes, one a coefficient nobody had measured.

**Two were safety holes**, in the sense that a contaminated run reads clean:

* `for start in range(len(seen))` reduced to `range(1)` passed every test. The
  scan then reads only the first window of the transcript. A leak lands wherever
  the solver happened to dump the array, which in a 400 kB stream is essentially
  never the first hundred bytes.
* `strayed`'s `/tmp` exemption dropping its `"/scratchpad" not in path` clause
  passed every test. The exemption exists for ordinary temp files; the scratchpad
  under `/tmp` is where every banked card, ledger and `meta.json` on this box
  lives, and a scorecard carries complete median arrays for every game played on
  it.

**Two were coefficients**, and the honest way to settle a coefficient is to
measure it rather than argue it. 77 preserved streams × 16 recoverable median
arrays:

| variant | fires | new fires 3n does not make |
|---|---|---|
| `window = 3n` (shipped) | 43 | — |
| `window = 10n` | 46 | 3 |
| `window = 30n` | 99 | **56** |
| order-insensitive, 3n | 53 | **10** |

The 56 extra at `30n` include `bp35`, `cd82`, `lp85`, `tn36`, `vc33` and `su15`
— most of the bank — each "leaking" a *different* game's medians by coincidence.
The 10 extra for order-insensitivity land on four banked clean rollouts. So both
widenings void runs that are not contaminated, and a false void is the failure
this scan is least able to survive: it discards a real result, and unlike a false
clean nobody ever goes looking for it.

That measurement also closes the argument that ignoring order is "strictly
stronger". It is not, and the gap it leaves — a solver that sorts the array
before printing — is narrower than four discarded runs. Recorded rather than
fixed.

All eleven mutants are now caught, including the two calibration points, whose
tests carry the table above in their docstrings so the next person to widen the
window sees what it costs.

## The network gate, mutation-audited (2026-08-09)

`arc_proxy` is the only thing standing between the solver and `/api/games`,
which returns `baseline_actions` for all 25 environments. Twelve mutants, three
survivors, all three now closed or explained.

* **`set_budget` drops the session, and the adopted one has to survive it.** A
  card is reachable only from a session carrying its four `AWSALBAPP-*`
  stickiness cookies — measured live, 8 of 8 card reads succeed direct and 1 of 8
  through a shim without them. `proxy_for()` adopts the driver's cookies when a
  sweep shares a card; `build_without_baselines` then calls `set_budget`, which
  resets the session so a new game cannot inherit the previous one's pinning. The
  two coexist only because `_adopted` is stored apart from the live jar and
  re-seeded. Nothing tested either half, and its own docstring names the hazard:
  storing them in the jar alone "would make the order of these two calls silently
  load-bearing". Deleting the reset, collapsing the store, or clearing
  `_card_is_lent` all passed the suite. This is on the sweep's path — a shim that
  loses its pinning fails its first RESET with `game <id> not found`, a message
  that names the game and means the session.
* **The exhaustion refusal must not name the cap.** The cap is the withheld
  per-level total times `budget_multiple`, and `budget_multiple` defaults to 5.0
  in this package's source on the solver's `PYTHONPATH`. `CCARC3_MAX_ACTIONS` is
  popped from the child's environment precisely so the figure stays out of reach;
  a mutant that formatted the ceiling into the refusal handed it back, and passed
  all 110 proxy tests.
* **Equivalent mutant, third of three recorded.** `p.match(path)` →
  `p.search(path)` in `_allowed` changes nothing, because every `ALLOW` pattern
  is `^…$` and without `re.MULTILINE` a leading `^` matches only at position 0.
  That equivalence is a property of the four patterns, not of the code, so the
  anchoring is now asserted directly — removing either anchor from any pattern is
  caught, and the `search` mutant stays a documented equivalence rather than an
  untested coincidence.

(The other two recorded equivalent mutants: `scoring.capped()` `>`→`>=`, and the
reap deadline `>`→`>=`.)

## The three solver-facing modules, mutation-audited (2026-08-09)

`grids.py`, `rules.py` and `ledger.py` were the last on-path modules with line
coverage and no mutation audit. **95 mutants** across the three, all of them
kept in `tools/mutation_battery_ccarc3.py` so the audit can be re-run rather
than believed; 27 survived the suite as written. **Four were real defects; the rest were test holes.** The
split is worth stating that way round, because "the module was correct and
nothing was checking it" is the ordinary outcome here and is still worth the
work: an unchecked correct module is one edit away from an unchecked wrong one.

### The four defects

1. **`grids.logical()` returned the caller's own array on one of its two
   branches.** It copies when it reduces and aliased when it does not, so which
   contract applied was a property of the *data*. A solver writing to the result
   was writing to the ledger's frame. The aliasing branch is the one real games
   take — a board scaled into the 64×64 viewport rarely has an integer block
   factor, which the function and the doctrine both already say — and the
   copying branch is the one an `np.kron` fixture takes, which is why the tests
   only ever saw the safe half. `collapse()` had the same split on its
   empty-grid branch. Both now always return a fresh array.

2. **`grids.cell_boundaries()` pooled frames of different shapes silently**, and
   bounded the answer by whichever frame came last. It is documented to take a
   whole trace and ARC-AGI-3 changes board shape at a level boundary, so the
   ordinary call was also the broken one. Measured: a 12×12 frame with a
   boundary at row 6, pooled with a 6×6 frame, returned `[0, 3]` — the 12×12's
   real boundary dropped for being out of range of a board it never belonged
   to, and the survivor reported as an index into the wrong frame. It now
   refuses, for the reason `diff()` already refuses a shape change.

3. **`ledger.TraceWriter.append()` recorded score 0 for a frame carrying
   `score: None` beside a real `levels_completed`.** The fallback was keyed on
   key-absence (`frame.get("score", fallback)`), not on value-absence, so the
   compatibility seam between `arc_agi_3`'s `score` and `arcengine`'s
   `levels_completed` had a third door its own code comment did not cover. The
   blast radius is larger than it looks: `infer_levels` reads score increments
   as level boundaries, so a zeroed score collapses every level in a trace into
   level 0.

4. **`counts` was missing from `grids.__all__` and `ledger_facts` from
   `session.__all__`.** Harmless only by luck — `ccarc3/__init__` imports by
   name, which ignores `__all__` — but that is the *fifth* stale enumerated list
   in this repo. Replaced with the derived rule rather than a sixth manual
   entry: anything the package exports must be exported by the module defining
   it, and no module may promise a name it does not have.

### The test holes worth naming

`rules.py` had no code defects at all, and four of its seven survivors were the
exact failure the module exists to prevent, in the places nothing checked:

- `VerifyResult.vacuous` reading `holds == 0` instead of `applicable == 0`
  relabels a **refuted** rule as "never applicable" — the module's founding
  distinction, inverted, passing every test.
- `PredictionReport.perfect` without its `correct > 0` clause reports a forward
  model that declined every transition as perfect. The docstring tells solvers
  `perfect` is the thing to chase.
- `PredictionReport.accuracy` counting skips in the denominator punishes a model
  for declining honestly.
- `regressions()` broadened from "applicable and violated" to "not vacuous"
  fires on a mechanic that is **working**. Its own docstring says why that is
  fatal — "a boolean predicate would have fired on both and trained everyone to
  ignore the alarm" — and nothing made it stay quiet. There is now a test that
  runs a four-rule book where nothing is wrong and asserts silence.

In `ledger.py`, `load()` chained `before` from `grids[-1]`; taking `grids[0]`
passed everything, because the two agree until an action renders more than one
frame — which `grids.py` opens by warning is the normal case. And
`TraceWriter.append` could stop writing `full_reset` altogether without a single
failure, a flag that `board_replaced` and `level_pace` both read.

### Three of my own new tests survived their mutants

Written to kill a specific mutant, they passed against it. The instructive one:
`monotone_rows`' threshold defaults to 0.9, and a nine-of-ten fixture sits
*exactly* on it, so the test passed whether or not the skipped pair was counted
in the denominator. The fixture agreed with the mutant. Same defect as the ones
above, one level up — which is the argument for running the battery against new
tests too, not just against old code.

**Totals: 95 mutants, 93 caught, 2 equivalent and argued below.** Re-verify
with `.venv/bin/python tools/mutation_battery_ccarc3.py`, which exits non-zero
on any survivor it is not expecting.

## The card-facing scoring path, mutation-audited (2026-08-09)

The RHAE core of `scoring.py` carries mutation evidence in its own docstrings.
The functions added later for **card corroboration** did not, and they are what
the 25-game shared-scorecard sweep runs through when deciding whether a run may
be banked. 25 mutants; **11 survived**.

**No code defects** — every pristine behaviour held up under a live probe. All
eleven were holes, and two are worth naming because of where they sat:

- **`score_run` could be made to score the *first* play and nothing failed**,
  even though scoring the *last* play was caught. Every fixture in the suite had
  its best play first or last, so "best" was only ever pinned from one side. The
  shape that separates best from first, last and worst simultaneously is
  *middle-is-best*, and no such fixture existed. There is one now.
- **`card_disagreement` could accept a card one level behind the result, or read
  the last of an attempt's plays instead of the best, and stay silent.** That
  function exists because a frozen card once let a run finish 8/8 against a card
  stuck at level 3 with no other symptom. A hole in it is a hole in the only
  check independent of our own trace.

The rest: the opening RESET able to open an empty play; both `score_run`
tie-breaks (efficiency, then fewest actions) unpinned, so the choice between two
capped plays was incidental rather than deterministic;
`server_actions_per_level` defaulting to the *first* play rather than the most
recent — which under a shared card means reading a possibly-discarded earlier
attempt; the same function able to return `[]` for a missing card instead of
raising, which would make `disagreements_with_server` compare nothing and report
agreement, a missing card reading as a clean one; and
`disagreements_with_server` able to report only the direction where our count
came in lower.

### The tie-break fixture had to be searched for

`score_run` breaks a score tie on `raw` and then on fewest actions, and those two
rules almost always agree — higher efficiency normally means fewer actions. A
fixture where they agree cannot tell them apart, so it cannot kill the mutant
that deletes one. Rather than assume the mutant was equivalent, the pair was
found by search: against baselines `[100, 200]`, a play spending `60 + 95`
actions and one spending `94 + 60` both cap out at the same score, but efficiency
prefers the first (raw 1.1500 vs 1.1439) and fewest-actions prefers the second
(154 vs 155). **"I could not construct a discriminating case" is not the same as
"no case exists"**, and the difference is a search.

### One small correction, and one guard

`score_run`'s `if not candidates:` branch is unreachable — `plays()` always
yields at least one list, so an empty ledger arrives as `[[]]` and is scored by
the live path. The branch is harmless, but its comment claimed it was what
handled the empty ledger, which would have misled anyone relaxing `plays()`. The
comment now says what is true.

`card_disagreement` coerced a missing or zero `playthroughs` to 1 with `or 1`,
which let a **negative** through — where it becomes a slice length and silently
selects a different set of the card's rows to compare. A non-positive count is
now refused. This function is the one check that does not read our own trace, so
a corrupted `result.json` is exactly what it may be needed to catch; computing a
comparison from the corruption is the one thing it must not do.

**Totals: 25 mutants, 23 caught, 2 equivalent and argued below.**

## Equivalent mutants, recorded rather than tested around (2026-08-09)

A mutant that survives is either a gap in the tests or a change that cannot alter
behaviour. Conflating the two is how a suite acquires tests that assert
coincidences, so the eight found so far are written down with the argument for
each:

| mutation | why it cannot change the answer |
|---|---|
| `scoring.capped()` `>` → `>=` | the boundary value maps to the same result on both sides |
| the card-reap deadline `>` → `>=` | same, at a one-second granularity nothing observes |
| `arc_proxy._allowed` `p.match` → `p.search` | every `ALLOW` pattern is `^…$` and no pattern is `MULTILINE`, so `^` matches only at position 0. The anchoring is now asserted directly, so the equivalence is held rather than assumed. |
| `clean_rollouts._outstanding` dropping the `GAMES.index` tie-break | the generator yields in `GAMES` order and `sorted` is guaranteed stable, so equal attempt counts already retain it |
| `grids.collapse` dropping `.copy()` on `arr[:, keep_cols]` | numpy *advanced* indexing always returns a copy, so the call is belt-and-braces; verified with `np.shares_memory` and `.base is None`. Note this does **not** extend to the sibling `arr[::k, ::k]` in `logical`, which is basic slicing and does alias -- the two look alike and behave oppositely. |
| `scoring.score_run`'s `if not candidates` branch | `plays()` always yields at least one list — its `starts` begins with a literal `0` — so an empty ledger arrives as `[[]]`, one empty play, and is scored by the live path. The branch is a defensive fallback that cannot fire today. Its old comment claimed it was what handled the empty ledger, which was false and would have misled anyone relaxing `plays()`; the comment now says what is true. |
| `scoring.card_disagreement`'s `else 0` when the play slice is empty | Both paths return early before reaching it: with a known attempt boundary an empty slice gives `0 < plays` and returns, and without one the `len(done) < plays` check has already returned. **This became equivalent only when the non-positive `playthroughs` guard landed** — a negative count was previously usable as a slice length and could empty the slice. Verified exhaustively over every `(done, playthroughs, boundary)` shape up to length 4: zero combinations reach it. |
| `grids.cell_boundaries` `r < height` → `r <= height` | a boundary is `j + 1` for `j` at most `height - 2`, so none can reach `height`. **This became equivalent only when the ragged-input guard landed**: with mixed shapes a boundary from a taller frame could equal the shorter frame's height, and the two comparisons differed. It survived as a genuine gap before the fix and as an equivalence after it. |

The last two of the original four are the instructive pair: both equivalences are properties of the
*data* the code runs on, not of the code, so each is only safe while something
else holds that property. The anchoring is now a test; the `GAMES` ordering is
stated in the docstring and would have to be broken deliberately.

## Still open, and not a finding

The 25-game sweep onto one shared scorecard is blocked on quota, not
on any of the above. The four sweep-relevant fixes here (3, 7, 15, and the
`attempt_1` hole) all land before it runs.
