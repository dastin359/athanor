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
  `_card_is_lent` all passed the suite. This is on task #35's path — a shim that
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

## Still open, and not a finding

Task #35, the 25-game sweep onto one shared scorecard, is blocked on quota, not
on any of the above. The four sweep-relevant fixes here (3, 7, 15, and the
`attempt_1` hole) all land before it runs.
