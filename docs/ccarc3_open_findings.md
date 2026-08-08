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

## Still open, and not a finding

Task #35, the 25-game sweep onto one shared scorecard, is blocked on quota, not
on any of the above. The four sweep-relevant fixes here (3, 7, 15, and the
`attempt_1` hole) all land before it runs.
