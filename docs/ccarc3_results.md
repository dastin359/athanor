# CCARC3 results

Durable log of ARC-AGI-3 runs. Design and findings: `ccarc3_design.md`. How to
run it: `ccarc3.md`.

## Where this stands against the published Opus 5 result

The ARC-AGI-3 leaderboard entry for Claude Opus 5 (24 July 2026) reports
**30.16%**, at High reasoning effort, on the **public demo set of 25
environments**. The metric is *percentage of the 25 environments completed*.

| | |
|---|---|
| Opus 5, published | **30.16%** = ~7.5 of 25 |
| CCARC3, games won | **6** |
| CCARC3 on the same metric | **24.00%** |
| needed to pass | **8** of 25 = 32.00% |

**So this is not ahead yet, and the denominator is why.** Six games have been
won out of six attempted, which is the encouraging figure, but eighteen
environments have never been played and an unplayed environment scores zero. The
run in flight (`sc25`) would make seven; one more clears the bar.

Three things any comparison drawn from this must carry:

- **Same model, different scaffold.** Both are `claude-opus-5` at effort `high`.
  The difference under test is the harness, which is the point of the project —
  but it also means the comparison says nothing about the model.
- **`cd82` took two runs.** It scored 0/6 on its first attempt and 6/6 on a
  second, against a harness that had changed in between. ARC-AGI-3's own scoring
  is best-of across plays, so that is legitimate inside the benchmark's rules; it
  is still not a single-shot result and should not be quoted as one.
- **The games played were chosen, not sampled.** Batch 1 picked one game per tag
  type; batch 2 deliberately picked `keyboard_click`, including a re-run of the
  only failure. A 6-of-6 rate on a chosen subset is weak evidence about the
  remaining eighteen.

The remaining eighteen are queued cheapest-baseline-first (`scratchpad/batch6.py`).
That ordering is not cosmetic: every environment is worth 1/25 whatever its
length, a run costs turns × roughly $0.10, and baselines span 317 to 1843 — so
cheap games buy the same score for a fraction of the budget.

> **Best single result so far: `sb26-7fbdac44` — 8 of 8 levels in 125 actions
> against a 213 baseline (0.59×), zero deaths, $3.04.**

---

## Run 1 — `ls20-9607627b`, 2026-08-02

**6 of 7 levels cleared in 370 actions. Zero deaths, zero wasted actions, zero
full resets.**

| | |
|---|---|
| model / effort | `claude-opus-5` / `high` |
| budget | 388 actions (`budget_multiple` 0.5 of a 776 baseline) |
| cost | **$13.15**, 116 turns, 41.6 min |
| rule book | 42 mechanics, 13 refutations |
| exit | clean, voluntary — the cap was never hit |

### Per level

Actions are trace-derived: **every** action spent while on that level, including
abandoned attempts.

| level | actions | baseline | ratio |
|---|---|---|---|
| 0 | 17 | 22 | 77% |
| 1 | 59 | 123 | 48% |
| 2 | 67 | 73 | 92% |
| 3 | 49 | 84 | 58% |
| 4 | 83 | 96 | 86% |
| 5 | 94 | 192 | 49% |
| 6 | 1 | 186 | not attempted |
| **0–5** | **369** | **590** | **63%** |

The baseline is a playthrough by someone who already knows the rules. This
solver was discovering them and still finished six levels at 63% of it.

### It stopped on purpose, not on the cap

18 actions remained. The solver's own account:

> *"Level 6 was reached with 18 actions left against a baseline of 186, so it
> wasn't attemptable."*

That is the §2.6 control law working as designed: the published per-level
baseline told it the remainder could not finish the level, so it declined to
spend it. A solver without that number has no way to distinguish "this level is
long" from "I have misunderstood it", and would have burned the remainder.

**The budget was the only binding constraint.** At its demonstrated 63% pace the
full seven levels cost roughly 486 actions against a 776 baseline — comfortably
inside a `1.0x` budget. The halved cap was chosen deliberately to bound a first
run; it is what stood between this and a win, not the solver.

### What it worked out about the game

Reverse-engineered from nothing but frames, with no external information:

- A 12×13 logical board rendered at 5 px per cell.
- Win condition: step into a framed panel with the HUD's **current** 3×3 pattern
  matching the panel's **target** pattern, in both shape *and* colour.
- Four item transforms: a white plus rotates the icon 90° clockwise; a second
  white item applies an arbitrary but deterministic map on shapes; a four-colour
  pinwheel advances colour one step (blue→green→red→orange→blue); yellow rings
  refuel.
- A colour-1 bar on a **wall** edge marks the adjacent floor cell as a launcher —
  entering it slides the cursor away from that wall until blocked, in a single
  action, stopping one cell short of a goal panel.
- Items **patrol**. Walking onto an item's starting cell twice did nothing
  because the item had moved; interception means moving into the cell it is
  moving *into*.

### A counting discrepancy worth keeping

The solver's own summary reported level 0 as 18 actions and level 4 as 62; the
trace says 17 and 83. Level 4 is the real gap.

The scorecard records `resets: [2]` against zero deaths, so the solver twice
reset a level deliberately to re-approach it. Its table counts the actions of the
**winning route**; the trace counts **everything spent on that level**. Both are
meaningful, and they answer different questions — but only the trace number is
the one a budget is spent in, which is why `collect_outcome` reads the ledger and
never the solver's account.

### Harness behaviour

Everything the harness is for either fired correctly or was never needed:

- The level gate held at all six boundaries and was acknowledged each time.
- Zero wasted actions — the solver never acted while dead.
- Zero full resets — the RESET-after-advance trap never triggered.
- Two deliberate level resets, which the doctrine now describes as a legitimate
  retry rather than only a death recovery.
- Ledger and server agree exactly on per-level action counts.

---

## Probe — is the `ls20` bar lethal? 2026-08-02

Scripted, no model involved: open a scorecard, wander for 120 actions, watch
row 61.

**It is not lethal.** The bar holds 42 yellow cells, drains exactly one per
action, reads 0 at actions 42 and 85, and **refills to 42 automatically** — with
zero cells changing outside the bar rows at the refill, so nothing was collected
to cause it. A clean 43-action cycle, twice, no `GAME_OVER` in 120 actions of
aimless movement.

This corrects the doctrine, which said a depleting per-level resource means
*"exhausting it kills you"*. That came from an SDK prompt hint
(`_llm.py:580`, "you GAME_OVER if you run out") which belongs to some other
game, generalised further than the evidence supported. On `ls20` the display is
a cycle, and a solver told it was lethal would rush, skip experiments and waste
the exploration budget the rest of the doctrine tells it to spend.

It also explains the zero deaths across run 1's 370 actions: on this game there
was nothing to die of by wandering.

The doctrine now says find the display, then find out what it does — letting it
run out once, early in a level, is a cheap experiment whose answer changes how
the whole game is played.

Still open: whether a post-`GAME_OVER` RESET opens a new scorecard play row.
Neither run has died, and on `ls20` dying by exhaustion appears impossible, so
this needs a different game.

---

## Probe — what a death actually costs, 2026-08-02

Scripted, no model. Random legal actions on four games until something died.

| game | tag | actions | outcome |
|---|---|---|---|
| `cd82-fb555c5d` | keyboard_click | 71 | no death |
| `ft09-0d8bbf25` | — | 71 | no death |
| `r11l-495a7899` | click | 38 | **GAME_OVER**, on level 1 |
| `sb26-7fbdac44` | keyboard_click | 71 | no death |

`r11l` answers the last open question in the design note. Across the death and
the RESET that followed: `total_plays` stayed **1**, `levels_completed` stayed
**[1]**, and `full_reset` was **not set**.

- **A post-`GAME_OVER` RESET does not open a new play row.** A play belongs to a
  scorecard, not to a life, so `Card.scores` and `Card.actions` are per-play and
  dying inside one adds no entry.
- **It is a level reset** — §2.3 replicated on a second game and a different tag.
- **The completed level survived the death.** This is the direct confirmation of
  the doctrine's central claim: dying costs no progress, so it is a legitimate
  experiment.

Deaths are reachable on some games and evidently not on others. `ls20` cycles
its bar rather than killing; three of four probed games survived 71 random
actions. So "avoid GAME_OVER" is not merely wrong as a terminal goal — on
several games it is not even an available failure mode by wandering.

Residual, one sample only: the scorecard's `states` still read `['GAME_OVER']`
after the RESET. Do not read `states` as the live state.

---

## Run 2 — `ls20-9607627b` **WON**, 2026-08-02

**All 7 levels cleared in 489 actions against a 776 baseline — 63%. Zero
deaths, zero wasted actions, zero full resets.**

| level | actions | baseline | ratio |
|---|---|---|---|
| 0 | 17 | 22 | 77% |
| 1 | 59 | 123 | 48% |
| 2 | 67 | 73 | 92% |
| 3 | 49 | 84 | 58% |
| 4 | 83 | 96 | 86% |
| 5 | 94 | 192 | 49% |
| **6** | **120** | **186** | **65%** |
| **total** | **489** | **776** | **63%** |

66 mechanics and 16 refutations recorded. Model `claude-opus-5`, effort `high`.

### The rate was stable, not lucky

Run 1 reached 63% of baseline over levels 0-5 and stopped without attempting
level 6. Level 6 — the longest, and the one no run had touched — came in at 65%.
The prediction made when run 1 stopped was "roughly 486 actions for the full
game at the demonstrated pace". It took **489**.

### An accidental reproducibility check

The resume was supposed to continue from level 6 and instead replayed the game
from level 0 (see the bug below). That accident is the most useful control this
project has produced: **levels 0-5 took 369 actions in run 1 and 369 in the
replay**, in independent sessions three hours apart. Identical, action for
action.

### The resume bug — diagnosed

`result.json` reported `actions_used: 860` because the ledger spans both
playthroughs. The honest figure for a complete game is **490** — the full reset
plus the 489 spent across the seven levels; the other 370 were the resume
re-walking ground already covered.

That correction no longer has to be made by hand. `collect_outcome` now splits
the trace at the last full reset and reports `playthroughs`,
`actions_final_playthrough` and `levels_reached_final_playthrough` alongside the
total, and the batch summary rates against the final playthrough while showing
the total it cost. The restarting RESET counts in the playthrough it starts,
because it was billed — which is why this reads 490 where the per-level table
below sums to 489. The reset belongs to no level.

The cause is legible in two consecutive lines of the trace:

| i | level | action | full_reset |
|---|---|---|---|
| 369 | 5 → **6** | ACTION2 | false |
| 370 | 6 → **0** | **RESET** | **false** |

Run 1 ended one action after clearing level 5. Its final action advanced the
level, which is the single state in which RESET performs a **full game reset**
rather than a level reset (§2). The client refuses exactly that call. The
refusal never fired.

**`_last_advanced`, the flag that arms it, was not persisted.** It lived only in
memory, and a CC solver takes every action in a new process — so it defaulted to
`False` the moment the resume started, and the guard stood down.

Three things had to line up, and all three did:

1. The resume prompt *did* tell the solver "Do not RESET to 'start clean'; that
   discards real progress." It reset anyway, as its first action. This is the
   project's most-repeated lesson arriving again: **doctrine that stays prose
   gets ignored, and the version that survives is a function.**
2. The refusal — the function version — was unarmed across the process
   boundary that the whole design depends on.
3. **The server reported `full_reset: false` on a transition that took the game
   from level 6 to level 0.** So nothing downstream noticed either: the client's
   counter read zero, and `board_replaced` — documented as *the check a spatial
   rule wants* — returned `False` on the largest board replacement in an
   860-action trace.

Fixed at all three layers. `_last_advanced` is saved and restored with the rest
of the state; a level that goes *down* is treated as a full reset by both the
client and the ledger regardless of what the server claims; and the per-level
action counter restarts on one. The server's flag is now a hint, not the fact.

The claim in an earlier draft that the resume opened "a fresh scorecard" was
wrong — the scorecard was the same one. The game was reset inside it.

Separately, `run_game` had opened `stream.jsonl` with `"w"`, so starting the
resume truncated the record of the handoff. A previous stream is now renamed
aside, and `run_game` writes `resume_state.json` before the solver starts.

---

## Batch 1 — four games across every tag type, 2026-08-02

| game | tag | levels | actions | baseline | rate | deaths |
|---|---|---|---|---|---|---|
| `ls20-9607627b` | keyboard | **7/7 WON** | 489 | 776 | 63% | 0 |
| `ft09-0d8bbf25` | untagged | **6/6 WON** | 75 | 208 | **36%** | 0 |
| `r11l-495a7899` | click | **6/6 WON** | 82 | 233 | **35%** | 0 |
| `tr87-cd924810` | keyboard | **6/6 WON** | 357 | 414 | 86% | 1 |
| `cd82-fb555c5d` | keyboard_click | **0/6** | 337 | 171 | — | 1 |

**Four wins from five**, spanning 35-86% of a baseline set by someone who already
knew the rules. One total failure that never left level 0, burning 6.1x that
level's 55-action baseline.

**The variance between games dwarfs everything else measured on this project.**
A single-game result — which is all `ls20` was — is correspondingly weak
evidence, and the 63% figure quoted from it now looks like the *worst* of the
three wins rather than a representative number.

### What the failure looks like from the trace

| game | action types used | actions that changed nothing |
|---|---|---|
| `cd82` | **7** | **20%** (67 of 336) |
| `ls20` | 5 | 0% |
| `ft09` | **1** | 0% |
| `r11l` | **1** | 0% |

**Every win wasted zero actions. The failure wasted one in five**, and clicks
were the worst of it — 46 of 157 `ACTION6`s left the board untouched.

**Hypothesis, n=1 on the key cell: a wide action space is what breaks it.**
`cd82` is the only game here where keyboard *and* click are both available, and
the solver spread effort across all seven types rather than establishing which
modality mattered. The two fastest wins used exactly **one** action type each.

This is a testable prediction rather than a conclusion, and it is an
uncomfortable one: **13 of the 25 public games are `keyboard_click`**, so if it
holds, the harness is systematically weak on more than half the set. The next
batch should be drawn from `keyboard_click` games specifically, to confirm or
kill it.

> **Killed.** `cd82` was re-run in batch 2 and won 6/6 in 121 actions while
> using six action types. A wide action space is evidently survivable on the
> very game that produced the hypothesis. What remains true is the *correlation*
> between wasted actions and failure — see batch 2 — not the explanation offered
> for it here.

### What it argues the harness should do

The doctrine already says to trust `available_actions` over the tags. That is
necessary and evidently not sufficient: **available is not the same as
effective**. Nothing establishes which of the available actions actually do
anything, and a solver facing seven of them has no cheap way to find out.

Roughly seven actions spent probing each available action once, against a
possible saving of hundreds, is the trade. This is the same shape as every other
finding here — doctrine that stays prose gets ignored, and the version that
survives is a function.

### What a level costs when you understand it — 26 attempts, five games

Computed from the traces, final playthrough only, after the full-reset fix made
that split possible.

| | |
|---|---|
| levels cleared at **≤ 0.92×** their baseline | **24 of 25** |
| median cleared level | **0.52×** |
| attempts above 1.0× | 2 — `tr87` L3 at 3.44× (cleared), `cd82` L0 at 6.13× (never cleared) |

**A solver that understands a level beats the published baseline**, usually by
about half. That figure is a playthrough by someone who already knows the rules,
and it still contains their hesitation; a solver executing a verified plan does
not hesitate. So the baseline is not a par score to aim at — it is a ceiling that
normal play sits well under, and *reaching* it is already the anomaly.

**There is no threshold to find here, and that is the finding.** The sample is
empty between 0.92× and 3.44×, so every cutoff in that range separates the data
identically — one false alarm, one catch. This harness had shipped 2.0×, chosen
by eye. It has been lowered to 1.0×, not because 1.0 fits better (nothing fits
better) but because the costs are asymmetric: warning at the bottom of the gap
costs a re-read, warning at the top costs the hundreds of actions in between.

The relative signal is sharper than the absolute one. `tr87` L3 was 3.44× its
baseline but **6.2× the median of the levels the same run had already cleared**.
A game can run above or below baseline throughout — `r11l` never exceeded 0.50×,
`ls20` reached 0.92× — so the run's own history is the better reference.
`arc.level_pace()` reports it.

*Caveat on n.* 25 cleared levels is a decent sample; **one** non-cleared level is
not. The "24 of 25 finish under 1.0×" half is well-sampled and is what the
doctrine now teaches. Whether the warning reliably *catches* failures rests on a
single observation and is not established.

---

## Batch 2 — `cd82` re-run, 2026-08-02 *(batch in progress)*

Drawn from `keyboard_click` games to **test** the wide-action-space hypothesis
rather than confirm it, since 13 of the 25 public games carry that tag and the
hypothesis rested on one game. `cd82` itself was re-run as the only paired
measurement available.

### `cd82-fb555c5d` — **WON 6/6 in 121 actions**, having previously cleared none in 337

| level | actions | baseline | ratio |
|---|---|---|---|
| 0 | 24 | 55 | 0.44× |
| 1 | 8 | 8 | 1.00× |
| 2 | 42 | 41 | 1.02× |
| 3 | 14 | 21 | 0.67× |
| 4 | 16 | 23 | 0.70× |
| 5 | 16 | 23 | 0.70× |
| **total** | **120** | **171** | **0.70×** |

Zero deaths, zero wasted actions, zero full resets. 28 mechanics, 8 refutations.
Level 0 alone had taken **6.13×** its baseline in the losing run; here it took
0.44×.

**This does not establish that the harness changes caused it, and the run's own
telemetry is why.** Between the two runs the harness gained
`effective_actions()`, the pace ratio in `status()`, and the doctrine sections
describing both. Counting what the winning solver actually executed:

| call | times in 83 tool blocks |
|---|---|
| `client.act()` | 91 |
| `arc.render()` | 64 |
| `client.status()` | 44 |
| `arc.png()` | 8 |
| `arc.diff()` | 6 |
| **`arc.effective_actions()`** | **2** |
| `predict`, `Rule`, `verify`, `shortest_path` | 0 |

So the new function was used — twice. That is real but thin, and `status()`
being read 44 times means the pace line was in front of it constantly without
any way to tell what it did. Run-to-run variance on this benchmark is also
uncharacterised: the one estimate available (`ls20` replaying levels 0–5 in
exactly 369 actions twice) came from **replaying a route already known**, which
is the lowest-variance case imaginable and says little about a novel
exploration. A 14× improvement on level 0 is far outside anything variance has
been shown to produce here, but "far outside" is not a measurement.

The run also carried the pre-§2 doctrine — its workspace was built minutes
before that correction landed — so batch 2 is not byte-identical across its
games.

**Sixth consecutive run in which the rule engine, forward model and planner were
never called.** See §9.8a of the design note.

### `sb26-7fbdac44` — **WON 8/8 in 125 actions** against a 213 baseline (0.59×)

Zero deaths, zero wasted actions, zero full resets — and **100% effective**, in
the sense that every one of its 118 in-level actions changed the board. 51
mechanics and 12 refutations recorded, the most of any run so far.

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| ratio | 0.50× | 0.54× | 0.83× | 0.79× | 0.55× | 0.83× | 0.29× | 0.06× |

Eight levels, none above 0.83× — the tightest run on record, and the strongest
single data point for the "cleared levels come in under baseline" finding.

It is also the most *economical*: **35 tool calls** for an eight-level win,
against 83 for `cd82`'s six. Whatever varies between runs, it is not only
actions.

### What a run costs, and what drives it

Measured across four runs, now that `result.json` carries turns and cost:

| run | levels | turns | cost | cost/turn | input tokens | tool output |
|---|---|---|---|---|---|---|
| `cd82` | 6/6 | 84 | $10.17 | $0.121 | 17.8M | 126K chars |
| `sb26` | 8/8 | 36 | $3.04 | $0.084 | 5.3M | 82K chars |
| `ls20` | 7/7 | 80 | $8.51 | $0.106 | 16.5M | 109K chars |
| `r11l` | 6/6 | 90 | $9.01 | $0.100 | 17.4M | 111K chars |

**Cost is turns × roughly $0.10, and almost nothing else.** The spread in
cost-per-turn is 0.084–0.121, and it tracks average context size: context
accumulates within a run, so a longer run pays more per turn as well as paying
for more turns.

**It is not driven by tool output.** Every run's entire tool output is around
100K characters — perhaps 25K tokens across the whole game — against 5–18M input
tokens. Rendered grids are 16–61% of that tool output and therefore a rounding
error in the bill. The context is dominated by the accumulated conversation, not
by what the harness hands back.

Two consequences worth acting on:

- **Budget in turns, not actions.** `cd82` and `sb26` took near-identical action
  counts (121 vs 125) and differed 3.3× in cost, because one needed 84 turns and
  the other 36. The harness caps *actions*, which is the right cap for the
  benchmark and the wrong one for the bill.
- **Do not optimise the renderer for tokens.** An earlier instinct here was that
  1-char-per-cell rendering was the big saving. At 25K tokens of tool output
  against 18M of context, halving it would change a $10 run by pennies.

### The batch's own verdict on the hypothesis it was built to test

Two `keyboard_click` games, two wins, both comfortably under baseline. Combined
with `cd82`'s six action types, the batch-1 proposal that a wide action space is
what breaks the solver is **refuted**, not merely unsupported.

What remains is the observation the hypothesis was invented to explain — the
losing run wasted one action in five — and that is now recorded as a rank order
without a threshold, because a later run won at 93% effective.

### The click space is easier, not harder

Recorded because it contradicts a prediction stated twice while building this.
`ACTION6` covers a 64x64 coordinate space against four directions for keyboard,
so clicks looked like the harder search. `r11l` is the second-most efficient run
in the set and never probed a single unavailable action.

The likely reason is the one §9.7 already identified: a click acts *directly* on
a target the solver has located in the frame, while a keyboard game must **route**
to it. Verification is cheap here and planning is expensive, so the modality that
skips the routing wins.

`r11l` also took **zero deaths on the one game where a scripted random policy
died in 38 actions** — the solver avoided entirely a failure mode that random
play walks straight into.
