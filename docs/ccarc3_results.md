# CCARC3 results

Durable log of ARC-AGI-3 runs. Design and findings: `ccarc3_design.md`. How to
run it: `ccarc3.md`.

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
