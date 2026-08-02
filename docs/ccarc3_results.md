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
