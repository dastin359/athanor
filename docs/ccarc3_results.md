# CCARC3 results

Durable log of ARC-AGI-3 runs. Design and findings: `ccarc3_design.md`. How to
run it: `ccarc3.md`.

## Read this before quoting any number here

**Two caveats govern everything below: the set, and the budget.**

**1. These are public-set scores, and ARC states outright that public-set scores
are not a measure of progress.** From the ARC-AGI-3 technical report:

> "Because it is impossible to ensure that system designers don't use the
> public environments as part of their work, and because the public set is
> materially easier than the private set, **we will never report public set
> scores of any system on the official leaderboard.** The public set is to be
> used strictly as a demonstration of what ARC-AGI-3 is – evaluating on it is
> **emphatically not** a valid measure of progress towards AGI."

They also demonstrate why: *"we are releasing an open-source 'harness' which
scores 100% on all public environments, **using human replay**."* A recorded
route replayed on a known environment scores 100%, so any public-set figure is
uninterpretable without knowing how much prior knowledge went into it.

So everything below is a **harness-development signal** — it tells us whether a
change helps, on a fixed set, against a fixed reference. It is not a benchmark
claim and cannot be turned into one. The comparison against Opus 5's published
public-demo numbers is like-for-like in the sense that both are public-set and
neither is leaderboard-eligible.

**2. Runs up to and including `sc25` were given 40% of the official action
budget. From `s5i5` (2026-08-04) they are not.** ARC's technical report §4.3:
*"we impose an action budget of five times the human-baseline median action
count per level. That is, for a level with a human median of n actions to
completion, the agent is terminated after 5n actions."* That is **per level**,
with no game-wide pool — so an agent that maxed every level would spend
`5 × baseline_total`.

This harness used a per-game cap of **2.0 ×** the baseline total, 40% of that
ceiling. `tn36` was stopped at 634 actions where ARC would have allowed 1585,
having cleared 6 of 7 levels. **Every figure produced before `s5i5` is therefore
understated**, and `tn36`'s 0.449 is the clearest case: it hit our cap, not
ARC's.

The current configuration, settled 2026-08-04:

| | |
|---|---|
| game-wide cap | **5.0 × baseline_total** — ARC's own ceiling |
| per-level 5n cap | **none** (`e05e839`) |
| cap disclosed to the solver | **no** (`c78e87b`) — passed via `CCARC3_MAX_ACTIONS` |

**The 5n rule was briefly enforced here and then removed, on evidence.** It is
not a property of the game: the live API does not apply it — three level-attempts
in this project ran past it, `su15` L7 to 23.6×, every action accepted — and the
report places it in §4.3 *Leaderboards*, justified by *"the computational cost of
evaluating high-reasoning frontier models ... tens of thousands of dollars in API
costs"*. It is the organisers capping their own spend. Harness results go to the
**community leaderboard**, which the report says is self-reported and which *"the
ARC Prize foundation will not verify"*, so nothing imposes it on us.

The cap is also no longer announced. ARC's `FrameResponse` carries no budget
field, and the report designed *away* from a per-environment allowance — *"we
won't ... encourage AI to waste actions on levels because they're still 'under
budget' for a given environment"*. A solver told its total paces against a number
it would not have at test time.

> **This section used to call correcting it "the single highest-value change
> outstanding". Measured across all 19 runs on record, that overstates it.** The
> cap has actually bound in **two** runs — `tn36`'s control at 631/634 (99.5%)
> and `su15` baseline-free at 701/722 (97%) — and those two are the project's
> two worst scores. Every other run finished with room to spare: ten of the
> twelve baseline-free runs used **under 40% of the cap they were given**, and
> the median is 29%.
>
> So raising it is a **tail-risk fix, not a broad uplift**. It would have
> rescued two bad results; it would not have moved twelve good ones by a single
> action. And `tn36` — the case this paragraph was built on — has since been
> won at **1.000 in 220 actions**, 35% of the same cap that stopped its control.
> The environment was never budget-limited; that one run was.
>
> The difference that does matter is **structural, not a magnitude**. ARC
> terminates per *level*, so spending 5n on level 1 leaves level 2's allowance
> untouched. Our cap is a single game-wide pool, so one runaway level can eat
> the budget every later level needed — which is exactly how `su15` lost, with
> 450 actions on two levels and nothing left for the ninth. Raising 2.0× to 5.0×
> would not fix that; only enforcing the per-level rule does. **That rule went
> live on 2026-08-04 (`e57b007`), from `s5i5` onward** — the arm now resolves the
> medians through `arc.baselines_for()` instead of blanking them, so
> `hide_baselines` keeps the solver blind while `level_budget` bites. Everything
> up to and including `sc25` ran with it inert.

### How this harness differs from ARC's own conditions, on every axis at once

The facts are scattered across three sections below and they point in **opposite
directions**, which is easy to miss one at a time. Collected:

| axis | ARC | here | direction |
|---|---|---|---|
| `baseline_actions` | served by the public API, absent from the OpenAPI spec | withheld from the solver four ways | **stricter** |
| per-level 5n termination | applied by ARC when *they* run a model; not by the API | **none** — enforced briefly on 2026-08-04, then removed | matched to the game, not to ARC's cost control |
| total action budget | no game-wide pool; 5n per level → `5 × baseline_total` | **5.0 × baseline_total** from `s5i5`; 2.0 × before | **matched now**, stricter before |
| budget disclosed to the agent | no budget field in `FrameResponse` | **not disclosed** from `s5i5`; told outright before | **matched now**, more generous before |
| score during play | none in `FrameResponse` | completion cap only, derivable from `levels_completed` / `win_levels` | neutral |
| prior knowledge | — | **471-line doctrine** learned from these 25 environments | **far more generous** |

**The last row dominates everything above it.** The doctrine is not general
reasoning advice: it is that a RESET one action after a level advance silently
destroys the game, that the server scores the *best* play so a replay is free,
that deaths are cheap and confusion is expensive, that rules are level-scoped
while mechanics are game-scoped. Every one was learned by losing runs on these
public environments. Only five lines still name a specific game, which
undercounts it — the rules *are* the residue of those games.

ARC makes the same point structurally: they ship a reference harness that scores
**100% using human replay**, to demonstrate that prior knowledge is what a
public-set score measures. This doctrine is a weaker form of the same thing.

So "we withhold the baselines" is true and narrow. **The setup is
information-minimal about human medians and information-rich about how to play
ARC-AGI-3**, and the second is the larger term by far. The two middle rows are
were, until 2026-08-04, simply *wrong in opposite directions* rather than
conservative. Both are now aligned; **the doctrine row is the one that remains,
and it is the largest.**

**And the reason to match ARC here is comparability, not compliance.** The live
API does not enforce 5n — three level-attempts in this project blew past it,
`su15` L7 at 23.6×, every action accepted. Harness-driven results go to ARC's
**community leaderboard**, which the technical report says is self-reported and
which "the ARC Prize foundation will not verify". Nobody would impose the cap on
us. But Opus 5's published 40.68% is an *official*-leaderboard number produced
under it, so running uncapped and comparing against that inflates the margin.

## Where this stands against the published Opus 5 result

The ARC-AGI-3 leaderboard entry for Claude Opus 5 (24 July 2026) reports
**40.68%**, at High reasoning effort, on the **public demo set of 25
environments** — the unweighted mean of its 25 per-environment scores
(1016.9 / 25 = 40.676%).

> An earlier draft of this section said 30.16%, taken from a summary of the
> results page rather than from the per-environment values. The correct figure
> is 40.68%, and the gap it implies is three times larger. Recorded rather than
> quietly corrected: a number lifted from a summariser is not a measurement, and
> this one set the project's target for several hours.

**The metric is not "games finished".** It is *Relative Human Action
Efficiency*, implemented in `athanor.ccarc3.scoring` and checked against the
rubric's published validation example:

```
S_l   = min(1.15, (h_l / a_l) ** 2)     per completed level, 0 if not completed
E_raw = Σ(l · S_l) / Σ(l)               weights are the 1-indexed level numbers
C     = Σ(1..k) / Σ(1..n)               k = sequential levels completed
E     = min(C, E_raw)
T     = mean(E) over the evaluated set
```

The ratio is **squared and then capped**, so twice the human's actions scores
0.25 rather than 0.5; later levels carry more weight; and the completion cap
stops early speed from paying for late failure.

### The score

| environment | levels | E_raw | cap | **E** | limited by |
|---|---|---|---|---|---|
| `lp85-305b61c3` | 8/8 | 1.150 | 1.000 | **1.000** | completion cap — `raw` at the theoretical maximum |
| `ls20-9607627b` | 7/7 | 1.150 | 1.000 | **1.000** | completion cap |
| `vc33-5430563c` | 7/7 | 1.150 | 1.000 | **1.000** | completion cap |
| `su15-1944f8ab` | 9/9 | 1.055 | 1.000 | **1.000** | completion cap |
| `tu93-0768757b` | 9/9 | 1.077 | 1.000 | **1.000** | completion cap — narrowest margin of any win |
| `ft09-0d8bbf25` | 6/6 | 1.150 | 1.000 | **1.000** | completion cap |
| `r11l-495a7899` | 6/6 | 1.150 | 1.000 | **1.000** | completion cap |
| `sb26-7fbdac44` | 8/8 | 1.144 | 1.000 | **1.000** | completion cap |
| `cd82-fb555c5d` | 6/6 | 1.108 | 1.000 | **1.000** | completion cap |
| `sc25-635fd71a` | 6/6 | 1.040 | 1.000 | **1.000** | completion cap |
| `tr87-cd924810` | 6/6 | 0.947 | 1.000 | **0.947** | efficiency, level 3 |
| `tn36-ef4dde99` | **6/7** | 0.449 | 0.750 | **0.449** | efficiency — hit its action cap |

> **`tn36`'s 0.449 is this arm's number, not this harness's best.** The
> baseline-free arm later scored **1.000** on the same environment — 7/7 in 220
> actions against this run's 631, with 65% of the budget unspent. The table is
> left alone deliberately: the two arms are separate experiments and the standing
> rule below is not to merge them into one total. But the sentence *"the harness
> scores 0.449 on `tn36`"* is no longer true as a claim about capability, and the
> opening caveat that this run "hit our cap, not ARC's" is now demonstrated
> rather than argued. See the ablation section.

| | |
|---|---|
| Opus 5, published | **40.68%** |
| CCARC3, 12 environments scored, 13 unplayed scored 0 | **45.58%** |
| mean over environments actually played | 94.97% |
| margin | **+4.91 percentage points — 1.229 environment-units ahead** |

**Ahead as of 2026-08-03, on 12 of 25 environments.** The four wins that closed
the gap were `su15` (9/9, 0.47x), `lp85` (8/8, 0.24x), `vc33` (7/7, 0.51x) and
`tu93` (9/9, 0.53x).

| | CCARC3 | Opus 5 |
|---|---|---|
| environments scoring >=99% | **10** | **5** |
| environments at 0% | 13 (unplayed) | 3 |
| aggregate | **45.58%** | 40.68% |

Opus 5's published distribution is five environments at 100%, then 98.8, 77.8,
58.3, 56.3, 47.6, 47.6, 44.8, 28.6 and a tail down to zero. **It fully clears
half as many environments as this harness does** and earns the rest of its total
on partial progress — levels cleared in games it did not finish.

**Read the margin with three caveats attached.** Thirteen environments are still
unplayed and scored zero, so this is a lead held while touching under half the
set. Those thirteen were not randomly withheld: batches 1-2 chose games to test
hypotheses and batch 3 plays cheapest-baseline-first, so the remaining games are
longer and harder by construction — `wa30` alone carries a 1843-action baseline,
six times `tn36`'s. And every run here used roughly 40% of the official action
budget, which bounds the result from below rather than above.

RHAE gives partial credit for levels cleared under the completion cap, so the
thirteen remaining games do not all have to be *wins* to extend the lead: a run
that clears four of six levels still scores. What cannot be recovered is an
environment never attempted, which is exactly what the thirteen zeros are — and
they are the only reason a 11-from-12 record sits at 45.58% rather than near 95%.

**The margin is one bad stretch wide.** Thirteen unplayed environments are worth
52 percentage points between them; the current lead is 4.91. This is a position,
not a result.

### `tn36-ef4dde99` — the first partial credit, and the first cap hit

**6 of 7 levels, 631 actions against a 634 cap, RHAE 0.449.** The first run in
this project not to win, and the first to be stopped by its own budget rather
than by finishing.

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ratio | 0.53× | **2.57×** | 0.35× | **1.52×** | 0.70× | **5.62×** | 0.47× |
| score | 1.15 | 0.15 | 1.15 | 0.43 | 1.15 | 0.03 | 0.00 |

The pattern is alternating, not degrading: levels 0, 2, 4 and 6 were all cleared
well under baseline while 1, 3 and 5 blew out. Level 5 alone cost 5.62× its
baseline and scores 0.03 — and carries weight 6 of 28, so it does most of the
damage on its own.

**It also cost $39.78** — 155 turns, nearly four times the previous most
expensive run and thirteen times `sb26`, which won eight levels for $3.04. The
cost model holds (turns × context) but the context grew: this run averaged
$0.26 a turn against the $0.084–0.121 of the earlier set. **A struggling run is
not merely slower, it is superlinearly dearer**, because the accumulated
conversation is what gets re-read every turn.

Partial credit is doing real work here: 0.449 of an environment from a game
that was never won, worth +1.8 points on the benchmark. Under a
games-won metric it would have scored zero.

### The replay question, settled by probing the live scorecard

**Can you replay a whole game to learn its route, and does it cost you?** The
rubric does not say how multiple playthroughs combine into `a_l`, and this
section argued both readings before measuring. The measurement:

```
RESET, +2 actions        plays=1  actions=[2]     actions_by_level=[[]]   resets=[0]
RESET mid-level          plays=1  actions=[3]     actions_by_level=[[]]   resets=[1]
RESET with counter at 0  plays=2  actions=[3, 0]  actions_by_level=[[],[]] resets=[1,0]
+1 action                plays=2  actions=[3, 1]  actions_by_level=[[],[]]
```

The API keeps, per game: `total_plays`, `actions` **per play**,
`actions_by_level` **per play** (a list of lists), and `total_actions` as the
sum across plays. A `RESET` while the counter is non-zero is a *level* reset —
`plays` unchanged, `resets` incremented. A `RESET` when the counter is zero,
which is the state immediately after a level advance, **starts a new play**: new
guid, new `actions` row, new `actions_by_level` row.

**So a one-level game with baseline 7, won in 10 and then replayed and won in 7,
records `actions_by_level = [[10], [7]]` and `total_actions = 17`, and `a_l` is
7.** Per-level counts are never summed. The 17 is budget consumed, which is a
different field answering a different question.

Two consequences:

- **`ls20` scores 1.000, not 0.799.** An earlier version of this section
  switched to the summed reading and called it conservative. The argument was
  that `total_actions` accumulates — which conflated a game-level total with a
  per-level denominator. The server keeps both, separately, and the probe shows
  which one `a_l` comes from.
- **Replaying a game is not punished by the score, only by the budget.** A run
  that wins a long game sloppily can be replayed against a known route to raise
  that environment toward the cap, paying only in actions and money.

### The games are deterministic, so a known route replays exactly

The remaining doubt about replaying was whether a recorded action sequence
reproduces. `ls20` has **patrolling items** — objects that move on their own
schedule — and if their phase depended on a fresh seed or wall-clock, a replay
would desync and the route would be worthless.

Replaying the first 40 actions of `ls20`'s winning route on a fresh play:
**40 of 40 frames identical**, cell for cell, and the same level reached at the
same action. The game is deterministic; the patrol phase is a function of the
action count, not of anything external.

That closes the loop on a strategy the scoring permits:

1. explore freely, spending whatever it takes to learn each level's rule;
2. after any level advance, RESET — which starts a **new play**;
3. execute the known-optimal route, which is the play that gets scored.

The exploration play's waste never enters `a_l`. The only cost is
`total_actions`, the budget field.

**It is worth almost nothing to this project today, and the arithmetic is worth
stating so.** An environment scores `min(completion cap, raw)`, and with every
level cleared the cap is 1.0 — so six of the seven wins are already at ceiling
and a replay gains them exactly zero. The whole lever is worth **+0.053 units
(+0.21 points)**, all of it on `tr87`, against **+4.0 points** for one new
environment. A 19:1 loser at current margins. It becomes material only on a
game won badly enough that `raw` falls well below 1.0.

Recorded because it is a property of the benchmark rather than of this harness:
**RHAE's efficiency term is defeatable by any agent that can replay a
deterministic game**, and an efficiency figure obtained that way is not a
first-discovery result. Nothing in this document used it.

### What the rubric says about where the remaining margin is

**Efficiency is already saturated, and this is the finding that matters.** On
six of the seven environments `E_raw` exceeds 1.0 — between 1.04 and the
theoretical maximum of 1.15 — and is then clipped by the completion cap. Every
level is being finished so far inside the human baseline that the rubric stops
paying for it.

So for this harness, RHAE has collapsed into *a count of environments finished*,
and the two levers are worth wildly different amounts:

- **A new environment won: +1.000 units = +4.0 percentage points.**
- **Perfecting `tr87`, the one environment where efficiency binds: +0.053 units
  = +0.21 points.** Its level 3 took 155 actions against a 45 baseline —
  `(45/155)² = 0.08` — and that level carries weight 4 of 21.

A new environment is worth **nineteen times** more than repairing the worst
level in an existing win. That is why `scratchpad/batch6.py` plays the remaining
eighteen cheapest-baseline-first: every environment is worth the same 1/25, a
run costs turns × roughly $0.10, and baselines span 317 to 1843.

**One qualification the corrected Opus 5 distribution adds.** "RHAE has
collapsed into a count of environments finished" is true of *this harness's
results so far*, not of the rubric. Opus 5 earns most of its 40.68% from
environments it never finished — 98.8, 77.8, 58.3, 56.3, 47.6 — because the
completion cap pays for levels cleared, not only for games won. Every run here
has so far been all-or-nothing, and the one that was nothing (`cd82` first
attempt, 0 of 6 levels) scored a true zero. A run that stalls on level 5 of 8
is worth roughly 0.4 of an environment, so **a game that looks unwinnable is
still worth playing to its budget rather than abandoned.**

### Caveats any published comparison must carry

- **Same model, different scaffold.** Both sides are `claude-opus-5` at effort
  `high`. What is under test is the harness; nothing here says anything about
  the model.
- **`cd82` took two runs.** 0/6 first, 6/6 second, against a harness that had
  changed in between. Its score above is the better of the two. ARC-AGI-3 scores
  best-of across plays, so that is legitimate inside the benchmark's own rules,
  but it is not a single-shot result.
- **The games played were chosen, not sampled.** Batch 1 took one game per tag
  type; batch 2 deliberately took `keyboard_click`, including a re-run of the
  only failure. Seven-of-seven on a chosen subset is weak evidence about the
  eighteen that remain — and the eighteen are, on average, *longer* games.
- **`baseline_actions` from the API is assumed to be the rubric's `h_l`** — the
  upper-median action count among humans who completed that level. It is the
  only baseline data available; if the leaderboard used a different figure, every
  number here moves.
- **The primary evidence for six of the seven environments no longer exists.**
  On 2026-08-02 a verification subagent in a code-review workflow wrote test
  fixtures over the live run directories, destroying the traces for `ft09`,
  `r11l`, `tr87`, `cd82`, `sb26` and `sc25`. Only `ls20` survives, because it had
  been archived separately. The per-level figures in this document were computed
  and committed *before* the loss and are what they were, but they can no longer
  be recomputed from source, and nobody else can check them. Treat every number
  above for those six as reported rather than reproducible, and re-run the games
  if the result ever needs to be defended.

> **Best single result so far: `sb26-7fbdac44` — 8 of 8 levels in 125 actions
> against a 213 baseline (0.59×), zero deaths, $3.04.**

---

## Ablation — are the human baselines load-bearing? 2026-08-03 *(in progress)*

> # ⚠ The first twelve runs did not withhold the baselines. Read this first.
>
> **`meta.json` carried the full array into every workspace, and all twelve
> solvers read it.** `build_workspace` writes the whole `GameInfo` to
> `meta.json`; `strip_baselines()` rewrote `session.py`, `CLAUDE.md` and
> `DOCTRINE.md` and never touched it. Measured, not inferred: the exact
> per-level array appears in a tool result in **12 of 12** streams, and `tu93`
> went further and ran
> `bl = json.load(open('meta.json'))['baseline_actions']` followed by
> `arc.score_run(ts, bl)` — it computed its own RHAE from the numbers the arm
> was built to hide.
>
> **The root cause is a check that could only ever pass.** `strip_baselines()`
> ended by scanning for leaks in exactly the three files it had just rewritten.
> A leak check scoped to what you edited confirms your edits; it says nothing
> about the artefact you shipped. It now scans every file in the workspace, and
> that scan is tested against a leak planted in a file the function does not
> edit — the precise class it was blind to.
>
> **So these twelve are not an information ablation. They are a *presentation*
> one:** the baselines were removed from the workspace `CLAUDE.md`, from
> doctrine §6/§6a, and from the pace line in `client.status()`, while remaining
> in a JSON file every solver opened during orientation. That is a real and
> interesting treatment — it asks whether the *scaffolding* around the number
> matters once you have the number — but it is not the question the section
> title asks, and every "withheld" below should be read as "demoted".
>
> **What they do establish, and it was on the wanted list.** Twelve paired
> re-runs under near-identical information are the best evidence yet on
> **run-to-run variance**, which `ccarc3_design.md` still lists as *not
> established*. **Eight of twelve reproduced their control's score exactly**;
> the four that moved were `tn36` +0.551, `tr87` +0.053, `sp80` −0.022 and
> `su15` −0.200. Variance is small in the body and has a long tail — which is
> also the sharpest possible caution against reading `tn36`'s swing as an
> effect.
>
> **The remaining thirteen games will be genuinely baseline-free**, so this arm
> will have two halves that are not comparable to each other. That is the lesser
> evil: continuing with a known leak would produce twenty-five runs of a claim
> nobody could make.
>
> **One channel remains open, and it cannot be closed by sanitising files.** The
> solver holds an API key and imports the package, so `arc.list_games()` returns
> `baseline_actions` for all 25 environments on demand. Checked across every
> baseline-free stream: **no solver has ever called it** — the one apparent hit,
> `tu93`, was the `meta.json` read. So the channel is open and unused, which
> makes "baseline-free" a claim about what the harness *offers*, not what it can
> *enforce*. Closing it would mean removing `list_games` from the solver's
> namespace, which changes the harness rather than the experiment, so it is
> recorded rather than patched. Any future run of this arm should re-check that
> grep before its numbers are quoted.

**Why this exists.** `baseline_actions` is not in ARC's published `/api/games`
schema — the docs list `game_id` and `title`; the live server also returns
`tags` and `baseline_actions`. Every figure above was produced by a solver that
could read that undocumented field, through `session.py`, the `CLAUDE.md` table,
doctrine §6, and the pace ratio in `client.status()` (called 256 times across six
runs). If the semi-private set withholds it, none of those numbers transfer.

**Design.** Paired. Each game re-run with `baseline_actions=()`, the CLAUDE.md
rows removed and §6/§6a cut, against its own recorded result as control. Same
model, same effort, same action cap, everything else identical. Only the
per-level array is withheld; §0 (how scoring works) and the cap stay, because
cutting those would ablate different variables.

| game | control E | no-baseline E | actions | turns |
|---|---|---|---|---|
| `ft09` | 1.000 | 1.000 | 76 → 83 | ×2.17 |
| `sb26` | 1.000 | 1.000 | 125 → 127 | ×1.97 |
| `tr87` | 0.947 | **1.000** | 358 → 194 | ×0.83 |
| `ls20` | 1.000 | 1.000 | 860 → 612 | ×1.65 |
| `lp85` | 1.000 | 1.000 | 94 → 96 | ×1.43 |
| `r11l` | 1.000 | 1.000 | 83 → 96 | ×1.03 |
| `vc33` | 1.000 | 1.000 | 230 → 271 | ×1.11 |
| `cd82` | 1.000 | 1.000 | 121 → 171 | ×1.40 |
| `tu93` | 1.000 | 1.000 | 246 → 270 | ×0.58 |
| `sp80` | 1.000 | **0.978** | 329 → 300 | ×1.14 |
| `su15` | 1.000 | **0.800** | 168 → 701 | ×1.10 |
| `tn36` | 0.449 | **1.000** | 631 → 220 | ×0.57 |
| `sc25` | 1.000 | 1.000 | 204 → 414 | — |
| `s5i5` | 1.000 | 1.000 | 353 → 510 | ×1.31 |

`sc25` is the **first genuinely baseline-free run** — `meta.json` sanitised, the
whole-workspace leak scan clean, `CCARC3_HIDE_BASELINES=1` on the child. Everything
above it in this table had the array in context.

**Twelve pairs. Score 11.778 against 11.396 — the demoted-baseline arm is ahead
by 0.383,** on eleven wins to the controls' eleven. **Eight** exact ties, two
small losses, two gains. (An earlier draft said nine ties; it was eight.)

**And the aggregate rests on one game.** `tn36` alone is +0.551, more than the
whole margin; without it the arm is 0.169 *behind*. Read the sign of the total
as an artifact of a single result until it replicates.

**`sp80` is the first score loss, and it is the case predicted three pairs
earlier.** It won all six levels but two ran over — L1 at 1.69× (score 0.349) and
L5 at 1.30× (0.590) — and `raw` landed at **0.9785**, a fraction below the
completion cap. When `cd82` and `vc33` came in at 1.0296 and 1.0268 this section
recorded that "on a game with fewer levels to spread the weight, one such level
would tip the environment below 1.000". A six-level game, two bad levels, and it
did.

The irony is that `sp80` looked like the arm's strongest result early: it cleared
the opening level in **66** actions where its control needed **126** (3.23×, the
worst opening this project has recorded). It gave the advantage back on L5, 125
against 66.

### `su15` — the arm's first lost game, and the finding is about the arm

`su15` is the first environment where something went badly wrong inside a
baseline-free run, and it exposed that **this arm has never been a
one-variable experiment.**

The arm withholds the medians by setting `GameInfo.baseline_actions=()`. That is
the same array `client.level_budget` derives ARC's official per-level
termination rule from — *"for a level with a human median of n actions, the
agent is terminated after 5n actions"*. With the array empty, `baseline_here` is
None, `level_budget` returns 0, and the rule silently stops existing. Every
workspace in `ablate_nobaseline/` writes `level_budget_multiple=5.0` into its
`session.py` and none of them can enforce it:

```
baseline-free arm : baseline_here = None | level_budget = 0
control arm       : baseline_here = 31   | level_budget = 155
```

So the arm does not measure *"the solver cannot see the baselines"*. It measures
*"the solver cannot see the baselines **and** the benchmark's termination rule
has been switched off"* — and the second half flatters it, because the real
benchmark would have ended these runs earlier, not later.

**What that cost here.** The exploration play cleared five levels comfortably and
then jammed:

| level | baseline | play 1 | ratio | ARC would have stopped it at |
|---|---|---|---|---|
| L6 | 31 | **268** | 8.65× | 155 |
| L7 | 8 | **182** | 22.75× | 40 |

Those two levels spent 450 actions — **62% of the game's entire 722 budget** —
and fixed the play's ceiling at **0.8199**: RHAE is a weighted mean over levels
already finished, so no play from there could have reached 1.000 however
perfectly it finished. Under the official rule the run would have ended on L6 at
155 actions, five levels cleared, **E = 0.333**.

**The solver then did the right thing, and it is the first time in this project
that any solver has.** Across every run on record, `su15` is the only one whose
stream contains a `restart_for_replay()` call — four of them. (`ls20` has a
`full_reset` on its ledger, but that was the accidental RESET trap of §2.3, the
one the client now refuses; it is the opposite of a deliberate replay.) Having
cleared L7 — the exact state where the action
counter is zero and a RESET starts a new play rather than resetting a level — it
called `restart_for_replay()`. That call has existed since the doctrine's §0a
reversal earlier the same day, which established that the server scores the
**best** play and a replay can therefore only raise a score. This is its first
use in the wild, and it came from a solver that could see none of the arithmetic
above.

The replay was extraordinary. Eight levels, every one at the 1.15 per-level cap:

| | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 22 | 42 | 26 | 115 | 36 | 31 | 8 | 40 | 41 |
| play 1 | 16 | 22 | 16 | 18 | 5 | **268** | **182** | — | — |
| replay | 14 | 22 | 16 | 9 | 5 | **11** | **7** | 34 | — |
| ratio | 0.64× | 0.52× | 0.62× | 0.08× | 0.14× | 0.35× | 0.88× | 0.85× | — |

`raw` = 0.920, and it still lost: the completion cap binds at 8/9 = **0.800**,
because the 528 actions the abandoned play had already spent left only 194, and
level 9's baseline is 41. It ran out with 55 actions on that level.

**So the replay was correct and insufficient, and those are separate facts.**
Correct: play 1's ceiling was 0.8199 and the replay realised 0.8000, a wash —
but it was a wash only because the exploration had been so expensive. Had those
two levels cost anything like their baselines, the replay had budget to spare
and wins at 1.000. Insufficient: the cost was already sunk when the decision was
made, and no decision available at that point could recover it.

**The lesson is not about replaying.** It is that nothing stopped the 268- and
182-action levels while they were running. In the control arm the pace line
would have said `268/31 on this level = 8.6x` and `OVER BASELINE` for two
hundred consecutive actions. In this arm it said nothing, and the 5× rule that
exists precisely to end such a level had been disabled by the same edit that
silenced the warning.

**A budget-fraction warning was considered and rejected — the separation is real
and the mechanism is not.** The doctrine's §0a already carries a reserve rule
("do not spend more than half your action cap before you understand the game;
keep one baseline in reserve"), it was **present in `su15`'s workspace** —
`strip_baselines()` cuts §6/§6a, not §0a — and the solver spent 73% of its cap
before replaying. Across all eighteen finished runs the line separates perfectly:

| | runs | won |
|---|---|---|
| stayed at or under half the cap | 16 | **16** |
| crossed it | 2 | **0** |

The two crossings are this project's only two losses. That looked like a warning
worth building, and it is not, because replaying the trace shows it would have
fired uselessly. At exactly half its cap `su15` had cleared **six of nine
levels**, 346 of its 361 actions were accounted for by levels it had *finished*,
and only 15 were sunk in the level it was on. Nothing was wrong. The damage came
afterwards, in the 182 actions L7 then took against an 8-action baseline.

So the threshold is a *consequence* of struggling, not a trigger that precedes
it — a run that has used a lot of actions is a run that is in trouble, which is
true and useless. The only instrument that catches this while it is happening is
the per-level pace ratio, firing at 1.0×, and **that is the instrument the
ablation had disabled.** Recorded here because the 16/18-against-0/2 table is
exactly the kind of clean-looking separation this document has been fooled by
before.

Both halves are now fixed in the client, off by default:

- `hide_baselines` withholds the medians from every solver-facing surface
  through one gated property, while `level_budget` reads the enforced value. ARC
  withholding a number from an agent does not stop ARC applying it.
- `show_score` reports `score_now`, `score_ceiling` and `completion_cap` in
  `status()`, and names `restart_for_replay()` when the ceiling drops below
  1.000. `completion_cap` needs no baselines — it is which levels fell, not how
  fast — so it survives the baseline-free setup.

They are off because ten of this arm's twenty-five games are banked without
them. Switching a variable halfway makes two half-experiments. **The remaining
games should finish as they started, and the faithful configuration — cap
enforced, numbers hidden — is a separate arm.**

**The turn-cost question is closed as unanswerable at this n.** The ten ratios
are 0.58, 0.83, 1.03, 1.11, 1.14, 1.40, 1.43, 1.65, 1.97, 2.17 — median 1.27,
range nearly fourfold, no central tendency worth reporting. This document called
the effect real at n=2, refuted at n=3, real again at n=5, and it is now clear
that each of those readings was noise. Recorded as a caution against the habit,
not as a finding.

**Through ten pairs, withholding the baselines had not cost a single win. `su15`
ends that**, at −0.200, taking the aggregate to −0.169. One loss in eleven is
well inside what this design can call noise — McNemar on a single discordant
pair is p=0.5 — so the finding is not "the baselines are load-bearing". It is
that the one game where a level went badly wrong is the one game the arm lost,
and that the arm had disabled the rule which exists to stop exactly that.

### `s5i5` — the same instruction fires again, and this time buys nothing

`s5i5` is the first run whose budget conditions match ARC's: no per-level cap, a
game ceiling of `5 × baseline_total` (3190), and **the ceiling not disclosed to
the solver**. It won 8/8 in 269 actions, every level under the human median, and
its control needed 353.

Then it replayed, and the replay was worth **exactly zero**:

| | actions | `raw` | `cap` | **E** |
|---|---|---|---|---|
| play 0 | 269 | **1.1500** | 1.000 | 1.0000 |
| play 1 | 241 | **1.1500** | 1.000 | 1.0000 |

Both plays already stood at `raw = 1.15`, the theoretical maximum — the score
could not move. The replay spent **241 actions, 47% of the run**, and $10 of its
$26, for nothing.

**This is the doctrine paragraph added hours earlier doing what it says.** Its
second half reads: *"If you cannot compute `raw`, replay anyway when you have the
budget."* The solver followed it exactly, and could not have known better — which
is the situation the sentence describes.

**And across all three replays on record the instruction is strongly positive.**
Scoring each play separately: `su15` +0.4146, `sc25` +0.1934, `s5i5` +0.0000 —
a mean of **+0.203 per replay**, against a cost of a few hundred actions from a
budget that is now 5× the baseline total. Two of three bought real score. An
earlier draft of this section treated the instruction as mostly wasteful on the
strength of `s5i5` alone; that was one data point read without the other two.

What saves it from being harmful is that it costs money rather than score, and
only because the budget moved to 5×. Under the previous 2.0× cap (1276) this
replay would have been 40% of the allowance — the same shape that lost `su15`.

**The fix is already written two paragraphs above it**, and was not applied to
the decision: *"You also do not need a baseline to know you fumbled. The gap
between what the route turned out to be and what you spent finding it is the
signal."* `s5i5`'s first play had no blown level and no grinding. A solver asked
*"did I actually fumble?"* answers no and saves 241 actions. The unconditional
instruction should be conditioned on that self-assessment — **`sp80` is the case
it is right about, `s5i5` is the case it is wrong about**, and telling them apart
needs judgement the solver already has.

### `sc25` — the post-clear replay instruction fires, first time out

The doctrine gained a paragraph a few hours before this run: once every level is
cleared `cap` is 1.0, so `E` **is** `raw`, and a replay is worth `1.0 − raw` —
take it whenever the budget is there, because without baselines you cannot tell
whether you left anything behind. `sc25` is the first solver to read it.

It won 6/6 at action 283, sat in `state=WIN` with the environment cap already at
1.000, and instead of stopping called `restart_for_replay()` and played the whole
game again:

| | L1 | L2 | L3 | L4 | L5 | L6 | total |
|---|---|---|---|---|---|---|---|
| baseline | 36 | 6 | 32 | 83 | 143 | 50 | 350 |
| first play | 60 | 7 | 29 | 38 | 41 | **108** | 283 |
| **replay** | **13** | 6 | **14** | **25** | **36** | **36** | **130** |
| ratio | 0.36× | 1.00× | 0.44× | 0.30× | 0.25× | 0.72× | |

**`raw` 1.1357, `E` 1.0000**, 414 actions of 700, $19.24. The replay walked the
game in **46% of the first play's actions** and beat the human median on five of
six levels.

**Correction — the replay was worth +0.1934, not nothing.** This section first
said it "changed the behaviour, not the number", on the reasoning that the
control also scored 1.000. That comparison cannot see the effect: the final score
*already contains* the replay's benefit. Scoring the plays separately,
**play 0 stood at E = 0.8066** — L1 at 1.67× and L6 at 2.16× held `raw` to 0.8066
against a `cap` of 1.000 — and the replay lifted it to 1.000.

The run was also never in danger: several ticks of live monitoring reported L6
climbing to 4.8× as though it were failing, when it was in its first play and the
solver understood the position better than the observer did.

What it does establish is that the instruction is **followed**, unprompted, in
exactly the state that cost `sp80` 0.0215 — win, `raw` below the cap, budget
spare, and no way to compute how much was being left. Whether it ever converts a
loss into a win is still untested.

### `tn36` — the arm's largest result, and the one to trust least

`tn36` was the gain arm by construction: at the time the arm was designed its
control was the only environment this project had ever lost, at 0.4487, so it
was the one game where withholding the baselines could plausibly *help*. It did. **E = 1.0000, +0.5513**, in 220
actions against the control's 631 — a third of the actions for more than twice
the score, with 65% of the budget unspent.

| | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|
| baseline | 32 | 72 | 26 | 40 | 30 | 55 | 62 |
| actions | 27 | 38 | 11 | 16 | 20 | **65** | 42 |
| ratio | 0.84× | 0.53× | 0.42× | 0.40× | 0.67× | **1.18×** | 0.68× |

`raw` = 1.0570, so the cap binds and the single over-baseline level costs
nothing. Six of seven levels sat at the 1.15 per-level cap.

**What differed is the level that decided both runs.** The 55-baseline L6 is
where the control died: 309 actions, 5.62×, with 95 of them landing on boards it
had already stood on. It was not flying blind while it did that — its pace line
fired `OVER BASELINE` **26 times** and climbed through 2.0× → 3.5× → 5.3×. The
baseline-free run, with no pace signal at all, cleared the same level in 65.

**Which is why this result is the least trustworthy in the arm, not the most
convincing.** It is a single run against a single control, and "run-to-run
variance is small" is still listed as **not established** in the design note.
The honest reading is that the control's 309-action collapse looks like the tail
of a distribution rather than the mean, and a baseline-free run drawing a normal
sample would beat it regardless of what it could see. `tr87` carries the same
caveat at a tenth the magnitude. **The arm's total is positive because of this
game and would be negative without it — so the sign of the aggregate is a claim
about one run, and should not be reported as a property of the arm until it
replicates.**

The safe statement across twelve pairs: withholding the baselines cost one game
outright and left the other eleven at or above their controls.

**Five cautions, and they matter more than the table.**

*The arm is confounded, and `su15` is where it shows.* See that section above:
blanking `baseline_actions` also disables ARC's 5n per-level termination rule,
so this arm is strictly more permissive than the benchmark it is meant to
predict. The ten wins banked before `su15` are unaffected in *outcome*: the
worst level in any of them is `vc33` L1 at 2.29×, and `su15` is the only run in
the arm with a second play, so no abandoned overrun is hidden behind a scored
count. But the design claim *"only the per-level array is withheld"* was false
for all eleven.

*The `tr87` result is the least attributable, not the most impressive.* There is
no repeat of `tr87` **with** baselines, so a 358 → 194 improvement is equally
consistent with run-to-run variance on a game whose control had one bad level.
"Run-to-run variance is small" is listed as **not established** in the design
note, and this is exactly where that gap bites. The two ties carry more weight
than the apparent win.

*One run was killed and must not be counted.* A session-worker restart
SIGTERMed `sc25` five actions in; `collect_outcome` still wrote a `result.json`
with `won: false` and `exit_code: 143`, and the arm's own log counted it as a
loss — `ablate.log` reads 8/9 from that point and cannot be repaired, being
append-only. The run is quarantined outside the batch directory and will be
re-run. Two lessons: **a `result.json` is not proof a game was played** — check
`exit_code > 128` for a signal — and **`pathlib.Path.glob("*/...")` matches
dot-directories**, so renaming the workspace with a leading dot did *not* hide
it from the snapshotter, which re-inserted the false 0.000 within minutes.

*Mechanism is unanswerable for seven of the thirteen.* `ft09`, `ls20`, `r11l`,
`cd82`, `sb26`, `sc25` and `tr87` had their `stream.jsonl` destroyed with their
traces, so whether those control solvers ever consulted the baseline cannot be
checked. Of the six that survive, engagement ranged from 30% of tool blocks
(`lp85`) to 0.6% (`tn36`, then the only loss). The win/loss comparison is valid
for all thirteen; the *why* is answerable for six.

~~That 0.6% is worth restating now that `tn36` has been played both ways.~~
**Retracted.** This said the winning run "could not consult them at all". It
could: `meta.json` carried the array and its solver read that file, like all
twelve. Both runs of `tn36` had the baselines available, so the comparison
measures nothing about consulting them and the observation collapses.

**Do not merge these into the headline figure.** Both arms score against the
same 25-environment denominator, so summing the ledger without filtering on
`batch` would report a meaningless combined total. `snapshot_results.py` tags
every row; readers must filter.

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
