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
> **One channel remained open, was recorded rather than patched, and a solver
> walked straight into it the next morning.** The reasoning was: the solver holds
> an API key and imports the package, so `arc.list_games()` returns
> `baseline_actions` on demand; but a grep across every stream showed **no solver
> had ever called it**; so the channel was open and unused, and closing it would
> change the harness rather than the experiment. Recorded, not patched.
>
> That was wrong, and the specific error is worth naming: **"no solver has done
> it yet" was treated as evidence about the channel, when it was only evidence
> about the eleven samples drawn so far.** On 2026-08-04 `cd82` ran `dir(arc)` on
> its first orientation turn — the obvious move in an unfamiliar package — and the
> listing handed it `'actions_per_level', 'as_grid', 'baselines_for',
> 'block_size'`. It went looking for nothing. The name was simply there, and by
> then `baselines_for` had been added to the package as a *harness* helper that
> ignored the withholding flag by design, so the array was one call away.
>
> A channel nobody has used is not a channel nobody will use, and adding a
> convenience to the shared namespace widened this one while the note above said
> it was being watched. The flag is now a real boundary — `baselines_for` raises
> under it, `list_games` has no bypass argument, and the name is gone from
> `dir(arc)` — so "baseline-free" is now a claim about what the harness
> *enforces* for incidental access, while remaining unenforceable against a
> hand-rolled HTTP request. Re-check the grep anyway before quoting numbers.

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
| `bp35` | — | **0.4667** | — → 307 | **timed out** |

`bp35` is the first **untouched** environment (no control) and the first run lost
to the **wall clock** rather than to the game. Six of nine levels, `raw` 0.4970
*above* its cap of 0.4667, every cleared level near the human median — and
**307 of 3255 actions used, 9%**. It ran out of the 2-hour ceiling at 23 s per
action.

That pace is ordinary: median across finished runs is 10.5 s/action, and `bp35`
sits with `r11l` (23.8) and `lp85` (22.0). **Raising the action budget to 5×
made actions cheap, so the solver began spending minutes on each — and the clock
became the binding constraint instead.** Fixing one cap exposed the next.

The timeout now scales as `12 s × baseline_total` on a **4 h floor** (operator
instruction, raised from 2 h). **`bp35`'s 0.4667 measures this harness, not the
environment** — the same defect as `tn36`'s old 0.449 — and it is being re-run.

~~`sc25` is the **first genuinely baseline-free run**.~~ **It is not, and the
correction is smaller than it looks but worth having exactly right.** `sc25`'s
`meta.json` had no `baseline_actions` — that fix had landed — but it still
carried `action_budget`, which at the time was a fixed multiple of
`baseline_total` and therefore discloses that total to anyone who divides. The
first *fully* clean workspace is **`s5i5`**, relaunched from scratch on
instruction a day later, followed by `bp35` and everything after.

Ordered by when the workspace was actually built, rather than by queue position,
the boundary is sharp:

| built | run | `meta.json` carried |
|---|---|---|
| Aug 3, 08:49 → 17:41 | ft09, sb26, tr87, ls20, lp85, r11l, vc33, cd82, tu93, sp80, **su15**, **tn36** | the per-level array |
| Aug 4, 04:07 | sc25 | `action_budget` only |
| Aug 4, 05:57 → | s5i5, bp35, … | nothing |

Two things follow. The "first twelve" above is a **chronological** claim, not a
queue-order one — `su15` and `tn36` sit at queue positions 13 and 14 and are
inside it, `sc25` sits at position 9 and is outside. And all thirteen are being
re-run: the twelve to `ablate_leaked/`, `bp35` to `ablate_timeout/`, ledger rows
re-tagged so a re-run cannot land beside the run it replaces under the same batch
name. `s5i5`'s 1.000 stands and is not re-run — a perfect score cannot improve.

### The first clean re-run: `cd82` — same score, half the actions

`cd82` is the first game re-run under the closed boundary, and it is the first
observation of what withholding actually costs. **It costs nothing, and the
mechanism is not the one the section was built to look for.**

| | E | `raw` | per-level actions | total | plays | turns | cost | wall |
|---|---|---|---|---|---|---|---|---|
| leaked | 1.0000 | 1.0296 | 38, 6, 74, 16, 16, 20 | 170 | 1 | 118 | $16.52 | 54 min |
| **clean** | **1.0000** | **1.1500** | **5, 6, 16, 14, 13, 16** | **70** | 2 | 98 | $13.01 | 40 min |
| baseline | — | — | 55, 8, 41, 21, 23, 23 | 171 | — | — | — | — |

Identical scores, because both are pinned at the completion cap of 1.0. The
difference is entirely in the headroom underneath it: the leaked run finished at
`raw` 1.0296, a hair over the cap, while the clean run finished at **1.1500 —
the theoretical maximum**, every one of the six levels clamped. It used **70
actions to the leaked run's 170**, against a human total of 171.

**The cause looks like the withheld score, not the withheld baselines.** The
clean run played twice: 108 actions working the game out, then a replay that
took all six levels in 71. The leaked run played once, in 171. Doctrine §0a tells
a solver to replay after clearing, and adds *"if you cannot compute `raw`, replay
anyway when you have the budget"* — which is exactly the branch a baseline-free
run is forced down, because `score_now` returns `None` without baselines. The
leaked run *could* compute `raw`, saw 1.0296 clear the cap, and stopped. Knowing
the score told it the replay was worthless; not knowing made it replay, and the
replay cost 100 actions. **What it bought is corrected below: nothing.**

**Why that matters even though E is unchanged.** It is invisible here only
because `cd82` was won outright, where the cap binds and swallows the gain. On
any game that does *not* clear every level the cap sits below 1.0, `raw` becomes
the binding term, and 1.15-vs-1.03 is the whole score. This is a mechanism that
pays exactly where the current results are weakest — and it is an argument
*against* `show_score`, which the operator has separately established is absent
from the live API at test time.

One observation, one game, and the cheapest environment in the set. It wants the
other eleven before it is a finding.

### `ft09` declines to reproduce it, and that is the useful part

| | E | `raw` | per-level actions | total | plays |
|---|---|---|---|---|---|
| leaked | 1.0000 | 1.1500 | 6, 7, 14, 21, 21, 13 | 82 | 1 |
| **clean** | 1.0000 | 1.1500 | 4, 7, 14, 16, 21, 13 | 75 | 1 |
| baseline | — | — | 43, 12, 23, 28, 65, 37 | 208 | — |

Both arms at the maximum `raw`, one playthrough each, seven actions apart across
six levels. **No replay, and nothing to explain.**

That is not a failed prediction so much as the boundary of the `cd82` one. The
replay in `cd82` was thought to be worth 0.12 of `raw` because its *first* play
looked clumsy —
170 actions, `raw` 1.0296. `ft09`'s first play was already at the clamp on every
level, so there was no headroom for a second play to recover and the withheld
score changed nothing. The mechanism is **"not knowing the score rescues a bad
first play"**, not "not knowing the score helps." Which games it pays on is
therefore a question about first-play variance, and two games say nothing about
that.

### `sb26` replicates its leaked run almost exactly

| | E | `raw` | per-level actions | total |
|---|---|---|---|---|
| leaked | 1.0000 | 1.1436 | 11, 15, 15, 15, 17, 19, 17, 17 | 126 |
| **clean** | 1.0000 | 1.1436 | **10**, 15, 15, 15, 17, 19, 17, 17 | 125 |
| baseline | — | — | 18, 28, 18, 19, 31, 23, 58, 18 | 213 |

**Seven of eight levels identical to the action**, the eighth off by one, `raw`
equal to four decimal places. This is the tightest paired observation in the
project and it says something the aggregate cannot: on a game the solver
understands, the run is close to deterministic. Whatever variance produced
`tn36`'s old +0.551 swing is not a property of every environment.

`sb26` also shows where `raw` leaks below the clamp: level 8 has `h`=18 against
`a`=17, a ratio of 1.06 that squares to 1.12 and so misses the 1.15 ceiling.
Seven levels at the clamp and one at 1.1218 is what 1.1436 is made of. A level
where the human is already efficient cannot be beaten enough to matter — the
clamp only pays where the baseline is generous.

### The replay is not free, and `r11l` is where that shows

| game | E | `raw` | actions clean → leaked | plays | cost clean → leaked |
|---|---|---|---|---|---|
| cd82 | 1.0000 | 1.1500 | **70** → 170 | 2 | $13.01 → $16.52 |
| ft09 | 1.0000 | 1.1500 | 75 → 82 | 1 | ~$13.78 → $6.17 |
| sb26 | 1.0000 | 1.1436 | 125 → 126 | 1 | $4.72 → $4.85 |
| r11l | 1.0000 | 1.1500 | **69** → 95 | 2 | **$18.77 → $12.02** |
| s5i5 | 1.0000 | 1.1500 | 240 | 2 | — |

`r11l` replayed, cut its actions from 95 to 69 — and gained **nothing**, because
its leaked run had already reached `raw` 1.1500 in a single play. The replay cost
**56% more money and 55% more wall clock** ($18.77 against $12.02, 59 min against
38) to move a number that was already at its ceiling.

That sharpens the `cd82` reading rather than softening it. Withholding the score
does not make the solver better; **it makes the solver replay**, because §0a's
"replay anyway when you cannot compute `raw`" is the only branch available. That
replay is a *bet*: it pays when the first play was clumsy (`sc25`, +0.0837 E)
and is pure cost when it was not (`r11l`, `s5i5`). Three of five clean runs
replayed; exactly one of those three was worth it.

Which means the honest summary so far is **not** "hiding the score helps." It is:
hiding the score buys a replay on every game, at roughly 1.5× cost, and the
replay is only redeemable where the completion cap does not already swallow it.
Every game here was won outright, so the cap swallowed all of it and E is 1.0000
either way. The bet only becomes visible on a game that does *not* clear — and
none of the five did that.

### `tn36` — the first clean loss, and it is variance, not withholding

**E = 0.5357 against the leaked run's 1.0000.** Five of seven levels, `raw`
0.5904 above a completion cap of 0.5357, so the cap is what binds — the first
game in the clean arm where it does.

| level | h | clean `a` | leaked `a` |
|---|---|---|---|
| 0 | 32 | 21 | 27 |
| 1 | 72 | **81** (S 0.79) | 38 |
| 2 | 26 | 11 | 11 |
| 3 | 40 | 16 | 16 |
| 4 | 30 | 20 | 20 |
| 5 | 55 | **gave up after 139** | 65 |
| 6 | 62 | — | 42 |

Levels 2, 3 and 4 are identical to the action. The two runs diverge on exactly
the two levels either found hard.

**It quit voluntarily, with 82% of its actions and 63% of its wall clock
unspent** — 289 of a 1585 cap, 89 minutes of 240 — signing off with *"I've run
out of viable hypotheses for level 5."* `exit_code` 0, `subtype: success`, not
killed, not timed out. Idea exhaustion, not resource exhaustion.

**And it was not flying blind when it did.** An initial reading of this run
claimed the completion cap had never been surfaced; that was a grep for the
Python identifier rather than the rendered string, and it was wrong. The solver
saw `[cap …]` on 28 status lines, ending on `[cap 0.536 = 5/7 levels]`, together
with the harness telling it `7/139 actions on this level changed nothing` and
`21/139 returned the board to a state already seen`. It knew the score, it knew
it was going in circles, and it stopped anyway.

**The most likely reading is variance, and `tn36` is the worst possible game to
read anything else into.** Its three scored runs are **0.449, 1.0000, 0.5357** —
this is the environment whose +0.551 swing the section above already flags as the
sharpest caution against treating a `tn36` movement as an effect. A third
data point at a third value is confirmation of that caution, not a finding about
baselines. The clean run was also worse on level 1 (81 actions against 38), which
withholding does not explain: no per-level baseline was visible in *either* arm's
level-1 decisions, since the leaked arm's numbers sat in `meta.json` rather than
in the pace line.

What it does establish is the first case where the replay bet *could* have paid
and was never placed. §0a's instruction fires after clearing every level; a run
that gives up at 5/7 never reaches it. That is the gap worth thinking about — not
the missing score, which was there.

`sc25` follows and lands where `sb26` did — E 1.0000 either way, 124 clean
actions against 130 leaked, `raw` 1.1500 against 1.1357, both arms replaying
after one death apiece. Six actions and a hundredth of `raw` between them, which
is the ordinary case: **five of seven clean runs are within noise of their
leaked pair, one is better (`cd82`), one is worse (`tn36`).**

### `su15` wins 9/9, which its leaked pair did not

**E 1.0000 against 0.8000**, and the shape of the win is the interesting part:

| level | h | clean | leaked |
|---|---|---|---|
| 0 | 22 | 7 | 14 |
| 1 | 42 | 8 | 22 |
| 2 | 26 | 13 | 16 |
| 3 | 115 | 9 | 9 |
| 4 | 36 | 6 | 5 |
| 5 | 31 | 10 | 11 |
| 6 | 8 | 7 | 7 |
| 7 | 40 | **10** | **34** |
| 8 | 41 | **12** | **never reached** |

The clean run cleared all nine in **250 actions to the leaked run's 320-plus for
eight**, and its final playthrough took the whole game in 83. Level 7 is where
the two part: 10 actions against 34, and the leaked run arrived at level 8 with
its budget and patience spent.

**Paired ledger so far — seven pairs, five exact ties, one win, one loss:**

| | clean | leaked | Δ |
|---|---|---|---|
| cd82, ft09, r11l, sb26, sc25 | 1.0000 | 1.0000 | 0.0000 |
| **su15** | **1.0000** | 0.8000 | **+0.2000** |
| **tn36** | 0.5357 | 1.0000 | **−0.4643** |

Net −0.2643 across seven pairs, on two movements in opposite directions. Neither
is evidence about baselines: `tn36` has now scored 0.449, 1.0000 and 0.5357 on
three runs, and `su15` was the arm's other known high-variance environment — it
is the game whose earlier run reproduced the ceiling bug at 23.6× on level 7,
which is precisely the level that separates these two runs. **The two games that
moved are the two games already known to move.** Five of seven are exact ties.

`lp85` is another tie — E 1.0000 both ways, `raw` 1.1500 both ways, 79 clean
actions against 95 leaked. Six exact ties in eight pairs.

### The completion cap binds in 28 of 30 runs, so efficiency is almost never worth anything

Prompted by a question about `tn36`: if that run had restarted and replayed
levels 0–4 optimally, would its score have moved? **No — and not by a little.**

```
clean run as played                          raw=0.5904 cap=0.5357  E=0.5357
levels 0-4 at their best-ever cost           raw=0.6161 cap=0.5357  E=0.5357
levels 0-4 in ONE action each (impossible)   raw=0.6161 cap=0.5357  E=0.5357
```

The arithmetic is forced. `S_l` is clamped at 1.15, so with `k` of `n` levels
cleared, `raw ≤ 1.15 × C` **always** — and `E = min(C, raw)`. Perfect play on the
levels you did clear cannot lift `E` above the cap those clearings already set.
Clearing level 5 would have been worth **+0.208**, level 6 another **+0.256**, at
the very action costs the leaked run actually paid.

**This corrects a claim made earlier in this section.** After `cd82` it said the
replay's `raw` gain was invisible only because that game was won outright, and
that "on any game that does *not* clear every level the cap sits below 1.0, `raw`
becomes the binding term." That is wrong. Since `raw ≤ 1.15 × C`, `raw` exceeds
`C` whenever the cleared levels average at or better than the human median —
**how many levels you cleared has nothing to do with which term binds.**

Measured across every scored run in the project:

| binding term | runs |
|---|---|
| completion cap — only *clearing* helps | **28** |
| `raw` — efficiency helps | **2** |

The two exceptions are `runs3/tn36` (0.4487 against a cap of 0.7500 — the run
that put 309 actions into level 5 and hit the old 2× action budget) and
`ablate_leaked/sp80` (0.9785 against 1.0000 — two levels run over the human
median). Both are cases of going *badly* over baseline, not of ordinary
inefficiency. **In all nine clean runs under the current config, the cap binds.**

Three consequences, none of them small:

1. **The efficiency apparatus is scoring-irrelevant almost all the time.** Pace
   warnings, the action budget, per-level ratios — across 104 cleared levels the
   median was 1.70× better than the human median and 92 of 104 beat it, which is
   exactly the regime where `raw > C` and none of it counts.
2. **§0a's post-clear replay is usually pure cost.** It fires after clearing
   every level, where `C = 1.0` and `raw` is already over 1.0 — so it buys
   nothing while spending real budget. `r11l` is the clean demonstration: $18.77
   against its pair's $12.02, 59 minutes against 38, `raw` 1.1500 either way,
   **E identical at 1.0000**. The `cd82` replay was the same shape, and worse than
   reported — see the correction below.
3. **Persistence on an unsolved level dominates everything else.** `tn36` walked
   away from +0.208 to save actions it had no use for — 82% of its budget went
   unspent — while the doctrine's loudest signals were all about spending less.

### When the replay pays, measured within each run rather than across runs

**Correction, and the same methodological error twice.** This section said
`cd82`'s replay was "worth 0.12 of `raw`". It was not. That 0.12 is the gap
between the clean run's *final* `raw` (1.1500) and its **leaked pair's** (1.0296)
— two different runs — and attributing it to the replay repeats exactly the
mistake already recorded for `sc25`: comparing arm to control instead of before
to after, when the final number already contains whatever the replay did.

Measured properly, by splitting each trace at its full reset:

| run | plays | play 1 `raw`/E | final `raw`/E | replay changed E |
|---|---|---|---|---|
| nobaseline/cd82 | 2 | 1.1500 / 1.0000 | 1.1500 / 1.0000 | **+0.0000** |
| nobaseline/lp85 | 2 | 1.1486 / 1.0000 | 1.1500 / 1.0000 | +0.0000 |
| nobaseline/r11l | 2 | 1.1500 / 1.0000 | 1.1500 / 1.0000 | +0.0000 |
| nobaseline/s5i5 | 2 | 1.1500 / 1.0000 | 1.1500 / 1.0000 | +0.0000 |
| nobaseline/su15 | 2 | 1.1500 / 1.0000 | 1.1500 / 1.0000 | +0.0000 |
| **nobaseline/sc25** | 2 | **0.9163** / 0.9163 | 1.1500 / 1.0000 | **+0.0837** |
| **leaked/sc25** | 2 | **0.8066** / 0.8066 | 1.1357 / 1.0000 | **+0.1934** |
| **leaked/su15** | 2 | **0.3854** / 0.3854 | 0.9200 / 0.8000 | **+0.4146** |

`cd82`'s first play was **already at `raw` 1.1500**, the theoretical maximum,
with all six levels clamped in 107 actions. The replay cut that to 70 and moved
the score by nothing at all.

**The separator is clean, 8 for 8: the replay pays if and only if play 1's `raw`
was below 1.0.** Which is not an empirical accident but `E = min(C, raw)` restated
— once `raw ≥ C`, and `C = 1.0` when every level clears, raising `raw` is pushing
on the term that is not binding.

Two consequences that pull in opposite directions, and both are true:

- **The exact rule is "replay iff `raw < 1.0`", and a baseline-free solver cannot
  evaluate it**, because `raw` needs the medians. §0a's substitute — *replay
  anyway when you cannot compute `raw`* — is therefore right 1 time in 6 in the
  clean arm and wasted the other 5.
- **But replaying cannot lower the score.** ARC scores best-of-plays, and E is
  monotone in levels cleared, so a wasted replay costs money and wall clock, never
  points. Across the six clean replays it bought +0.0837 for roughly 1.5× the
  spend on five of them. As a *score* policy under uncertainty that is defensible;
  as a *cost* policy it is poor.

So "usually pure cost" above is right about the money and wrong to imply the
instruction is a mistake. The sharper version: §0a is a blind bet that is free in
points and expensive in dollars, and the condition that would make it cheap is
the one piece of information the arm deliberately withholds.

Staged rather than applied: reordering the doctrine around "clear the next level"
over "spend fewer actions" changes the treatment mid-arm, so it waits for the
batch boundary alongside the replay-conditioning change already queued.

**Why the mean is under 1.0, and it is one game.** Eight of nine clean runs score
1.0000; `tn36` alone scores 0.5357. That single partial game costs the mean
0.0516, and the reason it costs so much is the completion cap's non-linearity:
`tn36` cleared **5 of 7 levels — 71% of the levels — for 53.6% of the score**,
because `C = Σ(1..5)/Σ(1..7) = 15/28`. The two levels it did not reach were worth
6 and 7 of the 28 weight units between them, more than the first four combined.
Nothing is wrong with the harness or the arm; one game gave up early on an
environment already known to swing.

`tr87` is a seventh tie — E 1.0000 both ways, `raw` 1.1500 both ways, 155 clean
actions against 193 leaked. Its replay was another that bought nothing: play 1
was already at the clamp, and the second play spent 156 of the run's 349 actions
to move the score by 0.0000.

`vc33` is the ninth tie on score and the widest gap underneath it: E 1.0000 both
ways, but `raw` 1.1500 clean against 1.0268 leaked, on **167 actions against
270**. Level 3 is the whole difference — 21 actions clean, 84 leaked.

Running total, cleanly scored: **11 of 25 — ten at 1.0000 and `tn36` at
0.5357**, mean 0.9578; **ten pairs, eight exact ties, net −0.2643**. These are
the eleven cheapest environments in the set, so the
run of 1.0000s that preceded `tn36` said more about the cheapest-first ordering
than about the harness — and `tn36`, at a baseline total of 317, is the first
game big enough to have somewhere to go wrong. Leak probes clean across every
channel — `/api/games`, `ARC_API_KEY`,
`list_games(`, `baselines_for`, `urllib`, `requests.`, `httpx`, `curl` — and from
`ft09` onwards even the string `baseline_actions` is absent from the stream, the
line-deletion strip having landed for it.

One caveat on `ft09`'s cost line: it reads $8.57, but that covers only the second
attempt. `run_cost` reads `stream.jsonl` alone while `ledger_facts` reads the
trace, which carries across attempts — so a resumed run understates its spend by
whatever the killed attempt burned, here about $5.21. Real total ≈ $13.78.

**Verification.** Full probe of the finished 1366-line stream: zero hits on
`/api/games`, `ARC_API_KEY`, `list_games(`, `baselines_for`, `urllib`,
`requests.`, `httpx` and `curl`; `cd82`'s array `55, 8, 41, 21, 23, 23` appears
nowhere. The workspace built for the next game, `ft09`, carries no
`baseline_actions` line at all.

---

**The twelve leaked pairs, retained for the variance evidence.** Score 11.778
against 11.396 — the demoted-baseline arm is ahead by 0.383, on eleven wins to
the controls' eleven. **Eight** exact ties, two small losses, two gains. (An
earlier draft said nine ties; it was eight.) These are being replaced game by
game as the clean arm runs.

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

### `sp80-589a99af` — **5/6 for E=0.7143**, and the first run to stop *voluntarily*

Twelfth game of the baseline-free arm, and the first that neither won nor was
cut short by a cap. It ended at `stop_reason: end_turn`, `is_error: false`, with
**95% of its action budget and 0.8 h of its 4 h clock unused** — 137 of 2,590
actions in 3.18 h.

| | |
|---|---|
| E | **0.7143** (= cap; `raw` 0.8214) |
| levels | 5 of 6 |
| actions | 137 billed, 1 death, 0 full resets |
| wall | 3.18 h of a 4.00 h cap · 83 s/action |
| cost | $26.49 over 132 turns |

**Every cleared level hit the 1.15 efficiency ceiling.** Not one was merely under
baseline — all five were capped:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 11 | 9 | 10 | 23 | 40 | — |
| human | 39 | 58 | 25 | 148 | 96 | 152 |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 0.00 |

This is the completion cap binding in its purest form: `raw` (0.8214) exceeds
`E` (0.7143) purely because level 6 is unfinished. Efficiency contributed nothing
to the score and could not have — the 29th of 30 runs where that is true.

**The stop was reasoned, not a give-up.** The solver built a flow model that
reproduced all 13 recorded releases exactly, cleared five levels with it, then
searched **63.7M platform configurations** on level 6 plus a relaxed run with the
adjacency rule disabled. It concluded that the two left-facing cups each require
the single left-sliding chute in a different row, and that every solution needs
two cells that the "no platform orthogonally adjacent to a cup" rule forbids. Its
own verdict: *"there's a mechanic I didn't find… I'd rather flag that gap than
dress up a guess."*

That is the behaviour the doctrine asks for, and it costs nothing: it correctly
noted that once level 6 was unfinishable the remaining experiments were
score-free, because completed levels' counts were already banked and the replay
window closes once you act on a new level.

**A prediction this refuted.** Mid-run, at 76 s/action, this was called as a
second `bp35` — a run that would lose to the clock. It did not: the pace was the
solver spending minutes per action on a genuine search, and it finished its search
before the cap. *Slow is not the same as clock-bound*, and the earlier inference
read one as the other.

**What it does not tell us.** `sp80` ran on the pre-`b5f4651` runner, so it has no
thinking capture and no ARC-notation action counting — `clean` tier on the audit
page, not `latest`. Zero runs are on the current config; game 13 is the first.

### `tu93-0768757b` — **WON 9/9 for E=1.0000**, and the first run on the current config

Thirteenth game of the baseline-free arm, third attempt at this environment (the
first was SIGTERM'd by a container restart, the second quarantined for baseline
exposure — see below). It is the **first run in the whole project carrying both**
`--thinking-display summarized` **and ARC-notation action counting**.

| | |
|---|---|
| E | **1.0000** (`raw` 1.1259, `cap` 1.0000) |
| levels | 9 of 9, won |
| actions | 452 billed / 454 trace rows, against a 462 baseline total |
| wall | 0.91 h · 4 deaths · 1 full reset · 2 playthroughs |
| cost | $12.20 over 96 turns |

#### The replay rule holds, and this is the first run where it paid

Every prior replay in this project was wasted money — eight for eight, because
play 1 was already at or above `raw` 1.0 and ARC scores best-of-plays. The rule
derived from those was: **replay pays iff play-1 `raw` < 1.0.** `tu93` is the
first run to satisfy the antecedent, and the consequent followed:

| | actions | E | raw |
|---|---|---|---|
| play 1 | 268 | 0.8286 | 0.8286 |
| play 2 | 186 | **1.0000** | **1.1259** |

A gain of **+0.1714**, from a replay that cost 186 actions out of a 2,310 cap.
Play 2 finished eight of nine levels at the 1.15 efficiency ceiling:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| agent | 18 | 10 | 19 | 17 | 29 | 28 | 14 | 21 | 29 |
| human | 19 | 16 | 34 | 42 | 123 | 80 | 14 | 23 | 111 |
| `S_l` | 1.11 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.00 | 1.15 | 1.15 |

#### Both new instruments verified live

- **Thinking capture:** 78 blocks with non-empty bodies. Every earlier run in the
  set has zero — `redacted: true` on every THINK span.
- **ARC-notation counting:** `trace_rows` 454 against `actions_used` 452. The
  delta is exactly 2, one unbilled opening `RESET` per playthrough, which is
  precisely what ARC does not bill.

#### Why no earlier run has either flag

Both landed in `b5f4651`, and **neither could reach a runner that was already
running**: `session.py` is imported once at runner startup and Python caches the
module, so editing the file changes nothing for the live process. That is the
fourth time this project has lost a config change to import caching.
`restart_after_sp80.sh` existed only to wait for a game boundary and relaunch,
and `tu93` is the first game the relaunched runner started. The audit page marks
this as the `latest` tier; the twelve runs before it are `clean` — valid scores on
older instrumentation, not comparable on thinking volume or action counting.

#### The attempt that had to be thrown away

Between the interrupted run and this one sits a third, killed deliberately. Five
and a half minutes in, that solver issued a hand-rolled
`urllib.request` to `client.root + "/api/games"` with `client._key` and read back
`baseline_actions` for all 25 environments, its own included
(`[19, 16, 34, 42, 123, 80, 14, 23, 111]` — the exact array in the table above).
The baseline watcher caught it within one 5-minute poll.

**Nothing instructed it to.** `strip_baselines` already deletes §6 and §6a from
the shipped doctrine, so the workspace copy jumps §5 → §7 and never tells the
solver to budget against a published median. It went looking on its own, holding
a key and knowing the API — which is capable behaviour, not misbehaviour.

`CCARC3_HIDE_BASELINES=1` closes the in-process path and did its job; it cannot
close a raw HTTP call, because `ARC_API_KEY` is in the solver's own environment
and anything the client can read to authenticate, the solver can read too. The
only structural fix is moving the key into a separate process behind a shim that
forwards game actions but never `/api/games`. Until that exists the posture is
detect-and-discard, which bounds the cost to wasted quota rather than a corrupted
result — one reach in thirteen runs, caught in under five minutes, quarantined at
`ablate_exposed/tu93-0768757b-exposed` with both streams intact.

Deliberately **not** fixed with a doctrine prohibition: the distance between "a
baseline exists" and "the baseline is 55" is the entire experiment, and an
explicit ban advertises the target.

### `s5i5-18d95033` — **WON 8/8 at `raw` 1.1500**, a perfect efficiency sweep, and a replay that could not possibly pay

Thirteenth scored game, second attempt at this environment (the first was
quarantined for baseline exposure). Second `latest`-tier run.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 8 of 8, won, **0 deaths, 0 wasted actions** |
| actions | 539 billed / 541 trace rows, against a 638 baseline total |
| wall | 2.53 h of 4.00 h · 121 thinking blocks |
| cost | $36.54 over 145 turns |

`raw` of exactly 1.1500 is the ceiling: `S_l = min(1.15, (h/a)²)` saturated on
**every level**, not one below it.

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 13 | 26 | 37 | 30 | 28 | 25 | 45 | 36 |
| human | 20 | 89 | 106 | 54 | 162 | 38 | 86 | 83 |
| ratio | 0.65× | 0.29× | 0.35× | 0.56× | **0.17×** | 0.66× | 0.52× | 0.43× |

#### The replay was structurally incapable of helping, and cost 241 actions

| | actions | E | raw |
|---|---|---|---|
| play 1 | 300 | 1.0000 | **1.1500** |
| play 2 | 241 | 1.0000 | **1.1500** |

Play 1 had **already cleared all eight levels at the ceiling**. Since ARC scores
best-of-plays and `E = min(cap, raw)`, a replay can raise E only when play 1's
`raw` is below 1.0. This is the ninth wasted replay in the project and the rule
that predicts it has now separated 9 for 9.

**But the solver could not have known.** That is the finding. `score_now` and
`score_ceiling` return `None` without baselines — by design, since withholding
the medians is the whole experiment — so the solver cannot compute `raw` and
cannot apply "replay iff `raw` < 1.0". **Withholding baselines has a measurable
action cost, and this run priced it: 241 of 3,255 actions, ~45% of the run's
wall clock and spend, on a play that could not change the score.**

The fix does not need the baselines back. After clearing every level, `cap` is
1.0 and E = min(1.0, `raw`), so a replay pays only if the agent averaged *worse*
than the human median — and a run with **zero deaths and zero wasted actions**
almost certainly did not. That is computable from structure alone, and it is the
doctrine line to add:

> **A clean sweep is already banked.** If you have cleared every level with no
> deaths and no refused actions, do not replay. Your cap is 1.0 and a replay can
> only help if you were slower than the human median throughout, which a clean
> sweep is evidence against. Replay after a run that *died*, not one that won.

Filed against the staged-changes task rather than applied mid-arm, so the
remaining twelve runs stay comparable to the thirteen already scored.

#### Integrity

No level exceeded ARC's per-level 5n cap. Zero baseline-reach markers across
both streams — the exposure that quarantined the first attempt did not recur.

### `ar25-0c556536` — **WON 8/8 at `raw` 1.1500**, second perfect efficiency sweep

Fourteenth scored game, third `latest`-tier run, and the first scored under the
two-wide claiming pool.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 8 of 8, won, 0 deaths |
| actions | 529 billed / 531 trace rows, against a 748 baseline total |
| wall | 0.70 h · 72 thinking blocks |
| cost | $11.98 |

Every level at the 1.15 ceiling, several by a wide margin:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 15 | 12 | 40 | 22 | 28 | 53 | 37 | 47 |
| human | 32 | 50 | 75 | 37 | 89 | 159 | **233** | 73 |

Level 7 is the standout: **37 actions against a 233 human median**, 0.16×.

**Tenth wasted replay, and the rule is now 10 for 10.** Play 1 finished all eight
levels at `raw` 1.1500 — already the maximum — so play 2's 255 actions could not
raise E under best-of-plays scoring. Same mechanism as `s5i5`: without baselines
the solver cannot compute `raw` and so cannot apply "replay iff `raw` < 1.0". The
structural stand-in already filed against the staged-changes task — *a clean sweep
with no deaths is already banked, do not replay* — would have saved 255 of 3,740
actions here, about 48% of the run.

Two independent confirmations in two consecutive games make that doctrine line the
highest-value item in the staged set.

#### Integrity
No level exceeded ARC's per-level 5n cap. Zero baseline-reach markers.

### `ka59-38d34dbb` — **WON 7/7 at `raw` 1.1318**, and the run that corrected how a lost card is scored

Fifteenth scored game, and the most instructive one so far: it was killed by the
~8h session rotation at level 3, lost its scorecard to a server-side reap, and
won anyway.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1318**, `cap` 1.0000) |
| levels | 7 of 7, won, 0 deaths, 0 wasted actions |
| actions | **364 scored** / 504 on the ledger, against a 730 baseline total |
| wall | 1.34 h across two attempts · 167 turns |
| cost | $22.50 |

Six of seven levels at the 1.15 ceiling:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 35 | 40 | 36 | 46 | 20 | 87 | **100** |
| human | 28 | 109 | 51 | 51 | 33 | 132 | **326** |
| ratio | 1.25× | 0.37× | 0.71× | 0.90× | 0.61× | 0.66× | **0.31×** |
| `S_l` | 0.64 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

Level 1 is the only one below the cap, and it is the cheapest level in the game —
7 actions over the human median on a 28-action level. Level 7 is the standout:
**100 actions against a 326 human median**.

#### A lost card costs budget, not score — and that corrects an earlier claim

The rotation SIGTERMed this solver at 03:09:12 with 141 actions spent and level 3
reached. It was relaunched 59.9 minutes later, which is well past the window in
which ARC keeps an idle game: the scorecard 404'd and every `/api/cmd` answered
`game not found`. The solver opened a fresh card and replayed.

It had been reported here that those re-played actions inflate the RHAE
denominator — that this run would be scored on 504. **That is wrong**, and this
run is the proof. `actions_per_level` counts the play that finished, matching
ARC's own best-of-plays selection, so the score is computed on **364**. The 140
actions of the reaped play cost budget and 60 minutes of wall clock, and nothing
else. `scorecard.json` confirms it independently: `total_plays: 1`,
`total_actions: 364` — ARC never saw the lost play at all.

Stronger still, `disagreements_with_server` returns **empty**: our trace and
ARC's `actions_by_level` agree on all seven levels, `[35, 40, 36, 46, 20, 87,
100]`. This is the first run to check every level against the server rather than
trusting the trace, and the two are identical.

#### The replay was reconstructed, not blundered

Rather than re-explore, the solver wrote `notes/replay.py` and `notes/run.py` — a
driver that re-executes the *recorded effective actions* against the fresh card,
dropping actions whose frame did not change and checking each replayed frame
against the recording. That is why a rerun of a 141-action position cost so
little and still reached the ceiling on six levels. The technique was invented
in-run and is a candidate for the harness proper.

#### Integrity
No level exceeded ARC's per-level 5n cap (0 real refusals; an earlier heartbeat
alert reporting one was a false positive — it matched the f-string in
`client.py:968` that the solver had read, not a runtime message). Zero
baseline-reach markers; no `ARC_API_KEY` mention anywhere in the stream.

### `ls20-9607627b` — **WON 7/7 at `raw` 1.1500**, and the run that nearly broke the scorer

Sixteenth scored game. Every one of seven levels at the 1.15 ceiling.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 7 of 7, won, 1 death, 0 wasted actions |
| actions | **325 scored** / 2,116 on the ledger over **5 plays**, against a 776 baseline total |
| wall | 2.27 h · 186 turns |
| cost | $33.71 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 13 | 45 | 49 | 43 | 50 | 72 | **53** |
| human | 22 | 123 | 73 | 84 | 96 | 192 | **186** |
| ratio | 0.59× | 0.37× | 0.67× | 0.51× | 0.52× | 0.38× | **0.28×** |

#### The scorer's "final play" convention is not ARC's "best play" convention

`score_run` counts the play that finished, and its docstring asserts that this is
"what best-of-plays scoring selects". `ls20` is the first run with enough plays to
test that, and it shows the two are **not** equivalent — it just happened to agree
here:

| play | actions | `raw` | state |
|---|---|---|---|
| 1 | 757 | 0.8659 | WIN |
| 2 | 349 | **1.1500** | WIN |
| 3 | 440 | **1.1270** | WIN |
| 4 | 245 | — | NOT_FINISHED (5/7) |
| 5 | 325 | **1.1500** | WIN |

Play 3 is a *completed* play that scored **worse** than play 2. Had the run stopped
there, our scorer would have reported 1.1270 while ARC's best-of-plays would score
1.1500. The trajectory is not monotone, so "last" is not a safe proxy for "best".
Filed as a scorer correction; it changes no result recorded so far, because every
run's final play has happened to be its best.

#### Third and fourth confirmation of the do-not-replay doctrine

Play 2 reached `raw` 1.1500 — the arithmetic maximum. Plays 3, 4 and 5 cost a
further **1,010 actions, 48% of the run**, and could not raise E by construction.
Same for `cn04` below. With `s5i5` and `ar25` that is four independent
confirmations of the staged doctrine line — *a clean sweep with no deaths is
already banked, do not replay* — which remains the highest-value unshipped item.

The replay rule itself holds: play 1 scored 0.8659 < 1.0, so the **first** replay
was correctly taken. It is the third, fourth and fifth that were free.

#### Integrity
Zero disagreements with ARC: `actions_by_level` matches our trace on all seven
levels. No level exceeded the per-level 5n cap. Zero baseline-reach markers.

### `cn04-2fe56bfb` — **WON 6/6 at `raw` 1.1500** on 0.22× the human action total

Seventeenth scored game, and the widest efficiency margin in the arm so far.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 6 of 6, won, 0 deaths, 0 wasted actions |
| actions | **174 scored** / 564 on the ledger over 2 plays, against a **789** baseline total |
| wall | 1.93 h · 142 turns |
| cost | $29.86 |

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 14 | 29 | 20 | **29** | 44 | 38 |
| human | 29 | 54 | 85 | **300** | 208 | 113 |
| ratio | 0.48× | 0.54× | 0.24× | **0.10×** | 0.21× | 0.34× |

Level 4 is the standout of the whole arm: **29 actions against a 300 human
median**, one tenth. The scored play finished the entire game in 174 actions
against a human total of 789.

Replay rule again correct — play 1 scored 0.8910 < 1.0 and the replay took it to
1.1500, from 390 actions to 174. That is **12 for 12**.

#### Integrity
Zero disagreements with ARC on all six levels. No 5n cap hits. Zero baseline-reach
markers.

### Arm standing after 17 games

| | |
|---|---|
| scored | 17 of 25 |
| wins | **15 / 17** |
| sum `E` | **16.2500** |
| mean `E` | 0.9559 |
| floor over all 25 (unscored counted as zero) | **65.00%** |

Only two games are not wins: `sp80` (5/6, E 0.7143) and `tn36` (5/7, E 0.5357).
The floor of 65.00% requires no extrapolation — it assumes the eight unplayed
environments score zero — and stands against the 40.68% published for Opus 5 on
the public demo set (24 Jul 2026, High effort). ARC states public-set scores are
"emphatically not" a valid measure of progress, and that caveat applies here.

### `sk48-d8078629` — **5/8 for E=0.4167**, and the pattern every loss in the arm shares

Eighteenth scored game, and the first loss since the arm went baseline-free.

| | |
|---|---|
| E | **0.4167** (`raw` 0.4792, `cap` **0.4167** — the cap binds) |
| levels | 5 of 8, **0 deaths** |
| actions | **192 scored** / 632 on the ledger over 2 plays, against a 1,070 baseline total |
| wall | 2.37 h · 155 turns |
| cost | $45.75 |

Every level it *reached* was at the ceiling, and not narrowly:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 14 | 40 | 33 | 36 | 69 | — | — | — |
| human | 61 | 177 | 101 | 103 | 230 | 181 | 125 | 92 |
| ratio | 0.23× | 0.23× | 0.33× | 0.35× | 0.30× | — | — | — |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 0.00 | 0.00 | 0.00 |

Nothing here is an efficiency failure. `raw` 0.4792 is low purely because three levels
score zero; on the five it cleared it was between three and four times faster than
the human median. The whole loss is completion.

The replay was textbook: exploration cost 429 actions across five levels, the solver
restarted at a legal boundary and walked the known routes in 192, and `raw` rose
0.4161 → 0.4792. Thirteenth confirmation of the replay rule. Zero disagreements with
ARC on every level it scored.

#### Every loss in this arm is a voluntary stop, not an exhausted budget

`sk48` ended with **632 of 5,350 actions used — 12%**. It was not killed, did not time
out, and exited 0. It wrote a closing report explaining that level 6 requires pushing
red blocks through a gap that neither chain can enter, and stopped.

That is not an isolated call. It is what all three losses have in common:

| game | E | levels | budget used | exit | wall |
|---|---|---|---|---|---|
| `sp80` | 0.7143 | 5/6 | **137 / 2,590 — 5%** | 0 | 3.18 h |
| `tn36` | 0.5357 | 5/7 | **289 / 1,585 — 18%** | 0 | 1.48 h |
| `sk48` | 0.4167 | 5/8 | **632 / 5,350 — 12%** | 0 | 2.37 h |

None timed out. None was signal-killed. **No run in the arm has ever exhausted its
action budget** — in eighteen games the binding constraint has never once been the
thing the budget measures. Fifteen games stopped because they won; three stopped
because the solver concluded it was stuck, with 82–95% of its actions unspent.

This is now the largest recoverable loss on the board. The three losses hold 1.333 of
the 1.667 points missing from a perfect eighteen. Taking `sk48` alone from 5/8 to 8/8
is worth **+0.58** — more than every efficiency gain in the arm put together, since
efficiency is already capped at 1.15 nearly everywhere and `E` is capped at 1.0.

Filed as a doctrine item alongside the do-not-replay line: **a stuck level with budget
remaining is a reason to change technique, not to stop.** The counterpart to
"dying is cheap" is that concluding is expensive, and only one of the two is currently
written down.

### Arm standing after 18 games

| | |
|---|---|
| scored | 18 of 25 |
| wins | **15 / 18** |
| sum `E` | **16.6667** |
| mean `E` | 0.9259 |
| floor over all 25 (unscored counted as zero) | **66.67%** |

### `g50t-5849a774` — **WON 7/7 at `raw` 1.1500**, every level at the ceiling

Nineteenth scored game.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 7 of 7, won, 1 death |
| actions | **307 scored** / 1,097 on the ledger over 2 plays, against an 879 baseline total |
| wall | 3.03 h · 186 turns |
| cost | $47.88 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 17 | 31 | 68 | **35** | 64 | 49 | 43 |
| human | 78 | 175 | 179 | **230** | 96 | 54 | 67 |
| ratio | 0.22× | 0.18× | 0.38× | **0.15×** | 0.67× | 0.91× | 0.64× |

Level 4 is the standout — **35 actions against a 230 human median**.

Fourteenth confirmation of the replay rule, and one of its clearest cases: play 1
won all seven levels but only at `raw` 0.8941, below 1.0 and so worth replaying.
Play 2 walked the known routes in 307 actions against play 1's 790 and landed at
the 1.1500 ceiling. Both plays are recorded WIN, so this is also the second run
where best-of-plays and last-play would agree only because the last play happened
to be the better one.

#### Integrity
Zero disagreements with ARC on all seven levels. No 5n cap hits. Zero
baseline-reach markers.

### Arm standing after 19 games

| | |
|---|---|
| scored | 19 of 25 |
| wins | **16 / 19** |
| sum `E` | **17.6667** |
| mean `E` | 0.9298 |
| floor over all 25 (unscored counted as zero) | **70.67%** |

### `bp35-0a0ad940` — **WON 9/9 at `raw` 1.1481**, and clean this time

Twentieth scored game, and the one that needed re-running. `bp35`'s first attempt
was the second baseline exposure in this project: it printed the game-id list
after reaching `/api/games` with the key from its own environment. That run was
discarded to `ablate_exposed/`. This is the replacement, and it is clean —
**zero `ARC_API_KEY` mentions, zero `api/games` references, zero 5n cap hits** in
the entire stream.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1481**, `cap` 1.0000) |
| levels | 9 of 9, won, 6 deaths |
| actions | **374 scored** / 1,278 on the ledger over 3 plays, against a 651 baseline total |
| wall | 3.89 h · 230 turns |
| cost | $57.49 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| agent | 19 | 43 | 34 | 27 | 31 | 54 | 43 | 45 | 78 |
| human | 21 | 48 | 44 | 38 | 33 | 87 | 86 | 131 | 163 |
| ratio | 0.90× | 0.90× | 0.77× | 0.71× | 0.94× | 0.62× | 0.50× | 0.34× | 0.48× |

Eight of nine at the ceiling; level 5 misses at 0.94×. This is the tightest
baseline in the set — a 651-action human total across nine levels — so the margins
are thinner than elsewhere and the run still cleared it.

Three plays, monotonically improving: `raw` 1.1161 → 1.1365 → **1.1481**, all three
recorded WIN. The only run in the arm where every play won and each was better
than the last, which is exactly the shape that makes best-of-plays and last-play
agree by luck rather than by construction.

### `m0r0-492f87ba` — **WON 6/6 at `raw` 1.1500** on 0.17× the human action total

Twenty-first scored game, and the widest margin in the arm — beating `cn04`'s 0.22×.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 6 of 6, won, **0 deaths** |
| actions | **190 scored** / 456 on the ledger over 2 plays, against a **1,107** baseline total |
| wall | 0.70 h · 101 turns |
| cost | $13.20 |

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 15 | 23 | 53 | 11 | **45** | 43 |
| human | 30 | 111 | 203 | 26 | **500** | 237 |
| ratio | 0.50× | 0.21× | 0.26× | 0.42× | **0.09×** | 0.18× |

Level 5 is the single widest margin recorded here: **45 actions against a 500
human median**, one eleventh. The whole game fell in 190 actions against a human
total of 1,107, for $13.20 — the cheapest win in the arm by a wide margin.

**Fifth confirmation of the do-not-replay doctrine.** Play 1 already reached `raw`
1.1500, the arithmetic maximum. Play 2's 190 actions could not raise `E` by
construction, and were spent anyway because without baselines the solver cannot
compute `raw` and so cannot apply "replay iff `raw` < 1.0". With `s5i5`, `ar25`,
`ls20` and `cn04` that is five independent confirmations of the same unshipped
line — *a clean sweep with no deaths is already banked, do not replay* — which
remains the highest-value staged item.

### Arm standing after 21 games

| | |
|---|---|
| scored | 21 of 25 |
| wins | **18 / 21** |
| sum `E` | **19.6667** |
| mean `E` | 0.9365 |
| floor over all 25 (unscored counted as zero) | **78.67%** |

Four environments remain: `wa30`, `lf52`, `re86`, `dc22` — the last three in
flight now.

### `dc22-fdcac232` — **WON 6/6 at `raw` 1.1500**, every level at the ceiling

Twenty-second scored game.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 6 of 6, won, **0 deaths** |
| actions | **434 scored** / 1,806 on the ledger over 3 plays, against a 1,228 baseline total |
| wall | 2.01 h · 235 turns |
| cost | $50.58 |

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 20 | 42 | 45 | 62 | 110 | **155** |
| human | 59 | 102 | 67 | 98 | 324 | **578** |
| ratio | 0.34× | 0.41× | 0.67× | 0.63× | 0.34× | **0.27×** |

The two deepest levels carry the widest margins — level 6 at **155 actions against
a 578 human median**. This is a game where the human baseline grows steeply with
depth and the agent's cost does not.

**Sixth confirmation of the do-not-replay doctrine.** Play 1 won all six levels at
`raw` 1.1445, play 2 reached 1.1500, play 3 spent a further 434 actions at exactly
1.1500. **All three plays scored E=1.0000, so plays 2 and 3 were both free** —
887 actions, 49% of the run, that could not move the number.

An earlier version of this entry got that wrong. It claimed play 1 at 1.1445 was
"worth improving, since `raw` still fed a `min(cap, raw)` that the cap had not yet
pinned", and concluded the standing rule *replay iff `raw` < 1.0* was too strict.
The cap **had** pinned it: clearing every level makes `cap` exactly 1.0, so
`E = min(1.0, raw)` and every `raw` at or above 1.0 scores the same 1.0000.
Raising `raw` from 1.1445 to 1.1500 buys nothing. The standing rule was right and
the proposed refinement was an arithmetic error.

### Arm standing after 22 games

| | |
|---|---|
| scored | 22 of 25 |
| wins | **19 / 22** |
| sum `E` | **20.6667** |
| mean `E` | 0.9394 |
| floor over all 25 (unscored counted as zero) | **82.67%** |

All 25 environments are now scored or in flight — `wa30`, `re86` and `lf52` are
the last three, all running.

### `re86-8af5384d` — **WON 8/8 at `raw` 1.0974**, and the first game the best-of-plays fix saved

Twenty-third scored game. It is also the run that proves the scoring change
shipped earlier today was not academic.

| | |
|---|---|
| E | **1.0000** (`raw` 1.0974, `cap` 1.0000) |
| levels | 8 of 8, won, **0 deaths** |
| actions | **928 scored** / 1,248 on the ledger over 2 plays, against a 1,255 baseline total |
| wall | 2.35 h · 201 turns |
| cost | $39.13 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 25 | 36 | 49 | 66 | **63** | 64 | 374 | 251 |
| human | 26 | 42 | 86 | 108 | **189** | 139 | 424 | 241 |
| ratio | 0.96× | 0.86× | 0.57× | 0.61× | **0.33×** | 0.46× | 0.88× | 1.04× |

Six of eight at the ceiling. Level 8 is the only one above the human median, at
1.04×, on the hardest level of a 1,255-action game.

#### The scoring convention decided this result

| convention | E | `raw` | levels |
|---|---|---|---|
| **best of plays** (current) | **1.0000** | 1.0974 | **8 / 8** |
| last play (previous) | 0.5833 | 0.6708 | 6 / 8 |

Play 1 won all eight levels in 928 actions. Play 2 was a replay, 257 actions and
six levels in, when a container restart moved the agent proxy to a new loopback
port and the solver died on `Connection refused`. Under the old convention the
run would be scored on that severed replay and recorded as a 6-of-8 **loss at
0.5833**. ARC's own card says otherwise — `states: ['WIN', 'NOT_FINISHED']` — and
takes the maximum.

**A +0.4167 swing on one game.** When the change landed the note here read "no
recorded result moves, every run's final play has so far also been its best".
That is no longer true, and the first counterexample is worth more than every
efficiency gain recorded in the arm.

#### The same restart cost two games, and nearly banked them as losses

`wa30` (GAME_OVER at 5 of 9) and `lf52` (NOT_FINISHED at 6 of 10) died in the same
event, both with `exit 1` and no error field. `collect_outcome` marks
signal-killed runs retryable — that guard was written for `ft09` — but a plain
non-zero exit fell straight through it, so both would have been banked
permanently as losses under `if prior and not prior.get("error"): skip`. Fixed:
a non-zero exit that is not a timeout and did not win is now an interruption. A
win is exempt, which is exactly why `re86` above stays banked.

### Arm standing after 23 games

| | |
|---|---|
| scored | 23 of 25 |
| wins | **20 / 23** |
| sum `E` | **21.6667** |
| mean `E` | 0.9420 |
| floor over all 25 (unscored counted as zero) | **86.67%** |

`wa30` and `lf52` are re-running now.

### `lf52-271a04aa` — **WON 10/10 at `raw` 1.0530**, the game that was nearly banked as a loss

Twenty-fourth scored game, the longest environment in the set at ten levels, and
the direct vindication of the crash-detection fix.

| | |
|---|---|
| E | **1.0000** (`raw` 1.0530, `cap` 1.0000) |
| levels | 10 of 10, won, **0 deaths**, 1 playthrough |
| actions | **941**, against a 1,339 baseline total |
| wall | 3.55 h across two attempts · 366 turns |
| cost | $77.77 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 9 | 56 | 53 | 74 | 127 | **230** | 154 | 68 | 127 | **43** |
| human | 32 | 81 | 60 | 71 | 205 | **148** | 244 | 109 | 164 | **225** |
| ratio | 0.28× | 0.69× | 0.88× | 1.04× | 0.62× | **1.55×** | 0.63× | 0.62× | 0.77× | **0.19×** |

Eight of ten at the ceiling. Level 6 is the one real overrun at 1.55×, and level
10 — the deepest level in the arm — went in **43 actions against a 225 human
median**.

#### This is the run the crash guard was written for

`lf52` died at 6 of 10 levels with `exit 1` when a container restart moved the
agent proxy. With no error field it was, for about fifteen minutes, a permanent
6-of-10 **loss at E=0.4909**. Marking a non-zero exit as an interruption is what
sent it back to the pool, and it came back a 10-of-10 win. The fix is worth
**+0.5091** on this game alone.

#### And it corrects the reap deadline recorded here

The resume gap was **12.2 minutes and the card survived** — one playthrough, 941
actions, ARC's card reading `plays=1, states=['WIN']`, and not one
`game not found` in the stream. The note in `snapshot_scorecard` had put the
deadline at "about 12 minutes". Six resumes now bracket it to **(12.2, 43.8]
minutes**:

| gap | outcome |
|---|---|
| `ft09` 9.1 min | card live, finished |
| `sb26` 11.9 min | card live, finished |
| **`lf52` 12.2 min** | **card live, won 10/10** |
| `bp35` 43.8 min | 404 — replayed from level 0 |
| `ka59` 59.9 min | 404 — replayed from level 0 |
| `tu93` 191.1 min | 404 — discarded a restored level 7 |

Nothing here narrows it further, and it is not a documented ARC policy, so the
bracket is the claim.

### Arm standing after 24 games

| | |
|---|---|
| scored | 24 of 25 |
| wins | **21 / 24** |
| sum `E` | **22.6667** |
| mean `E` | 0.9444 |
| floor over all 25 (unscored counted as zero) | **90.67%** |

`wa30` is the last environment, re-running now.

### `wa30-ee6fef47` — **WON 9/9 at `raw` 1.1500**, every level at the ceiling, and the arm's last game

Twenty-fifth scored game, the second crash-recovery win, and the largest human
baseline in the set at 1,843 actions.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 9 of 9, won, 3 deaths |
| actions | **667 scored** / 2,502 on the ledger over 3 plays, against a **1,843** baseline total |
| wall | 2.34 h across two attempts · 221 turns |
| cost | $36.25 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| agent | 45 | 58 | 93 | 52 | 101 | 57 | 50 | 145 | **66** |
| human | 71 | 119 | 183 | 98 | 368 | 68 | 79 | 442 | **415** |
| ratio | 0.63× | 0.49× | 0.51× | 0.53× | 0.27× | 0.84× | 0.63× | 0.33× | **0.16×** |

All nine at the ceiling. Level 9 went in **66 actions against a 415 human
median**. Three plays, monotonically improving — `raw` 1.0842 → 1.1367 → 1.1500,
all three recorded WIN by ARC.

Like `lf52`, this run had been banked as a loss: it died at 5 of 9 with `exit 1`
in the proxy-move incident and was marked retryable by the crash guard. It came
back a 9-of-9 sweep. The two recoveries together are worth **+1.0924**.

---

## The baseline-free arm: final result

All 25 ARC-AGI-3 public environments, played with the human medians withheld —
`CCARC3_HIDE_BASELINES` on, no control arm, every run scored against ARC's own
`actions_by_level`.

> **Corrected 2026-08-06.** `ka59` is void — it queried `/api/games` for its own
> baselines and disclosed doing so. The figures below are stated both ways; the
> 24-game row is the honest one. See the contamination section for how the door
> was left open.

| | as first published | `ka59` removed |
|---|---|---|
| environments scored | 25 of 25 | **24 of 24** |
| wins | 22 / 25 | **21 / 24** |
| sum `E` | 23.6667 | **22.6667** |
| mean `E` | 0.9467 | **0.9444** |
| **total score** | 94.67% | **94.44%** |
| total cost | $696.19 | $673.69 |

Published Opus 5 on the same public demo set, 24 Jul 2026, High effort: **40.68%**.

### Every environment

| game | E | `raw` | levels | agent actions | human | ratio |
|---|---|---|---|---|---|---|
| `lp85` | 1.0000 | 1.1500 | 8/8 | 79 | 388 | 0.20× |
| `m0r0` | 1.0000 | 1.1500 | 6/6 | 190 | 1107 | **0.17×** |
| `su15` | 1.0000 | 1.1500 | 9/9 | 82 | 361 | 0.23× |
| `cn04` | 1.0000 | 1.1500 | 6/6 | 174 | 789 | 0.22× |
| `r11l` | 1.0000 | 1.1500 | 6/6 | 69 | 233 | 0.30× |
| `ar25` | 1.0000 | 1.1500 | 8/8 | 254 | 748 | 0.34× |
| `dc22` | 1.0000 | 1.1500 | 6/6 | 434 | 1228 | 0.35× |
| `g50t` | 1.0000 | 1.1500 | 7/7 | 307 | 879 | 0.35× |
| `sc25` | 1.0000 | 1.1500 | 6/6 | 124 | 350 | 0.35× |
| `ft09` | 1.0000 | 1.1500 | 6/6 | 75 | 208 | 0.36× |
| `wa30` | 1.0000 | 1.1500 | 9/9 | 667 | 1843 | 0.36× |
| `tr87` | 1.0000 | 1.1500 | 6/6 | 155 | 414 | 0.37× |
| `vc33` | 1.0000 | 1.1500 | 7/7 | 167 | 447 | 0.37× |
| `s5i5` | 1.0000 | 1.1500 | 8/8 | 240 | 638 | 0.38× |
| `cd82` | 1.0000 | 1.1500 | 6/6 | 70 | 171 | 0.41× |
| `ls20` | 1.0000 | 1.1500 | 7/7 | 325 | 776 | 0.42× |
| `bp35` | 1.0000 | 1.1481 | 9/9 | 374 | 651 | 0.57× |
| `sb26` | 1.0000 | 1.1436 | 8/8 | 125 | 213 | 0.59× |
| `ka59` | ~~1.0000~~ **void** | 1.1318 | 7/7 | 364 | 730 | 0.50× |
| `tu93` | 1.0000 | 1.1259 | 9/9 | 185 | 462 | 0.40× |
| `re86` | 1.0000 | 1.0974 | 8/8 | 928 | 1255 | 0.74× |
| `lf52` | 1.0000 | 1.0530 | 10/10 | 941 | 1339 | 0.70× |
| `sp80` | 0.7143 | 0.8214 | 5/6 | 93 | 518 | 0.18× |
| `tn36` | 0.5357 | 0.5904 | 5/7 | 149 | 317 | 0.47× |
| `sk48` | 0.4167 | 0.4792 | 5/8 | 192 | 1070 | 0.18× |

### What the shape says

**Efficiency is not the constraint.** Twenty-two of 25 sit at `E` 1.0000 with the
completion cap binding, and 16 of those are at the `raw` ceiling of 1.1500 —
meaning the score is clipped, not earned at the margin. Even the three losses
were *efficient* on the levels they cleared: `sk48` at 0.18× and `sp80` at 0.18×
of the human action count.

**Completion is the only axis left.** The 1.333 points missing from a perfect 25
sit entirely in three games that each cleared exactly five levels and stopped.
None was killed, none timed out, all three exited 0 — with **5%, 18% and 12%** of
their action budgets unspent. No run in this arm ever exhausted its budget. The
binding constraint was never the thing the budget measures.

### Caveats, stated plainly

- ARC states that public-set scores are "emphatically not" a valid measure of
  progress. This is the public demo set and the comparison inherits that caveat.
- Every scored run was cross-checked against ARC's own `actions_by_level`;
  `disagreements_with_server` returned empty for all of them.
- No level in any scored run exceeded ARC's per-level 5n cap.
- Baseline integrity: zero `ARC_API_KEY` references and zero `api/games` hits in
  any scored run's stream. The two runs that did reach for baselines (`tu93`,
  `bp35`) were discarded to `ablate_exposed/` and re-run clean.
- Three results turned on harness fixes shipped mid-arm: best-of-plays scoring
  (`re86`, +0.4167) and crash-as-interruption (`lf52` +0.5091, `wa30` +0.5833).
  Without those three fixes the same runs would total **22.1667 → 88.67%**.

### Caveat on the three losses: `tn36` has a 7/7 run on record

The arm scored `tn36` at **E=0.5357, 5 of 7 levels**. A *different* run of the
same environment, recorded in `results.jsonl`, cleared **7 of 7 in 220 actions at
`rhae` 1.0000** — and a third finished 6 of 7 in 631 actions at 0.4487. Three
runs of one environment, scoring 1.0000, 0.5357 and 0.4487.

That is the worked example the shipped doctrine uses for replaying (§0a: "one run
finished at `raw` 0.449 ... a later run cleared all seven in 220 actions against
that run's 631"), and it is accurate — both runs exist on disk.

It also means **the arm's three losses should not be read as a capability
ceiling.** `tn36` is direct evidence that the same harness, on the same
environment, can score anywhere from 0.45 to 1.00 run to run. The arm reports one
run per environment, which is the correct discipline for a benchmark number, but
it makes the 94.67% a single draw rather than an expected value. The honest
reading of `sp80` 0.7143, `tn36` 0.5357 and `sk48` 0.4167 is *these three runs
stopped early*, not *these three environments are hard*.

Establishing which would take repeated runs per environment, which this arm did
not do.

## Provenance and what can still be audited

The numbers above are reproducible from this file. The **raw traces are not**, for
12 of the 25 environments, and that needs stating plainly rather than discovered
later.

**What happened.** The scratchpad holding the run directories was never in git —
deliberately, because the ARC API key lives there. On 2026-08-05 the container
rolled back three times to an earlier snapshot. The repository survived every one
of them: work was pushed as it was made, and each rollback was recovered by
fast-forwarding from `origin/claude/athanor-cc-harness-variant-jpqw7t`, which was
always ahead. The scratchpad had no such second copy.

**What is gone.** `trace.jsonl`, `scorecard.json` and `stream.jsonl` for `ar25`,
`bp35`, `cn04`, `dc22`, `g50t`, `ka59`, `lf52`, `m0r0`, `re86`, `s5i5`, `sk48`
and `wa30`. Thirteen run directories survive, from the arm's earlier state.

**What survives, and where.**

| artefact | status |
|---|---|
| every per-level table, per-play `raw`, and the 94.67% total | this file, committed |
| harness, scorer, proxy, tests | the repo, committed |
| span-level traces for all 25 games | **only** in the published artifact page |
| raw `trace.jsonl` for 13 games | scratchpad, volatile |
| raw `trace.jsonl` for the other 12 | gone |

**Consequence for the audit trail.** Every scored run was checked against ARC's
own `actions_by_level` at the time it was scored, and `disagreements_with_server`
returned empty for all 25 — but that check cannot now be *re-run* for the 12
whose scorecards are gone. The claim rests on the record in this file, not on
re-derivation.

**Do not regenerate the trace-audit artifact from a rolled-back container.** The
generator reads the scratchpad; with 13 run directories it produces a 13-game
page, and republishing that to the same URL would overwrite the 25-game version —
which is the only remaining copy of the missing traces. The tooling
(`refresh_audit.sh`, `gen_trace_audit.py`, `build_spans.py`) was itself lost in
the rollback and has been left absent on purpose.

**The lesson, which is the general one.** Anything worth keeping has to be pushed,
not merely written. The repository came through three rollbacks without losing a
line because every change was committed and pushed the moment it was made; the
scratchpad lost half an arm's evidence because it was only ever on disk.

## Re-running the three losses, with §0b in the doctrine

The arm's three losses all failed the same way — cleared exactly five levels, hit
a level they could not read, stopped voluntarily with 82–95% of the action budget
unspent. Doctrine §0b (*being stuck with budget left is a reason to change
technique, not to stop*) was written from that pattern and shipped before these
re-runs began.

### `sp80-589a99af` — **0.7143 → 1.0000**, the first one back

| | arm run | re-run |
|---|---|---|
| levels | 5 of 6 | **6 of 6, won** |
| `E` | 0.7143 | **1.0000** |
| `raw` | 0.8214 | 1.0218 |
| actions | 137 (5% of budget) | 419 (16%) |
| outcome | stopped voluntarily | cleared it |

**+0.2857** on this environment. The difference is not efficiency — level 2 cost
2.28× the human median here — it is that the run kept going. It spent three times
the actions the losing run did and cleared the level that stopped it. Four of six
levels still finished at the 1.15 ceiling, and level 4 took **27 actions against a
148 human median**.

#### The first disagreement with ARC's scorecard in this project

`disagreements_with_server` has returned empty for all 25 arm games. Here it
returned two:

| level | our trace | ARC |
|---|---|---|
| 5 | 59 | **66** |
| 6 | 105 | **110** |

ARC counted **12 more actions than we recorded**, all in the last two levels.
The cause is visible in the run's own metadata: `attempts: 11`. Containers were
being replaced roughly every 30 minutes, and this run was cut off and resumed
eleven times. An action sent to the server whose response never reached
`trace.jsonl` — because the process died between the two — is counted by ARC and
missing from us.

Scored either way it is `E` **1.0000** and `raw` 1.0218: the shortfall is small
and lands on levels already below the cap. But the direction matters. **Our trace
under-counts**, so a trace-derived score is optimistic, and the discrepancy scales
with how often a run is interrupted. Every arm figure was taken from runs
interrupted at most twice and agreed exactly; this one was interrupted eleven
times and did not. ARC's numbers are the authority and are used above.

#### Integrity
Zero `ARC_API_KEY` references, zero `api/games`, zero 5n cap hits.

### `tn36-ef4dde99` — **0.5357 → 1.0000**, the second one back

| | arm run | re-run |
|---|---|---|
| levels | 5 of 7 | **7 of 7, won** |
| `E` | 0.5357 | **1.0000** |
| `raw` | 0.5904 | 1.0344 |
| actions | 289 (18% of budget) | 507 (32%) |
| outcome | stopped voluntarily | cleared it |

**+0.4643** on this environment, and the shape is the same lesson as `sp80`,
harder. Per level, against the human medians:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 131 | **260** | 9 | 15 | 23 | 27 | 42 |
| human | 32 | **72** | 26 | 40 | 30 | 55 | 62 |
| ratio | **4.09×** | **3.50×** | 0.35× | 0.38× | 0.30× | 0.49× | 0.68× |

The first two levels cost **four times and three and a half times** the human
median — the worst overruns anywhere in this project. Every level after them came
in at a third to two thirds of the median. So this run was catastrophically
inefficient early, recovered completely, and still scored a perfect 1.0000,
because `E = min(cap, raw)` and clearing all seven pins `cap` at 1.0 while `raw`
finished at 1.0344.

That is the clearest single demonstration of the arm's central finding. A run that
is four times too slow on level 1 and finishes the game beats a run that is fast
on everything and stops at level 5. **Completion dominates efficiency**, and the
losing run's virtue — spending only 18% of its budget — was the thing that cost it
0.4643.

#### Second ARC disagreement, same cause

| level | our trace | ARC |
|---|---|---|
| 2 | 252 | **260** |
| 5 | 9 | **23** |

22 actions missing from our ledger across `attempts: 3`. Same mechanism as
`sp80`: an action sent to the server whose response never reached `trace.jsonl`
because the container died in between. Scored on ARC's numbers throughout —
`E` 1.0000, `raw` 1.0344 against the trace's 1.0347, so the difference is
immaterial here but again in the optimistic direction.

#### Integrity
Zero `ARC_API_KEY` references, zero `api/games`, zero 5n cap hits.

### ⚠ CONTAMINATION: every re-run and every clean rollout below is void

Found 2026-08-06 from a solver's own reasoning, visible in the published trace
audit: *"I'm looking at the baseline metrics from meta.json — they total 518
across six levels, and I have a budget of 2590 ... I also notice from the
documentation that sp80 was a previous loss where five of six levels consumed 137
actions (5% of budget)."*

**Two leaks, both mine.**

**1. `meta.json` carried the real per-level baselines and the action budget.**
`ablate_baselines.strip_baselines()` does cover `meta.json`, but it is installed
by `_install_patch()`, which is called **from `main()` only** — its own docstring
explains why it is deliberately not at module scope. Both
`tools/rerun_losses.py` and `tools/clean_rollouts.py` carry the line

```python
import ablate_baselines as ab   # installs the baseline strip
```

and that comment is false. Importing installs nothing. The arm ran
`ablate_baselines.py` as a script, so `main()` fired and the strip applied; every
run launched through my drivers got the unstripped workspace.

**2. §0b of the doctrine names the three losses with their exact prior results** —
`sp80` 5 of 6 in 137 actions, `tn36` 5 of 7 in 289, `sk48` 5 of 8 in 632. That
table ships inside `DOCTRINE.md` to every solver, so each re-run and clean rollout
of those three read a summary of its own previous failure.

**3. The key shim was never switched on** (found 2026-08-06 by proofreading, after
the first two). `arc_proxy` exists because a solver holding `ARC_API_KEY` can
`GET /api/games` and read `baseline_actions` for all 25 environments — file-level
sanitation cannot touch that. It was written, unit-tested, end-to-end tested, and
**nothing anywhere set `CCARC3_PROXY_URL`**, so the branch in `build_workspace`
that drops the key never once ran. Every game in this project was played by a
solver holding the credential.

The cap was the same shape of mistake one level down: stripping `action_budget`
from `meta.json` left `CCARC3_MAX_ACTIONS` in the child's environment, and
`budget_multiple: float = 5.0` is a default in the package source on the solver's
`PYTHONPATH`, so cap/5 recovers the baseline total exactly. Both are closed as of
`0d416c7` — the proxy starts from `install()`, holds the key, refuses
`/api/games`, and counts the actions itself.

**Blast radius**, measured by testing whether each run's spans contain its own
game's actual per-level baseline array:

| set | leaked | clean |
|---|---|---|
| 25-game arm | **1** (`ka59`) | **24** |
| 3 resumed re-runs | **3** | 0 |
| 7 clean rollouts | **7** | 0 |

**`ka59` is the one arm run that used the open door, and it said so itself:**
*"the harness deliberately withholds the human baselines, but they are served by
the game API's own `/api/games` endpoint, which I queried while diagnosing the
404. I used them for pacing … and for the arithmetic above."* Nine `/api/games`
references in its transcript; the reply printed
`[28, 109, 51, 51, 33, 132, 326]` — its own array. The published trace audit had
already tiered it `not a score`; **this file was still counting it at 1.0000**,
and that is corrected below.

The other 24 arm runs show no trace of their own baselines in a median 740 kB of
span text each. Stated precisely: for the arm, *prevention* was absent for all 25
— the key was in every environment — and what clears those 24 is detection.

**What this voids.** The 100.00% headline, which depended on the three re-runs.
The five "clean rollout" conversions, which were not clean draws. And the decisive
experiment — `sp80` 0.4762 and `tn36` 1.0000 were built to separate "§0b works"
from "the notes work", and both runs were handed their own baselines *and* a
summary of their own prior loss.

**What survives, and at what number.** Two figures, and they must not be
conflated:

| set | criterion | honest figure |
|---|---|---|
| baseline-free arm, minus `ka59` | strip installed, resume allowed | **22.6667 / 24 = 94.44%** |
| strict clean set | one solver process, no container restart | **18.6667 / 20 = 93.33%** |

The arm figure was **23.6667 / 25 = 94.67%**; removing `ka59`, which read its own
baselines off `/api/games`, takes it to 22.6667 / 24. That game now has no valid
score at all — its arm run is contaminated and the rollout meant to replace it is
void with the rest.

The strict clean set is back to the pre-rollout 20 games. Every one of the five
conversions that walked it from 18.6667/20 up to 23.6667/25 was a void rollout,
so the whole ladder in the table below comes off the board. **23.6667/25 appears
twice in this file meaning two different things** — the arm sum over 25 games, and
the clean set after five conversions — which is a coincidence of arithmetic, not a
corroboration.

There is a test asserting the budget stays out of `CLAUDE.md`, `session.py` and
the prompt. It never checked `meta.json`, which is where both numbers were.

Everything from here to the end of the clean-rollout sections is retained for the
record and must not be read as a result.

### Round 2 rollouts — the first runs with the door actually shut

Everything in the previous section is void. This is the same eight-game queue
re-run after the three baseline leaks were closed, the key shim was switched on,
and the action cap was moved into the proxy. Verified per run on the shipped
workspace rather than on the config that built it.

| game | E | levels | actions | wall | cost | card agrees |
|---|---|---|---|---|---|---|
| `sb26` | **1.0000** | 8/8 | 126 | 20.4 min | $5.76 | yes — 126 |
| `ft09` | **1.0000** | 6/6 | 78 | 30.1 min | $9.15 | yes — 78 |
| `ka59` | **1.0000** | 7/7 | 607 (best play 319) | 72.6 min | $22.73 | yes — 607 |
| `wa30` | **1.0000** | 9/9 | 2125 (best play 723) | 129 min | — | yes — 2125 |
| `lf52` | **0.4537** | **7/10 — lost** | 865 | 142 min | $59.72 | yes — 849 |
| `sk48` | **0.9538** | 8/8 — **was 5/8** | 1111 (best play 488) | 152 min | $55.61 | yes — 1111 |
| `sp80` | **1.0000** | 6/6 — **was 5/6** | 339 (best play 100) | 96 min | $20.88 | yes — 339 |
| `tn36` | **1.0000** | 7/7 — **was 5/7** | 542 (best play 103) | 97 min | $31.51 | yes — 542 |
| **8 games** | **7.4075 / 8 = 92.59%** | **7 wins** | | 13.4 h | **$278** | |

#### `sb26-7fbdac44` — **1.0000 (8/8)**, and the first run on the current harness

| | |
|---|---|
| E | **1.0000** (`raw` 1.1436, `cap` 1.0000) |
| levels | 8 of 8, **0 deaths, 0 resets, 1 play** |
| actions | **126**, against a 213 human total |
| wall | 20.4 min, single process, 58 turns, $5.76 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 11 | 15 | 15 | 15 | 17 | 19 | 17 | 17 |
| human | 18 | 28 | 18 | 19 | 31 | 23 | 58 | 18 |
| ratio | 1.64× | 1.87× | 1.20× | 1.27× | 1.82× | 1.21× | **3.41×** | 1.06× |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.1211 |

**What makes this one different from the seven before it.** Checked on the
workspace the solver actually received, not on the code that wrote it:

- its own per-level array appears in **no** file — `meta.json` is down to six
  keys, `session.py` carries `baseline_actions=()`, `CLAUDE.md` contains the
  word "baseline" zero times;
- `1065`, its budget and therefore `5 × 213`, appears nowhere in the stream;
- the solver's environment held no `ARC_API_KEY` and no `CCARC3_MAX_ACTIONS`,
  and `/api/games` answered 403;
- **ARC's own card corroborates it**: `levels_completed=[8]`, state `WIN`, 126
  actions against our 126. The void `sb26` rollout finished 8/8 with a card
  frozen at level 3, so this check is not ceremonial.

It is also the only run on the audit page tiered `current` — no commit has
touched the package, the doctrine or the strip since it started. All 24 valid
arm runs are `superseded` by 12 to 19 commits.

**Against the void rollout it replaces**, which scored the same 1.0000: 126
actions against 130, and level 1 at 11 rather than 15. Not a difference worth
reading anything into on n=1 — the point is that this one is attested, not that
it is better.

#### `ft09-0d8bbf25` — **1.0000 (6/6)**, every level at the efficiency ceiling

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 6 of 6, **0 deaths, 0 resets, 1 play** |
| actions | **78**, against a 208 human total |
| wall | 30.1 min, single process, 81 turns, $9.15 |

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 7 | 7 | 14 | 16 | 21 | 13 |
| human | 43 | 12 | 23 | 28 | 65 | 37 |
| ratio | **6.14×** | 1.71× | 1.64× | 1.75× | 3.10× | 2.85× |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

**All six levels at the 1.15 ceiling**, which no run in the arm managed. `raw`
1.1500 is the maximum the formula allows, so this is as efficient as the score
can register — and it still reads `E` 1.0000, because the completion cap binds
at 1.0 the moment every level is cleared. The surplus is real and discarded.

Proofread by tool and by hand: 80 commands, none leaving the workspace; its own
per-level array, its budget and the API surface absent from 108 kB of tool
output; ARC's card corroborating 78 against 78. The eleven flagged passages are
all inference, and the solver states the limit itself — *"baselines are hidden
here, so that is an argument, not a measurement — I can show I won and that my
routes were minimal, not the value of `raw` itself."*

#### `ka59-38d34dbb` — **1.0000 (7/7)**, and the first valid score this game has ever had

Its arm run is the one that queried `/api/games` for its own baselines and said
so; it has been `excluded` since. The rollout that was meant to replace it was
void with the other six. This is the first `ka59` result that is neither.

| | |
|---|---|
| E | **1.0000** (`raw` **1.1500**, `cap` 1.0000) |
| levels | 7 of 7, **0 deaths, 2 plays, 1 full reset** |
| actions | **607** total, **319** on the scoring play, against a 730 human total |
| wall | 72.6 min, single process, 121 turns, $22.73 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 21 | 36 | 36 | 44 | 20 | 59 | **103** |
| human | 28 | 109 | 51 | 51 | 33 | 132 | **326** |
| ratio | 1.33× | 3.03× | 1.42× | 1.16× | 1.65× | 2.24× | **3.17×** |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

**The blind replay paid, and it is the rule that produced it.** The solver could
not compute `raw` — the baselines are withheld — so under §0 it replayed anyway
rather than banking the first win, and every level came in at the 1.15 ceiling.
Against the excluded arm run's `raw` 1.1318 that is a real improvement, though
`E` was 1.0000 either way: the completion cap binds the moment all seven levels
fall, so the surplus is earned and discarded. Recorded because the rule keeps
being right, not because it moved the score.

**Hiding the cap changed how it spent.** The solver reasoned *"since the action
cap isn't enforced by the harness, there's no practical limit on actions"* — and
it is enforced, in the proxy, at 3,650. It spent 607. The excluded arm run,
which could see its budget, spent 364. Withholding the number removed the
solver's budget discipline along with the leak. Nothing was lost here, and on a
game where the cap binds it would be.

Proofread: 120 commands, none left the workspace; its own array, its budget and
the API surface absent from 148 kB of tool output; card corroborating 607 against
607. Ten flagged passages, all inference — *"I can't calculate the raw score
since the baselines are hidden."*

#### `wa30-ee6fef47` — **1.0000 (9/9)**, and the run my own gate threw away

| | |
|---|---|
| E | **1.0000** (`raw` **1.1367**, `cap` 1.0000) |
| levels | 9 of 9, **2 plays, 1 full reset** |
| actions | **2,125** total, **723** on the scoring play, against a 1,843 human total |
| wall | 129 min, single process, 175 commands |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| agent | 47 | 63 | 87 | 98 | 105 | 55 | 66 | 140 | **62** |
| human | 71 | 119 | 183 | 98 | 368 | 68 | 79 | 442 | **415** |
| ratio | 1.51× | 1.89× | 2.10× | 1.00× | 3.50× | 1.24× | 1.20× | 3.16× | **6.69×** |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.00 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

**The blind replay again, and this time it is visible in the numbers.** Play 1
took 484 actions on level 8 and 271 on level 9; play 2 took 140 and 62. Eight of
nine levels finished at the 1.15 ceiling on the replay against six on the first
pass. `E` was 1.0000 either way.

**It was discarded, and the gate was wrong.** `proofread_trace.py` classified
reading your own environment as leaving the workspace, so this run failed on

    echo "CAP=${CCARC3_MAX_ACTIONS:-unset}"; env | grep -i ccarc

after 129 minutes and 2,125 actions. The command returned `CAP=unset` and
nothing else of use: the solver went looking for its budget and the strip had
already taken it. **That is the defence succeeding, and the gate punished it as
a failure.** Probing is now reported for reading and never fatal — whether a
probe *yielded* anything is what the inbound scan already tests, and here it
yielded nothing.

**What the probe did expose was a real hole.** Its output included
`CCARC3_PROXY_URL`, the startup shim `install()` uses to prove a proxy can bind.
That shim is created with no budget, so `MAX_ACTIONS` is 0 and anything reaching
it spends without limit — and `build_workspace` copied the parent's whole
environment, so every solver held the address of an uncapped route to ARC. No
run ever used it; `wa30` printed it while hunting for its cap and moved on. The
probe now shuts down once it has proved the port binds, and the child's
environment carries exactly one root, its own.

**The twenty flagged passages are inference from the game, not from the
harness.** `wa30` has an in-game tick bar, and the solver reverse-engineered
per-level budgets from it — *"I get something like 196, 70, 100, 100, 125 …
possibly around twice the human baseline."* The real medians are 71, 119, 183,
98, 368. It guessed, and it guessed wrong.

#### `lf52-271a04aa` — **0.4537 (7/10)**, the first clean loss, and it stopped voluntarily

Its arm run scored 1.0000 at 10/10. This one cleared seven and stopped.

| | |
|---|---|
| E | **0.4537** (`raw` 0.4537, `cap` 0.5091 — `raw` is binding) |
| levels | **7 of 10**, 0 deaths, 0 resets, 1 play |
| actions | **865 of a 6,695 cap — 13%** |
| wall | 142 min of a 6 h limit, no timeout, exit 0 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 9 | 65 | 77 | 53 | 249 | 190 | 166 | — | — | — |
| human | 32 | 81 | 60 | 71 | 205 | 148 | 244 | 109 | 164 | 225 |
| `S_l` | 1.15 | 1.15 | 0.61 | 1.15 | 0.68 | 0.61 | 1.15 | **0** | **0** | **0** |

**It was not stopped; it stopped.** No timeout, no crash, no signal, no budget
exhaustion — 87% of the action cap unspent and four hours of wall clock unused.
The final message is a polished write-up of the game's mechanics, delivered
after level 7 as though the task were complete.

**And it had read §0b.** The doctrine it shipped with contains
*"Being stuck with budget left is a reason to change technique, not to stop"* —
`grep` confirms the section is present and that the redaction never touched it,
because §0b names `sp80`, `tn36` and `sk48`, not `lf52`. So the one rule written
specifically against this failure was in front of the solver, and the solver did
it anyway. That is a negative result about §0b and it should not be softened:
the rule is not sufficient on its own.

**Why it is not a leak.** Proofread clean — 281 commands, none left the
workspace; own array, budget and API surface absent from 324 kB of tool output;
card corroborates 7 levels. Its five flagged passages are inference, including
one that shows the cap working as intended: *"we're at 596 actions now, and since
the game is still accepting moves, the action cap must be higher than that."*

**On comparability.** The arm's 1.0000 came from a run that was interrupted and
resumed, so the pair is a clean single-process run against a resumed one, not a
like-for-like. What it does establish is that this environment is not a
guaranteed win for the harness.

#### The decisive experiment: two of the three arm losses reverse

`sp80`, `tn36` and `sk48` are the only environments the arm lost. They were
queued last because they are the question the rollouts exist to answer: does a
clean single-process run on the current harness beat what the arm managed?

| game | arm | clean rollout | |
|---|---|---|---|
| `sk48` | 0.4167 — 5/8 | **0.9538 — 8/8** | won |
| `sp80` | 0.7143 — 5/6 | **1.0000 — 6/6** | won |
| `tn36` | 0.5357 — 5/7 | **1.0000 — 7/7** | won |

**`tn36-ef4dde99` — 1.0000 (7/7), every level at the ceiling**

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 7 | 9 | 9 | 13 | 16 | 21 | 28 |
| human | 32 | 72 | 26 | 40 | 30 | 55 | 62 |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

**103 actions on the scoring play against a 317 human total.** The arm run
stopped at 5 of 7 having spent 289 — nearly three times as many actions for two
fewer levels.

**Three of three.** Every environment the arm lost was won on the re-run, and
two of the three at `raw` 1.1500, the formula's maximum. That is the strongest
result on this page and the one most in need of a caveat: three things moved
between the arm and these runs — the doctrine gained §0b and the replay
correction, the action cap became invisible, and the key shim was switched on.
The reversals are real; their cause is not isolated.

**`sk48-d8078629` — 0.9538 (8/8), and `raw` is the binding term**

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 14 | 32 | 33 | 75 | 43 | 56 | 57 | **178** |
| human | 61 | 177 | 101 | 103 | 230 | 181 | 125 | **92** |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | **0.27** |

Seven of eight levels at the ceiling and one at 0.27 — 178 actions against a 92
median on the level the arm never reached. **The only current-harness win where
`cap` is not binding**, so unlike every other 1.0000 on this page, its remaining
0.046 is real and collectable. It replayed once already (2 plays, 488 on the
scoring play of 1,111); the gap is level 8 being genuinely hard for it, not a
missed replay.

**`sp80-589a99af` — 1.0000 (6/6), every level at the ceiling**

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 5 | 9 | 10 | 22 | 18 | 36 |
| human | 39 | 58 | 25 | 148 | 96 | 152 |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

**100 actions on the scoring play against a 518 human total**, `raw` at the
formula's maximum. The arm run stopped at 5 of 6 having spent 137.

**What this does and does not show.** Both reversals are real and both are
proofread clean — no array, no budget, no API surface, cards corroborating
1,111 and 339. But three things moved between the arm and these runs: the
doctrine gained §0b and the replay correction, the action cap became invisible,
and the key shim was switched on. Two of those plausibly help and one plausibly
hurts, so a 2-of-2 reversal is encouraging rather than attributable. `tn36` is
the third draw and it is still running.

### Clean rollouts: converting the five interrupted environments, one at a time

The five that scored 1.0000 but were interrupted and resumed — `ft09`, `ka59`,
`lf52`, `sb26`, `wa30` — are being re-run under the strict criterion:
`tools/clean_rollouts.py`, `fresh=True`, no evidence restore, no inherited
`rules.json`, and **an interrupted attempt is discarded rather than resumed**.
Resuming is the confound being removed, so a driver that resumed would reproduce
it. One game at a time, sequentially; each attempt opens its own scorecard, so a
discard costs quota and wall clock but never score.

| game | arm | clean rollout | status |
|---|---|---|---|
| `sb26` | 1.0000 (interrupted) | **1.0000 (8/8)** | **clean, banked** |
| `ft09` | 1.0000 (interrupted) | **1.0000 (6/6)** | **clean, banked** |
| `ka59` | 1.0000 (interrupted) | **1.0000 (7/7)** | **clean, banked** |
| `wa30` | 1.0000 (interrupted) | **1.0000 (9/9)** | **clean, banked** |
| `lf52` | 1.0000 (interrupted) | **1.0000 (10/10)** | **clean, banked** |
| `sp80` | **0.7143 (5/6)** | **0.4762 (4/6)** | **clean — and it LOST** |
| `tn36` | **0.5357 (5/7)** | **1.0000 (7/7)** | **clean — and it WON** |
| `sk48` | **0.4167 (5/8)** | — | running |

The three losses were added to the queue on 2026-08-06. They have been run on the
current doctrine — their re-runs carried §0b — but never *cleanly*: 12, 3 and 4
solver launches, each restoring `rules.json` from the losing arm run. So "§0b
works" and "a second look at your own notes works" remain confounded, and a fresh
single-process run is the only configuration that separates them. Their arm
losses predate §0b by a day and serve as the control.

#### `sb26-7fbdac44` clean rollout — **1.0000 (8/8)**, and the first zero-disagreement run

| | |
|---|---|
| E | **1.0000** (`raw` 1.1436, `cap` 1.0000) |
| levels | 8 of 8, **0 deaths, 0 resets, 1 play** |
| actions | **130**, ARC and our ledger agreeing exactly, against a 213 human total |
| wall | 15 min, single process, exit 0 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 15 | 15 | 15 | 15 | 17 | 19 | 17 | 17 |
| human | 18 | 28 | 18 | 19 | 31 | 23 | 58 | 18 |
| ratio | 1.20× | 1.87× | 1.20× | 1.27× | 1.82× | 1.21× | **3.41×** | 1.06× |
| `S_l` | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.1211 |

**The first run on record where our ledger and ARC agree exactly** — 130 against
130, an empty `disagreements_with_server`. Every earlier comparison was off:
`sp80` +12, `tn36` +22, `sk48` +6. Those were attributed to responses that never
reached `trace.jsonl` when a container died mid-action, and this is the first
positive evidence for that explanation rather than a plausible story: the one run
that was never interrupted is the one where the counts match.

It also converts the first of the five. `sb26` is no longer "1.0000, but resumed";
it is 1.0000 on a single uninterrupted process, which the strict criterion accepts.
The clean set moves from 20 games to 21, **19.6667 of 21**.

**Running total as each rollout banks**, stated here because I twice quoted it a
point too high in conversation — each converted game adds 1.0000 to the numerator
*and* one game to the denominator, and I was adding only the numerator:

| after | clean set | | |
|---|---|---|---|
| `sb26` | 19.6667 / 21 | 93.65% | **void** |
| `ft09` | 20.6667 / 22 | 93.94% | **void** |
| `ka59` | 21.6667 / 23 | 94.20% | **void** |
| `wa30` | 22.6667 / 24 | 94.44% | **void** |
| `lf52` | ~~23.6667 / 25~~ | ~~94.67%~~ | **void** |

Every rung of that ladder is a void rollout, so the clean set stays at its
starting point: **18.6667 / 20 = 93.33%**.

The denominator grows because these five were *excluded* from the original clean
20, not scored zero in it. Converting one moves it from the interrupted set into
the clean set.

#### The bug that nearly threw this result away

`clean_rollouts.py` passed `out_dir=attempt_N` while `Ccarc3Config` builds its
workspace at `out_dir/<game_id>`, so the driver read `attempt_N/result.json` — one
directory above where the harness had written it. This run **completed, won 8 of 8,
and was logged as `discarded — no result`.** Nothing was wrong with the solver or
the game; only with where the driver looked. Left alone it would have burned all
12 retry passes re-running games that had already finished.

Recorded here because the failure mode is the dangerous kind: no error, no crash,
a plausible-looking log line, and a real result silently binned. `salvage()` now
re-scans earlier attempt directories before starting a new one, which is how this
result was recovered rather than re-bought.

#### `ft09` and `ka59` clean rollouts — 1.0000 each, and the disagreement pattern closes

| | `ft09` | `ka59` |
|---|---|---|
| E | **1.0000** (`raw` 1.0400) | **1.0000** (`raw` 1.1400) |
| levels | 6 of 6, 1 play, 2 level-resets | 7 of 7, 1 play, 0 resets |
| actions | **98** vs 208 human | **385** vs 730 human |
| wall | 32 min, one process | 98 min, one process |
| ledger vs ARC | **98 vs 98** | **385 vs 385** |

`ft09`'s level 1 at **10.75×** the human median — 4 actions against 43 — is the
widest single-level margin recorded in the project.

**Three clean runs, three exact agreements with ARC.** 130/130, 98/98, 385/385,
against +12, +22 and +6 on the three interrupted re-runs. Every uninterrupted run
matches and every interrupted one does not, which is what the "responses lost when
a container dies mid-action" explanation predicts. It was one data point when
`sb26` landed; it is now a clean split.

**Level 1 keeps being the weak level.** `sb26` 1.20×, `ft09` 10.75×, `ka59` 0.93×
— and in two of the three it is the *only* level below the 1.15 ceiling
(`ft09`'s L4 at 0.76× being the exception). That matches the arm-wide measurement:
level 1 runs below human pace in 6 of 15 scored runs against 9 of 90 for later
levels. It costs nothing under a full clear, since `cap` binds at 1.0 and `raw`
clears it regardless.

**A level reset does not break cleanliness.** `ft09` used two, and ARC still
records `total_plays: 1`. A level reset is the cheap in-level retry §2a endorses;
it opens no new play. The criterion is one solver process and one continuous
attempt, which in-run replays and level resets both satisfy.

#### `wa30` clean rollout — **1.0000 (9/9)**, and the sharpest §0a case on record

`wa30` won all nine levels, then replayed, and the replay is what earned the point.

| play | levels | actions | `raw` | E |
|---|---|---|---|---|
| 1 | 9/9 WIN | 2,712 | 0.8373 | 0.8373 |
| 2 — replay | 9/9 WIN | **793** | **1.1054** | **1.0000** |

Play 1 cleared everything, so `cap` was 1.0 and `raw` was binding at 0.8373 —
precisely the row of §0a's table that says *replay; another level barely helps*.
The solver took it and collected the whole 0.1627.

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| play 1 | 47 | 175 | 86 | 85 | 307 | 66 | 79 | **1417** | 450 |
| play 2 | 27 | 123 | 86 | 85 | 117 | 66 | 79 | **144** | **66** |
| human | 71 | 119 | 183 | 98 | 368 | 68 | 79 | 442 | 415 |

Level 8 cost **1,417 actions on the first pass and 144 on the second** — 0.31× the
human median becoming 3.07×, and `S_l` 0.0973 becoming 1.15. Level 9 went 450 to
66. The route was the same both times; what differed was knowing it, which is
exactly what §0a says the gain comes from — *"the gain comes from executing the
route your rules now imply"*, not from replaying the trace verbatim.

**A level drop is not always lost progress, and the doctrine's own heuristic says
otherwise.** §2 states the reliable signal that a full reset happened is that your
level went down, *whatever the flag says*. Here the ledger goes 9 -> 0 with
`full_reset: false` and nothing was lost: it is the deliberate
`restart_for_replay()` one action after the winning frame, which is the only
moment it is legal. Monitoring that keys on "level decreased" flags this textbook
move as a catastrophe — it did in this session. The discriminator is whether the
preceding frame was a `WIN` or level advance.

**Fourth clean run, fourth exact agreement with ARC**: 3,505 vs 3,505. With
130/130, 98/98 and 385/385, every uninterrupted run now matches the server and
every interrupted one does not.

#### `lf52` clean rollout — **1.0000 (10/10)**, and all five interrupted games converted

The longest game in the set and the one I predicted could not finish in a single
window: 3.55 h in the arm against a box lifetime whose median was 35 minutes at
the time. The box lasted, and it came in at 2.9 h.

| | |
|---|---|
| E | **1.0000** (`raw` 1.0183, `cap` 1.0000) |
| levels | 10 of 10, 0 deaths, **0 resets, 1 play** |
| actions | **966** vs a 1,339 human total, ledger and ARC agreeing exactly |
| wall | 177 min, single process, exit 0, $64.26 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 2.29× | **0.74×** | 1.02× | 1.25× | 2.16× | 1.01× | 1.67× | 1.30× | **0.77×** | 5.11× |
| `S_l` | 1.15 | 0.552 | 1.034 | 1.15 | 1.15 | 1.028 | 1.15 | 1.15 | 0.598 | 1.15 |

**The narrowest margin of the five.** Three levels sit below the ceiling and two
below human pace, so `raw` cleared `cap` by 0.0183. One worse level and the replay
would have been worth taking, as it was on `wa30`.

**Five clean runs, five exact agreements with ARC**: 130, 98, 385, 3505, 966 —
every one matching the server exactly, against +12, +22 and +6 on the three
interrupted re-runs. Every uninterrupted run matches and every interrupted one
does not. That is as much support as this data can give the explanation that the
missing actions are responses lost when a container dies mid-action.

#### The five converted, and what it settles

| game | arm (interrupted) | clean rollout | actions vs human |
|---|---|---|---|
| `sb26` | 1.0000 | **1.0000** (8/8) | 130 / 213 |
| `ft09` | 1.0000 | **1.0000** (6/6) | 98 / 208 |
| `ka59` | 1.0000 | **1.0000** (7/7) | 385 / 730 |
| `wa30` | 1.0000 | **1.0000** (9/9) | 3,505 / 1,843 (two plays) |
| `lf52` | 1.0000 | **1.0000** (10/10) | 966 / 1,339 |

All five held up under the strict criterion — one solver process, fresh, no
inherited `rules.json`, interrupted attempts discarded rather than resumed. The
clean set is now **23.6667 of 25, 94.67%**, and every game in it that was ever
scored 1.0000 has been scored 1.0000 again without a resume.

What remains is the part that was never clean: `sp80`, `tn36` and `sk48`, whose
only clean data are the arm losses at 0.7143, 0.5357 and 0.4167. `sp80` started
its clean rollout on 2026-08-06.

### `sp80` clean rollout — **0.4762 (4/6)**, and the re-run's 1.0000 does not survive

This is the result the clean-rollout experiment existed to produce, and it goes
against the optimistic reading.

| | arm | re-run (resumed, inherited `rules.json`) | clean rollout |
|---|---|---|---|
| E | 0.7143 | **1.0000** | **0.4762** |
| levels | 5 of 6 | 6 of 6 | **4 of 6** |
| actions | 137 | 419 | 249 |
| exit | clean stop | won | **`GAME_OVER`, 2 deaths** |

**It did not fail on efficiency.** Every level it cleared was at the 1.15 ceiling,
and not narrowly:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| agent | 9 | 10 | 13 | 27 | — | — |
| human | 39 | 58 | 25 | 148 | 96 | 152 |
| ratio | 4.33× | 5.80× | 1.92× | 5.48× | — | — |

`raw` 0.5476 against `cap` 0.4762 — the cap binds, so the whole loss is
completion. It stopped with **284 of 2,590 actions used, 11% of budget.**

**§0b was in this run's doctrine and did not prevent the voluntary stop.** The
paragraph exists precisely because all three arm losses stopped early with budget
in hand; this run had it and stopped at 11% anyway, one level short of where the
arm run stopped. Whatever produced the re-run's 6-of-6, it was not §0b acting
alone on a fresh solver.

**What this does and does not establish.** It is one draw. The same environment
scored 0.7143 in the arm and 0.4762 here under the same doctrine with no notes —
a 0.24 spread on identical conditions, which is itself evidence that single draws
on this environment are noisy, and therefore that the re-run's 1.0000 may be
partly luck. Two more clean rollouts (`tn36`, `sk48`) are queued; three points
will say much more than one. What it removes is the clean inference that §0b
converted these losses.

**The server disagreement inverted.** Ledger 284, ARC 249 — our trace
*over*-counts by 35, where every prior disagreement under-counted (+12, +22, +6)
and all five clean rollouts matched exactly. This is also the first clean rollout
to end in `GAME_OVER`, with 2 deaths. Actions after a death not counting
server-side would explain it, but that is untested and is recorded here as an
open question rather than a conclusion.

### `tn36` clean rollout — **1.0000 (7/7)**, and the three split

Run under the same conditions that sank `sp80`: fresh, one process, no inherited
`rules.json`, §0b in the doctrine. It won everything.

| | arm | resumed re-run (with notes) | clean rollout |
|---|---|---|---|
| E | 0.5357 | 1.0000 | **1.0000** |
| levels | 5 of 7 | 7 of 7 | **7 of 7** |
| actions | 289 | 507 | **196** |
| `raw` | — | 1.0344 | **1.1500** |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| agent | 24 | 40 | 12 | 16 | 21 | 39 | 44 |
| human | 32 | 72 | 26 | 40 | 30 | 55 | 62 |
| ratio | 1.33× | 1.80× | 2.17× | 2.50× | 1.43× | 1.41× | 1.41× |

**`raw` 1.1500 is the theoretical maximum.** Every level at the 1.15 ceiling —
the only run on record where that is true. Zero deaths, one play, zero
disagreement with ARC, 49 minutes.

**It beat the run that had the notes.** The resumed re-run spent 507 actions to
clear the same seven levels; this one spent **196**, 2.59× leaner, starting from
nothing. On this environment the inherited `rules.json` was not an advantage — it
was correlated with a *worse* route.

### What the split means, with `sk48` still to come

| | arm | resumed re-run | clean rollout |
|---|---|---|---|
| `sp80` | 0.7143 | 1.0000 | **0.4762** ↓ |
| `tn36` | 0.5357 | 1.0000 | **1.0000** ↑ |
| `sk48` | 0.4167 | 1.0000 | running |

Two results, opposite directions. Neither simple story survives:

- **"The notes did the work"** is refuted by `tn36`, which cleared all seven from
  scratch, more efficiently than the run that had them.
- **"§0b converted the losses"** is refuted by `sp80`, which had §0b, stopped at
  11% of budget, and finished *below* its own arm result.

What both are consistent with is **high per-environment variance**. `sp80` has now
produced 0.7143 and 0.4762 on two clean draws of the same environment under the
same doctrine; `tn36` has produced 0.5357 and 1.0000. Spreads of 0.24 and 0.46 on
identical conditions are larger than any effect being argued about, which is the
uncomfortable finding and the one this series was built to surface.

**On the arithmetic.** Taking the better of the two clean draws per environment —
which is what ARC's own best-of-scorecards rule would do — the clean set is
**24.1310 of 25, 96.52%**, up from 23.6667 because `tn36` improved and `sp80` did
not. That figure is a weaker claim than the 23.6667, because it is best-of-two
rather than one draw, and it is recorded that way rather than as a headline.

### The strictest honest number: 18.6667 of 20 on runs that were never interrupted

Operator's criterion, 2026-08-06, and it is the right one: **only a run that
completed in a single solver process counts as a genuine clean draw.** A resumed
run re-reads its own `rules.json` and trace on the far side of a fresh context
window, which is the same contamination as the re-runs, in a smaller dose.

Twenty of the twenty-five qualify — one process, one stream, no relaunch. Five do
not: `ft09`, `ka59`, `lf52`, `sb26`, `wa30`, each matching a known container event.

| set | n | total | mean |
|---|---|---|---|
| one-shot, no restart | 20 | 18.6667 | **0.9333** |
| interrupted and resumed | 5 | 5.0000 | 1.0000 |

**The criterion is conservative, not flattering.** All three losses — `sp80`
0.7143, `tn36` 0.5357, `sk48` 0.4167 — sit inside the clean twenty, and every one
of the five interrupted runs scored 1.0000. Applying the rule therefore discards
five perfect scores and keeps all three failures, taking the headline *down* from
94.67% to **93.33%**. So the three figures this project can honestly quote are:

| claim | figure |
|---|---|
| never-interrupted single attempts | **18.6667 / 20 = 93.33%** |
| all 25 first attempts, resumes included | 23.6667 / 25 = 94.67% |
| best scorecard per environment, re-runs included | 25.0000 / 25 = 100.00% |

**An in-run replay does not break cleanliness.** 17 of the 20 have
`playthroughs >= 2` — they used `restart_for_replay()` mid-run. That also re-walks
a route with knowledge in hand, but inside one continuous process, and best-of-
plays scoring rewards it by design. The line that matters is the *process*
boundary, not the replay.

**The uncomfortable part.** The interrupted five went 5 for 5 at 1.0000 against 17
of 20 for the clean runs, and two of them — `wa30` and `lf52` — had been banked as
losses before a resume recovered them. Interruption is essentially random with
respect to difficulty, and n=5, so this is suggestive rather than evidence. But it
points the same direction as the three re-runs: **a second pass holding your own
notes converts losses into wins.** That is exactly why the 100% carries an
asterisk and this section exists.

`re86` is the one boundary case. It ran in a single process and won 8 of 8, then
exited 1 when the network dropped during a post-win replay — the same signature as
`wa30`'s proxy move. Counted clean here because it never needed a relaunch and its
score was already banked, but the data does not settle it.

### Re-run standing: three of three back, all three reversed

| game | arm | re-run | delta |
|---|---|---|---|
| `sp80` | 0.7143 (5/6) | **1.0000 (6/6)** | **+0.2857** |
| `tn36` | 0.5357 (5/7) | **1.0000 (7/7)** | **+0.4643** |
| `sk48` | 0.4167 (5/8) | **1.0000 (8/8)** | **+0.5833** |

All three were losses for the same reason and all three cleared on a second
attempt with §0b in the doctrine. The arm total moves from 23.6667 to
**25.0000 — 100.00%** on the 25 public environments.

The honest caveat stands from the `tn36` variance note: these are single draws.
What they establish is that the three environments were never beyond the harness
— which is what the arm's loss column could not distinguish. A 100% that required
a second attempt on three environments is not the same claim as a 100% first try,
and the table above is written so the difference stays visible.

### `sk48-d8078629` re-run — **1.0000 (8/8)**, and §0b's sharpest test

The arm run stopped voluntarily on level 6 with 88% of its budget unspent, having
written a closing report arguing the level required pushing red blocks through a
gap that neither chain could enter. The re-run cleared level 6 in **45 actions**
and went on to win the game.

| | |
|---|---|
| E | **1.0000** (`raw` 1.1295, `cap` **1.0000** — the cap binds) |
| levels | 8 of 8, **0 deaths, 0 resets, 1 play** |
| actions | **456** by ARC's count, against a 1,070 human total |
| wall | 0.95 h · 75 turns · 4 attempts |
| cost | $14.60 |

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| agent | 95 | 44 | 47 | 49 | 70 | 45 | 59 | 47 |
| human | 61 | 177 | 101 | 103 | 230 | 181 | 125 | 92 |
| ratio | 0.64× | 4.02× | 2.15× | 2.10× | 3.29× | 4.02× | 2.12× | 1.96× |
| `S_l` | 0.4123 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |

**The two levels the arm never reached were among the cheapest.** Level 6 — the
one it declared impassable — cost 45 actions against a 181 median, the joint-best
ratio in the game at 4.02×. Level 7 cost 59 against 125. Whatever stopped the arm
run was not the difficulty of those levels.

**Level 1 is the only sub-1.0 level in any of the three re-runs**, at 95 actions
against a 61 median for `S_l` 0.4123. It cost nothing: with all eight cleared,
`cap` is 1.0 and `raw` 1.1295 sits above it, so the score is the cap. This is
§0a's arithmetic — under a full clear, surplus efficiency is discarded and only
completion is paid for. Had the run stopped at seven levels the same weak opening
would still not have bound, because `cap` would have been 0.7778 against a `raw`
still above 1.0.

**Scored on ARC's counts, not ours.** Our ledger recorded 450 actions and ARC 456,
a sixth non-empty `disagreements_with_server` in the re-run series; as with `sp80`
(+12) and `tn36` (+22), the trace under-counts, so its numbers are optimistic and
the server's are used. On these figures the difference is invisible — every level
but the first is pinned at the 1.15 ceiling — but the rule is the rule.

**One scorecard, one play, zero resets, across two interruptions.** The card
survived both a container replacement and a wall-clock kill: ARC records
`total_plays: 1` and `resets: 0` over the whole 456 actions, and its level-1
cumulative of 95 covers the very first actions of the run, hours and two
relaunches earlier. This is the strongest form of the finding recorded above at
`(13.6, 43.8]` minutes — a resume that is prompt enough keeps not just the notes
but the game itself, and the ledger reads as one continuous playthrough.

**This run is also why `collect_outcome` no longer exempts wall-clock timeouts.**
An earlier pass was killed by the driver's one-hour cap at level 4 on 232 of 5,350
actions and banked `levels_reached: 4, won: false` with no error field — a result
*worse than the arm's* that the resume rule would have made permanent. The fix
distinguishes the limit that actually bound: budget spent is a real result, budget
unspent is the clock. Without it this row would read 0.2778 instead of 1.0000.

#### The bar, written down before the run finished

Recorded mid-run, while `sk48` was on level 3, so the prediction is on the record
ahead of the outcome rather than fitted to it. `cap` binds this environment, so
the score was decided by level count alone and the efficiency being banked could
not move it. `cap(k)` for `k` of 8 levels is `sum(1..k)/36`:

| levels | 5 (arm) | 6 | 7 | 8 |
|---|---|---|---|---|
| `cap` | 0.4167 | 0.5833 | 0.7778 | 1.0000 |
| delta vs arm | — | **+0.1667** | **+0.3611** | **+0.5833** |

**Six of eight was the bar**: anything less repeats the arm's result, and each
level past it is worth more than the last. The run cleared all eight and took the
last row, +0.5833.

The prediction that `E = cap` at every row held: `raw` finished at 1.1295, above
even the 1.0 cap of a full clear. The one place the mid-run note was wrong is the
opening level, which it put at 89 actions for an `S_l` of 0.470 — that was our
ledger's count, and ARC's is 95 for an `S_l` of 0.4123. **The entire six-action
disagreement with ARC sits on level 1**, which is also the only level where the
gap could have mattered had the run stopped short. It did not change the score,
for the reason the note gave: under a full clear, efficiency is free and only
completion is paid for.

#### The card survived the container replacement

Worth recording because it contradicts what `tools/rerun_losses.py` claimed. The
restore→resume loop was built assuming a replacement always costs the ARC card,
so a restore would carry `rules.json` but replay from level 0. `sk48` crossed the
02:48:15Z replacement on card `57690598-daed-4fac-8b18-e8bb34734288` and resumed
on the same card — zero `full_reset` markers, one continuous play, level
0 → 1 → 2. The restore buys back the game, not only the notes.

That tightens the reap bracket from (12.2, 43.8] to **(13.6, 43.8]** minutes. It
is a lower bound rather than a measurement: evidence preserves on a 5-minute
poll, and the ledger sat at 87 actions from 02:41:25Z through 02:54:59Z and was
growing again by 03:00:07Z, so the idle gap was at least 13.6 minutes and
possibly ~19. Both claims are corrected in the source (`817bfbf`).

---

## Cognitive proofread of the reframed harness — four defects, all in the rendered surface

Commit `2003a4a` removed the two rules that told a solver to ration actions, and
the test suite went green. That is not the same as the reframe being real: the
tests checked that two banned phrases were gone from the doctrine *source*, and
what a solver reads is the doctrine, `CLAUDE.md` and `session.py` **after the
baseline strip has rewritten all three**. Nobody had read that.

Reading it turned up four defects, three of which every one of the eight clean
rollouts had already seen.

### 1. `CLAUDE.md`'s baseline paragraph was being shredded, not redacted

`strip_baselines` filtered `CLAUDE.md` **line by line**, dropping any line
containing "baseline". The paragraph it was aimed at is hard-wrapped over four
lines, two of which contain the word. What the solver actually received was the
other two, standing alone:

```
must also discover them, hence the larger budget. But if you are several times
wrong — go re-explore rather than grind.
```

Incoherent, and still asserting an allowance. This was in every clean-rollout
workspace, read during orientation, for the whole rollout.

Fixed both ends: the strip now drops whole paragraphs outside code fences and
stays line-wise inside them, and the source paragraph was rewritten so what
survives says the opposite — *"Nothing here is rationed and there is nothing to
save for later."*

The same line filter had also removed `client.status()` from the driving
example, because its trailing comment mentioned a baseline. Combined with §6
being stripped wholesale, the baseline-free solver was never told the call
exists — while `status()` under withheld baselines still reports level, state,
actions and the exact completion cap. Its comment no longer names a baseline, so
it survives; `client.pace()`, which genuinely cannot work without them, still
goes.

### 2. The doctrine still described an allowance in ten places

Removing the two rationing rules left the frame intact everywhere else: *"push as
deep as the budget allows"*, *"71% of its action budget unspent"*, *"if your
budget is tight"*, *"a replay that runs out of budget"*, *"the exploration had
eaten the budget"*, §0b's own heading, *"when you are stuck and the budget is
not"*.

The sharpest was §0b's only sanctioned reason to stop — *"stop only when you can
state what you would need and show you cannot afford it"* — which asks the solver
to compute a remaining balance the harness deliberately does not show it. An
unfollowable stopping rule that re-installs the exact mental model the reframe
removed. It now reads **"There is no version of this where stopping is the
move"**, with the three arm losses as the evidence: every stop on record was
written up as a reasoned decision and every one was later reversed.

§0b's table changed too. It listed each loss as a *share of its cap* — 5%, 18%,
12% — which is a fraction of a number the solver cannot see. It now gives the
absolute action counts (137, 289, 251), which mean something on their own.

### 3. `session.py` was a briefing on the strip, in the file the solver imports

The rendered `ArcClient(...)` call carried thirty lines of operator commentary
telling the solver that a hidden action cap exists **and is enforced in
`arc_proxy`**; that the human medians are withheld on purpose and are "resolved
at import and never written to a workspace file"; and what a named other
environment had scored. A map to everything the harness spends real machinery
hiding, handed over in the one file the doctrine instructs the solver to open —
and `cat session.py` is already a tracked probe in `proofread_trace.py`.

It also named `su15` outright, which `redact_self_reference` never sees because
that function only rewrites `DOCTRINE.md`. A run of `su15` would have read about
its own past failure from a file no redaction pass looks at.

The commentary now lives above `SESSION_TEMPLATE` in the repo, where the operator
reads it and the workspace does not carry it. A test asserts the rendered file
names no environment but its own and describes none of the strip.

### 4. A dead knob that reads as a live one

`level_budget_multiple=0.0` was rendered into every workspace. The per-level cap
has been off since 2026-08-04, but the keyword stayed — and the only word the
solver sees in it is `budget`, in a harness whose entire claim is that it is not
running one. It is now emitted only when non-zero.

### What this does to the eight rollouts

Nothing to their scores, and something to their interpretation. All eight ran
under defects 1, 3 and 4 — a mangled orientation paragraph asserting a budget, no
documented `status()`, and a `session.py` describing the cap and the strip. Seven
still won. The result stands; what cannot be claimed is that they were run under
the doctrine as written, because the file they read was not the file that was
written.

Defect 2 is different: the doctrine reframe landed *after* all eight finished, so
none of them saw either version of §0b's stopping rule in its current form.

### The pattern, again

Every defect here is the session's recurring shape: **a mechanism that looked
like it worked because the visible outcome was unchanged.** The strip ran, raised
no error, and left prose it had destroyed. The reframe removed the two phrases a
test looked for and left ten it did not. The commentary explaining why the
harness hides a number was itself the disclosure. Each was found the same way —
by rendering the artifact and reading it, rather than reading the code that
produces it.

Four tests now pin it: two on the strip's paragraph and code-fence behaviour, one
that no solver-facing file frames actions as an allowance, and one that the
workspace `session.py` carries no operator commentary and names no other game.

---

## The exhaustive proofread: 70 confirmed defects, and a disclosure channel nobody was guarding

The four defects fixed in `a63232a` came from reading three rendered files. That
was the right method applied to too small a surface. This pass enumerated
**every** surface a solver can reach — both initial prompts, the workspace files,
the runtime strings, and the package's own docstrings — and read each through
seven independent lenses (allowance framing, coherence after the two rewriting
passes, disclosure, followability of every documented API call, runtime strings,
the prompts, and whether the whole thing read end-to-end induces the intended
objective). Every finding then went to an adversarial verifier whose default was
REFUTED and which had to prove the quoted text exists *and* reaches a solver
under the shipped config.

**70 confirmed, 10 refuted, 8 of them high severity.**

### The channel: `help()` is a solver-visible surface

The harness guards *files*. A strip rewrites every workspace file, a proxy holds
the API key so `/api/games` is unreachable, and a redaction pass rewrites the
doctrine. All of it is file-shaped.

`help()` is not a file. The solver is handed `client`, `gate` and `arc` by name
in `session.py`, and `help(client)` / `dir(arc)` / `pydoc athanor.ccarc3.gate` are
the standard way to find out what an unfamiliar object does — the first thing an
orienting agent does. Every public docstring in the package renders through that
path, and those docstrings were written for the operator:

- `help(arc.GameInfo)` printed **"real games span 171 to 1843 baseline actions"** —
  the envelope of the secret across the whole public set, on a class the workspace
  `session.py` imports by name.
- `arc.gate.__doc__` named **1843** outright and justified the gate by saying
  re-derivation is "unaffordable".
- `help(client)` rendered `baseline_here`'s docstring under *Readonly properties*:
  it announced that a per-level human median exists, that the solver is being
  denied it deliberately, and named `_baseline_here_enforced` — a pointer straight
  to `info.baseline_actions`.
- `arc_proxy.__doc__` was a complete description of the defence: which env var
  closes the in-process route, that `baselines_for` refuses and `list_games`
  filters, that the ceiling is `budget_multiple` times the withheld total and
  therefore inverts to it, and which two runs had already fetched the numbers by
  hand.
- `help(client.level_budget)` asserted a per-level termination rule at 5× baseline,
  "enforced whether or not it told the agent" — a rule this harness stopped
  modelling on 2026-08-04, in a property that returns 0 on every shipped run.

This is not hypothetical browsing. `client.py`'s own note records `cd82` reading
`baselines_for` straight out of `dir(arc)` **"having gone looking for nothing at
all"** — the incident that motivated the env-var guard. The guard was added; the
docstring explaining the guard was left in the listing beside it.

**Fix:** rationale moves to `#` comments, which `help()` cannot reach. Docstrings
state what the thing does and nothing else. Four tests in
`tests/test_ccarc3_solver_reachable_docs.py` now walk every reachable docstring —
and render the real `pydoc` page for the three objects `session.py` binds — and
fail on a real baseline, allowance vocabulary, or any description of the
withholding.

### A real human baseline was printed in the doctrine

§0a's worked proof that ARC scores the *best* play gave two plays as
`21 actions → level_scores [65.533]` and `7 actions → [115.000]`, then spelled out
`65.533 = (17/21)²×100`. **17 is a real per-level human median**, and the pair
`(21, 65.533)` inverts to it without the derivation line: `21·√0.65533 = 17.0`.
Neither rewriting pass touches it — the strip removes §6, and the redaction only
rewrites lines naming the game being played. Every solver read it, in the first
file it is told to open. The demonstration is load-bearing, so it survives in
symbolic form; the numbers do not.

### The resumed prompt

The first user turn of every resumed run — the highest-authority position in
context, read *before* DOCTRINE.md — said re-deriving mechanics *"wastes budget
you have already spent"*. Incoherent on its own terms (spent budget cannot be
wasted again), and the exact allowance frame §0 exists to deny, with no
counterweight in front of it. It also collapsed the three-valued rule book into
settled fact, told the solver not to RESET at all when only one RESET is
dangerous and mid-level resets are legitimate, read `rules.json` unconditionally
though a fresh build deletes it, and ended on a comma splice with a duplicated,
mid-sentence capitalised "Continue". Rewritten whole.

### The rest

- **`quiet_pace`, formerly `hide_baselines`.** The keyword is written literally
  into the workspace `session.py` the solver imports, and repeated in
  `help(ArcClient)`'s signature — announcing in one word that per-level medians
  exist and are being kept from it. A knob named for the secret is a signpost to
  the secret; the new name describes the visible effect.
- **Runtime refusals** told the solver actions "cost budget" and cited `§2.4` of
  a design note that is not in the workspace. They now give the real reason: a
  discarded action teaches nothing and moves nothing.
- **Three documented calls did not work as written.** `arc.monotone_rows()` takes
  `(before, after)` pairs, not a bare call. `arc.shortest_path`/`arc.reachable`
  default to `actions=(1,2,3,4)`, which is *wrong for every click game* and fails
  silently — an empty route or a tiny reachable set, which the doctrine then tells
  you to read as "your model puts a wall where there is none". §10 told the solver
  to stamp its hypothesis into the action's `reasoning` field; `act()` has no such
  parameter, so the instruction had no correct form.

### What this does to the banked runs

Nothing to the eight rollouts' scores. What it changes is what can be claimed
about them: they ran under a harness whose docstrings would have handed over the
baseline envelope to anyone who typed `help(arc.GameInfo)`, and whose doctrine
printed one real median outright. No trace shows a solver having done either —
`proofread_trace.py` clears all eight — but "we withheld the baselines" was a
stronger claim than the evidence supported, and it is now closer to true than it
was.

### The pattern, for the third time

Every defect in this pass is the same shape as the last two: **a mechanism that
looked like it worked because the visible outcome was unchanged.** The strip ran
and left prose it had destroyed. The reframe removed the phrases a test looked
for. And the guard against `dir(arc)` was shipped with a docstring beside it
explaining what it guards. Each was found by rendering the artifact and reading
it, rather than reading the code that produces it — and the surface was larger
each time I looked.

---

## Rollout 9: `r11l-495a7899` — **1.0000 (6/6)**, and the surface repair confirmed live

The first game run under the repaired solver surface (`a63232a`, `f66cc91`), and
the first of the seventeen environments that had no result under the current
harness.

| | |
|---|---|
| levels | **6 of 6** |
| actions | 166 total across 2 plays (94 + 72) |
| `raw` | **1.1500** — the ceiling, on *both* plays |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths | 0 |
| wasted actions | 0 |
| wall clock | 46 min, one-shot, exit 0 |
| cost | $14.69 over 116 turns |

Per-level against the published baselines `[22, 33, 51, 26, 52, 49]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| play 1 actions | 12 | 19 | 15 | 13 | 17 | 18 |
| ratio | 0.55× | 0.58× | **0.29×** | 0.50× | **0.33×** | 0.37× |

Every level cleared at or under 0.58× of its human median, so every one of them
clipped at the 1.15 per-level ceiling and `raw` landed exactly on 1.1500. With
all six levels cleared the completion cap is 1.0 and binds, which is the usual
shape here: **the 30th of 33 scored runs where `cap`, not `raw`, decided the
score.**

### The replay was free, correct, and worth nothing — which is the point

The solver replayed after winning, cutting 94 actions to 72. In score terms it
gained exactly zero: both plays were already at `raw` 1.1500 with the cap
binding at 1.0. It said so itself, without being able to check:

> I can't compute `raw` — the API returns no per-level baselines, so I know the
> ceiling is 1.0 but not my distance below it. **I replayed because that bet costs
> actions and cannot cost score, not because I measured a gain.**

That is §0a executed exactly as written. The doctrine tells a solver with no
baselines to replay anyway, on the grounds that the server keeps the best play
and the bet cannot lose score. Across the 25-environment arm that bet paid about
+0.35 `E` in total and was wasted roughly seven times. This is the eighth waste,
and it cost 72 actions and nothing else.

### The proofread: 8 passages, all of them the strip holding

Mechanical checks clean — 115 commands, none left the workspace, and the inbound
scan found no per-level array, no ceiling figure, no `api/games`, no pace line.
The card corroborates: 6 levels, 166 actions, matching `result.json` exactly.

All 8 flagged passages are the solver reasoning about *not* having the numbers:
*"I don't know the human baselines"*, *"The scorecard doesn't include baseline
level scores, so I can't calculate the raw score"*, *"if they're typical for
ARC-AGI-3 (often in the tens)"*. It guessed at the magnitude and got no
confirmation. That is what a working strip reads like from the inside.

### It ran `cat session.py`, and that is the whole proofread in one line

On its first orientation turn the solver ran:

```
cat meta.json resume_state.json session.py && ls notes/
```

and the tool result contains, verbatim:

```python
client = ArcClient(
    'r11l-495a7899',
    trace_path=HERE / "trace.jsonl",
    info=INFO,
    gate=gate,
    max_actions=0,
    quiet_pace=True,
    show_score=True,
)
```

Eight lines and nothing else. **Three hours earlier the same command would have
returned thirty lines of operator commentary** telling the solver that a hidden
action ceiling exists and is enforced in `arc_proxy`, that the human medians are
withheld deliberately and are "resolved at import and never written to a
workspace file", and what a different environment had scored — plus the keyword
`hide_baselines=True`, which names the secret in the one file the doctrine
instructs the solver to import.

The proofread's argument for removing all of it was that `cat session.py` is
"one of the first things a careful solver does". The first solver to run under
the fix did it on turn one. The prediction was not clever; it was just a matter
of reading what the harness hands over and asking who reads it.

### Standing

**9 clean rollouts, 8.4075 / 9 = 93.42%**, eight wins. Sixteen environments
outstanding.

---

## Rollout 10: `tr87-cd924810` — **1.0000 (6/6)**, and a transient blip that cost more than the game

| | |
|---|---|
| levels | **6 of 6** |
| actions | 259 total across 2 plays (140 + 119) |
| `raw` | **1.1500** — the ceiling, on both plays |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | 0 / 0 |
| wall clock | 30 min, one-shot, exit 0 |
| cost | $8.73 over 79 turns |

Against baselines `[54, 58, 40, 45, 71, 146]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| actions | 23 | 32 | 26 | 21 | 14 | 24 |
| ratio | 0.43× | 0.55× | 0.65× | 0.47× | **0.20×** | **0.16×** |

The two deepest levels — the ones carrying 5/21 and 6/21 of the environment —
were the two it played fastest, at 0.20× and 0.16×. That inversion keeps
recurring: the levels humans find hardest are not the ones that cost this solver
anything.

Proofread clean: 78 commands, none left the workspace, nothing inbound, card
corroborates 6 levels and 259 actions. And like `r11l` before it, its first
orientation command was `cat session.py` — **two for two** on the behaviour the
surface repair was aimed at.

### The blip: one dead endpoint, thirteen launches, and 82 minutes lost

At 07:19 a transient network failure made `list_games()` return
`[Errno 111] Connection refused`. The endpoint was healthy again within a minute
— 25 games on the next call — but the damage was already done, in a chain worth
recording because every link behaved *as designed*:

1. `sc25` failed at `build_workspace`, before any solver launched.
2. The circuit breaker fired, correctly: a dead endpoint means every remaining
   game fails identically, so abort rather than burn the queue.
3. **But twelve more games had already launched.** The breaker raises `SystemExit`
   inside a worker thread; `one_pass` only sees it via `as_completed`, and with
   seventeen futures failing in milliseconds, thirteen ran before the main thread
   processed the first. `m0r0`, `vc33`, `lp85`, `g50t`, `s5i5`, `tu93`, `re86`,
   `cn04`, `ar25`, `ls20`, `bp35`, `dc22` all started and failed. The abort that
   exists to stop the queue burning through arrived last.
4. The driver exited, orphaning its live solvers to init.
5. The supervisor started a fresh driver, which ran `kill_orphan_solvers()` —
   correctly, by design — and killed them.

`cd82` was one of them: **82 minutes and 365 actions, discarded** at level 2 of 6.
`su15` lost 3 minutes. Neither is a scoring loss (a discarded attempt costs quota,
never score) but `cd82` was the single most expensive thing in flight.

The cost was low only because the failure happened at `build_workspace`, before
any solver was launched. Had the endpoint died a step later, thirteen solvers
would have been launched into it.

Two defects, both now understood and neither yet fixed:

- **The breaker cannot stop its siblings.** Raising `SystemExit` in one thread
  does nothing to the sixteen already running. It needs a flag that `_take_slot`
  and `_run_one` check, so a game that has not started never does.
- **Aborting kills in-flight games.** "Stop and let the supervisor restart" is
  right for a dead endpoint and wrong for the running solvers, which are
  independent of it. The supervisor already has a polite-stop that waits for
  `result.json`; the driver's own abort does not.

### Standing

**10 clean rollouts, 9.4075 / 10 = 94.08%**, nine wins. Fifteen environments
outstanding, of which `cd82` and `su15` need re-running after the kill.

---

## Rollout 11: `lp85-305b61c3` — **1.0000 (8/8)**, and the replay that finally moved `raw`

| | |
|---|---|
| levels | **8 of 8** |
| actions | 179 total across 2 plays (100 + 79) |
| `raw` | 1.1486 (play 1) → **1.1500** (play 2, the ceiling) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | 0 / 0 |
| wall clock | 36 min, one-shot, exit 0 |
| cost | $12.66 over 98 turns |

Against baselines `[17, 38, 31, 16, 41, 60, 26, 159]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| play 1 | 7 | 11 | 16 | 15 | 11 | 22 | 8 | **10** |
| ratio | 0.41× | 0.29× | 0.52× | **0.94×** | 0.27× | 0.37× | 0.31× | **0.06×** |
| play 2 | 5 | 8 | 16 | 12 | 9 | 19 | 5 | **5** |
| ratio | 0.29× | 0.21× | 0.52× | 0.75× | 0.22× | 0.32× | 0.19× | **0.03×** |

**Level 8 carries a 159-action human baseline and fell in 10, then in 5.** That
is 0.03× — the most extreme instance yet of the inversion this project keeps
finding: the level humans find hardest, by a factor of four over any other level
in the game, cost this solver less than any of them. The arm run of the same
environment did it in 9. Whatever makes that level expensive for a human is not
what makes a level expensive here.

### The first replay on record that moved `raw` at all

Ten rollouts in, every replay had been worth exactly zero: both plays already at
the 1.1500 ceiling, cap binding at 1.0. This one started at `raw` **1.1486** —
fractionally short — because level 4 came in at 0.94×, the only level all session
to finish near its baseline rather than far under it. The replay took it to 0.75×
and `raw` to the ceiling.

**It still changed the score by nothing.** `E = min(cap, raw)` and `cap` was
already 1.0000, so 1.1486 and 1.1500 are the same environment score. The replay
was free and correct and moved a number that does not matter — which is exactly
what §0a predicts, and a good illustration of why the doctrine tells solvers to
replay on the *cap* argument rather than on measured efficiency they cannot see.

### Proofread

Clean. 97 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 8 levels and 179
actions. Its orientation included importing `session` and printing `client.status()`
— **three for three** on solvers inspecting the harness surface on turn one.

### Standing

**11 clean rollouts, 10.4075 / 11 = 94.61%**, ten wins. Of these, **three ran
under the repaired solver surface** (`r11l`, `tr87`, `lp85`) and all three scored
1.0000. Fourteen environments outstanding.

---

## Rollout 12: `vc33-5430563c` — **1.0000 (7/7)**, the first run a replay genuinely rescued

| | |
|---|---|
| levels | **7 of 7** |
| actions | 436 total across 2 plays (269 + 167) |
| `raw` | 1.0688 (play 1) → **1.1500** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths | **1** — the first death in twelve clean rollouts |
| wall clock | 44 min, one-shot, exit 0 |
| cost | $10.54 over 71 turns |

Against baselines `[7, 18, 44, 61, 131, 34, 152]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| play 1 | 9 | 8 | 23 | 72 | 88 | 20 | 49 |
| ratio | **1.29×** | 0.44× | 0.52× | **1.18×** | 0.67× | 0.59× | 0.32× |
| play 2 | 3 | 7 | 23 | **21** | **44** | 20 | 49 |
| ratio | 0.43× | 0.39× | 0.52× | 0.34× | 0.34× | 0.59× | 0.32× |

**This is the first rollout where a solver actually went over baseline** — twice,
on levels 1 and 4 — and the first where the replay did real work rather than
shaving a rounding error. Level 4 went 72 → 21 actions and level 5 went 88 → 44,
carrying `raw` from 1.0688 to the 1.1500 ceiling.

It also still changed the environment score by nothing: `E = min(cap, raw)`, and
with all seven levels cleared `cap` was already 1.0000. **That is now three
distinct shapes of the same lesson** — a replay that was pure waste (`r11l`,
`tr87`), one that moved `raw` by 0.0014 (`lp85`), and one that moved it by 0.08
(`vc33`) — and in all three the environment score was identical either way.
Efficiency has not been worth a single point in this project since the completion
cap started binding.

The level-1 overrun is worth noting on its own: a 7-action human baseline, spent
9 on the first level of an unfamiliar game. There is no version of exploration
that beats that, and the squared ratio makes 1.29× cost more than it looks —
which is precisely why it is a `raw` problem and not a `cap` problem, and why it
cost nothing.

### Proofread

Clean. 70 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 7 levels and 436
actions. One probe, reading its own workspace.

### Standing

**12 clean rollouts, 11.4075 / 12 = 95.06%**, eleven wins. **Four under the
repaired solver surface** — `r11l`, `tr87`, `lp85`, `vc33` — all four 1.0000.
Thirteen environments outstanding.

---

## Rollout 13: `sc25-635fd71a` — **1.0000 (6/6)**, and the replay was worth **+0.4156**

The most expensive run of the session, the hardest, and the one that finally
proves the doctrine's blind-replay rule pays.

| | |
|---|---|
| levels | **6 of 6** |
| actions | 839 total across 2 plays (674 + 165) |
| `raw` | **0.5844** (play 1) → **1.1357** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** — from **0.5844** without the replay |
| deaths | **5** |
| wall clock | 89 min, one-shot, exit 0 |
| cost | **$27.76** over 148 turns |

Against baselines `[36, 6, 32, 83, 143, 50]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| play 1 | 113 | 6 | 25 | 68 | **355** | 107 |
| ratio | **3.14×** | 1.00× | 0.78× | 0.82× | **2.48×** | **2.14×** |
| play 2 | 14 | 6 | 14 | 32 | **61** | 39 |
| ratio | 0.39× | 1.00× | 0.44× | 0.39× | **0.43×** | 0.78× |

### This corrects something I have written three times

The last three write-ups all said some version of *"efficiency has not been worth
a single point in this project since the completion cap started binding."* That
was true of the twelve runs it described and **it is false as a general claim**,
and this run is the counterexample.

`sc25` cleared all six levels on its first play, so `cap` was 1.0000 — and `raw`
came in at **0.5844**, far below it. The binding term was efficiency, not
completion, for the first time in the clean rollout series. Play 1 alone would
have scored **0.5844**. The replay scored 1.1357 and the environment takes the
best play, so the run banks **1.0000**.

**The replay was worth +0.4156** — more than `sk48`'s +0.5833 completion gain is
the only thing in this project that beats it, and that one came from clearing
three more levels. This one came from walking a route it already knew.

### The solver could not see any of that, and replayed anyway

It has no baselines. What it had was §0a's rule — *if you cannot compute `raw`,
replay anyway* — and an estimate:

> Level 4 consumed around 355 actions while level 5 took roughly 107, and **if the
> human baseline is somewhere near 60 actions, that ratio squared would be
> devastatingly low** — raw would be far below 1.0, making a replay potentially
> worth a significant score boost.

It guessed the baseline's order of magnitude from its own action counts, reasoned
that the squared ratio would be brutal, and replayed. The real level-5 baseline is
143 and it had spent 355 — 2.48×, scoring 0.163 on a level carrying 5/21 of the
environment. Its instinct was right and its arithmetic was conservative.

Across the 25-environment arm, blind replay paid three times, was wasted seven
times, and was worth about +0.35 `E` in total. **This single run is worth more
than that entire arm's replay bet.**

### Five deaths and 3.14× on level 1 — and none of it mattered

839 actions against a 350-action baseline: 2.4× the entire game's human cost, five
deaths, and three levels finished over baseline including the opening one at
3.14×. Under the old doctrine — the one that told solvers to keep a baseline in
reserve and not spend more than half their cap before understanding the game —
this run reads as a disaster. It scored 1.0000.

That is the reframe working exactly as intended: the run that ignored every
efficiency instinct and kept going is the run that won, and the instrument that
saved it was available only because it had cleared the final level and could
restart from a level advance.

### Proofread

Clean. 147 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 6 levels. All 19
flagged passages are the solver reasoning about the **in-game timer** — rows
draining per action, which this game genuinely meters — not about a harness action
budget. Two are the replay decision quoted above.

### Standing

**13 clean rollouts, 12.4075 / 13 = 95.44%**, twelve wins. **Five under the
repaired solver surface** — `r11l`, `tr87`, `lp85`, `vc33`, `sc25` — all five
1.0000. Twelve environments outstanding.

---

## Rollout 14: `tu93-0768757b` — **1.0000 (9/9)**, replay worth **+0.1202**, against the solver's own advice

| | |
|---|---|
| levels | **9 of 9** |
| actions | 450 total across 2 plays (269 + 181) |
| `raw` | **0.8798** (play 1) → **1.1259** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** — from **0.8798** without the replay |
| deaths | 5 |
| wall clock | 71 min, one-shot, exit 0 |
| cost | $18.55 over 123 turns |

Against baselines `[19, 16, 34, 42, 123, 80, 14, 23, 111]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| play 1 | 21 | 26 | 24 | 42 | 36 | 31 | 20 | 32 | 33 |
| ratio | 1.11× | **1.62×** | 0.71× | 1.00× | 0.29× | 0.39× | **1.43×** | **1.39×** | 0.30× |
| play 2 | 18 | 10 | 19 | 17 | 29 | 28 | 14 | 21 | 29 |
| ratio | 0.95× | 0.62× | 0.56× | 0.40× | 0.24× | 0.35× | 1.00× | 0.91× | 0.26× |

**Two consecutive rollouts where the replay carried the score.** After twelve
runs in which it was worth nothing, `sc25` (+0.4156) and now `tu93` (+0.1202)
both had `raw` below a `cap` of 1.0000 and were rescued by the second play. The
claim I corrected on `sc25` is now doubly wrong, and the shape is clearer: a run
that clears every level but *overruns several of them* is exactly the case blind
replay exists for, and it took thirteen rollouts to produce one.

`tu93` is also the environment §0a already cites as a case where blind replay paid
in the 25-environment arm — **0.8286 → 1.0000** there, 0.8798 → 1.0000 here. The
finding replicates on the same game, a day later, under a rebuilt harness, with a
different solver process that could not read the earlier result.

### It replayed against its own estimate

This game displays a depleting bar the solver reads as a per-level action meter,
so unlike previous runs it had *something* to estimate `raw` from. It used it, and
the estimate said not to bother:

> **The measured budgets suggest replaying wouldn't help since the raw score
> exceeds the cap**, but the real baselines are unknown, **so the doctrine
> recommends** [replaying anyway]

Its own arithmetic said the replay was worthless. The doctrine's blind rule said
replay regardless. **The doctrine was right and the estimate was wrong by 0.12.**

That is the strongest evidence yet for §0a as written. The rule is not "replay
when you calculate a gain" — it is "replay when you *cannot* calculate one",
precisely because the in-game proxies a solver reaches for are not the human
baselines and will mislead it. A solver that trusted its own meter would have
banked 0.8798.

### Proofread

Clean. 122 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 9 levels and 450
actions. All 16 flagged passages concern the in-game drain bar, not a harness
budget.

### Standing

**14 clean rollouts, 13.4075 / 14 = 95.77%**, thirteen wins. **Six under the
repaired solver surface** — `r11l`, `tr87`, `lp85`, `vc33`, `sc25`, `tu93` — all
six 1.0000. Eleven environments outstanding.

---

## Rollout 15: `s5i5-18d95033` — **1.0000 (8/8)**, and a note on how I have been reporting these

| | |
|---|---|
| levels | **8 of 8** |
| actions | 511 total across 2 plays (267 + 244) |
| `raw` | 1.1333 (play 1) → **1.1500** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 83 min, one-shot, exit 0 |
| cost | $25.01 over 112 turns |

Against baselines `[20, 89, 106, 54, 162, 38, 86, 83]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| play 1 | 27 | 26 | 44 | 30 | 28 | 25 | 51 | 36 |
| ratio | **1.35×** | 0.29× | 0.42× | 0.56× | **0.17×** | 0.66× | 0.59× | 0.43× |

Zero deaths across 511 actions on an eight-level game — the first fully
death-free run since `lp85`. Level 5 carries the largest baseline in the game
(162) and fell in 28, at 0.17×. The only level over baseline was the first, at
1.35×, which the replay took to 0.65×.

The replay moved `raw` 1.1333 → 1.1500 and `E` by nothing, `cap` already binding.
That is the ordinary case again after two runs where it mattered: of eight
replays now on record under the repaired surface, **two carried the score
(`sc25` +0.4156, `tu93` +0.1202) and six moved nothing.** The bet remains
correct and remains mostly wasted, which is exactly what §0a claims for it.

### Correcting how I have been quoting the standing figure

Through several updates above I reported a combined count — *"14 clean rollouts,
13.4075/14 = 95.77%"* — which pools two different harnesses and implies they are
one experiment. They are not. Eight of those runs read a solver surface that
printed a real human median in the doctrine, briefed the solver on where the
action ceiling is enforced, and framed actions as an allowance; the rest read the
repaired one. The combined number hides the treatment difference it should be
showing.

The audit page has tiered them correctly the whole time (`current` vs
`superseded`); the prose was the sloppy part. From here the standing figure is
reported split, and `scratchpad/standing.sh` derives it by reading each run's own
preserved `session.py` rather than trusting commit timestamps.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **7 of 25** | **7.0000 / 7 = 100%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 10 | — |

Seven for seven at 1.0000 under the current harness. Worth stating plainly that
this is **seven games**, that four of them (`r11l`, `tr87`, `lp85`, `vc33`) are
among the shorter environments in the set, and that the superseded eight include
the two hardest — so the gap between 100% and 92.59% is not yet evidence of a
harness effect and should not be quoted as one.

---

## Rollout 16: `ar25-0c556536` — **1.0000 (8/8)**, the cheapest win on record

| | |
|---|---|
| levels | **8 of 8** |
| actions | 521 total across 2 plays (268 + 253) |
| `raw` | **1.1500** on **both** plays — the ceiling twice over |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 34 min, one-shot, exit 0 |
| cost | **$7.83** over 75 turns |

Against baselines `[32, 50, 75, 37, 89, 159, 233, 73]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| actions | 26 | 14 | 41 | 22 | 28 | 53 | **37** | 47 |
| ratio | 0.81× | 0.28× | 0.55× | 0.59× | 0.31× | 0.33× | **0.16×** | 0.64× |

**The largest baseline in the whole 25-environment set is level 7 here — 233
actions — and it fell in 37.** That is 0.16×, on the level carrying 7/36 of the
environment. `lp85`'s 159-baseline level in 10 was the previous high-water mark
for this inversion; this is a bigger baseline cleared at a comparable ratio.

`ar25` is also the **cheapest win on record at $7.83**, on a game whose published
human cost (748 actions) is the second-highest in the set. It took 75 turns,
fewer than any other eight-level run, with no deaths and no wasted actions. The
expensive runs in this series have been the ones that had to re-derive a mechanic
they got wrong (`sc25` at $27.76, `s5i5` at $25.01); this one appears to have got
its model right early and simply executed.

Both plays hit `raw` 1.1500, so the replay was worth nothing again — nine replays
under the repaired surface now, **two of which carried the score**.

### Proofread

Clean. 74 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 8 levels and 521
actions. One probe, `cat session.py meta.json resume_state.json` — the fourth
solver in eight to open `session.py` on turn one.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **8 of 25** | **8.0000 / 8 = 100%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 9 | — |

Eight for eight at 1.0000. The caveat from the last entry still stands and is now
worth restating with a number: the eight current runs have a mean published
baseline total of 486 actions against the superseded eight's 421, so this is no
longer a "the easy ones went first" story — but two of the superseded eight
(`lf52`, `sk48`) are the only environments in the project that have *ever* scored
below 1.0 on a clean run, and until one of those is re-run under the current
surface the comparison stays confounded.

---

## Rollout 17: `cn04-2fe56bfb` — **1.0000 (6/6)**, and a 300-action level cleared in 33

| | |
|---|---|
| levels | **6 of 6** |
| actions | 396 total across 2 plays (222 + 174) |
| `raw` | 1.1397 (play 1) → **1.1500** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 40 min, one-shot, exit 0 |
| cost | $11.78 over 92 turns |

Against baselines `[29, 54, 85, 300, 208, 113]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| play 1 | 30 | 33 | 25 | **33** | 58 | 43 |
| ratio | 1.03× | 0.61× | 0.29× | **0.11×** | 0.28× | 0.38× |

**Level 4 carries a 300-action human baseline and fell in 33 — 0.11×.** That now
displaces `ar25`'s 233-in-37 from yesterday as the largest absolute gap on record,
and it is the third such case in ten runs. The pattern is no longer anecdotal:
across the current generation the levels with the biggest human baselines are
routinely the ones this solver clears fastest, and whatever makes a level
expensive for a first-time human player is close to orthogonal to what makes it
expensive here.

`cn04` is also the third environment in this batch whose published human cost
(789) is near the top of the set, cleared in half that. Only level 1 came in over
baseline, at 1.03×, which the replay took to 0.48×.

### Proofread

Clean. 91 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 6 levels and 396
actions.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **9 of 25** | **9.0000 / 9 = 100%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 8 | — |

Nine for nine. The confound named in the last two entries is unchanged and still
governs: `lf52` and `sk48` are the only environments that have ever scored below
1.0 on a clean run, both sit in the superseded set, and until one is re-run under
the current surface the 100% is not a harness effect.

---

## Rollout 18: `g50t-5849a774` — **1.0000 (7/7)**, ten for ten

| | |
|---|---|
| levels | **7 of 7** |
| actions | 684 total across 2 plays (399 + 285) |
| `raw` | 1.0387 (play 1) → **1.1500** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 84 min, one-shot, exit 0 |
| cost | $18.24 over 107 turns |

Against baselines `[78, 175, 179, 230, 96, 54, 67]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| play 1 | 55 | 31 | 77 | 75 | 50 | **68** | 43 |
| ratio | 0.71× | **0.18×** | 0.43× | 0.33× | 0.52× | **1.26×** | 0.64× |
| play 2 | 17 | 31 | 64 | **31** | 50 | 49 | 43 |
| ratio | 0.22× | 0.18× | 0.36× | **0.13×** | 0.52× | 0.91× | 0.64× |

The inversion again, and cleanly separated this time. The three largest baselines
in the game — levels 2, 3 and 4 at 175, 179 and 230 — were cleared at 0.18×,
0.43× and 0.33×. The **only** level to run over was level 6, whose baseline is
54, the second-smallest in the game. Across the current generation this is now the
rule rather than a curiosity: a large human baseline predicts an *easy* level for
this solver, and the levels that cost it are small-baseline ones where a human's
route is short and the solver's model of the mechanic has to be exactly right.

`raw` opened at 1.0387 — under the ceiling but above the cap, so the replay was
worth nothing again. Ten replays under the repaired surface now, **two of which
carried the score**.

### Proofread

Clean. 106 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 7 levels and 684
actions.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **10 of 25** | **10.0000 / 10 = 100%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 7 | — |

Ten for ten, and now a **majority of the environments that have been attempted at
all** under the current surface. The comparison remains confounded in the way
named three entries ago and nothing has changed it: `lf52` and `sk48` are the only
environments ever to score below 1.0 on a clean run and both sit in the
superseded set. What has changed is the sample — ten runs at a mean published
baseline of 528 actions, against the superseded eight's 421 — so "the easy ones
went first" is now clearly false, while "the two hardest have not been retried"
remains true.

---

## Rollout 19: `m0r0-492f87ba` — **1.0000 (6/6)**, a 500-action baseline in 56, and a hole in my own repair

| | |
|---|---|
| levels | **6 of 6** |
| actions | 424 total across 2 plays (245 + 179) |
| `raw` | **1.1500** on both plays |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 70 min, one-shot, exit 0 |
| cost | $13.43 over 110 turns |

Against baselines `[30, 111, 203, 26, 500, 237]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| play 1 | 20 | 43 | 65 | 16 | **56** | 45 |
| ratio | 0.67× | 0.39× | 0.32× | 0.62× | **0.11×** | 0.19× |
| play 2 | 15 | 23 | 53 | 11 | **39** | 38 |
| ratio | 0.50× | 0.21× | 0.26× | 0.42× | **0.08×** | 0.16× |

**Level 5 carries a 500-action human baseline — the largest single-level baseline
in the set — and fell in 56, then 39.** That is 0.08×, and it displaces `cn04`'s
300-in-33 from an hour ago. Every level came in under baseline on both plays.

### The run that found a hole in the surface repair

This solver ran, on its own initiative:

```
python -c "
from session import client, gate, arc
print([n for n in dir(client) if not n.startswith('_')])
import inspect
print(inspect.getsource(client.restart_for_replay))
print(client.status())
"
```

`dir(client)` then `inspect.getsource`. That is the fifth solver in eleven to
inspect the harness on turn one, and the first to use `inspect`.

The `f66cc91` repair moved operator rationale **out of docstrings and into `#`
comments**, on the argument that `help()` renders docstrings and cannot see
comments. That argument is correct and the fix works for `help()`. It does not
work for `inspect.getsource` **on the class**:

```
inspect.getsource(ArcClient)  ->  1011 lines
  "Renamed from hide_baselines ... because the solver reads the keyword"   PRESENT
  "help(client) told the solver it was under a per-level termination rule" PRESENT
  "cd82 ran dir(arc) ... read baselines_for straight out of the listing"   PRESENT
```

Every comment I moved for safety is in there. `getsource` on a single method
starts at its `def` and excludes preceding comments — which is why the property
above came back clean — but `getsource` on the *class* spans the whole block,
comments included. I closed one standard introspection path and left an equally
standard one open, and a solver was one command away from it in this very run.

**What it would have disclosed, and what it would not.** Not the numbers: the
strip leaves `baseline_actions=()` in the shipped workspace, so
`_baseline_here_enforced` returns `None` however it is reached — verified against
this run's own `session.py`. What it discloses is the *mechanism* — that medians
exist, that they are withheld deliberately, which flag does it, and the history of
previous leaks. By the standard the proofread itself set ("a map to where they
live is a leak"), that is a finding, and the repair it came from was incomplete.

Recorded here rather than quietly patched, because the pattern is now four for
four today: **every fix I have shipped closed the path I was looking at and left
an adjacent one open** — the proxy port for the driver but not the evidence
daemon, the abort for the raising thread but not its siblings, `help()` but not
`inspect`.

### Proofread

Clean. 109 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 6 levels and 424
actions.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **11 of 25** | **11.0000 / 11 = 100%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 6 | — |

---

## Rollout 20: `bp35-0a0ad940` — **0.7252 (9/9)**. It won every level and the harness told it to stop.

The first sub-1.0 score under the repaired surface, and it is the harness's fault,
not the solver's.

| | |
|---|---|
| levels | **9 of 9** — every level cleared |
| actions | 990 in a **single play** |
| `raw` | **0.7252** |
| `cap` | 1.0000 |
| **`E`** | **0.7252** |
| deaths | **14** |
| wall clock | **292 min** (4 h 52 m), one-shot, exit 0 |
| cost | **$78.96** over 301 turns — the most expensive run of the project |

Against baselines `[21, 48, 44, 38, 33, 87, 86, 131, 163]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| actions | 30 | 75 | 42 | 33 | 36 | **309** | 90 | 100 | **275** |
| ratio | 1.43× | 1.56× | 0.95× | 0.87× | 1.09× | **3.55×** | 1.05× | 0.76× | **1.69×** |

`playthroughs: 1`, `full_resets: 0`. **It never replayed**, and a clean replay at
the winning frame was worth up to **+0.2748**.

### Why it stopped, in its own words

> All nine levels are cleared with a perfect score, so the game is won. **I should
> wrap this up with a summary for the user since there's nothing left to
> accomplish.** All nine levels are cleared — `level 9/9 state=WIN`, **cap 1.000**.

It read `cap 1.000` off `client.status()` and took it for its score. `cap` *was*
1.000 — that is true and the harness printed it correctly. But
`E = min(cap, raw)`, and with the medians withheld `status()` printed only the
`cap` half:

```
[cap 1.000 = 9/9 levels]
```

Half the formula, with nothing to indicate the other half existed. At 9 of 9 that
renders as a clean `1.000` and reads, reasonably, as a perfect score. §0a's rule —
*if you cannot compute `raw`, replay anyway* — was an hour of context earlier and
lost to a number on screen that appeared to settle the question.

**This is the mirror image of `tu93`.** There the solver estimated `raw` from an
in-game meter, concluded a replay would not help, and replayed anyway because the
doctrine said to — gaining +0.1202. Here the estimate came from the *harness
itself*, carried more authority, and pointed the wrong way. A rule that survives
the solver's own arithmetic does not survive the harness contradicting it.

### The fix

`status()` now labels the number as what it is, and speaks at the one moment the
instrument is available:

```
[cap 1.000 = 9/9 levels — a CEILING, not your score: efficiency is
 unmeasurable here and can only lower it]
WON — and `restart_for_replay()` is legal RIGHT NOW and illegal after any
 further action. Your score is min(cap, raw); cap is 1.000 and raw is unknown
 to you, so a clean replay can only raise it. Doctrine §0a: if you cannot
 compute raw, replay anyway.
```

The replay is legal only while the server's action counter is zero — the frame
straight after the final level advance — and any further action ends it
permanently. Saying so at that frame is worth more than the doctrine paragraph
saying it an hour earlier. Two tests pin both halves.

### What this run also shows about the game

Level 6 ran to 3.55× and level 9 to 1.69×, with 14 deaths and 990 actions against
a 651-action baseline. It is the hardest environment encountered so far and it
still fell completely — which is §0b working exactly as intended. **The solver's
persistence was not the failure; only its exit was.** Under the pre-reframe
doctrine, a run 1.5× over the whole game's baseline at level 6 would have been a
candidate to stop, and stopping there would have scored roughly 0.28.

### Proofread

Clean. 299 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 9 levels and 990
actions.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **12 of 25** | **11.7252 / 12 = 97.71%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 5 | — |

The first blemish on the current generation, and worth stating plainly: it is a
*harness* defect that cost 0.2748, found only because the run happened to clear
every level badly. Nine of the previous eleven never had a `raw` below `cap`, so
the display had no opportunity to mislead them.

### Generation note: the `status()` fix is a boundary the digest cannot see

The `bp35` fix changes `client.py`, and `client.py` is **not** covered by
`surface_digest`. That digest hashes the four workspace files plus the proxy's
allowlist — what the solver *reads*. `status()` is what the solver is *told*,
every turn, and it just changed materially.

I tried adding it and reverted. `client.py` is imported off `PYTHONPATH` and
never copied into a workspace, so a finished run holds no record of the version
it saw; hashing the repo's current copy on the reference side while every stored
digest predates the field makes them all mismatch by construction. All twelve
`current` runs flipped to `superseded` at once and the page rendered empty — the
same failure the digest's own docstring already warned about, from the first time
it happened with the environment.

So the boundary is recorded here by hand instead:

- **Rollouts 9–20** (`r11l` … `bp35`) ran against the `status()` that printed
  `[cap N = k/n levels]` with no indication that `cap` is only half of
  `min(cap, raw)`.
- **Rollouts from 21 on** get the labelled ceiling and the win-frame replay
  prompt.
- Of the twelve, **only `bp35` could have been affected**: the other eleven never
  had `raw` below `cap`, so the display had nothing to mislead them about. The
  fix is not retroactively invalidating; it closes a hole one run fell through.

`preserve_evidence.sh` now copies `client.py` and `gate.py` beside every run, so
the next generation boundary can be asserted by digest on both sides rather than
narrated.

**Three of the twelve are still in flight as this lands** — `ls20`, `dc22` and
`re86`. `client.py` is re-imported on every action, by design, so they pick the
new `status()` up mid-run. That makes them mixed-surface runs, and it is worth
knowing rather than discovering later: `ls20` in particular has been going 4 h 21 m
and started long before the fix.

---

## Rollout 21: `ls20-9607627b` — **1.0000 (7/7)** in five plays, and the controlled experiment `bp35` set up

`ls20` finished eighteen minutes after `bp35`. Their first plays are nearly
identical and their decisions were opposite, which makes this the cleanest natural
experiment the project has produced.

| | `bp35` | `ls20` |
|---|---|---|
| levels cleared, play 1 | **9 of 9** | **7 of 7** |
| `raw`, play 1 | **0.7252** | **0.7236** |
| `cap` | 1.0000 | 1.0000 |
| what it did next | **stopped** | **replayed, four times** |
| **`E`** | **0.7252** | **1.0000** |

Two solvers reached the same place — every level cleared, `raw` around 0.72, `cap`
at 1.0 — and the one that kept going scored **+0.2764** more. `bp35` left
+0.2748 behind. The two numbers are the same quantity seen from either side.

### The five plays

| play | levels | actions | ratios | `raw` | `E` |
|---|---|---|---|---|---|
| 1 | 7 | 898 | 1.09, 0.65, 0.89, 0.58, **1.38**, **1.57**, **1.32** | 0.7236 | 0.7236 |
| 2 | 4 | 439 | 0.59, 0.37, 0.56, 0.51 | 0.4107 | 0.3571 |
| 3 | 7 | 737 | …, **2.38**, 0.38, **1.59** | 0.7882 | 0.7882 |
| 4 | 7 | **313** | 0.59, 0.37, 0.56, 0.51, 0.46, 0.38, **0.30** | **1.1500** | **1.0000** |
| 5 | 7 | 365 | 0.59, 0.37, 0.56, 0.51, 0.46, 0.38, 0.58 | 1.1500 | 1.0000 |

Play 2 was **abandoned at four levels** — it scored 0.3571 and was simply worse
than what came before. That is §0a's "a replay you cannot finish costs nothing but
the actions" happening in practice: the environment keeps the best play, so a
discarded attempt is invisible in the score. Play 3 recovered the full clear but
still overran levels 5 and 7. Play 4 got it right and hit the ceiling.

**Levels 5, 6 and 7 are where the whole gain lives.** They went 132/302/246 on
play 1 to 44/72/55 on play 4 — the route was there all along and the first pass
was paying to find it. This is exactly the case §0a describes: *"what differed was
spending them rather than finding them."*

### What this does and does not show about the `status()` fix

The fix landed at 06:56 PDT; `ls20` finished at 07:07 and had been replaying for
hours before that. **Its behaviour is not evidence the fix works** — it replayed
on its own, under the same display that misled `bp35`. What the pair shows is that
the *decision* was worth ~0.28 and that solvers were already split on it, which is
the thing the fix is meant to stop leaving to chance.

`ls20` is also a mixed-surface run by the note above: `client.py` is re-imported
every action, so its final play saw the new `status()`. Play 4 had already reached
the ceiling by then, so nothing in this result turns on it.

### Cost

2752 actions, 271 minutes, $35.78 over 205 turns, 1 death. The 2752 actions are
3.5× the game's published baseline of 776 — and cost nothing, because only the
best play is scored. `bp35` spent 990 actions and lost 0.27; `ls20` spent 2752 and
lost nothing.

### Proofread

Clean. 204 commands, none left the workspace; no per-level array, no ceiling
figure, no `api/games`, no pace line inbound; card corroborates 7 levels and 2752
actions.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **13 of 25** | **12.7252 / 13 = 97.89%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 4 | — |

---

## Rollout 22: `dc22-fdcac232` — **1.0000 (6/6)**, three plays, all three at the ceiling

| | |
|---|---|
| levels | **6 of 6** |
| actions | 1574 across **3 plays** (668 + 458 + 448) |
| `raw` | **1.1500 on every play** |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 113 min, one-shot, exit 0 |
| cost | $42.11 over 189 turns |

Against baselines `[59, 102, 67, 98, 324, 578]` — a 1228-action published total,
the second-largest in the set:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| play 1 | 39 | 56 | 59 | 69 | 224 | **221** |
| ratio | 0.66× | 0.55× | 0.88× | 0.70× | 0.69× | **0.38×** |
| play 3 | 20 | 56 | 51 | 62 | 110 | **149** |
| ratio | 0.34× | 0.55× | 0.76× | 0.63× | 0.34× | **0.26×** |

**Level 6 carries a 578-action baseline — the largest single level in the entire
25-environment set — and fell in 221 on the first attempt, 149 on the third.**
That beats `m0r0`'s 500-in-56 on absolute size, though not on ratio.

The first play already hit `raw` 1.1500, so both replays were worth nothing. It
replayed twice anyway, which is §0a followed exactly: the solver could not see
that it was already at the ceiling, and the bet cost 906 actions and no score.
Eleven replays now under the repaired surface, **two of which carried the score**.

### Proofread

Clean. 188 commands, none left the workspace; nothing inbound; card corroborates
6 levels and 1574 actions.

### The heartbeat's unscored check was broken, and this run exposed it

`dc22` finished at 07:57 PDT and no `UNSCORED` event fired. The check grepped
`docs/ccarc3_results.md` for the bare game id — and **all 25 environments already
appear in this file from the 25-game arm**, so it matched arm-era prose and could
never report anything. It had been silently vacuous since I added it; `bp35` had
already slipped through a different hole in the same check an hour earlier.

Anchoring to `^## Rollout .*\`<id>\`` fixed that but then flagged the eight
superseded-generation runs, which were written up before the rollout numbering
existed and will never gain such a heading. So `scratchpad/unscored.sh` now also
requires the game's own preserved `session.py` to carry `quiet_pace` — the
measured marker of the current surface, the same one `standing.sh` uses. A run
from an older generation is not unscored, it is scored differently.

That is the fifth check today that reported success while doing nothing.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **14 of 25** | **13.7252 / 14 = 98.04%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 3 | — |

---

## Rollout 23: `cd82-fb555c5d` — **1.0000 (6/6)** on the third attempt, and the queue's discipline paying off

| | |
|---|---|
| levels | **6 of 6** |
| actions | 245 across 2 plays (175 + 70) |
| `raw` | 1.0267 (play 1) → **1.1500** (play 2) |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 40 min, one-shot, exit 0 |
| cost | $8.85 over 70 turns |
| attempt | **3** |

Against baselines `[55, 8, 41, 21, 23, 23]`:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| play 1 | 54 | 6 | **72** | 14 | 13 | 16 |
| ratio | 0.98× | 0.75× | **1.76×** | 0.67× | 0.57× | 0.70× |
| play 2 | **5** | 6 | 16 | 14 | 13 | 16 |
| ratio | **0.09×** | 0.75× | 0.39× | 0.67× | 0.57× | 0.70× |

### Two prior attempts, both killed, neither wasted

`cd82` is the shortest environment in the set (171-action baseline) and took three
attempts to bank — not because it is hard, but because it kept being killed:

| attempt | outcome |
|---|---|
| 1 | **discarded** — killed by SIGTERM after 82 min, 365 actions, at level 2 of 6 |
| 2 | **discarded** — killed by SIGTERM after 68 min, 655 actions |
| 3 | **CLEAN in 40 min**, 6/6, `E` 1.0000 |

Both kills were session-worker restarts, not game failures: the first at 00:19
PDT, the second at 08:16. Each orphaned the solver to init, and the next driver's
`kill_orphan_solvers()` collected it — behaving exactly as designed, and costing
150 minutes of play across the two.

**It cost nothing in score and that is the design working.** A discarded attempt
opens its own ARC scorecard and the environment keeps the best card, so an
abandoned run is invisible in the result. The driver's *fewest-attempts-first*
ordering then deprioritised `cd82` behind every zero-attempt game — which is why
it ran last rather than blocking the queue while it kept dying. Both properties
were written for exactly this and both did their job without supervision.

### The replay, once more

Play 1 came in at `raw` 1.0267 — above the cap, so the replay was worth nothing —
and the solver replayed anyway. Level 1 went 54 actions to **5**, a 0.09× ratio.
Twelve replays now under the repaired surface, **two of which carried the score**.

### Proofread

Clean. 69 commands, none left the workspace; nothing inbound; card corroborates 6
levels and 245 actions.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **15 of 25** | **14.7252 / 15 = 98.17%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| never run | 2 | — |

---

## Rollout 24: `re86-8af5384d` — **1.0000 (8/8)**, the largest environment in the set

| | |
|---|---|
| levels | **8 of 8** |
| actions | 1962 across 3 plays (737 + 644 + 581) |
| `raw` | 1.0512 → **1.1500** |
| `cap` | 1.0000 |
| **`E`** | **1.0000** |
| deaths / wasted | **0 / 0** |
| wall clock | 98 min, one-shot on attempt 2, exit 0 |
| cost | $33.44 over 176 turns |

Against baselines `[26, 42, 86, 108, 189, 139, 424, 241]` — **1255 actions, the
largest published total of the 25**:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| play 1 | 25 | 44 | 47 | 71 | 73 | 64 | **139** | **274** |
| ratio | 0.96× | 1.05× | 0.55× | 0.66× | 0.39× | 0.46× | **0.33×** | **1.14×** |
| play 2 | 20 | 36 | 47 | 44 | 67 | 114 | 106 | **210** |
| ratio | 0.77× | 0.86× | 0.55× | 0.41× | 0.35× | 0.82× | 0.25× | 0.87× |

Level 7 carries a **424-action** baseline — the second-largest single level in the
project — and fell in 139, then 106. The one level that resisted was level 8 at
1.14× on the first play, taken to 0.87× on the second, which is what lifted `raw`
from 1.0512 to the ceiling.

Like `cd82`, this is attempt 2: attempt 1 was killed by the 08:16 PDT worker
restart after 86 minutes and 904 actions. Also like `cd82`, it cost nothing —
the environment keeps the best card and the discarded attempt is invisible.

### Proofread

Clean. 175 commands, none left the workspace; nothing inbound.

### Standing, split by generation

| | environments | score |
|---|---|---|
| **repaired surface** | **16 of 25** | **15.7252 / 16 = 98.28%** |
| superseded surface | 8 | 7.4075 / 8 = 92.59% |
| **remaining** | **1** (`su15`, in flight) | — |

**One environment left in the entire queue.** When `su15` banks, all 25 will have
a clean one-shot result and 17 of them will be under the repaired surface.
