# Deep-proofread findings not yet adjudicated

Recovered from `wf_166ed0b5-984`'s journal after a container restart killed the
run mid-verify. **20 hunted, 3 verified (all confirmed), 8 since fixed.** The rest
are UNVERIFIED — the sweep's own verify phase never reached them, so treat each as
a claim to check, not a defect to fix. Two of tonight's findings were refuted on
inspection and one proposed fix would have broken correct code.

| # | file | axis | blocks sweep | claim |
|---|---|---|---|---|
| 3 | `clean_rollouts.py:346` | guards that pass by not running | no | `uncorroborated()`: "A run nobody but us can confirm is not a clean run... a card that has seen fewer levels than we claim to have cleared is a card t |
| 5 | `test_proofread_reach.py:206` | tests that cannot fail | no | The test `test_an_inbound_median_array_actually_fails_the_run` says its subject is `FAILING_VERDICTS`, "the tuple that decides pass/fail by prefix", a |
| 6 | `test_build_trace_audit_best_play.py:23` | tests that cannot fail | no | The module docstring says `arc_actions_per_level` used a "furthest first, then cheapest at that depth" proxy instead of the RHAE maximum, that this pr |
| 7 | `test_arc_proxy_allowlist.py:86` | tests that cannot fail | no | `test_close_response_loses_every_baseline_field` and `test_score_fields_are_stripped_at_every_depth` present themselves as pinning the response filter |
| 9 | `client.py:205` | cross-file claim | no | In the `HIDE_BASELINES_ENV` docstring: "`session.build_cli_args` sets it for the child only, and nothing sets it on the runner." This is the only stat |
| 10 | `build_trace_audit.py:603` | cross-file claim | no | In `surface_digest`'s docstring, justifying excluding the child environment ("whether `ARC_API_KEY` and `CCARC3_MAX_ACTIONS` reach the child") from th |
| 11 | `arc_proxy.py:62` | cross-file claim | no | Above the `ALLOW` tuple: "Every path a solver legitimately needs, and nothing else. Derived from the only four call sites in client.py that reach the  |
| 15 | `clean_rollouts.py:792` | sweep end-to-end | no | Lines 786-796: "Catch a split while it is still recoverable. `verify_one_card.py` is the end-of-sweep check; by then a card that was reaped mid-sweep  |
| 16 | `CCARC3_DOCTRINE.md:567` | figure re-derivation | **yes** | §6a, of the level whose `status()` line reads `95/308 actions returned the board to a state already seen on this level`: "That is the level the lost r |
| 17 | `CCARC3_DOCTRINE.md:273` | figure re-derivation | no | §0b: "`tn36` has five runs on record: one cleared 5 of 7 and the rest cleared all seven, the fastest in **220 actions**." |
| 18 | `rules.py:282` | figure re-derivation | no | `effective_actions` docstring: "A run that failed outright spent **20% of its actions on no-ops** -- 46 of 157 clicks left the board untouched" — the  |
| 19 | `ccarc3_results.md:2535` | figure re-derivation | no | Arm caveat: "Three results turned on harness fixes shipped mid-arm: best-of-plays scoring (`re86`, +0.4167) and crash-as-interruption (`lf52` +0.5091, |

## Detail

### 3 — `clean_rollouts.py:346`

**Claimed:** `uncorroborated()`: "A run nobody but us can confirm is not a clean run... a card that has seen fewer levels than we claim to have cleared is a card that stopped listening." The guard is `best = max(done) if done else 0` against `data['levels_reached']`, and is the only independent check before a run is banked.

**Asserted reality:** `levels_completed` is a list with one entry **per play**, and under the sweep's shared card that list accumulates across *attempts* of the same game, not just plays within one attempt. `max(done)` therefore includes the high-water mark of every discarded earlier attempt. A game whose attempt_1 reached level 8 and was discarded, and whose attempt_2 has a card entry that stopped updating (the `sb26` failure this guard was written for -- 8/8 with a card frozen at level 3), still reads `best=8 >= reached=8` and banks as corroborated. The guard's meaning silently changed when the shared card landed on 2026-08-08 and none of its three copies were updated.

**How they say they checked:** Decoded a real preserved card: `evidence/ccarc3/clean_rollouts/su15-1944f8ab/attempt_3/su15-1944f8ab/scorecard.json.gz` -> `cards['su15-1944f8ab']['levels_completed'] == [9, 9, 9]` with `total_plays: 3` and three `actions_by_level` rows, i.e. one entry per play. `result.json` for that attempt records `playthroughs: 3`, so plays and entries correspond 1:1. Under a per-game card those three plays all belong to one attempt; under the sweep's shared card (`clean_rollouts.sweep_card` -> `ab.use_shared_card` -> `build_without_baselines` sets `config.card_id`, and `ArcClient.open` at client.py:661-667 keeps the lent card) the same `cards[gid]` entry is appended to by every attempt. Contrast `ArcClient._assert_server_agrees` (client.py:715-717), which for the same field deliberately takes `done[-1]`, "the play now in flight" -- the two readers of the identical field disagree about which plays are in scope.

**Proposed fix:** Compare only the plays this attempt produced. `result.json` already carries `playthroughs`, so: `n = data.get('playthroughs') or 1; mine = done[-n:]; best = max(mine) if mine else 0`. Apply identically at `proofread_trace.py:442` and `restore_clean_rollouts.py:93`. Additionally, make `len(done) < n` itself a failure -- a card holding fewer plays than the trace recorded is a card that stopped listening, which is the condition the guard is named for.

### 5 — `test_proofread_reach.py:206`

**Claimed:** The test `test_an_inbound_median_array_actually_fails_the_run` says its subject is `FAILING_VERDICTS`, "the tuple that decides pass/fail by prefix", and closes with a loop over all five prefixes — presenting itself as pinning that every verdict prefix the proofreader can emit actually fails a run ("a verdict nothing acts on is a check that reports by not running").

**Asserted reality:** Lines 206-207 are a tautology: `for prefix in (...): assert any(v.startswith(prefix) for v in [prefix + ": x"])`. The list being scanned is built from `prefix` itself, so `startswith` is unconditionally true and `pt.FAILING_VERDICTS` is never read. The only real assertion is line 203, `"INBOUND" in pt.FAILING_VERDICTS`. So of the five prefixes the tool emits (LEAK at proofread_trace.py:432, REACH at 376, CARD at 444/446, NOT ONE-SHOT at 451, INBOUND at 396), exactly one is pinned. Grep confirms `FAILING_VERDICTS` appears in only two places outside the tool itself, both on line 203 and in that docstring.

**How they say they checked:** Mutation: `FAILING_VERDICTS = ("LEAK", "REACH", "CARD", "NOT", "INBOUND")` → `("INBOUND",)` in tools/proofread_trace.py:155 left the suite at 765 passed, 5 skipped — no test noticed that the leak, reach, card-disagreement and not-one-shot verdicts had stopped failing a run. Read of tools/proofread_trace.py:474 confirms the predicate is `any(v.startswith(FAILING_VERDICTS) for v in verdicts)` and nothing else consults the tuple.

**Proposed fix:** Replace lines 206-207 with `assert set(pt.FAILING_VERDICTS) >= {"LEAK", "REACH", "CARD", "NOT", "INBOUND"}` — or better, drive `pt.main()` end-to-end once per verdict on a fixture that provokes it and assert exit 2, which would also pin the prefixes to the strings actually appended and remove the drift the code comment is worried about.

### 6 — `test_build_trace_audit_best_play.py:23`

**Claimed:** The module docstring says `arc_actions_per_level` used a "furthest first, then cheapest at that depth" proxy instead of the RHAE maximum, that this produced every `E` and `raw` in evidence/ccarc3/trace_audit/runs.json.gz and on the published page, and that "this test is the constructed case where they [disagree], so the proxy cannot come back unnoticed."

**Asserted reality:** The test never touches `tools/build_trace_audit.py`. It imports only `athanor.ccarc3.scoring` and calls `score_environment` twice on two hand-written per-level lists, asserting `a.score > b.score` — a property of the RHAE weighting, not of the audit's play selector, which lives at tools/build_trace_audit.py:351-366. Worse, the constructed pair could not express the failure even if it were wired up: `play_a=[30,10,10]` and `play_b=[10,10,30]` have equal depth AND equal totals (the test asserts the equality itself at line 37), so the old proxy key `(len(plays[i]), -last_cumulative)` ties and `max` returns index 0 — which is also the rubric's winner. The two selectors agree on this fixture.

**How they say they checked:** (1) Coverage over the whole suite: `tools/build_trace_audit.py  542 stmts  542 miss  0%  26-1107` — not one line executes under any test. (2) Mutation: replacing tools/build_trace_audit.py:362 `best = max(range(len(plays)), key=_rank)` with the old proxy `max(range(len(plays)), key=lambda i: (len(plays[i]), -(plays[i][-1][1] if plays[i] else 0)))` left the suite at 765 passed, 5 skipped. (3) Hand-evaluating the proxy key on the test's own two plays gives an exact tie, and `max` returns the first maximal element.

**Proposed fix:** Call the real function: build a two-play `scorecard.json` under a tmp game dir, monkeypatch `athanor.ccarc3.client.baselines_for` to return the fixture medians, and assert `build_trace_audit.arc_actions_per_level(dir, gid)` returns the rubric's play. Use a pair where the two selectors genuinely diverge — e.g. baselines (10,10,10) with play A `[[1,30],[2,40],[3,50]]` and play B `[[1,10],[2,20],[3,45]]`: same depth, B is cheaper so the proxy picks B, but A scores 0.852 against B's 0.580.

### 7 — `test_arc_proxy_allowlist.py:86`

**Claimed:** `test_close_response_loses_every_baseline_field` and `test_score_fields_are_stripped_at_every_depth` present themselves as pinning the response filter — the second half of the withholding defence, after the allowlist — with the file docstring framing `arc_proxy` as the thing whose "entire job is to withhold" the medians.

**Asserted reality:** The single fixture `CLOSE_BODY` (line 56) exercises only 4 of the 6 entries in `arc_proxy.HIDDEN_FIELDS` (arc_proxy.py:240-247). Nothing anywhere in the suite contains the keys `baseline_actions` (line 242) or `scores` (line 245), so neither is pinned: both can be deleted from the frozenset with every test still green. `baseline_actions` is the exact key `GET /api/games` returns for all 25 environments — the one field name the whole module exists to name and remove.

**How they say they checked:** Two mutations, each run against the full suite: deleting `"baseline_actions",` from HIDDEN_FIELDS → 765 passed, 5 skipped; deleting `"scores",` → 765 passed, 5 skipped. Grepping the tests for either bare key returns no fixture that carries it, so the assertion `for banned in (...)` at line 86 can only ever confirm the three names the fixture happens to include.

**Proposed fix:** Parametrize the filter test over `arc_proxy.HIDDEN_FIELDS` itself rather than a hand-listed subset: for each field name, plant it at three depths in a synthetic body and assert it does not survive `_filtered`. That makes the test grow with the frozenset instead of drifting behind it, and would fail on any entry silently removed.

### 9 — `client.py:205`

**Claimed:** In the `HIDE_BASELINES_ENV` docstring: "`session.build_cli_args` sets it for the child only, and nothing sets it on the runner." This is the only statement anywhere of who arms layer 3 of the withholding stack (docs/ccarc3_withholding.md §3).

**Asserted reality:** Nothing in `athanor.ccarc3.session` sets `CCARC3_HIDE_BASELINES`. `build_cli_args` assembles argv and never touches `env` at all; `build_workspace`, which builds the child's env dict, mentions the variable only in a comment at line 366. The single assignment in the repo is `tools/ablate_baselines.py:503` (`ws.env[HIDE_BASELINES_ENV] = "1"`), inside `build_without_baselines`. So layer 3 exists only on the strip path: a plain `run_game()` hands the child a workspace with the real per-level medians, an ARC key, and no flag.

**How they say they checked:** `grep -rn 'HIDE_BASELINES' src tools tests` -> the only assignment is tools/ablate_baselines.py:503. `inspect.getsource(session.build_cli_args)` contains neither "HIDE_BASELINES" nor "env". Ran `session.build_workspace(Ccarc3Config(game_id='zz99-abc'), GameInfo(...,(10,20,30)))` with `CCARC3_PROXY_URL` unset: `'CCARC3_HIDE_BASELINES' in ws.env` is False, `'ARC_API_KEY' in ws.env` is True, rendered session.py carries `baseline_actions=(10, 20, 30)`.

**Proposed fix:** Replace the sentence with the truth — "`tools/ablate_baselines.build_without_baselines` sets it on the child's env; nothing in `session.py` does, so a plain `run_game()` withholds nothing" — or, better, set it in `session.build_workspace` beside the `env.pop("ARC_API_KEY")` so the claim becomes true and the mainline gains the layer the docs already credit it with.

### 10 — `build_trace_audit.py:603`

**Claimed:** In `surface_digest`'s docstring, justifying excluding the child environment ("whether `ARC_API_KEY` and `CCARC3_MAX_ACTIONS` reach the child") from the digest: "The environment is checked directly instead, per run, by `proofread_trace.py`, which reads it from the live process rather than inferring it." (sentence spans lines 603-605)

**Asserted reality:** `proofread_trace.py` never reads any process's environment. `main()` (line 335) takes a workspace path and reads `stream.jsonl` (line 346); `--gz` mode is explicitly for preserved evidence, where no process exists. Its only environment-related rule is `PROBES` (line 214), a regex over the recorded transcript that flags when the solver *ran* `os.environ` / `printenv` / `env |` / `CCARC3_MAX_ACTIONS` — and the comment above it (lines 200-212) states such a probe "is reported for reading and never fatal". That is inference from the transcript, and only when the solver happened to look.

**How they say they checked:** `grep -n 'environ|/proc|ARC_API_KEY|CCARC3_MAX_ACTIONS|psutil' tools/proofread_trace.py` -> hits only in the ESCAPES/PROBES regexes and the SYSTEM_ROOTS literal; no `/proc/<pid>/environ` read anywhere. Read `main()` lines 335-351: input is `ws / "stream.jsonl"`.

**Proposed fix:** Either delete the false clause and say plainly that the child environment is recorded nowhere and is a known gap, or make it true: have `session.collect_outcome` write the child's env *key names* (never values) — e.g. `{"child_env": {"ARC_API_KEY": false, "CCARC3_HIDE_BASELINES": true, "CCARC3_ARC_ROOT": true}}` — into `result.json`, and have `proofread_trace.py` assert on it.

### 11 — `arc_proxy.py:62`

**Claimed:** Above the `ALLOW` tuple: "Every path a solver legitimately needs, and nothing else. Derived from the only four call sites in client.py that reach the network."

**Asserted reality:** client.py has five network call sites, not four: `_get(f"{root}/api/games")` at line 254 (inside `list_games`), `_post(.../api/scorecard/open)` at 668, `_post(.../api/scorecard/close)` at 753, `_get(.../api/scorecard/{card}/{game})` at 964, and `_post(.../api/cmd/{name})` at 1074. The fifth is the one endpoint the allowlist exists to deny: `/api/games` returns `baseline_actions` for all 25 environments (session.py:366-370, ablate_baselines.py:485-492). ALLOW is client.py's network surface minus one, deliberately — not a transcription of it.

**How they say they checked:** `grep -n '_post(f"\|_get(f"' src/athanor/ccarc3/client.py` -> lines 254, 668, 753, 964, 1074 (five). `len(arc_proxy.ALLOW)` is 4 and `arc_proxy._allowed('/api/games')` is False.

**Proposed fix:** Rewrite as: "Derived from client.py's network call sites **minus `/api/games`**. `list_games` (client.py:254) is the fifth call site and is deliberately excluded and must stay excluded — it returns `baseline_actions` for all 25 environments."

### 15 — `clean_rollouts.py:792`

**Claimed:** Lines 786-796: "Catch a split while it is still recoverable. `verify_one_card.py` is the end-of-sweep check; by then a card that was reaped mid-sweep has already cost every game after it. One read per finished game turns that into a line in the log at the moment it happens." It compares `_card_of(workspace)` (the `card_id` in the game's `trace.state.json`) against `_shared_card_id()` (the `card_id` in `shared_card.json`).

**Asserted reality:** The comparison cannot ever be unequal in this driver, so the check never fires. `_run_one` passes `fresh=True` (l.768-772); `build_workspace` unlinks `trace.state.json` when fresh (session.py:293-296); with no state file `ArcClient._resume()` is never entered, and lines 569-572 of client.py are the ONLY place `self.card_id` can become anything other than the injected sweep card (and the only place `foreign_card` is ever set — client.py:571 is its sole assignment). `open()` then takes the `if self.card_id:` branch (client.py:662-665) and `_save_state()` writes back exactly the id it was handed. So `got == want` by construction, for every game, always. The same construction makes `verify_one_card.py`'s `foreign_card` branch (l.68-73) dead for this driver too.

**How they say they checked:** `grep -n 'self\.card_id *=\|self\.foreign_card *=' src/athanor/ccarc3/client.py` -> assignments only at 571, 572 (inside `_resume`), 670 (only when no card was injected) and 745 (close). Traced `fresh=True` -> session.py:293-296 unlink of `trace.state.json` -> `_resume` guarded by `self.state_path.exists()` (client.py:547) -> `_resumed` False. `_card_of` (l.700-706) reads only that file.

**Proposed fix:** Compare against the sweep's history rather than against the game's own copy of the injected id. After each finished game, scan `OUT` for the distinct `card_id`s recorded in every `*/attempt_*/*/result.json` (or `trace.state.json`) plus the live `_shared_card_id()`; if more than one appears, print the split and set `_aborted` so the pass stops instead of spending the remaining queue. That is cheap, local, and fires on the first game after a succession.

### 16 — `CCARC3_DOCTRINE.md:567`

**Claimed:** §6a, of the level whose `status()` line reads `95/308 actions returned the board to a state already seen on this level`: "That is the level the lost run never cleared." The whole §6a rule — "over pace, many revisits ... More of the same will not break out. Change *what* you are trying" — rests on this single observation being a failure.

**Asserted reality:** That level (tn36 L5) WAS cleared. It cost 5.62x its reference count and scored 0.03, and the level that was never cleared is the last one, at 0.47x. The doctrine's own arithmetic proves it: §0a line 229 records `raw` 0.442 "with five of seven cleared" and §0a line 117 records 0.449 with "six of seven levels" — the 0.007 gap is exactly L5's weighted contribution, 6 x 0.0317 / 28 = 0.0068. A level that was never cleared contributes 0.

**How they say they checked:** docs/ccarc3_results.md:233-236 tn36 per-level table: `| ratio | 0.53x | 2.57x | 0.35x | 1.52x | 0.70x | 5.62x | 0.47x |` / `| score | 1.15 | 0.15 | 1.15 | 0.43 | 1.15 | 0.03 | 0.00 |`. A non-zero level score is only reachable for a completed level (scoring.py:132-134 sets score 0.0 when `agent is None`). Recomputed by hand: (1*1.15+2*0.15+3*1.15+4*0.43+5*1.15+6*0.03+7*0)/28 = 12.55/28 = 0.448 -> the doc's 0.449; dropping L5's term gives 12.37/28 = 0.442, the exact figure the doctrine prints for "five of seven cleared". Cross-checks: results.md:178 scores this run `6/7`; client.py:410 tabulates the same row as `tn36 L5  5.62x  31%  yes, barely` — while client.py:431, ten lines below in the same docstring, says "`tn36` L5 ran at 5.62x and 31% and never fell", and docs/ccarc3_design.md:70 says `cleared? no`.

**Proposed fix:** Rewrite doctrine:567 to "That is the level that cost the lost run its game. It did eventually fall — at 5.62x its reference count, for a level score of 0.03 — and the actions it ate are why the last level was never reached." Demote the §6a table row from "More of the same will not break out" to "expect this level to be expensive, not impossible" (which is what design.md:105 already concluded: "a cost signal, not a failure signal"). Fix the two copies that disagree: client.py:431 "never fell" and design.md:70 `cleared? no`.

### 17 — `CCARC3_DOCTRINE.md:273`

**Claimed:** §0b: "`tn36` has five runs on record: one cleared 5 of 7 and the rest cleared all seven, the fastest in **220 actions**."

**Asserted reality:** Of the five, one cleared 5 of 7, one cleared **6 of 7**, and three cleared all seven. The 6-of-7 run is the doctrine's own flagship worked example, described 155 lines earlier in the same file.

**How they say they checked:** doctrine:117-118 (§0a) states "one run finished at `raw` **0.449** against a `cap` of 0.750, having cleared six of seven levels" — and cap 0.750 = sum(1..6)/sum(1..7) = 21/28, which is arithmetically only reachable at six cleared levels. docs/ccarc3_results.md:178 tables that run as `6/7`; results.md:2542-2546 enumerates the environment's runs as "scoring 1.0000, 0.5357 and 0.4487", the 0.4487 being that 6-of-7 control. Preserved runs on disk confirm the spread: evidence/ccarc3/ablate_nobaseline/tn36-ef4dde99/result.json.gz (7/7, 220 actions), evidence/ccarc3/rerun_losses/tn36-ef4dde99/result.json.gz (7/7), plus the clean-rollout and void copies.

**Proposed fix:** "`tn36` has five runs on record: one cleared 5 of 7, one 6 of 7, and three cleared all seven, the fastest in 220 actions." The point survives intact — three of five swept the game the other two called impossible.

### 18 — `rules.py:282`

**Claimed:** `effective_actions` docstring: "A run that failed outright spent **20% of its actions on no-ops** -- 46 of 157 clicks left the board untouched" — the fraction is offered, via the em-dash, as the evidence for the percentage.

**Asserted reality:** 46/157 = 29.3%, not 20%. The two figures are different measurements welded together: 20% is the whole-run no-op rate (67 dead of 336 actions = 19.9%), while 46 of 157 is the ACTION6-only rate. The doctrine renders this same evidence correctly.

**How they say they checked:** Arithmetic: 46/157 = 0.293. The doctrine's §7a (doctrine:599-601) states the identical evidence as "a run that never cleared a level spent **almost one action in three** changing nothing at all — 46 of its 157 clicks landed on dead ground". The 20% denominator is client.py:1356-1357, which records that same failing run as "(336 tried, 67 dead, 17 repeats)" — 67/336 = 19.9% — and client.py:1428-1429 confirms the click split, "it never fired once in 336 actions: ``ACTION6`` was not dead, it was 111/157" (157 - 111 = 46).

**Proposed fix:** "A run that failed outright spent almost one action in three on no-ops -- 46 of 157 clicks left the board untouched, 20% across all 336 of its actions -- while every winning run in the same batch wasted none." Both numbers are then true and their denominators are visible.

### 19 — `ccarc3_results.md:2535`

**Claimed:** Arm caveat: "Three results turned on harness fixes shipped mid-arm: best-of-plays scoring (`re86`, +0.4167) and crash-as-interruption (`lf52` +0.5091, `wa30` +0.5833). Without those three fixes the same runs would total **22.1667 -> 88.67%**." Line 2448 repeats it: "The two recoveries together are worth **+1.0924**."

**Asserted reality:** `wa30`'s +0.5833 is impossible given the level count stated two sections above it, and the total does not follow from its own three deltas under any reading. wa30 was banked at 5 of 9 levels, so its counterfactual E is capped at sum(1..5)/sum(1..9) = 15/45 = 0.3333 and the fix is worth at least +0.6667. 0.5833 = 21/36 is `sk48`'s 5/8 -> 8/8 completion gain, copied one game across; no k of 9 levels produces a cap of 0.4167 (it would need sum(1..k) = 18.75).

**How they say they checked:** results.md:2340 "`wa30` (GAME_OVER at 5 of 9)" and session.py:1063 "`wa30` was banked at 5 of 9 levels" fix k=5, n=9. scoring.py:169-170 computes cap = sum(range(1, completed+1)) / total_weight and score = min(raw, cap), so E <= 15/45 = 0.3333. results.md:3448 shows +0.5833 as sk48's own delta (0.4167 (5/8) -> 1.0000). Total: 23.6667 - (0.4167 + 0.5091 + 0.5833) = 22.1576 -> 88.63%, not the stated 22.1667 -> 88.67%; with the corrected wa30 delta it is 22.0742 -> 88.30%. (Secondary, same paragraph: `lf52`'s 0.4909 at line 2385 is also unreachable at the "6 of 10" stated at line 2340 — cap = 21/55 = 0.3818 — it is only reachable at 7 of 10, cap 28/55 = 0.5091.)

**Proposed fix:** Recompute from the stated level counts: `wa30` 5 of 9 -> cap 0.3333, fix worth +0.6667; make `lf52`'s counterfactual level count agree with its 0.4909 (7 of 10, cap 0.5091) or restate the E for 6 of 10 (0.3818, +0.6182); then re-derive both the pair total at line 2448 and the "without those three fixes" row at line 2536 from the corrected deltas.

