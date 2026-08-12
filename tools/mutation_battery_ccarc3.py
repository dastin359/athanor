"""Every mutant the solver-facing ccarc3 modules were audited with.

Run it to re-verify the audit rather than trusting this file's summary:

    .venv/bin/python tools/mutation_battery_ccarc3.py            # every module
    .venv/bin/python tools/mutation_battery_ccarc3.py grids      # one module

Exit status is the number of unexpected survivors. Mutants listed in
``EQUIVALENT`` are expected to survive and are *not* counted -- each one has an
argument recorded in ``docs/ccarc3_open_findings.md``, and a battery that quietly
tolerated them would be indistinguishable from one with two untested gaps.

It lives in ``tools/`` and not in the scratchpad on purpose: the scratchpad
reverts to an image snapshot when the container is replaced, and that has
already taken one audit harness with it.

The counter-practice, restated because it is the whole point: a test that passes
is not evidence until something has made it fail. Write the mutant for what the
test *claims* to protect, then confirm the suite goes red. A survivor is either
a gap or an equivalence, and telling those two apart is the work -- assuming
equivalence is how a suite acquires tests that assert coincidences.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "tools")
import mutation_check as mc  # noqa: E402

# Mutants that cannot change behaviour. Argued in docs/ccarc3_open_findings.md.
EQUIVALENT = {"collapse_alias", "cb_bounds", "run_empty_ledger", "cd_empty_is_zero",
              "budget_zero_means_none", "reap_ge"}

GRIDS = ("src/athanor/ccarc3/grids.py", [
    ("palette_upper", "accept 16 as a legal colour",
     "arr.max() >= PALETTE_SIZE", "arr.max() > PALETTE_SIZE"),
    ("palette_lower", "accept -1 as a legal colour",
     "arr.min() < 0", "arr.min() < -1"),
    ("ndim", "accept a 3-D grid",
     "if arr.ndim != 2:", "if arr.ndim != 2 and False:"),
    ("diff_order", "return changed cells in reverse order",
     "for y, x in zip(ys, xs)]", "for y, x in zip(ys, xs)][::-1]"),
    ("diff_beforeafter", "swap before and after in a Change",
     "int(a[y, x]), int(b[y, x])", "int(b[y, x]), int(a[y, x])"),
    ("blocksize_empty", "return 0 for an empty grid",
     "if h == 0 or w == 0:\n        return 1", "if h == 0 or w == 0:\n        return 0"),
    ("blocksize_smallest", "return the smallest factor, not the largest",
     "for k in range(min(h, w), 1, -1):", "for k in range(2, min(h, w) + 1):"),
    ("blocksize_minonly", "check block min only, so a non-uniform block passes",
     "blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))",
     "blocks.min(axis=(1, 3)), blocks.min(axis=(1, 3))"),
    ("logical_alias", "alias the input on the branch that does not reduce",
     "return arr.copy() if k == 1", "return arr if k == 1"),
    ("logical_alias2", "alias the input on the branch that does reduce",
     "else arr[::k, ::k].copy()", "else arr[::k, ::k]"),
    ("collapse_rows_only", "collapse rows but not columns",
     "keep_cols[1:] = (arr[:, 1:] != arr[:, :-1]).any(axis=0)", "keep_cols[1:] = True"),
    ("collapse_empty_alias", "alias an empty input",
     "if arr.size == 0:\n        return arr.copy()", "if arr.size == 0:\n        return arr"),
    ("collapse_alias", "alias on the reducing branch (EQUIVALENT: advanced indexing copies)",
     "return arr[:, keep_cols].copy()", "return arr[:, keep_cols]"),
    ("mono_threshold", "require strictly more than the threshold",
     "if c >= total * threshold", "if c > total * threshold"),
    ("mono_shape_skip", "count a shape-mismatched pair in the total",
     "if a.shape != b.shape:\n            continue\n        total += 1",
     "if a.shape != b.shape:\n            total += 1\n            continue\n        total += 1"),
    ("mono_unsorted", "return rows in Counter order, not sorted",
     "return sorted(r for r, c in hits.items() if c >= total * threshold)",
     "return list(r for r, c in hits.items() if c >= total * threshold)"),
    ("cb_offbyone", "report the row boundary one row early",
     "rows.update(int(i) + 1 for i in np.nonzero(changed_rows)[0])",
     "rows.update(int(i) for i in np.nonzero(changed_rows)[0])"),
    ("cb_offbyone_col", "report the column boundary one column early",
     "cols.update(int(i) + 1 for i in np.nonzero(changed_cols)[0])",
     "cols.update(int(i) for i in np.nonzero(changed_cols)[0])"),
    ("cb_no_zero", "omit the implicit boundary at 0",
     "rows: set[int] = {0}", "rows: set[int] = set()"),
    ("cb_bounds", "let a boundary equal the height (EQUIVALENT once the ragged guard exists)",
     "return sorted(r for r in rows if r < height), sorted(c for c in cols if c < width)",
     "return sorted(r for r in rows if r <= height), sorted(c for c in cols if c <= width)"),
    ("cb_ragged_guard", "pool across two board shapes again",
     "if height and arr.shape != (height, width):", "if False and arr.shape != (height, width):"),
    ("cb_ragged_heightonly", "compare only the height",
     "if height and arr.shape != (height, width):", "if height and arr.shape[0] != height:"),
    ("cb_ragged_widthonly", "compare only the width",
     "if height and arr.shape != (height, width):", "if height and arr.shape[1] != width:"),
    ("cb_single_grid", "accept a lone 2-D array instead of refusing it",
     "if isinstance(grids, np.ndarray) and grids.ndim == 2:",
     "if isinstance(grids, np.ndarray) and grids.ndim == 3:"),
    ("obj_conn", "treat any connectivity as 8",
     "offsets = _NEIGHBOURS_4 if connectivity == 4 else _NEIGHBOURS_8",
     "offsets = _NEIGHBOURS_8"),
    ("obj_bg_least", "infer the least common colour as background",
     "background = Counter(arr.ravel().tolist()).most_common(1)[0][0]",
     "background = Counter(arr.ravel().tolist()).most_common()[-1][0]"),
    ("obj_order", "sort smallest object first",
     "found.sort(key=lambda o: (-o.size, sorted(o.cells)[0]))",
     "found.sort(key=lambda o: (o.size, sorted(o.cells)[0]))"),
    ("obj_bbox", "return an exclusive bbox",
     "return min(ys), min(xs), max(ys), max(xs)",
     "return min(ys), min(xs), max(ys) + 1, max(xs) + 1"),
    ("counts_drop", "drop colours with a count of one",
     "return dict(Counter(arr.ravel().tolist()))",
     "return {k: v for k, v in Counter(arr.ravel().tolist()).items() if v > 1}"),
    ("all_counts", "drop counts from grids.__all__ again",
     '    "objects",\n    "counts",\n]', '    "objects",\n]'),
])

RULES = ("src/athanor/ccarc3/rules.py", [
    ("na_becomes_violated", "an unmet precondition counts as a refutation",
     "if not self.applies(t):\n            return Outcome.NOT_APPLICABLE",
     "if not self.applies(t):\n            return Outcome.VIOLATED"),
    ("na_becomes_holds", "an unmet precondition counts as a pass",
     "if not self.applies(t):\n            return Outcome.NOT_APPLICABLE",
     "if not self.applies(t):\n            return Outcome.HOLDS"),
    ("scope_validation", "accept any scope string",
     'if self.scope not in ("level", "game"):',
     'if self.scope not in ("level", "game", *()) and False:'),
    ("applicable_counts_na", "count not-applicable as applicable",
     "return self.holds + self.violated\n", "return self.holds + self.violated + self.not_applicable\n"),
    ("refuted_always", "call every rule refuted",
     "return self.counts.violated > 0", "return self.counts.violated >= 0"),
    ("verified_vacuously", "a rule never applicable counts as verified",
     "return self.counts.violated == 0 and self.counts.holds > 0",
     "return self.counts.violated == 0"),
    ("vacuous_wrong", "call a rule vacuous whenever it never held",
     "return self.counts.applicable == 0", "return self.counts.holds == 0"),
    ("verify_unscoped", "let verify refute using every level's transitions",
     "scoped = [t for t in transitions if t.level == level]", "scoped = list(transitions)"),
    ("verify_first_level", "default to the first level, not the level being played",
     "level = max((t.level for t in transitions), default=0)",
     "level = min((t.level for t in transitions), default=0)"),
    ("survey_one_level", "survey only the current level",
     "levels = sorted({t.level for t in transitions})",
     "levels = sorted({t.level for t in transitions})[-1:]"),
    ("survey_inst_all", "count every level as an instantiation",
     "return len(self.levels_applicable)", "return len(self.per_level)"),
    ("survey_applicable_any", "call a level applicable when the rule was only ever n/a",
     "return [lv for lv, c in sorted(self.per_level.items()) if c.applicable]",
     "return [lv for lv, c in sorted(self.per_level.items()) if c.total]"),
    ("reg_level_rules", "raise regressions for level-scoped rules too",
     'if rule.scope != "game":\n            continue',
     'if rule.scope != "game" and False:\n            continue'),
    ("reg_not_vacuous", "report a mechanic that merely stopped applying",
     "if result.refuted:\n            out.append(result)",
     "if not result.vacuous:\n            out.append(result)"),
    ("reg_any_applicable", "report any game rule that was applicable",
     "if result.refuted:\n            out.append(result)",
     "if result.counts.applicable:\n            out.append(result)"),
    ("eff_keeps_boardswap", "count a level swap as an effective action",
     "if t.before is None or t.board_replaced or t.wasted:\n            continue",
     "if t.before is None or t.wasted:\n            continue"),
    ("eff_keeps_wasted", "count a wasted action in the denominator",
     "t.board_replaced or t.wasted:\n            continue", "t.board_replaced:\n            continue"),
    ("eff_denominator", "only count tries that changed something",
     "row[1] += 1\n        if t.changed:\n            row[0] += 1",
     "if t.changed:\n            row[1] += 1\n            row[0] += 1"),
    ("eff_level_filter", "ignore effective_actions' level filter",
     "if level is not None and t.level != level:\n            continue\n        row =",
     "if level is None and t.level != level:\n            continue\n        row ="),
    ("pace_first_reset", "cut at the first full reset, not the last",
     "cut = max((i for i, t in enumerate(transitions) if t.full_reset), default=0)",
     "cut = min((i for i, t in enumerate(transitions) if t.full_reset), default=0)"),
    ("pace_no_cut", "count a replayed level twice",
     "for t in transitions[cut:]:", "for t in transitions:"),
    ("pace_zero_baseline", "divide by a zero baseline",
     "if 0 <= level < len(baselines) and baselines[level]:",
     "if 0 <= level < len(baselines):"),
    ("pace_negative_level", "index baselines with a negative level",
     "if 0 <= level < len(baselines) and baselines[level]:",
     "if level < len(baselines) and baselines[level]:"),
    ("pred_crash_is_wrong", "count a model that raises as a wrong prediction",
     "except Exception:  # noqa: BLE001 -- a model that crashes has not predicted\n            predicted = None",
     "except Exception:  # noqa: BLE001\n            predicted = 0"),
    ("pred_keeps_boardswap", "score a forward model on level boundaries",
     "if t.before is None or t.board_replaced or t.wasted:\n            skipped += 1",
     "if t.before is None or t.wasted:\n            skipped += 1"),
    ("pred_level_filter", "ignore predict's level filter",
     "if level is not None and t.level != level:\n            continue\n        if t.before is None",
     "if level is None and t.level != level:\n            continue\n        if t.before is None"),
    ("pred_perfect_vacuous", "a model that predicted nothing is perfect",
     "return self.wrong == 0 and self.correct > 0", "return self.wrong == 0"),
    ("pred_accuracy_total", "divide accuracy by every transition, skips included",
     "return self.correct / self.tested if self.tested else 0.0",
     "return self.correct / (self.tested + self.skipped) if (self.tested + self.skipped) else 0.0"),
    ("book_order", "record a refuted rule as verified when it also held",
     "if result.refuted:\n            self.refuted.append(entry)\n        elif result.verified:",
     "if result.verified:\n            self.verified.append(entry)\n        elif result.refuted:"),
    ("book_vacuous_verified", "file a never-applicable rule as verified",
     "        else:\n            self.open_questions.append(",
     "        elif False:\n            self.open_questions.append("),
    ("book_load_missing", "raise instead of starting an empty book",
     "if not p.exists():\n            return cls()", "if not p.exists() and False:\n            return cls()"),
])

LEDGER = ("src/athanor/ccarc3/ledger.py", [
    ("an_bool", "accept True as action id 1",
     "if isinstance(raw, bool):  # bool is an int subclass; reject it explicitly\n        raise ValueError(f\"not an action id: {raw!r}\")",
     "if isinstance(raw, bool) and False:\n        raise ValueError(f\"not an action id: {raw!r}\")"),
    ("an_no_upper", "keep a lowercase action name as given",
     "if isinstance(raw, str):\n        return raw.upper()", "if isinstance(raw, str):\n        return raw"),
    ("an_unknown_id", "invent a name for an unknown action id",
     'raise ValueError(f"no GameAction with id {raw}") from None', 'return f"ACTION{raw}"'),
    ("an_enum_no_upper", "keep an enum-like name as given",
     "if isinstance(name, str):\n        return name.upper()", "if isinstance(name, str):\n        return name"),
    ("board_replaced_narrow", "a full reset is not a board replacement",
     "return self.crosses_level or self.full_reset", "return self.crosses_level"),
    ("board_replaced_reset_only", "a level completion is not a board replacement",
     "return self.crosses_level or self.full_reset", "return self.full_reset"),
    ("score_delta_sign", "report the score delta backwards",
     "return self.score_after - self.score_before", "return self.score_before - self.score_after"),
    ("changed_first", "the first transition changed nothing",
     "if self.before is None:\n            return True", "if self.before is None:\n            return False"),
    ("changed_shape", "a shape change is not a change",
     "if self.before.shape != self.after.shape:\n            return True",
     "if self.before.shape != self.after.shape:\n            return False"),
    ("changed_invert", "invert the board comparison",
     "return not np.array_equal(self.before, self.after)",
     "return np.array_equal(self.before, self.after)"),
    ("is_win_state", "read WIN as GAME_OVER",
     'return self.state == "WIN"', 'return self.state == "GAME_OVER"'),
    ("tw_params_all", "record every action_input data key, not just x and y",
     'if k in ("x", "y")', 'if True'),
    ("score_key_only", "reach the fallback only when the key is absent",
     '    for key in ("score", "levels_completed"):\n        value = frame.get(key)\n        if value is not None:\n            return int(value)\n    return 0',
     '    return int(frame.get("score", frame.get("levels_completed", 0)) or 0)'),
    ("score_zero_falls_through", "treat an explicit zero as missing",
     "if value is not None:", "if value:"),
    ("score_one_name", "read only `score`, never `levels_completed`",
     '    for key in ("score", "levels_completed"):', '    for key in ("score",):'),
    ("score_other_name", "read only `levels_completed`, never `score`",
     '    for key in ("score", "levels_completed"):', '    for key in ("levels_completed",):'),
    ("tw_index", "never advance the record index",
     "self._index += 1\n        return record", "return record"),
    ("tw_reset_flag", "drop the full_reset flag",
     '"full_reset": bool(frame.get("full_reset", False)),', '"full_reset": False,'),
    ("tw_available", "record raw action ids instead of names",
     'action_name(a) for a in frame.get("available_actions", [])',
     'str(a) for a in frame.get("available_actions", [])'),
    ("rec_forgive_all", "forgive a corrupt line anywhere in the trace",
     "if number == len(lines):\n                return", "if True:\n                return"),
    ("rec_forgive_none", "refuse a half-written final line",
     "if number == len(lines):\n                return", "if False:\n                return"),
    ("rec_skip_bad", "skip a corrupt middle line and keep reading",
     "if number == len(lines):\n                return", "if True:\n                continue"),
    ("rec_missing_file", "raise on a ledger that does not exist yet",
     "if not p.exists():\n        return", "if not p.exists() and False:\n        return"),
    ("load_wasted_dropped", "drop a wasted action instead of recording it",
     "if previous is None:\n                continue\n            grids = [previous]",
     "continue\n            grids = [previous]"),
    ("load_wasted_flag", "never mark an action wasted",
     "wasted = not grids", "wasted = False"),
    ("load_after_first", "take the first rendered frame as `after`, not the last",
     "after=grids[-1],", "after=grids[0],"),
    ("load_intermediate_last", "keep only the final frame as intermediate",
     "intermediate=tuple(grids),", "intermediate=(grids[-1],),"),
    ("load_crosses_first", "let the first transition cross a level",
     'previous is not None and int(rec.get("level", 0)) > previous_level',
     'int(rec.get("level", 0)) > previous_level'),
    ("load_crosses_ge", "call an unchanged level a crossing",
     'int(rec.get("level", 0)) > previous_level\n                ),',
     'int(rec.get("level", 0)) >= previous_level\n                ),'),
    ("load_reset_no_downgrade", "trust the server's full_reset flag alone",
     'full_reset=bool(rec.get("full_reset", False)) or (\n                    previous is not None and int(rec.get("level", 0)) < previous_level\n                ),',
     'full_reset=bool(rec.get("full_reset", False)),'),
    ("load_chain", "chain `before` from an intermediate frame",
     "previous = grids[-1]\n        previous_score", "previous = grids[0]\n        previous_score"),
    ("infer_ge", "count an unchanged score as a level boundary",
     "if s > previous:", "if s >= previous:"),
    ("infer_decrement", "count a score decrease as a level boundary",
     "if s > previous:", "if s != previous:"),
    ("infer_seed", "seed the comparison from zero rather than the first score",
     "previous = scores[0] if scores else 0", "previous = 0"),
])

SCORING = ("src/athanor/ccarc3/scoring.py", [
    ("plays_opening_reset", "let the opening RESET start an empty play",
     "if t.full_reset and i > 0", "if t.full_reset"),
    ("plays_no_implicit_start", "drop the implicit start at 0",
     "starts = [0] + [i for i, t in enumerate(kept) if t.full_reset and i > 0]",
     "starts = [i for i, t in enumerate(kept) if t.full_reset and i > 0]"),
    ("plays_off_by_one", "start each play one transition late",
     "return [kept[a:b] for a, b in zip(starts, starts[1:] + [len(kept)])]",
     "return [kept[a + 1:b] for a, b in zip(starts, starts[1:] + [len(kept)])]"),
    ("run_last_not_best", "score the last play instead of the best",
     "    return max(\n        candidates,", "    return candidates[-1] or max(\n        candidates,"),
    ("run_first_not_best", "score the first play",
     "    return max(\n        candidates,", "    return candidates[0] or max(\n        candidates,"),
    ("run_min_not_max", "score the worst play",
     "    return max(\n        candidates,", "    return min(\n        candidates,"),
    ("run_tiebreak_raw", "drop the raw-efficiency tie-break between two capped plays",
     "            s.score,\n            s.raw,", "            s.score,\n            0,"),
    ("run_tiebreak_actions", "drop the fewest-actions tie-break",
     "            -sum(level.agent or 0 for level in s.levels),", "            0,"),
    ("run_select_validation", "accept any select string as 'best'",
     'if select != "best":', 'if select != "best" and False:'),
    ("run_empty_ledger", "delete the unreachable empty-candidates guard (EQUIVALENT)",
     "    if not candidates:", "    if False:"),
    ("run_cumulative_implies_last", "let cumulative fall through to best-of-plays",
     'if cumulative or select == "last":', 'if select == "last":'),
    ("sapl_no_diff", "return the server's cumulative counts undifferenced",
     "out.append(cumulative - previous)", "out.append(cumulative)"),
    ("sapl_order_guard", "accept levels arriving out of order",
     "if level - 1 != len(out):", "if level - 1 != len(out) and False:"),
    ("sapl_default_play", "default to the FIRST play rather than the most recent",
     "def server_actions_per_level(scorecard: dict, game_id: str, *, play: int = -1)",
     "def server_actions_per_level(scorecard: dict, game_id: str, *, play: int = 0)"),
    ("sapl_missing_card", "return [] for a game with no card instead of raising",
     'if not card:\n        raise KeyError(f"no card for {game_id} in this scorecard")',
     "if not card:\n        return []"),
    ("cd_shared_slice", "take the last N rows even when the attempt boundary is known",
     "        mine = done[before:]", "        mine = done[-plays:]"),
    ("cd_boundary_ignored", "ignore card_plays_at_open entirely",
     "    if before >= 0:", "    if False:"),
    ("cd_short_card_ok", "let a card holding fewer plays than the trace pass",
     '        if len(mine) < plays:\n            return (f"card holds',
     '        if False:\n            return (f"card holds'),
    ("cd_level_ge", "accept a card one level behind the result",
     "    if best < reached:", "    if best < reached - 1:"),
    ("cd_best_is_last", "read the last play's level instead of the best",
     "    best = max(mine) if mine else 0", "    best = mine[-1] if mine else 0"),
    ("cd_empty_is_zero", "treat an empty play slice as agreeing (EQUIVALENT: unreachable)",
     "    best = max(mine) if mine else 0", "    best = max(mine) if mine else 10**6"),
    ("cd_plays_guard", "use a non-positive playthrough count as a slice length",
     "    if plays < 1:", "    if False:"),
    ("cd_plays_zero_refused", "refuse a zero or absent count instead of coercing it to one",
     "    if plays < 1:", "    if plays < 2:"),
    ("dws_range", "compare past the levels the server has recorded",
     "for i in range(min(len(theirs), len(ours)))", "for i in range(len(ours))"),
    ("dws_direction", "only report levels where ours is lower",
     "if ours[i] != theirs[i]", "if ours[i] < theirs[i]"),
])

CLIENT = ("src/athanor/ccarc3/client.py", [
    ("done_is_gameover", "read a loss as a win",
     'return self.state == "WIN"', 'return self.state in ("WIN", "GAME_OVER")'),
    ("dead_is_win", "read a win as a loss",
     'return self.state == "GAME_OVER"', 'return self.state in ("GAME_OVER", "WIN")'),
    ("budget_reads_visible", "compute the level budget from the solver-visible baseline",
     "        base = self._baseline_here_enforced\n        if not base",
     "        base = self.baseline_here\n        if not base"),
    ("baseline_leaks", "show the baseline even when pace is withheld",
     "return None if self.quiet_pace else self._baseline_here_enforced",
     "return self._baseline_here_enforced"),
    ("score_leaks", "score the play even when baselines are withheld",
     "        if self.quiet_pace:\n            return None",
     "        if False:\n            return None"),
    ("budget_zero_means_none", "drop the zero-multiple guard (EQUIVALENT: int(base*0) is 0)",
     "if not base or not self.level_budget_multiple:", "if not base:"),
    ("free_level_scores_zero", "a level credited without an action scores 0, not the cap",
     "                    LEVEL_SCORE_CAP if cost <= 0", "                    0.0 if cost <= 0"),
    ("free_level_threshold", "only a negative cost counts as free, so zero divides",
     "if cost <= 0\n", "if cost < 0\n"),
    ("uncleared_scores_cap", "an uncleared level scores the cap even pessimistically",
     "            else:\n                score = 0.0",
     "            else:\n                score = LEVEL_SCORE_CAP"),
    ("optimistic_pessimistic", "the ceiling scores unreached levels at zero",
     "            elif optimistic:\n                score = LEVEL_SCORE_CAP",
     "            elif optimistic:\n                score = 0.0"),
    ("cap_not_applied", "ignore the completion cap in the current score",
     "        cap = 1.0 if optimistic else self.completion_cap", "        cap = 1.0"),
    ("ceiling_uses_cap", "hold the ceiling down with the completion cap",
     "        cap = 1.0 if optimistic else self.completion_cap",
     "        cap = self.completion_cap"),
    ("score_max_not_min", "take the larger of raw and cap",
     "        return min(raw, cap)", "        return max(raw, cap)"),
    ("weights_unweighted", "weight every level equally instead of by index",
     "            weighted += index * score", "            weighted += score"),
    ("weight_denominator", "divide by the level count rather than the weight sum",
     "        raw = weighted / sum(range(1, n + 1))", "        raw = weighted / n"),
    ("levels_missing_ok", "score with fewer baselines than levels",
     "if not baselines or len(baselines) < n or not n:", "if not baselines or not n:"),
    ("reap_ge", "reap exactly at the deadline (EQUIVALENT: argued in the source)",
     "            if idle > self.REAP_DEADLINE_S:", "            if idle >= self.REAP_DEADLINE_S:"),
    ("reap_disabled", "never refuse a stale resume",
     "            if idle > self.REAP_DEADLINE_S:", "            if False:"),
    ("reap_no_clock", "skip the reap check when the clock was never wound",
     "        if self.last_touched:", "        if False:"),
    ("card_behind_ok", "resume when the card is behind the ledger",
     "        if int(done) < self.level:", "        if False:"),
    ("card_behind_strict", "refuse when the card merely equals the ledger",
     "        if int(done) < self.level:", "        if int(done) <= self.level:"),
    ("card_none_refuses", "refuse when the card says nothing",
     "        if done is None:\n            return", "        if done is None:\n            pass"),
    ("wasted_keeps_tally", "keep a stale tally across a board replacement",
     "            if board_replaced:\n                self.level_tried = self.level_dead = self.level_repeats = 0",
     "            if False:\n                self.level_tried = self.level_dead = self.level_repeats = 0"),
    ("revisit_counts_noop", "count a no-op as a revisit",
     "        if key != previous:\n            if key in self._seen_keys:",
     "        if True:\n            if key in self._seen_keys:"),
    ("dead_is_change", "count a board that moved as a dead action",
     "        if key != previous:\n            return\n        self.level_dead += 1",
     "        if key == previous:\n            return\n        self.level_dead += 1"),
    ("repeat_keys_ignore_xy", "treat every ACTION6 click as the same action",
     'what = f"{name}:{payload.get(\'x\')},{payload.get(\'y\')}" if name == "ACTION6" else name',
     'what = name'),
    ("tried_not_counted", "never count an action as tried",
     "        self.level_tried += 1", "        pass"),
])

# The shim every solver action passes through. It holds the credential, the
# action ceiling and the allowlist, so a defect here is not a wrong number in a
# report -- it is a leaked baseline, a miscounted budget, or a sweep whose shared
# card gets closed halfway. It had 41 tests across four files and had never been
# mutated, which is the state this whole battery exists to distrust.
PROXY = ("src/athanor/ccarc3/arc_proxy.py", [
    ("allowlist_off", "forward every path, not just the four",
     "return any(p.match(path) for p in ALLOW)", "return True"),
    ("games_allowlisted", "restore /api/games, the one endpoint the arm withholds",
     '    re.compile(r"^/api/scorecard/open$"),',
     '    re.compile(r"^/api/games$"),\n    re.compile(r"^/api/scorecard/open$"),'),
    ("allowlist_sees_query", "check the allowlist against the query string too",
     '        path = self.path.split("?", 1)[0]', "        path = self.path"),
    ("refuse_without_drain", "answer a refusal with the request body unread",
     '        n = int(self.headers.get("Content-Length") or 0)\n'
     "        payload = self.rfile.read(n) if n else None\n\n"
     "        if not _allowed(path):\n            self._refuse(path)\n            return\n",
     "        if not _allowed(path):\n            self._refuse(path)\n            return\n"
     '        n = int(self.headers.get("Content-Length") or 0)\n'
     "        payload = self.rfile.read(n) if n else None\n\n"),

    ("cmd_never_charges", "bill nothing, so the ceiling never binds",
     "charge = bool(_CMD.match(path))", "charge = False"),
    ("cmd_always_charges", "bill scorecard reads as actions",
     "charge = bool(_CMD.match(path))", "charge = True"),
    ("exhausted_off_by_one", "allow one action past the ceiling",
     "if self.max_actions and self.actions_used >= self.max_actions:",
     "if self.max_actions and self.actions_used > self.max_actions:"),
    ("exhausted_disabled", "never refuse for budget",
     "if self.max_actions and self.actions_used >= self.max_actions:",
     "if False:"),
    ("charge_on_error", "bill a 502 the solver was never served",
     "if charge and not (200 <= code < 300):", "if False:"),
    ("refund_never", "keep the slot upstream declined",
     "            if self.actions_used > 0:\n                self.actions_used -= 1",
     "            if self.actions_used > 0:\n                pass"),
    ("refund_underflows", "refund below zero",
     "            if self.actions_used > 0:\n                self.actions_used -= 1",
     "            if True:\n                self.actions_used -= 1"),
    ("charge_never", "accept the action and bill nothing",
     "            self.actions_used += 1\n        return None",
     "            pass\n        return None"),
    ("budget_race", "restore the two-acquisition check: ask, then take",
     "            self.actions_used += 1\n        return None",
     "        import time as _t\n        _t.sleep(0.05)\n"
     "        with self._lock:\n            self.actions_used += 1\n        return None"),
    ("forward_full_path", "forward the unchecked path, query string and all",
     "            UPSTREAM + path, data=payload, method=method, headers=headers,",
     "            UPSTREAM + self.path, data=payload, method=method, headers=headers,"),
    ("used_not_seeded", "restart a resumed game's counter at zero",
     "self.max_actions, self.actions_used = int(max_actions), int(used)",
     "self.max_actions, self.actions_used = int(max_actions), 0"),

    ("wrong_game_fails_open", "the pre-08-07 hole: a body-less action skips the check",
     "if wanted and charge:", "if wanted and charge and payload:"),
    ("wrong_game_none_ok", "an action naming no game is billed to whoever received it",
     "            if asked != wanted:", "            if asked is not None and asked != wanted:"),
    ("wrong_game_never", "spend a neighbour's budget freely",
     "            if asked != wanted:", "            if False:"),

    ("lent_card_closable", "let a solver close the driver's shared card",
     'if self.state.card_is_lent and path == "/api/scorecard/close":',
     'if False and path == "/api/scorecard/close":'),
    ("adopt_not_lent", "adopt a card without marking it lent",
     "            self._card_is_lent = True", "            self._card_is_lent = False"),

    ("strip_off", "return the body with its baselines intact",
     "return {k: _strip(v) for k, v in node.items() if k not in HIDDEN_FIELDS}",
     "return {k: _strip(v) for k, v in node.items()}"),
    ("strip_shallow", "strip only the top level, not nested objects",
     "return {k: _strip(v) for k, v in node.items() if k not in HIDDEN_FIELDS}",
     "return {k: v for k, v in node.items() if k not in HIDDEN_FIELDS}"),
    ("strip_skips_lists", "never descend into a list, where per-game records live",
     "return [_strip(v) for v in node]", "return list(node)"),
    ("hidden_keeps_level_scores", "leave level_scores, which inverts to the medians",
     '    "level_scores",', ""),
    ("hidden_keeps_baseline_actions", "leave baseline_actions",
     '    "baseline_actions",', ""),
    ("hidden_keeps_level_baselines", "leave level_baseline_actions",
     '    "level_baseline_actions",', ""),
    ("filtered_fails_closed", "swallow an unparseable 5xx body the client needs",
     "    except (ValueError, UnicodeDecodeError):\n        return body",
     "    except (ValueError, UnicodeDecodeError):\n        return b''"),

    ("fresh_opener_each_call", "rebuild the session per call, unpinning the card",
     "            if self._opener is None:", "            if True:"),
    ("adopted_cookies_dropped", "build the session without the driver's pinning",
     "                for c in self._adopted:", "                for c in ():"),
    ("no_setcookie_relay", "swallow the upstream Set-Cookie headers",
     "        for value in set_cookies:", "        for value in ():"),
    ("budget_no_reset_session", "carry the previous game's pinning into the next",
     "        # A new game means a new card; carrying the previous game's pinning over\n"
     "        # is how a stale session survives into a run that did not open it.\n"
     "        self.reset_session()",
     "        # A new game means a new card; carrying the previous game's pinning over\n"
     "        # is how a stale session survives into a run that did not open it.\n"
     "        pass"),
])


TESTS = [
    "tests/test_ccarc3.py",
    "tests/test_ccarc3_client.py",
    "tests/test_audit_findings_stay_closed.py",
    "tests/test_a_grid_helper_never_aliases_its_input.py",
    "tests/test_pooling_cell_boundaries_across_boards_is_refused.py",
    "tests/test_a_verdict_earned_by_nothing_is_not_a_verdict.py",
    "tests/test_the_ledger_records_what_the_engine_actually_sent.py",
    "tests/test_the_card_facing_scoring_path_is_pinned.py",
    "tests/test_ccarc3_scoring.py",
    "tests/test_scoring.py",
    "tests/test_shared_card.py",
    "tests/test_card_corroboration_scopes_to_the_attempt.py",
    "tests/test_scoring_zero_actions_never_earns_the_cap.py",
    "tests/test_build_trace_audit_best_play.py",
    "tests/test_sweep_split_is_detectable.py",
    "tests/test_the_client_knows_a_win_from_a_loss.py",
    "tests/test_ccarc3_gate.py",
    "tests/test_resume_agreement.py",
    "tests/test_reap_clock_is_actually_wound.py",
    "tests/test_ccarc3_solver_reachable_docs.py",
    "tests/test_gave_up_is_not_a_result.py",
    "tests/test_arc_proxy_allowlist.py",
    "tests/test_arc_proxy_endtoend.py",
    "tests/test_the_allowlist_is_anchored_at_both_ends.py",
    "tests/test_the_shim_keeps_its_pinning_and_its_secret.py",
    "tests/test_the_action_ceiling_binds_under_concurrency.py",
    "tests/test_a_search_that_gave_up_says_so.py",
]

MODULES = {"grids": GRIDS, "rules": RULES, "ledger": LEDGER, "scoring": SCORING,
           "client": CLIENT, "proxy": PROXY}


def _check_equivalent_names_exist() -> None:
    """Every name in EQUIVALENT must be a mutant this battery actually runs.

    A set of names beside a list of mutants is two enumerated lists, and this
    repo has watched that shape go stale six times. `reap_ge` sat in EQUIVALENT
    with no matching mutant the moment its entry was dropped from the client
    list -- silent, because a name that matches nothing simply never fires. The
    check is cheap and turns the second list into a derived one.
    """
    defined = {name for _, muts in MODULES.values() for name, *_ in muts}
    orphans = sorted(EQUIVALENT - defined)
    if orphans:
        sys.exit(f"EQUIVALENT names with no matching mutant: {orphans}. "
                 "Either the mutant was renamed or removed; fix one or the other.")


def main(argv: list[str]) -> int:
    _check_equivalent_names_exist()
    wanted = argv[1:] or list(MODULES)
    unknown = [n for n in wanted if n not in MODULES]
    if unknown:
        sys.exit(f"unknown module(s): {unknown}; pick from {list(MODULES)}")

    total = unexpected = 0
    equivalents_that_died: list[str] = []
    for name in wanted:
        source, muts = MODULES[name]
        killable = [m for m in muts if m[0] not in EQUIVALENT]
        equivalent = [m for m in muts if m[0] in EQUIVALENT]
        total += len(muts)

        print(f"\n{'=' * 70}\n{name}: {len(muts)} mutants against {source}\n{'=' * 70}")
        unexpected += mc.run(source, TESTS, killable)

        # Run the argued equivalences separately rather than subtracting them
        # from a total. Subtracting assumes they survived; if a later test
        # tightened enough to kill one, the arithmetic would silently absorb a
        # real survivor elsewhere and report zero. Ask them directly instead.
        if equivalent:
            print(f"\n-- {name}: {len(equivalent)} argued-equivalent mutant(s), "
                  f"expected to survive")
            died = len(equivalent) - mc.run(source, TESTS, equivalent)
            if died:
                equivalents_that_died.append(f"{name} ({died})")

    print(f"\n{'=' * 70}")
    print(f"{total} mutants; {unexpected} unexpected survivor(s)")
    if equivalents_that_died:
        print(f"NOTE: an argued-equivalent mutant was caught in {', '.join(equivalents_that_died)}. "
              "That is not a failure -- it means a test now discriminates where the "
              "argument said nothing could. Re-read the argument in "
              "docs/ccarc3_open_findings.md and drop it from EQUIVALENT.")
    if unexpected:
        print("An unexpected survivor is a gap. Close it, or argue the "
              "equivalence in docs/ccarc3_open_findings.md and add it to EQUIVALENT.")
    return unexpected


if __name__ == "__main__":
    sys.exit(main(sys.argv))
