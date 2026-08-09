"""Every mutant the three solver-facing ccarc3 modules were audited with.

Run it to re-verify the audit rather than trusting this file's summary:

    .venv/bin/python tools/mutation_battery_ccarc3.py            # all three
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
EQUIVALENT = {"collapse_alias", "cb_bounds"}

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

TESTS = [
    "tests/test_ccarc3.py",
    "tests/test_ccarc3_client.py",
    "tests/test_audit_findings_stay_closed.py",
    "tests/test_a_grid_helper_never_aliases_its_input.py",
    "tests/test_pooling_cell_boundaries_across_boards_is_refused.py",
    "tests/test_a_verdict_earned_by_nothing_is_not_a_verdict.py",
    "tests/test_the_ledger_records_what_the_engine_actually_sent.py",
]

MODULES = {"grids": GRIDS, "rules": RULES, "ledger": LEDGER}


def main(argv: list[str]) -> int:
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
