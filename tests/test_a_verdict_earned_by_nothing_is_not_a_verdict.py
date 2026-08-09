"""Vacuous truth, in every place ``rules.py`` can report a verdict.

The module's whole reason to exist is that ``False`` means two different things
-- "this rule failed" and "this rule was never tested" -- and that collapsing
them kills true rules. ``VerifyResult.verified`` guards against it and is
tested. Three siblings guard against the same thing and were not:

- ``VerifyResult.vacuous`` -- reading ``holds == 0`` instead of
  ``applicable == 0`` calls a *refuted* rule "never applicable", which is the
  error the module exists to prevent, pointed the other way.
- ``PredictionReport.perfect`` -- dropping ``correct > 0`` makes a forward model
  that declined every single transition report as perfect. The docstring names
  ``perfect`` as "the thing to chase".
- ``PredictionReport.accuracy`` -- counting declined transitions in the
  denominator makes a model look worse the more honestly it declines.

And one alarm that has to stay quiet: ``regressions`` must fire only on
*applicable and violated*. Broadening it to "not vacuous" makes a mechanic that
is working perfectly raise a regression -- an always-on alarm, which this repo
has already found and fixed elsewhere and which is worse than no alarm.

All four mutants survived the suite before this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from athanor.ccarc3 import Outcome, Rule, predict, regressions, verify
from athanor.ccarc3.ledger import Transition


def _tr(index: int, level: int, action: str, before, after, **kw) -> Transition:
    from athanor.ccarc3 import as_grid

    return Transition(
        index=index, level=level, action=action, params={},
        before=None if before is None else as_grid(before),
        after=as_grid(after), intermediate=(as_grid(after),),
        score_before=0, score_after=0, state="NOT_FINISHED",
        full_reset=kw.get("full_reset", False), available_actions=(),
        crosses_level=kw.get("crosses_level", False), wasted=kw.get("wasted", False),
    )


def _ts(n: int = 3, level: int = 0) -> list[Transition]:
    return [_tr(i, level, "ACTION1", [[i % 2]], [[(i + 1) % 2]]) for i in range(n)]


# --------------------------------------------------------------------------- #
# vacuous is about applicability, not about holding
# --------------------------------------------------------------------------- #


def test_a_refuted_rule_is_not_vacuous():
    """It was applicable and it failed -- the opposite of untested."""
    always_wrong = Rule("always wrong", applies=lambda t: True, holds=lambda t: False)
    result = verify(always_wrong, _ts())
    assert result.counts.holds == 0
    assert result.counts.violated == 3
    assert not result.vacuous, "a rule that was tested and failed was tested"
    assert result.refuted


def test_a_rule_that_never_applied_is_vacuous():
    never = Rule("never applies", applies=lambda t: False, holds=lambda t: True)
    result = verify(never, _ts())
    assert result.vacuous
    assert not result.refuted and not result.verified


def test_the_three_verdicts_are_mutually_exclusive_and_exhaustive():
    """No transition set may produce two verdicts, or none."""
    cases = [
        Rule("all hold", applies=lambda t: True, holds=lambda t: True),
        Rule("all fail", applies=lambda t: True, holds=lambda t: False),
        Rule("never applies", applies=lambda t: False, holds=lambda t: True),
        Rule("mixed", applies=lambda t: True, holds=lambda t: t.index > 0),
    ]
    for rule in cases:
        r = verify(rule, _ts())
        flags = [r.verified, r.refuted, r.vacuous]
        assert sum(flags) == 1, f"{rule.name} gave {flags}"


def test_the_string_form_says_which_verdict_it_is():
    refuted = verify(Rule("r", applies=lambda t: True, holds=lambda t: False), _ts())
    vacuous = verify(Rule("v", applies=lambda t: False, holds=lambda t: True), _ts())
    assert "REFUTED" in str(refuted) and "VACUOUS" not in str(refuted)
    assert "VACUOUS" in str(vacuous) and "REFUTED" not in str(vacuous)


# --------------------------------------------------------------------------- #
# regressions: an alarm nobody can learn to ignore
# --------------------------------------------------------------------------- #


def test_a_mechanic_that_is_working_raises_no_regression():
    """"Not vacuous" is not the test -- a rule that holds is not a regression."""
    working = Rule("holds everywhere", applies=lambda t: True,
                   holds=lambda t: True, scope="game")
    assert regressions([working], _ts()) == []


def test_a_mechanic_that_broke_does_raise_one():
    broken = Rule("broke", applies=lambda t: True, holds=lambda t: False, scope="game")
    out = regressions([broken], _ts())
    assert [r.rule for r in out] == ["broke"]


def test_a_mechanic_that_merely_stopped_applying_stays_silent():
    gone = Rule("no longer applies", applies=lambda t: False,
                holds=lambda t: True, scope="game")
    assert regressions([gone], _ts()) == []


def test_the_alarm_is_quiet_on_a_book_where_everything_is_fine():
    """The whole point: a report that is always on is one people stop reading."""
    book = [
        Rule("a", applies=lambda t: True, holds=lambda t: True, scope="game"),
        Rule("b", applies=lambda t: False, holds=lambda t: False, scope="game"),
        Rule("c", applies=lambda t: t.index == 0, holds=lambda t: True, scope="game"),
        Rule("d", applies=lambda t: True, holds=lambda t: False, scope="level"),
    ]
    assert regressions(book, _ts()) == []


# --------------------------------------------------------------------------- #
# a forward model that predicted nothing has not predicted everything
# --------------------------------------------------------------------------- #


def test_a_model_that_declines_everything_is_not_perfect():
    report = predict(lambda b, a, p: None, _ts(), name="declines")
    assert report.skipped == 3 and report.tested == 0
    assert not report.perfect, "declining every transition is not a perfect model"


def test_a_model_that_crashes_on_everything_is_not_perfect():
    def explode(before, action, params):
        raise RuntimeError("no model here")

    report = predict(explode, _ts(), name="crashes")
    assert report.tested == 0 and not report.perfect


def test_a_model_that_predicted_one_transition_correctly_is_perfect():
    """The guard must not make ``perfect`` unreachable."""
    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]])]
    report = predict(lambda b, a, p: np.array([[1]]), ts, name="one")
    assert report.tested == 1 and report.perfect


def test_accuracy_is_over_what_was_tested_not_over_what_was_skipped():
    """A model is not punished for declining -- it is punished for being wrong."""
    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]),
          _tr(1, 0, "ACTION1", [[0]], [[1]]),
          _tr(2, 0, "ACTION1", [[0]], [[1]])]
    seen = {"n": 0}

    def answers_once(before, action, params):
        seen["n"] += 1
        return np.array([[1]]) if seen["n"] == 1 else None

    report = predict(answers_once, ts, name="shy")
    assert report.tested == 1 and report.skipped == 2
    assert report.accuracy == 1.0, "two declines must not dilute one exact hit"


def test_accuracy_of_nothing_is_zero_not_a_crash():
    assert predict(lambda b, a, p: None, _ts()).accuracy == 0.0


@pytest.mark.parametrize("outcome", list(Outcome))
def test_every_outcome_has_a_distinct_value(outcome):
    assert sum(o.value == outcome.value for o in Outcome) == 1
