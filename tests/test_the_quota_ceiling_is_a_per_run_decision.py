"""The supervisor's quota ceiling must be settable without editing the file.

`LIMIT=0.98` was a constant, "raised from 0.95 on 2026-08-03" — by editing the
line, which is how a choice made for one afternoon becomes the number the next
operator inherits without ever choosing it.

The case that forced the knob: on 2026-08-09 a `bp35` validation run was
authorised against a reported utilisation of 0.77, and that reading turned out to
be 43 hours stale — the true figure was 0.96. `bp35` needs 4.86 hours, so a
ceiling at 0.98 would have stopped it partway, and `clean_rollouts` *discards* an
interrupted attempt rather than scoring it. Stopping at the ceiling there spends
the quota and banks nothing.

Lifting it is a real trade and the log says so at launch: above ~0.98 the brake
stops being the supervisor's polite stop and becomes the API's own rate limiting,
which interrupts the run without waiting for a `result.json`.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "tools" / "supervisor.sh"
TEXT = SUPERVISOR.read_text(encoding="utf-8")


def _assignments() -> str:
    """The real LIMIT/RESUME lines, taken from the file rather than restated."""
    lines = [ln for ln in TEXT.splitlines()
             if re.match(r'^(LIMIT|RESUME)=', ln)]
    assert len(lines) == 2, f"expected one LIMIT and one RESUME line, got {lines}"
    return "\n".join(lines)


def _eval(env: dict[str, str]) -> tuple[str, str]:
    proc = subprocess.run(
        ["bash", "-c", _assignments() + '\nprintf "%s %s" "$LIMIT" "$RESUME"'],
        capture_output=True, text=True, check=True,
        env={"PATH": "/usr/bin:/bin", **env})
    limit, resume = proc.stdout.split()
    return limit, resume


def test_the_defaults_are_unchanged():
    """A knob appearing must not move the operator ceiling for anyone else."""
    assert _eval({}) == ("0.98", "0.90")


def test_the_ceiling_is_overridable():
    assert _eval({"CCARC3_QUOTA_LIMIT": "0.999"}) == ("0.999", "0.90")


def test_the_resume_point_is_overridable_independently():
    assert _eval({"CCARC3_QUOTA_RESUME": "0.5"}) == ("0.98", "0.5")


def test_an_empty_override_falls_back_to_the_default():
    """`CCARC3_QUOTA_LIMIT=` from an unset shell variable is absence."""
    assert _eval({"CCARC3_QUOTA_LIMIT": ""}) == ("0.98", "0.90")


def _announcement(env: dict[str, str]) -> str:
    """The real announcement block, run against the real assignments."""
    start = TEXT.index('# Say the ceiling out loud')
    end = TEXT.index("ceiling_stopped=0")
    body = _assignments() + "\n" + TEXT[start:end]
    proc = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          check=True, env={"PATH": "/usr/bin:/bin", "TZ": "UTC", **env})
    return proc.stdout


def test_a_raised_ceiling_announces_itself():
    out = _announcement({"CCARC3_QUOTA_LIMIT": "0.999"})
    assert "CEILING RAISED" in out, f"a lifted ceiling was silent: {out!r}"
    assert "0.999" in out, "the announcement did not name the new ceiling"
    assert "rate limiting" in out, (
        "the announcement did not say what replaces the brake above 0.98"
    )


def test_a_lowered_ceiling_announces_itself_too():
    """Any departure from the default is worth one line; only the default is quiet."""
    assert "CEILING RAISED" in _announcement({"CCARC3_QUOTA_LIMIT": "0.5"})


def test_the_default_ceiling_says_nothing():
    """An always-on banner is one people stop reading."""
    assert _announcement({}) == "", "the default ceiling announced itself"


@pytest.mark.parametrize("var", ["CCARC3_QUOTA_LIMIT", "CCARC3_QUOTA_RESUME"])
def test_the_knob_is_reachable_from_a_launched_supervisor(var):
    """Named in the file, so `env VAR=... supervisor.sh` is a documented route."""
    assert var in TEXT, f"{var} is not referenced in supervisor.sh"
