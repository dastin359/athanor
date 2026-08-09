"""`tools/verify_audit_findings.py` must run automatically and report zero open.

It is the project's own adjudication of 32 audit findings, and nothing ran it.
Discovered 2026-08-09 sitting at "2 REOPENED — do not report this audit as
closed", and both were false alarms that a LATER, CORRECT fix had created:

* u4 asserted `"190/h on this level" in DOCTRINE.md`. 190 is `tn36`'s, and
  810b72f ("the doctrine leaked two totals") is what removed it. So the entry
  required the leak to still be present, went open the moment it was fixed, and
  anyone closing it as written would have put the figure back.
* u11 asserted `"tn36` has five runs"`, which fbf8887 ("four figures that did
  not follow from their own evidence") had corrected to six. Improving the
  doctrine reopened the finding that the doctrine was wrong.

Both were anchored to exact prose rather than to the property the finding is
about -- reading a proxy, one level up. And a report that is permanently open is
a latched alarm: people stop reading the last line, which is precisely how these
two went unnoticed while the file printed its own warning every run.

Running it here means a genuine reopening fails the suite on the same day.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
VERIFIER = REPO / "tools" / "verify_audit_findings.py"


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), str(VERIFIER)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO),
    )


def test_no_audit_finding_is_open() -> None:
    out = _run()
    combined = out.stdout + out.stderr
    open_lines = [ln for ln in combined.splitlines() if ln.strip().startswith("OPEN")]
    assert not open_lines, (
        "audit findings have reopened:\n" + "\n".join(open_lines)
        + "\n\nBefore 'fixing' the code, check whether the ENTRY is stale: two of "
          "these were assertions on exact prose that a later, correct fix "
          "legitimately reworded, and one of them required a leaked figure to "
          "still be present."
    )
    assert "REOPENED" not in combined, combined[-2000:]


def test_every_finding_is_actually_adjudicated() -> None:
    """A verifier that checks nothing also prints no OPEN lines.

    The count is read from its own summary and cross-checked against the number
    of result lines, so deleting entries cannot quietly turn this green.
    """
    out = _run()
    m = re.search(r"(\d+)/(\d+) verified closed", out.stdout)
    assert m, f"the verifier no longer prints a summary:\n{out.stdout[-1500:]}"
    closed, total = int(m.group(1)), int(m.group(2))
    assert closed == total, f"{total - closed} finding(s) open"
    results = [ln for ln in out.stdout.splitlines()
               if ln.strip().startswith(("PASS", "OPEN"))]
    assert len(results) == total, (
        f"summary claims {total} findings but {len(results)} result lines were "
        f"printed -- the count and the checks have drifted apart"
    )
    # A ratchet, not a guess. Set to the count on 2026-08-09; a floor of 30 let
    # a deleted entry through at 31, which is the whole failure this case exists
    # to catch -- closing a finding by removing it is indistinguishable from
    # fixing it if nothing counts. Adding findings is fine and should raise this
    # number deliberately; dropping below it must fail.
    assert total >= 32, (
        f"only {total} findings are adjudicated, down from 32; entries appear "
        f"to have been deleted rather than closed"
    )
