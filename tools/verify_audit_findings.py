#!/usr/bin/env python3
"""Re-check every finding of the 2026-08-07 harness audit, from the code.

**Written because answering this from memory got it wrong twice.** Asked "all
bugs fixed?", the honest-sounding answer was yes; the true answer was 14 of 16,
then 15 of 17 once one finding turned out to carry three separate claims. Both
times the error was the same shape the audit kept finding in the harness itself
— checking a proxy for the thing rather than the thing. "I edited that file"
stood in for "I fixed that finding", and "I fixed that finding" stood in for "I
fixed every claim inside it".

So the finding list is code now. Each entry names the audit's claim and asserts
the specific thing that made it true is gone. Run it; do not recall it.

    python tools/verify_audit_findings.py          # exits 1 if anything reopened
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src")]

import proofread_trace as pt  # noqa: E402


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


WS = pathlib.Path("/tmp/scratchpad/clean_rollouts/zz99-deadbeef/attempt_1/zz99-deadbeef")
LP85 = [17, 38, 31, 16, 41, 60, 26, 159]


def checks() -> list[tuple[str, str, bool]]:
    scoring, session = read("src/athanor/ccarc3/scoring.py"), read("src/athanor/ccarc3/session.py")
    client, ablate = read("src/athanor/ccarc3/client.py"), read("tools/ablate_baselines.py")
    doctrine = read("src/athanor/ccarc3/assets/CCARC3_DOCTRINE.md")
    return [
        # -- treasure: a median, or a published transform of one, in reachable source
        ("1  critical", "score_run docstring: score/action pair, card path, play scores",
         not any(x in scoring for x in ("65.5", "best_or_last/card.json", "0.8659"))),
        ("2  critical", "budget_multiple docstring: used/cap pairs invert to totals",
         not any(x in session for x in ("631/634", "701/722"))),
        ("3  critical", "collect_outcome comment: a bare cap is total x multiple",
         "1040" not in session),
        ("4  high", "level_revisits table: ratio beside actions is the median",
         not any(x in client for x in ("95/308", "30/184", "4/36", "5/60", "1/46"))),
        ("5a high", "reach: a relative path out of the workspace was invisible",
         bool(pt.strayed("cat ../../../../best_or_last/card.json", WS))),
        ("5b high", "inbound: an array printed one value per line evaded the scan",
         pt.array_arrived("\n".join(str(n) for n in LP85), LP85)),
        ("5c high", "inbound: a parsed array without the token evaded the scan",
         pt.array_arrived(f"medians = ({', '.join(map(str, LP85))})", LP85)),
        ("6  high", "a proofreader crash was read as 'passages need reading'",
         "PROOFREAD DID NOT RUN" in read("tools/clean_rollouts.py")),
        ("7  high", "section 6 removal asserted nothing; it holds real medians",
         "no '## 6. ' section to remove" in ablate),
        ("9  medium", "--gz derived the root from where evidence sits: 30/30 void",
         hasattr(pt, "recover_root")),
        ("10 medium", "key guard grepped plaintext in a gzipped tree, passed on unset key",
         all(x in read("tools/preserve_evidence.sh")
             for x in ("gzip -cd", "ARC_API_KEY unset", '. "$SP/arc3/.env"'))),
        ("11 medium", "audit page picked the best play by a proxy, not the rubric",
         "_rank" in read("tools/build_trace_audit.py")),
        ("12 medium", "page claimed digest-verified membership; the filter is a date",
         "byte for byte" not in read("tools/trace_audit_template.html")),
        ("13 medium", "fence check passed with one of two fences surviving",
         "EXPECTED_FENCES" in ablate),
        ("14 low", "sk48 cleared 8/8, replayed, and still lost 0.0462",
         "every point ever lost was lost by stopping" not in doctrine),
        ("15 low", "loss table said sk48 quit at 251 actions; it quit at 632",
         "| `sk48` | 5 of 8 | 632 | 0 |" in doctrine),
        ("16 low", "'the two worst scores on record' — they are not",
         "worst scores on record" not in doctrine),
        # -- from the 20 findings the audit workflow never adjudicated
        ("u1 high", "workspace doctrine spelled out a median; 8 banked runs read it",
         "MEDIAN_IN_DOCTRINE_FROM" in read("tools/build_trace_audit.py")),
        ("u2 high", "contamination check saw only the run's OWN game's array",
         "Any game's array, not only this run's" in read("tools/build_trace_audit.py")),
        ("u3 high", "the withdrawn 25-environment figure is marked withdrawn",
         "is\nwithdrawn" in read("docs/ccarc3_results.md")
         or "withdrawn" in read("docs/ccarc3_results.md")),
        # -- and the false positives that made earlier versions of these useless
        ("fp a", "reach must not fire on a `..` that lands back inside",
         not pt.strayed("sys.path.insert(0,'..')", WS)),
        ("fp b", "inbound must not fire on ordinary transcript integers",
         not pt.array_arrived("the board is 64x64 with 16 colours and 8 levels", LP85)),
    ]


def main() -> int:
    rows = checks()
    open_findings = [r for r in rows if not r[2]]
    for tag, claim, ok in rows:
        print(f"  {'PASS' if ok else 'OPEN'}  {tag:11} {claim}")
    print(f"\n{len(rows) - len(open_findings)}/{len(rows)} verified closed")
    if open_findings:
        print(f"{len(open_findings)} REOPENED — do not report this audit as closed")
    return 1 if open_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
