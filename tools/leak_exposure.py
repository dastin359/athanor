#!/usr/bin/env python3
"""Did any banked run actually receive one of the eight leaked medians?

The leaks lived in the package every solver imports, for the whole of the
25-environment arm and the whole 17-run repaired-surface cohort. Removing them
says nothing about whether they were read; the streams do.

Searches every preserved stream for the exact strings as they stood before
`9eae0bd`, recovered from git rather than retyped. Exact strings, not values: a
bare `171` collides with everything, but `a 171 baseline` appears in one place in
the world.

Reports separately for
  * inbound  — tool results, where a leak actually arrives;
  * outbound — the solver's own text, where a leak that arrived gets used.
An inbound hit means the harness handed it over. An outbound hit with no inbound
one would mean the solver produced the number some other way, which is a
different and worse finding.
"""
import gzip
import json
import pathlib
import re
import subprocess
import sys

# Derived from this file's location (`<repo>/tools/`), not written: the
# hard-coded path is this container's clone location, and a fresh container
# or a different account gets a different one.
REPO = pathlib.Path(__file__).resolve().parents[1]
# The evidence ROOT, not one cohort under it. This was `.../clean_rollouts`
# and the scan below now reaches every batch via `EV.parent` -- so the leaf
# was dead, and worse than dead: the next reader to use `EV` directly gets
# a single-cohort scan that reports the whole arm clean, which is exactly
# the bug the comment down there records having already been fixed once.
EV = REPO / "evidence" / "ccarc3"

# The exact leaked text, per file, as of the commit before the purge.
WANTED = {
    "arc_proxy.py": [r'"level_baseline_actions": \[17, 38, 31, 16, 41, 60, 26, 159\]',
                     r'"level_actions":\s+\[21, 0, 0',
                     r'"level_scores":\s+\[65\.53'],
    "client.py":    [r"651-action baseline", r"31-baseline level", r"8-baseline level",
                     r"55-baseline level", r"a 171 baseline",
                     r"would have allowed up to\s*\n?\s*1585"],
    "gate.py":      [r"a published baseline runs to 1843 actions"],
    "cli.py":       [r"171 to 1843 baseline actions"],
    "session.py":   [r"span 171 to 1843 baseline actions"],
}


def verify_present_before_the_fix() -> None:
    """A scan for strings that were never there proves nothing. Check first."""
    missing = []
    for name, pats in WANTED.items():
        src = subprocess.run(
            ["git", "show", f"9eae0bd^:src/athanor/ccarc3/{name}"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
        for pat in pats:
            if not re.search(pat, src):
                missing.append(f"{name}: {pat}")
    if missing:
        sys.exit("these patterns did not exist pre-fix, so the scan is vacuous:\n  "
                 + "\n  ".join(missing))
    print(f"verified: all {sum(len(v) for v in WANTED.values())} patterns were "
          f"live in the package before 9eae0bd\n")


def blocks(stream: pathlib.Path):
    inbound, outbound = [], []
    opener = gzip.open if stream.suffix == ".gz" else open
    with opener(stream, "rt", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            content = (row.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    inbound.append(c if isinstance(c, str) else json.dumps(c))
                elif b.get("type") in ("text", "thinking"):
                    outbound.append(b.get("text") or b.get("thinking") or "")
    return "\n".join(inbound), "\n".join(outbound)


def main() -> int:
    verify_present_before_the_fix()
    pats = [(n, p, re.compile(p)) for n, ps in WANTED.items() for p in ps]
    # **It scanned one directory in one layout, and claimed the whole arm.**
    # `EV` pointed at `clean_rollouts` and the glob matched only the
    # nested `<game>/attempt_N/<game>/` shape, so the baseline-free arm
    # (`ablate_nobaseline`, flat) and every ad-hoc `rerun_*` directory were
    # invisible — while the module docstring said the leaks were checked across
    # "the whole of the 25-environment arm". A scan that cannot see a cohort
    # reports it clean.
    streams = sorted(
        s for root in EV.iterdir() if root.is_dir()
        for pattern in ("*/attempt_*/*/stream*.jsonl.gz", "*/stream*.jsonl.gz")
        for s in root.glob(pattern)
    )
    print(f"scanning {len(streams)} preserved streams\n")

    findings = 0
    for st in streams:
        run = f"{st.parts[-3]}/{st.parts[-4].split('_')[-1]}"
        inb, outb = blocks(st)
        for name, pat, rx in pats:
            for where, text in (("INBOUND", inb), ("outbound", outb)):
                if rx.search(text):
                    findings += 1
                    print(f"  !! {run}: {where} carries {name} :: {pat}")
    print(f"\n{len(streams)} streams, {findings} exposure(s) to any of the eight leaks")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
