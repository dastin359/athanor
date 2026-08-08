#!/usr/bin/env bash
# Report EVERY rate-limit window, not whichever event happened to come last.
# There are at least two — five_hour and seven_day — with different warning
# thresholds (0.90 vs 0.75) and wildly different costs of exhaustion (a ~2h
# pause vs a ~43h blackout). Watching one of them is watching none.
#
# Readings come from rate_limit_event lines in solver stream.jsonl files, so
# they only refresh while solvers are running. During a build-only stretch the
# data goes stale within the hour. It used to report status=unknown at that
# point, which is worse than useless: it reads as "no constraint" exactly when
# the last thing we knew was allowed_warning.
#
# Staleness is only safe in the pessimistic direction. Utilization rises
# monotonically inside a window, so an old reading is a LOWER bound:
#
#   stale + allowed_warning/rejected -> still binding. Utilization only grew.
#   stale + allowed                  -> unknown. It may have crossed since.
#   resetsAt in the past             -> void. The window rolled over.
S="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
python3 - <<'PY'
import json, glob, os, time

# Overridable so this script can be exercised against a fixture. It had no
# such hook, so three tests of its scan logic silently ran against the live
# scratchpad and "passed" by reporting the real numbers.
S = os.environ.get("CCARC3_SCRATCH") or "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
FRESH = 3600  # a reading younger than this is trusted as-is

streams = sorted(
    glob.glob(f"{S}/**/stream.jsonl", recursive=True),
    key=os.path.getmtime,
    reverse=True,
)

# Every window this is required to report on. Absence of one of these is a
# finding, not a silence -- see the two defects below.
EXPECTED = ("five_hour", "seven_day")

# **Scan until every expected window is found, not a fixed 40 files.**
# `streams[:40]` looked generous and was not: `seven_day` events are rare
# (five_hour dominates), and on this box they appeared at mtime-ranks 2, 3, 4, 5
# and then nothing until rank 43. Thirty-six further stream.jsonl writes evict
# all four -- and a 25-game sweep writes up to 75. Past that point this printed
# the five_hour row and a clean `status=allowed`, never mentioning that the
# window whose exhaustion costs days had not been looked at; `if not latest`
# only fires when EVERY type is missing, so per-window absence read as health.
# Downstream, supervisor.sh's `util_now` returns empty, its `[ -n "$u" ]` guard
# skips the whole stop/start block, and it neither stops at the ceiling nor
# restarts after a reset.
#
# **Only a real solver workspace counts.** On 2026-08-08 a mutation-testing agent
# left fixture trees under the scratchpad -- `fx2/top-b/stream.jsonl` carrying a
# hand-written `seven_day allowed util=0.13` -- and this scan, whose whole job is
# to read that directory, took them as the truth. The reported seven-day window
# went from 0.84 allowed_warning to 0.13 allowed, the supervisor relaunched three
# times against a fabricated all-clear, and the only reason it cost nothing is
# that there was no pending work. A submission sweep would have started on an
# invented number.
#
# The discriminator: a real stream never sits alone. `build_workspace` writes
# meta.json, session.py, rules.json and trace.jsonl beside it before the solver
# starts, and result.json after. Measured across the 178 streams on this box:
# zero sit alone, and 172 have a sibling result.json. A synthetic fixture that
# writes only stream.jsonl has none of them.
# Two harness layouts write solver streams here, and the first version of this
# list only knew one. `ccarc3` workspaces carry meta.json/session.py/rules.json/
# trace.jsonl/result.json; the older `cc_harness` runs under `effort_max/` carry
# initial_prompt.md, system_prompt.md and a `workspace/` directory instead. Three
# of those were being discarded as debris while holding genuine `seven_day`
# readings -- harmless only because they were 122h stale and something newer was
# kept. A filter that drops real data while reporting confidence is the same
# defect it was written to fix, one layer along.
#
# `run.log` alone is deliberately not a marker: it is generic enough to admit
# scratch directories that are not runs at all.
HARNESS_SIBLINGS = ("result.json", "trace.jsonl", "meta.json", "rules.json",
                    "session.py", "initial_prompt.md", "system_prompt.md",
                    "workspace")

def _is_real_workspace(path):
    d = os.path.dirname(path)
    try:
        names = set(os.listdir(d))
    except OSError:
        return False
    return any(n in names for n in HARNESS_SIBLINGS)

skipped = [p for p in streams if not _is_real_workspace(p)]
streams = [p for p in streams if _is_real_workspace(p)]

# Newest-first, so the common case still stops after a handful of files.
latest = {}  # rateLimitType -> (mtime, seq, info)
scanned = 0
for p in streams:
    if all(t in latest for t in EXPECTED):
        break
    scanned += 1
    mt = os.path.getmtime(p)
    try:
        fh = open(p)
    except OSError:
        continue
    with fh:
        # **Last write wins within a file, which `mt >` could not express.**
        # Both sides of `mt > latest[t][0]` are the same file's mtime, so it is
        # False by construction and the FIRST event in a file won -- the oldest
        # and lowest reading. Every solver that crosses a second threshold logs
        # two, and a later `rejected` was silently discarded in favour of the
        # earlier `allowed`, then stamped with the file's mtime and labelled
        # fresh. Measured here: reported 0.83 while the newest event said 0.84.
        for seq, line in enumerate(fh):
            if '"rate_limit_event"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "rate_limit_event":
                continue
            i = d["rate_limit_info"]
            t = i.get("rateLimitType", "?")
            if t not in latest or (mt, seq) > latest[t][:2]:
                latest[t] = (mt, seq, i)

latest = {t: (mt, i) for t, (mt, seq, i) in latest.items()}

# Loud, not silent: debris in the live scan root is a fault in its own right.
if skipped:
    print(f"  note: ignored {len(skipped)} stream.jsonl with no harness files "
          f"beside them (e.g. {os.path.dirname(skipped[0]).replace(S + '/', '')})")

if not latest:
    print("status=unknown reason=no-reading-on-disk")
    raise SystemExit

missing = [t for t in EXPECTED if t not in latest]
if missing:
    for t in missing:
        print(f"  {t:10} MISSING — no event in any of {scanned} stream.jsonl scanned")
    print(f"status=unknown reason=window-not-observed:{','.join(missing)}")
    raise SystemExit

now = time.time()
rank = {"allowed": 0, "allowed_warning": 1, "rejected": 2}
overall = "allowed"
any_blind = False

for t, (mt, i) in sorted(latest.items()):
    age = int(now - mt)
    status = i["status"]
    resets_at = i.get("resetsAt") or 0
    left = int(resets_at - now)
    u = i.get("utilization")

    if left <= 0:
        verdict, note = "allowed", "VOID (window reset since this reading)"
    elif age <= FRESH:
        verdict, note = status, "fresh"
    elif rank.get(status, 0) >= 1:
        verdict, note = status, "STALE but still binding (utilization only rises)"
    else:
        verdict, note = "unknown", "STALE and was allowed — may have crossed since"
        any_blind = True

    if rank.get(verdict, 0) > rank.get(overall, 0):
        overall = verdict
    print(
        f"  {t:10} {status:16} util={u if u is not None else '<threshold'}"
        f"  reset_in={max(left,0)//3600}h{(max(left,0)%3600)//60:02d}m"
        f"  age={age}s  [{note}]"
    )

if any_blind and overall == "allowed":
    print("status=unknown reason=stale-allowed-reading")
else:
    print(f"status={overall}")
PY
