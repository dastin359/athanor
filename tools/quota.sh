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
S=/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad
python3 - <<'PY'
import json, glob, os, time

S = "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
FRESH = 3600  # a reading younger than this is trusted as-is

streams = sorted(
    glob.glob(f"{S}/**/stream.jsonl", recursive=True),
    key=os.path.getmtime,
    reverse=True,
)

latest = {}  # rateLimitType -> (mtime, info)
for p in streams[:40]:
    mt = os.path.getmtime(p)
    try:
        fh = open(p)
    except OSError:
        continue
    with fh:
        for line in fh:
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
            if t not in latest or mt > latest[t][0]:
                latest[t] = (mt, i)

if not latest:
    print("status=unknown reason=no-reading-on-disk")
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
