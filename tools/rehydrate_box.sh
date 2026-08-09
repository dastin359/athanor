#!/usr/bin/env bash
# Put a freshly-replaced container back into a working state.
#
# Containers here are *replaced*, not restarted: a new boot_id, a working tree
# rewound to whatever the image snapshot held, and a scratchpad missing whatever
# was written since. On 2026-08-06 this happened three times in 45 minutes, each
# needing the same four steps by hand. This is those four steps.
#
# Every one of them is idempotent, so running it on a box that is already healthy
# does nothing and costs nothing. Run it first thing on any autopilot wake-up.
#
#   1. Fast-forward the repo from origin. The tree rewinds to the image commit
#      (bf90a03 as of this writing) and origin is always ahead -- everything
#      durable is pushed, so origin is the source of truth, never the local tree.
#   2. Re-point the scratchpad symlink the hourly refresh instruction invokes.
#   3. Log the box fingerprint, so replacement cadence stays measurable.
#   4. Restore the banked arm results, without which the arm runner would treat
#      13 finished games as unstarted. See tools/restore_banked_results.py.
#
# Then report what is running and whether anything finished, which is what the
# wake-up needs to decide next.
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
# Overridable via `CCARC3_BRANCH`; a session on another account works on
# another branch. Not derived from HEAD -- see preserve_evidence.sh.
BRANCH="${CCARC3_BRANCH:-claude/athanor-cc-harness-variant-jpqw7t}"
SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
cd "$REPO" || exit 1

# `--links-only` runs step 2 and stops: no fetch, no fingerprint, no restore, no
# network. Step 2 is the part that decides which copy of the baseline strip and
# the launcher a replaced box will run, and until this flag existed there was no
# way to exercise it without a git fetch and two restores. Same lesson as
# `clean_rollouts.verdict`, whose own comment explains that it was made
# importable so it could be tested and then never was: **logic that decides what
# is admissible has to be reachable without launching anything.**
# Referenced as `${LINKS_ONLY:-0}` below, never bare: step 1 is extracted and run
# as a standalone slice by tests/test_rehydrate_reports_what_it_did.py under
# `set -u`, where an unbound variable kills the block before it narrates.
LINKS_ONLY=0
[ "${1:-}" = "--links-only" ] && LINKS_ONLY=1

[ "${LINKS_ONLY:-0}" = 1 ] || \
    echo "box $(cat /proc/sys/kernel/random/boot_id) up $(cut -d' ' -f1 /proc/uptime)s"

# 1. Repo. Retry the fetch: a replaced box often races container networking.
#
# **"already current" used to cover three different outcomes.** HEAD is
# unchanged when the tree really is up to date, when all five fetch retries
# failed (the networking race the retry loop exists for), and when the ff-only
# merge was REFUSED because HEAD had diverged — unpushed local commits, which is
# precisely the state a rollback recovery has to notice. Both commands sent their
# errors to /dev/null and neither status was tested, so the one line printed was
# "already current" in all three, and the recovery step that this script's own
# header calls the load-bearing one reported success for having done nothing.
fetched=0
if [ "${LINKS_ONLY:-0}" = 1 ]; then fetched=skip; fi
for delay in 0 2 4 8 16; do
  if [ "$fetched" = skip ]; then break; fi
  [ "$delay" = 0 ] || sleep "$delay"
  git fetch origin "$BRANCH" -q 2>/dev/null && { fetched=1; break; }
done
[ "$fetched" != 0 ] || echo "repo   FETCH FAILED after 5 tries — origin unreachable; the tree is NOT known current"

before="$(git rev-parse --short HEAD)"
if [ "${LINKS_ONLY:-0}" = 1 ]; then
  :
elif git merge --ff-only "origin/$BRANCH" -q 2>/dev/null; then
  after="$(git rev-parse --short HEAD)"
  if [ "$before" = "$after" ]; then
    [ "$fetched" = 1 ] && echo "repo   $after (already current)"
  else
    echo "repo   $before -> $after (recovered from rollback)"
  fi
else
  echo "repo   FF-ONLY REFUSED — HEAD $before has diverged from origin/$BRANCH"
  echo "       (unpushed local commits). The tree is NOT current and this script"
  echo "       will not resolve it: rebase or push by hand."
fi

# 2. Standing instructions call these by their old scratchpad paths.
#
# **Symlinks, not copies, and this is why.** `heartbeat.sh` was a copy here and
# drifted five days behind the repo: the scratchpad version still watched
# `ablate_baselines.py` after `clean_rollouts.py` became the runner, so
# `runner_alive` was permanently false, the loop read that as "the arm ended",
# and it broke on its first poll while looking exactly like a healthy heartbeat.
# The scratchpad reverts to an image snapshot when the container is replaced --
# CLAUDE.md records that as the reason a rule kept only there is a rule that
# expires -- and the same is true of a script. A symlink cannot drift.
mkdir -p "$SP"
# **Derived, not enumerated.** This was the literal list
# `refresh_audit.sh heartbeat.sh snapshot_results.py`, and a hand-maintained list
# of the tools of the day goes stale without a symptom -- the same defect as
# `snapshot_results.py`'s batch list, which silently omitted 25 banked games, and
# the live panel's arm allowlist, which rendered "nothing running" rather than "I
# cannot see it". Both were fixed by replacing an enumeration with a rule; this
# one was not, and it cost exactly what that costs.
#
# Measured on the 02:44 PDT replacement of 2026-08-09, immediately after this
# script reported success: three of the five shadowing copies were re-pointed and
# two were left as pre-fix files, because nobody had added them to the list.
#
#     supervisor.sh          5,464 bytes   vs  16,369 in the repo
#     ablate_baselines.py   21,367 bytes   vs  36,205 in the repo
#
# The second is the **baseline strip**. A 21 kB copy predates `install()` and
# `assert_installed()` -- the fix for the defect that shipped real per-level
# medians into eight rollouts and three re-runs -- and the first is the launcher,
# without its key guard, its argv-exact orphan sweep or its sweep-directory
# awareness. AUTOPILOT.md's own launch block still says `bash
# scratchpad/supervisor.sh`, so following the standing instruction verbatim on a
# replaced box starts the pre-fix launcher.
#
# The rule: **any scratchpad entry sharing a basename with a file in `tools/`
# must be a symlink to it.** A tool added to the repo tomorrow is covered without
# anyone remembering, and a shadowing copy is repaired the first time this runs.
# Names that do not match a repo tool -- `batch6.py`, `ablate_baselines.py.pre-guard`
# -- are left alone, which is the whole discrimination.
#
# Repairs are printed with the size delta rather than done silently: a scratchpad
# copy that had drifted far enough to matter is evidence about the box, and the
# one thing worse than the drift is fixing it where nobody sees.
for src in "$REPO"/tools/*; do
    tool="$(basename "$src")"
    dst="$SP/$tool"
    [ -e "$dst" ] || [ -L "$dst" ] || continue      # only shadowing copies
    if [ -L "$dst" ] && [ "$(readlink -f "$dst")" = "$(readlink -f "$src")" ]; then
        continue                                    # already pointing at the repo
    fi
    if [ -f "$dst" ] && ! [ -L "$dst" ]; then
        echo "  rehydrate: $tool was a real file ($(wc -c <"$dst") bytes vs" \
             "$(wc -c <"$src") in the repo) — re-pointing at the repo"
    fi
    ln -sfn "$src" "$dst"
done
# These three are named because the scratchpad calls them something else, so no
# rule over basenames can find them.
for tool in refresh_audit.sh heartbeat.sh snapshot_results.py quota.sh; do
    ln -sfn "$REPO/tools/$tool" "$SP/$tool"
done

# **The standing agenda and the memory index live in the repo now.** Both were
# scratchpad-only until 2026-08-08, and `rehydrate_bootstrap.sh` had already
# written down why that is unsafe: there is no on-disk location that reliably
# survives a replacement, and origin is the only store that has never lost
# anything. AUTOPILOT.md came through one replacement by predating the snapshot
# -- luck, not durability -- and it is the file every wake-up is told to read, so
# losing it or letting it go stale is what makes an unattended session act on
# work that finished days ago.
ln -sfn "$REPO/docs/ccarc3_autopilot.md" "$SP/AUTOPILOT.md"
ln -sfn "$REPO/docs/ccarc3_memory.md"    "$SP/MEMORY.md"

# An `if`, not `[ ... ] && exit 0`. As the last command of the block that is
# what sets the exit status, so on the normal path the false test made step 2
# return 1 -- and tests/test_rehydrate_reports_what_it_did.py runs step 2 as a
# standalone slice and asserts it exits 0. A guard that changes the status of the
# thing it guards is the same trap as the one this file's step 1 comment is about.
if [ "${LINKS_ONLY:-0}" = 1 ]; then exit 0; fi

# 3+4. Fingerprint and banked results.
#
# **Failures are reported, not swallowed.** These two restores are what stop a
# replacement from presenting finished games as unstarted, so a driver relaunched
# after one silently re-runs them -- the exact failure the last block of this
# script refuses to tolerate from `refresh_audit.sh`. `2>/dev/null` here read as
# "nothing to restore" whether that was true or the script had crashed.
# **`&& echo "fingerprint logged"` tested `tail`, not the fingerprint.**
# box_fingerprint.sh runs without `set -e` or `pipefail` and ends with
# `tail -5 "$LOG"`, so its exit status was "the log file is readable" — a failed
# commit and four failed pushes both left it 0. It exits on its own success now.
if bash "$REPO/tools/box_fingerprint.sh" >/dev/null 2>&1; then
  echo "fingerprint logged"
else
  echo "fingerprint NOT recorded — the row is local only, or was never written"
fi
for restore in restore_banked_results restore_clean_rollouts; do
  # The rollout driver reads its banked markers off the scratchpad too, and
  # restores only attempts that finished without an error.
  if out="$(python3 "$REPO/tools/$restore.py" 2>&1)"; then
    echo "$out" | head -1
  else
    echo "restore: $restore.py FAILED -- finished games may look unstarted"
    echo "$out" | tail -3
  fi
done

# What is running, argv-element-exact -- a substring match finds this script.
# `clean_rollouts.py` is the driver actually in use and was missing from this
# list, so a wake-up after a replacement could not see whether it had survived.
# `supervisor.sh` stays: it is the quota duty-cycle loop and the first thing the
# autopilot check asks about. (It lives in `tools/` now -- this parenthesis used
# to say it did not, which was true when written and stopped being true without
# the line changing. A comment that contradicts the tree beside it is the same
# defect as a check that reads a proxy: both keep answering after the thing they
# describe has moved.)
for name in clean_rollouts.py ablate_baselines.py rerun_losses.py preserve_evidence.sh supervisor.sh; do
  n=0
  for d in /proc/[0-9]*; do
    [ -r "$d/cmdline" ] || continue
# **`2>/dev/null` on `tr` does not silence the redirection.** `< "$d/cmdline"` is
# opened by the SHELL before `tr` exists, so when the process exits between the
# glob and the read -- which it does, `/proc` is a live directory -- the failure is
# reported on the shell's stderr and the `2>/dev/null` never sees it. Measured
# 2026-08-09: `bash: line 43: /proc/19946/cmdline: No such file or directory`
# reached the heartbeat's output and the Monitor forwarded it as an alert. An
# alarm channel that emits noise is one people stop reading, which is the same
# reason the daemon report is not always-on. Braces put the redirection inside the
# group, so the group's stderr covers it.
    { tr '\0' '\n' < "$d/cmdline"; } 2>/dev/null | grep -qx ".*/${name}\|.*${name}" && n=$((n+1))
  done
  [ "$n" -gt 0 ] && echo "running: $name ($n)"
done

bash "$REPO/tools/refresh_audit.sh"
case $? in
  0) echo "refresh: nothing new" ;;
  10) echo "refresh: NEW RESULTS above — rebuild and republish" ;;
  *) echo "refresh: FAILED — investigate, do not read as a no-op" ;;
esac
