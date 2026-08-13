#!/usr/bin/env bash
# Bring the sweep back after the VM was taken out from under it.
#
# **Why this exists, and why it is not a systemd unit.** On 2026-08-12 at
# 12:51:27 PDT the WSL platform package was updated in place (MSIX 2.7.3.0 ->
# 2.7.11.0) with `ForceTargetApplicationShutdownOption`. That force-terminated
# the VM: the guest was still writing `stream.jsonl` at 12:51:46, the old vNIC
# was deleted at 12:51:49, a replacement VM booted at 12:53:32. Windows never
# rebooted and nothing inside Linux failed. The sweep then sat dead until a human
# happened to look.
#
# A systemd unit cannot fix that. When the VM is terminated, the systemd inside
# it is gone too -- there is nothing left in the guest to notice, and the VM does
# not come back on its own. The watchdog has to live on the Windows side, where
# it survives the event, and reach in. `wsl.exe` starts the VM if it is down,
# which is exactly the lever needed. So the scheduled task runs this script, and
# this script is deliberately dumb: idempotent, quiet when there is nothing to
# do, and safe to run every few minutes forever.
#
# **It cannot save the attempt that was killed.** `clean_rollouts.py` discards an
# interrupted attempt rather than resuming it -- resuming is the confound that
# disqualified the first five games -- so a mid-game kill always costs that game's
# elapsed time. What this converts is the *second* loss: a kill silently ending
# the whole sweep instead of costing one game.
#
# **The brake is the spend switch and it stays the operator's.** Nothing is
# launched unless `$CCARC3_SCRATCH/concurrency` reads a positive integer. That is
# the same file the driver polls every 10s, so "stop the sweep" remains one
# `echo 0` and is honoured here too.
#
#   tools/ccarc3_resume.sh            # what the scheduled task runs
#
# Times are Pacific at the source, per AGENTS.md: a convention applied by hand at
# report time is one that gets skipped.
set -uo pipefail
export TZ=America/Los_Angeles

SCRATCH="${CCARC3_SCRATCH:-/tmp/athanor-ccarc3-codex/scratchpad}"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
LOG="$SCRATCH/resume.log"
# Overridable so the guards can be exercised end-to-end against a stub instead of
# a real $79 game. Defaults to the real driver, so a caller that forgets gets the
# production path rather than a silent no-op.
DRIVER="${CCARC3_RESUME_DRIVER:-$REPO/tools/clean_rollouts.py}"
SUPERVISOR="${CCARC3_RESUME_SUPERVISOR:-$REPO/tools/supervisor.sh}"

log() { printf '%s %s\n' "$(date '+%H:%M %Z')" "$*" >> "$LOG"; }

# **A watchdog whose healthy state is silence cannot be distinguished from one
# that is not running.** Almost every tick is a correct no-op -- the brake is 0,
# or the driver is already up -- so logging each one would bury the real lines,
# and logging none leaves no evidence the Windows scheduled task ever fired.
# A single file's mtime is the cheapest signal that carries: `stat` it and you
# know when this last ran, with no log growth at all. Written before any guard,
# so it reports the watchdog's liveness rather than the sweep's.
mkdir -p "$SCRATCH" 2>/dev/null
: > "$SCRATCH/resume.heartbeat" 2>/dev/null

# **argv-element-exact, never a substring.** `pgrep -f` self-matches and has
# produced a wrong answer five times in this project; this script's own command
# line names the driver. Split on NUL, anchor the pattern with a slash.
pids_of() {
    local d
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        { tr '\0' '\n' < "$d/cmdline"; } 2>/dev/null | grep -Fqx -- "$1" && echo "${d#/proc/}"
    done
}

# ── 1. is a sweep sanctioned at all? ─────────────────────────────────────────
# Absent, empty, 0, or unparseable all mean "no". Only a positive integer is a
# licence to spend money, and it is read fresh every tick so revoking it is
# immediate.
brake="$(cat "$SCRATCH/concurrency" 2>/dev/null | tr -d '[:space:]')"
case "$brake" in
    ''|*[!0-9]*) exit 0 ;;
    0)           exit 0 ;;
esac

# ── 2. is the config for this sweep known? ───────────────────────────────────
# `CCARC3_ONLY` and `CCARC3_SWEEP_DIR` are per-run choices that live in the
# launching shell, and a shell does not survive the VM. Persisting them beside
# the brake is what lets an unattended restart resume the SAME sweep rather than
# quietly starting a different one -- a driver pointed at the default directory
# would skip every game already banked elsewhere and "complete" with the
# benchmark missing, which is the failure `CCARC3_SWEEP_DIR` exists to prevent.
SWEEP_ENV="$SCRATCH/sweep.env"
if [ -r "$SWEEP_ENV" ]; then
    set -a; . "$SWEEP_ENV"; set +a
else
    log "brake=$brake but $SWEEP_ENV is missing — refusing to guess which sweep to run"
    exit 1
fi

# ── 3. already running? ──────────────────────────────────────────────────────
# The driver holds an flock of its own, so a duplicate would refuse rather than
# double-play. This check exists to keep the log quiet, not for correctness.
driver_path="$(readlink -f "$DRIVER")"
if [ -n "$(pids_of "$driver_path")" ]; then
    exit 0
fi

# ── 4. the key ───────────────────────────────────────────────────────────────
# Checked on the VALUE, not the file: an .env that exists but sets nothing reads
# as present, and every ARC call would then 401 for hours.
ENVF="$SCRATCH/arc3/.env"
if [ -r "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
if [ -z "${ARC_API_KEY:-}" ]; then
    log "brake=$brake but no ARC_API_KEY in $ENVF — every ARC call would 401; not launching"
    exit 1
fi

# ── 5. launch ────────────────────────────────────────────────────────────────
export CCARC3_SCRATCH="$SCRATCH"
log "sweep sanctioned (brake=$brake, only=${CCARC3_ONLY:-<all>}, dir=${CCARC3_SWEEP_DIR:-clean_rollouts}) and no driver running — starting"

cd "$REPO" || { log "cannot cd to $REPO"; exit 1; }

# Detached into its own session so it does not die with whatever invoked this --
# `wsl.exe` returns as soon as the command finishes, and a child in that session
# would go with it.
( setsid nohup "$REPO/.venv/bin/python" "$DRIVER" \
    >> "$SCRATCH/$(basename "$DRIVER" .py).log" 2>&1 < /dev/null & )

# The supervisor is the quota duty cycle. It is separate from the driver on
# purpose: it stops the run politely at the ceiling, and it cannot start the
# first game itself on a box with no quota readings yet.
supervisor_path="$(readlink -f "$SUPERVISOR")"
if [ -z "$(pids_of "$supervisor_path")" ]; then
    ( setsid nohup bash "$SUPERVISOR" >> "$SCRATCH/supervisor.log" 2>&1 < /dev/null & )
    log "supervisor started"
fi
exit 0
