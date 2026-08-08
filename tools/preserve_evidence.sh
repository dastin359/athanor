#!/bin/bash
# Push run evidence to the remote continuously, because the disk is not durable.
#
# **Why this exists.** Five container rollbacks reverted this box to an older
# snapshot. The repository came through every one of them -- work was pushed as it
# was made, and each rollback was undone by fetching from origin. The scratchpad
# came through none of them: it lost the raw traces for twelve of the arm's
# twenty-five games, because it only ever existed on disk.
#
# The relaunch fix in ablate_baselines.py does NOT cover this. That handles a
# solver dying with the filesystem intact -- it relaunches in place and the client
# restores from trace.state.json. A rollback takes trace.state.json with it, so
# there is nothing to resume from. Different failure, and only pushing survives it.
#
# **What it costs to fix: almost nothing.** JSONL traces gzip about 65x -- a 32.8 MB
# trace.jsonl compresses to 0.5 MB. The whole 641 MB scratchpad is low tens of MB
# once the regenerable virtualenv under arc3/ and the secret are excluded. Size was
# never the reason this was not being pushed.
#
# **This script lives in the repo, not the scratchpad, and that is the point.**
# The first version sat in the scratchpad -- so a rollback would have deleted the
# backup script along with the data it was backing up, and nothing would have
# restarted it. A guard with the same durability as the thing it guards is not a
# guard. From here it survives every rollback the repository survives, and the
# recovery step is `git fetch && git merge --ff-only` followed by relaunching it.
#
# **Partial traces are worth pushing too.** A rollback mid-game cannot be resumed
# against ARC -- the gap always exceeds the ~12 minute window in which the card
# survives -- but rules.json carries the mechanics the run paid to learn, and
# trace.jsonl carries the ledger. Recovering those turns a total loss into a replay.
set -u

SP="/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
REPO="/home/user/athanor"
DEST="$REPO/evidence/ccarc3"
BRANCH="claude/athanor-cc-harness-variant-jpqw7t"
TICK=300

# **Source the key rather than inherit it.** `key_is_clean` refuses to commit
# without `ARC_API_KEY`, which is right -- the one state where the check cannot
# run must not read as a pass -- but it made a long-lived daemon's correctness
# depend on an undocumented inheritance from whichever shell happened to start
# it. That is the same shape as the proxy port the supervisor carried for hours
# after it had moved. Reading the file each start makes the daemon
# self-sufficient, and it is the same file every other component reads.
# shellcheck disable=SC1091
[ -f "$SP/arc3/.env" ] && . "$SP/arc3/.env"

# Everything that is evidence, and nothing that is a secret or regenerable.
# Deliberately a whitelist: a blacklist would ship the next new file by default.
# **The three files the solver reads are preserved too, not just the ones it
# writes.** Without them a finished run's *surface* -- what harness it actually
# saw -- is unrecoverable once the scratchpad copy goes, and the audit can only
# guess its generation from commit timestamps. That guess is what marked `sb26`
# superseded by a refactor that changed none of these three bytes. They are
# small and they compress; the trace is 65x larger and has never been in doubt.
KEEP=(trace.jsonl trace.state.json result.json scorecard.json rules.json
      resume_state.json meta.json CLAUDE.md DOCTRINE.md session.py)
# Copied in from the package, not written by the run: the solver imports these off
# PYTHONPATH rather than from its workspace, so nothing preserves them unless this
# does. They are the *runtime* surface -- what `status()` and every refusal say
# back -- and on 2026-08-07 a `status()` line cost a run 0.2748 by printing a
# completion cap that read as a score. Without a per-run copy there is no way to
# tell afterwards which version a finished run was talking to.
RUNTIME=(client.py gate.py)

# **Pacific at the source, per CLAUDE.md.** A convention applied by hand at
# report time gets skipped whenever a log line is pasted through, which is
# exactly how this daemon's lines reach a status report.
log() { echo "$(TZ=America/Los_Angeles date '+%H:%M:%S %Z') $*"; }

# **Refuse to commit if the key is anywhere in the staged tree.** The whitelist
# should make this impossible; that is exactly why it is worth asserting, because
# a guard that only fires when the whitelist is already broken is the one that
# matters. Compares against the live key without ever printing it.
# **It has to read through the gzip, and a missing key is not a pass.** The first
# version ran `grep -rlF` over $DEST straight after gz_atomic() had gzipped every
# file in it, so the plaintext needle could not match anything it was guarding --
# 780 of 785 files compressed. And `[ -n "$ARC_API_KEY" ] || return 0` reported
# CLEAN, and committed, whenever the daemon had no key in its environment: the
# one state in which the check is definitionally incapable of running.
key_is_clean() {
    local hits f body
    if [ -z "${ARC_API_KEY:-}" ]; then
        log "REFUSING TO COMMIT — ARC_API_KEY unset, so the key check cannot run"
        return 1
    fi
    hits=""
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        body=$(gzip -cd -- "$f" 2>/dev/null || cat -- "$f" 2>/dev/null)
        case "$body" in
            *"$ARC_API_KEY"*) hits="$hits $f";;
        esac
    done <<< "$(git -C "$REPO" diff --cached --name-only -- evidence 2>/dev/null)"
    if [ -n "$hits" ]; then
        log "REFUSING TO COMMIT — the API key appears in:$hits"
        return 1
    fi
    return 0
}

# **Write through a temp file and rename, because `>` is not atomic and this
# loop commits whatever it finds.** `gzip -c src > dest` truncates dest, then
# fills it over many write() calls. Anything reading dest in between sees a
# valid-looking prefix at a write-buffer boundary -- and this script's own
# `git status --porcelain evidence` runs every 5 minutes against the same tree it
# is rewriting.
#
# Caught on 2026-08-07: a stop-hook git check reported `wa30`'s stream.jsonl.gz
# as modified, 1060904 -> **786432** bytes. 786432 is exactly 768 KiB, which is a
# buffer multiple and not a compressed size. Re-reading it a moment later gave
# 1060904 again, byte-identical to HEAD and passing `gzip -t`. Nothing had
# changed; git had photographed a half-written file.
#
# That was a harmless false alarm only because a human-facing check saw it first.
# Had the 5-minute cycle landed in the same window, `git add evidence` would have
# staged the truncated 768 KiB blob and committed it over a good one -- silently
# destroying the only surviving copy of a banked 9/9 run's reasoning, since the
# scratchpad is not durable. rename(2) within a filesystem is atomic, so a reader
# sees either the old file or the new one and never a prefix of the new one.
gz_atomic() {
    local src="$1" dest="$2" tmp="$2.tmp.$$"
    if gzip -9 -n -c "$src" > "$tmp" 2>/dev/null; then
        mv -f "$tmp" "$dest"
    else
        rm -f "$tmp"
        log "WARNING: failed to compress $src; left $dest as it was"
    fi
}

preserve_dir() {
    local src="$1" name out f
    name=$(basename "$src")
    # Mirror the full path under the scratchpad rather than just the parent.
    # Deriving the destination from one directory level works only for a flat
    # <batch>/<game> layout; clean_rollouts nests <batch>/<game>/attempt_N/<game>,
    # so the old form produced "attempt_2/<game>" — the batch name dropped, and
    # every batch's attempt_1 colliding in one directory. Existing paths are
    # unchanged: for rerun_losses/<game> this still yields rerun_losses/<game>.
    out="$DEST/${src#$SP/}"
    mkdir -p "$out"
    for f in "${KEEP[@]}"; do
        [ -f "$src/$f" ] || continue
        # -n so the gzip header carries no name/timestamp: byte-identical output
        # for unchanged input, so git sees no diff and the log stays honest about
        # what actually changed.
        gz_atomic "$src/$f" "$out/$f.gz"
    done
    # Streams are the solver's reasoning, big and highly compressible. Kept
    # separately so a reader can fetch ledgers without them.
    # **Written once and never overwritten.** This loop runs every five minutes
    # over every run ever preserved, and it copies from the *repo*, not from the
    # run -- so re-copying restamps a finished run with whatever the package
    # looks like now. It did: on 2026-08-07 a client.py edit made 22 historical
    # runs' preserved copies byte-identical to source written hours after they
    # ended, including runs from that morning.
    #
    # That is the opposite of what these are for. Everything else here is
    # written by the run and re-copying is idempotent; these two are the only
    # entries whose source keeps moving, so they are the only ones that need
    # this. A record that tracks HEAD records nothing.
    # Only beside a real run. `preserve_dir` is called on the game-level directory
    # as well as on each attempt, and the game level holds no run artefacts -- so
    # this produced 25 directories whose entire contents were a copy of HEAD's
    # client.py and gate.py, recording nothing about any run.
    [ -f "$src/result.json" ] || [ -f "$src/trace.jsonl" ] || return 0
    for f in "${RUNTIME[@]}"; do
        [ -f "$REPO/src/athanor/ccarc3/$f" ] || continue
        [ -f "$out/$f.gz" ] && continue
        gz_atomic "$REPO/src/athanor/ccarc3/$f" "$out/$f.gz"
    done
    for f in "$src"/stream*.jsonl; do
        [ -f "$f" ] || continue
        gz_atomic "$f" "$out/$(basename "$f").gz"
    done
}

# **Re-read the outbound proxy every cycle, because the port moves.** Outbound
# HTTPS leaves this box through an agent proxy on a loopback port that changes
# when the session worker restarts, and a daemon keeps whatever it inherited at
# launch. This one ran for two hours pushing into a dead proxy. The session
# writes the live value to $SP/proxy_env; sourcing it per cycle costs nothing and
# is the difference between self-healing and silently useless.
refresh_proxy() { [ -f "$SP/proxy_env" ] && . "$SP/proxy_env"; }

mkdir -p "$DEST"
log "preserving evidence every ${TICK}s -> $DEST"

while true; do
    # **Three hardcoded names, and the submission sweep is not one of them.**
    # `clean_rollouts.py` is *forced* into a different directory for a
    # submission run -- all 25 games in `clean_rollouts` already hold a
    # `clean_result.json`, so `_run_one` skips every one of them -- which means
    # the ~$650 sweep this preserver exists to protect would have run in
    # `clean_rollouts_submission` and had nothing preserved. The header says the
    # disk is not durable and names five container rollbacks that lost twelve
    # games; this box has been replaced twice more since.
    #
    # Globbed, so a directory created after this line was written is covered by
    # it. That is the same fix `refresh_audit.sh` needed for the same reason.
    for base in $(cd "$SP" 2>/dev/null && for d in clean_rollouts* rerun_* ablate_nobaseline; do [ -d "$d" ] && echo "$d"; done); do
        [ -d "$SP/$base" ] || continue
        for d in "$SP/$base"/*/; do
            [ -d "$d" ] || continue
            preserve_dir "${d%/}"
            # clean_rollouts nests one level deeper: <game>/attempt_N/<game>/,
            # because Ccarc3Config(out_dir=X) builds the workspace at X/<game_id>.
            # A one-level glob walked straight past every rollout stream, which is
            # the one class of solver log this session can still capture.
            for a in "${d%/}"/attempt_*/*/; do
                [ -d "$a" ] || continue
                preserve_dir "${a%/}"
            done
        done
    done

    refresh_proxy
    cd "$REPO" || exit 1

    # **Refuse to work on the wrong branch.** `git push origin "$BRANCH"` pushes
    # the local ref NAMED $BRANCH, not HEAD, and exits 0 with "Everything
    # up-to-date" when that ref has not moved. Nothing compared the two, so a
    # checkout of any other branch (`main` exists locally) would have this daemon
    # committing evidence onto that branch while pushing an untouched
    # `claude/...` ref and logging "pushed N files" every cycle -- success on
    # every line, nothing preserved anywhere. Not silently redirected to
    # `HEAD:$BRANCH`: that would publish whatever happens to be checked out.
    cur="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo DETACHED)"
    if [ "$cur" != "$BRANCH" ]; then
        log "REFUSING — HEAD is on $cur, not $BRANCH; nothing preserved this cycle"
        sleep "$TICK"
        continue
    fi

    # **Drain a push backlog even on a cycle with nothing new.** The commit and
    # push both sat inside the change gate below, so once evidence stopped
    # changing the whole block was skipped: a push that failed on the last cycle
    # carrying new evidence -- exactly when a sweep's final result.json lands --
    # was never retried, and the daemon sat in its 300s loop looking healthy
    # while the work existed only on a disk that has been rolled back five times.
    #
    # Measured against `$BRANCH`, not `HEAD`, because that is the ref the push
    # sends. Local-only, so it costs nothing and needs no network.
    #
    # It deliberately does NOT reuse `$n` from inside the gate: this script runs
    # under `set -u` and `n` is unbound on any cycle that skips the gate, so
    # borrowing that message would abort the daemon on its first post-restart
    # cycle -- the container-rollback recovery this exists for.
    ahead="$(git rev-list --count "origin/$BRANCH..$BRANCH" 2>/dev/null || echo 0)"
    if [ "${ahead:-0}" -gt 0 ] 2>/dev/null; then
        drained=0
        for i in 1 2 3 4; do
            git push -q origin "$BRANCH" 2>/dev/null && { drained=1; break; }
            sleep $((2**i))
        done
        if [ "$drained" = 1 ]; then
            log "drained a backlog of $ahead unpushed commit(s)"
        else
            log "BACKLOG STUCK — $ahead commit(s) unpushed, proxy=${HTTPS_PROXY:-unset}"
        fi
    fi

    if [ -n "$(git status --porcelain evidence 2>/dev/null)" ]; then
        # **Stage first, then check.** `key_is_clean` reads
        # `git diff --cached -- evidence`, which is the INDEX, and it used to run
        # one line before `git add`. At that moment nothing is staged, so its
        # loop iterated over an empty list and returned clean every time: the
        # backstop guarding an API key against a public push had never inspected
        # a file. Its own comment says the whitelist should make a hit
        # impossible and that this is exactly why it is worth asserting -- and
        # then it asserted nothing.
        git add evidence
        if key_is_clean; then
            # **Count and commit the same set, and let that set be `evidence`.**
            # `n` counted the WHOLE index and the commit carried no pathspec, so
            # this daemon would commit anything another process happened to have
            # staged -- under an "evidence: preserve" message, and without
            # `key_is_clean` ever reading it, because that guard scans
            # `-- evidence` only. A session doing ordinary `git add` work in this
            # tree every few minutes is all it takes; the daemon ticks every 300s.
            # It has not fired in 60 preserver commits, checked by name, which is
            # luck rather than design. The pathspec makes the guard's scope and
            # the commit's scope the same set by construction.
            n=$(git diff --cached --name-only -- evidence | wc -l)
            git commit -q -m "evidence: preserve ccarc3 run artifacts ($n files)

Pushed continuously because the disk is not durable: five container rollbacks
reverted this box, the repository survived all of them by having been pushed, and
the scratchpad lost twelve games' traces by existing only on disk.

Gzipped ledgers, state, scorecards, rule books and streams. No secret and no
virtualenv -- the file set is a whitelist, and a guard refuses the commit if the
live API key appears anywhere under evidence/." -- evidence || {
                # **A failed commit must not be reported as a push.** Commits
                # here are ssh-signed through `/tmp/code-sign`, and on 2026-08-07
                # that signer returned 503: the commit died with "failed to write
                # commit object", and this loop then logged "pushed 10 files"
                # anyway -- because the push ran regardless, succeeded trivially
                # on an unchanged tree, and `n` had been counted before the commit
                # that never happened. A total preservation failure produced a
                # log line indistinguishable from success.
                #
                # It also left a stale `.git/index.lock`, which blocks every later
                # commit -- daemon and session alike -- until something removes
                # it. Clearing it here is safe only because no other git process
                # runs in this repo unattended; the check is not decoration.
                log "COMMIT FAILED — $n files staged, nothing preserved this cycle"
                if [ -f "$REPO/.git/index.lock" ] && ! pgrep -x git >/dev/null 2>&1; then
                    rm -f "$REPO/.git/index.lock"
                    log "  removed a stale .git/index.lock left by the failed commit"
                fi
                sleep "$TICK"
                continue
            }
            pushed=0
            for i in 1 2 3 4; do
                if git push -q origin "$BRANCH" 2>/dev/null; then
                    log "pushed $n files"; pushed=1; break
                fi
                sleep $((2**i))
            done
            # **Say so when the push fails.** It used to retry four times and
            # fall through in silence, so a daemon that could not reach the
            # remote looked identical to one with nothing to do -- and the only
            # symptom was commits piling up locally for someone else to notice.
            # That happened on 2026-08-07: this process started at 05:21 with
            # HTTPS_PROXY=127.0.0.1:37827, the agent proxy moved to :41751 when
            # the session worker restarted, and every push after that failed
            # without a word.
            [ "$pushed" = 1 ] || log "PUSH FAILED after 4 tries — proxy=${HTTPS_PROXY:-unset}; $n files committed locally only"
        else
            # **Unstage, or the refusal only delays the leak.** Staging now
            # happens before the check, so a refusal leaves the offending files
            # in the index and the next cycle's `git add` would sweep them into
            # a commit that passes -- because by then the key may no longer be
            # in the newly-added files, while the already-staged ones ride along.
            git reset -q -- evidence
            log "unstaged evidence after the key check refused"
        fi
    fi
    sleep "$TICK"
done
