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

log() { echo "$(date -u +%H:%M:%S) $*"; }

# **Refuse to commit if the key is anywhere in the staged tree.** The whitelist
# should make this impossible; that is exactly why it is worth asserting, because
# a guard that only fires when the whitelist is already broken is the one that
# matters. Compares against the live key without ever printing it.
key_is_clean() {
    local hits
    [ -n "${ARC_API_KEY:-}" ] || return 0
    hits=$(grep -rlF "$ARC_API_KEY" "$DEST" 2>/dev/null | head -3)
    if [ -n "$hits" ]; then
        log "REFUSING TO COMMIT — the API key appears in: $hits"
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
    for f in "$src"/stream*.jsonl; do
        [ -f "$f" ] || continue
        gz_atomic "$f" "$out/$(basename "$f").gz"
    done
}

mkdir -p "$DEST"
log "preserving evidence every ${TICK}s -> $DEST"

while true; do
    for base in clean_rollouts rerun_losses ablate_nobaseline; do
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

    cd "$REPO" || exit 1
    if [ -n "$(git status --porcelain evidence 2>/dev/null)" ]; then
        if key_is_clean; then
            git add evidence
            n=$(git diff --cached --name-only | wc -l)
            git commit -q -m "evidence: preserve ccarc3 run artifacts ($n files)

Pushed continuously because the disk is not durable: five container rollbacks
reverted this box, the repository survived all of them by having been pushed, and
the scratchpad lost twelve games' traces by existing only on disk.

Gzipped ledgers, state, scorecards, rule books and streams. No secret and no
virtualenv -- the file set is a whitelist, and a guard refuses the commit if the
live API key appears anywhere under evidence/."
            for i in 1 2 3 4; do
                git push -q origin "$BRANCH" 2>/dev/null && { log "pushed $n files"; break; }
                sleep $((2**i))
            done
        fi
    fi
    sleep "$TICK"
done
