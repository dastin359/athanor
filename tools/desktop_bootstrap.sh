#!/usr/bin/env bash
# Stand up a CCARC3 rollout box on hardware you own, and prove it can run.
#
# **Why this exists.** The cloud container is replaced on a ~6h cadence -- three
# times on 2026-08-12 alone (18:48, 02:48, 08:18 PDT). `clean_rollouts.py`
# DISCARDS an interrupted attempt rather than scoring it, and `bp35` needs ~4.9h,
# so a sweep on that box is a race against a clock it cannot see. Most of the
# durability apparatus in this repo -- the card vault, `rehydrate_box.sh`, the
# 300s evidence push, the keepalive -- is scaffolding for that one problem. On a
# machine that stays up, the scaffolding is optional and the sweep is not a race.
#
# **This script does not merely configure; it runs the things it configures.**
# That distinction is the whole point. The recurring defect in this project is a
# check that names one thing and reads a proxy for it, and its signature is that
# it PASSES BY NOT RUNNING: `--version` succeeded, the import resolved, the file
# existed. So every step below that can be executed is executed -- the API is
# called, the baseline strip is installed and then probed, quota.sh is parsed for
# the windows it is required to report. A bootstrap that only checks for the
# presence of things reports a healthy box that cannot score a game.
#
#   bash tools/desktop_bootstrap.sh                       # verify + set up
#   bash tools/desktop_bootstrap.sh --scratch ~/ccarc3    # choose the scratch dir
#   bash tools/desktop_bootstrap.sh --no-live             # skip the API calls
#
# Idempotent: safe to re-run, and re-running it is the right way to check a box
# that has been sitting idle.
set -uo pipefail

# ── where everything lives ────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
SCRATCH="${CCARC3_SCRATCH:-$HOME/ccarc3-scratch}"
export CCARC3_SCRATCH="$SCRATCH"
LIVE=1
for arg in "$@"; do
    case "$arg" in
        --scratch=*) SCRATCH="${arg#--scratch=}";;
        --scratch)   NEXT_IS_SCRATCH=1;;
        --no-live)   LIVE=0;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0;;
        *) if [ "${NEXT_IS_SCRATCH:-0}" = 1 ]; then SCRATCH="$arg"; NEXT_IS_SCRATCH=0
           else echo "desktop_bootstrap: unknown argument: $arg" >&2; exit 2; fi;;
    esac
done
export CCARC3_REPO_TOOLS="$REPO/tools"
VENV="$REPO/.venv"
PY="$VENV/bin/python"

FAILED=0
step()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()    { printf '   \033[32mok\033[0m   %s\n' "$*"; }
warn()  { printf '   \033[33mwarn\033[0m %s\n' "$*"; }
die()   { printf '   \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
fatal() { printf '\n\033[31m%s\033[0m\n' "$*"; exit 1; }

# ── 1. the interpreter ────────────────────────────────────────────────────────
# `python3 -V` is not the check. The check is that the interpreter this repo will
# actually be installed under satisfies `requires-python = ">=3.11"`, and on a
# desktop `python3` is frequently 3.9 (macOS system) or 3.13 (homebrew HEAD) --
# neither of which is what a `python3 -V` glance tells you about `python3.11`.
step "interpreter"
HOST_PY=""
for cand in python3.13 python3.12 python3.11 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
        HOST_PY="$(command -v "$cand")"; break
    fi
done
[ -n "$HOST_PY" ] || fatal "no python >= 3.11 on PATH. Install one (pyproject requires it) and re-run."
ok "$HOST_PY -- $("$HOST_PY" -V 2>&1)"

# ── 2. the Codex CLI ──────────────────────────────────────────────────────────
# Resolved exactly the way session.py resolves it, so this cannot agree with the
# bootstrap and disagree with the run.
step "Codex CLI"
CLI="${CODEX_BINARY:-$(command -v codex 2>/dev/null || echo codex)}"
if ! "$CLI" --version >/dev/null 2>&1; then
    die "\`$CLI --version\` failed. Install Codex, or set CODEX_BINARY to its path."
    fatal "cannot run a solver without the CLI."
fi
ok "$CLI -- $("$CLI" --version 2>&1 | head -1)"
# The benchmark depends on these isolation switches. Refuse a CLI that predates
# any of them instead of silently inheriting user configuration or memory.
HELP="$("$CLI" exec --help 2>&1)"
for flag in --ignore-user-config --ignore-rules --disable --skip-git-repo-check; do
    case "$HELP" in *"$flag"*) :;; *) fatal "Codex exec does not support required isolation flag $flag";; esac
done
ok "Codex exposes every required isolation flag"
# Being logged in is separate from being installed, and a headless solver that
# hits an auth prompt produces an empty run rather than an error anyone reads --
# so this asks the CLI to actually answer something. It is a real (tiny) call
# against the account, hence --no-live skips it, and it is bounded because an
# unauthenticated CLI can sit waiting on a prompt nobody is there to answer.
if [ "$LIVE" = 1 ]; then
    BOOT_WORK="$SCRATCH/bootstrap-codex"
    mkdir -p "$BOOT_WORK"
    if (cd "$BOOT_WORK" && timeout 180 "$CLI" exec --json \
        --ignore-user-config --ignore-rules --disable memories \
        --skip-git-repo-check --sandbox workspace-write \
        'reply with the single word: ready') >/dev/null 2>&1; then
        ok "CLI answered a headless prompt (this account is authenticated)"
    else
        die "the CLI is installed but an isolated headless prompt failed or timed out -- authenticate Codex and retry."
    fi
else
    warn "auth probe skipped (--no-live) -- installed is not the same as logged in"
fi

# ── 3. disk ───────────────────────────────────────────────────────────────────
# Measured, not guessed: a banked 25-game sweep is 635 MB of scratch and 121 MB
# of compressed evidence. 5 GB leaves room for the venv and several sweeps.
step "disk"
mkdir -p "$SCRATCH" || fatal "cannot create $SCRATCH"
AVAIL_KB="$(df -Pk "$SCRATCH" | awk 'NR==2 {print $4}')"
if [ "${AVAIL_KB:-0}" -lt 5242880 ]; then
    warn "$(( AVAIL_KB / 1024 ))MB free at $SCRATCH -- a 25-game sweep wants ~1GB, plus venv"
else
    ok "$(( AVAIL_KB / 1048576 ))GB free at $SCRATCH"
fi

# ── 4. the virtualenv ─────────────────────────────────────────────────────────
step "virtualenv"
if [ ! -x "$PY" ]; then
    "$HOST_PY" -m venv "$VENV" || fatal "venv creation failed"
    ok "created $VENV"
else
    ok "reusing $VENV -- $("$PY" -V 2>&1)"
fi
"$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1
if "$PY" -m pip install --quiet -e "$REPO[dev]" >/dev/null 2>&1; then
    ok "athanor installed editable, with dev extras"
else
    die "\`pip install -e .[dev]\` failed -- re-run it by hand to see the error"
fi

# ── 5. the secret ─────────────────────────────────────────────────────────────
# The key lives in the scratch tree at mode 600 and NEVER in the repository --
# `evidence/` is pushed to a public remote. If ARC_API_KEY is in the environment
# this writes it; otherwise the file must already be there.
step "ARC API key"
ENVF="$SCRATCH/arc3/.env"
mkdir -p "$SCRATCH/arc3"
if [ -n "${ARC_API_KEY:-}" ]; then
    umask 077
    printf 'ARC_API_KEY=%s\n' "$ARC_API_KEY" > "$ENVF"
    chmod 600 "$ENVF"
    ok "wrote $ENVF (mode 600) from the environment"
elif [ -s "$ENVF" ]; then
    chmod 600 "$ENVF"
    ok "using existing $ENVF"
else
    die "no key. Either export ARC_API_KEY and re-run, or write $ENVF yourself:"
    printf '        umask 077; echo "ARC_API_KEY=sk-..." > %s\n' "$ENVF"
    fatal "cannot reach ARC without a key."
fi
# `.env` carries no `export`, so sourcing it makes a SHELL variable that a python
# child cannot see. That exact gap made a drill pass while the real wiring
# failed. `set -a` is the fix, and every caller below uses it.
set -a; . "$ENVF"; set +a
case "${ARC_API_KEY:-}" in
    "") die "$ENVF parsed but ARC_API_KEY is empty";;
    *)  ok "key loaded (${#ARC_API_KEY} chars, value not printed)";;
esac

# ── 6. the import surface ─────────────────────────────────────────────────────
step "harness imports"
if "$PY" - <<'PYEOF'
import sys
from athanor.ccarc3 import Ccarc3Config
from athanor.ccarc3.client import ArcClient, list_games
from athanor.ccarc3 import scoring, ledger, rules, shared_card, arc_proxy
print("   ok   imported ArcClient / scoring / ledger / rules / shared_card / arc_proxy")
PYEOF
then :; else die "import failed -- the install is broken"; fi

# ── 7. the baseline strip ─────────────────────────────────────────────────────
# The one number a solver must never see is the real per-level baseline `h`, and
# this repo is on the solver's PYTHONPATH -- so the strip is the only thing
# standing between the published baselines and a contaminated run.
#
# `assert_installed()` is the project's own check and it runs here, but note what
# it is: a check on the patch, not on the data. The data check is in step 8,
# where a LIVE `list_games()` must come back with every `baseline_actions` empty.
# That one fails if the strip is broken; this one only fails if it is absent.
step "baseline strip"
if "$PY" - <<'PYEOF'
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(os.environ["CCARC3_REPO_TOOLS"]).resolve()))
import ablate_baselines as ab
ab.install()
ab.assert_installed()
print("   ok   ablate_baselines.install() + assert_installed() pass")
PYEOF
then :; else die "the baseline strip does not install here -- a sweep from this box would be contaminated"; fi

# ── 8. the live checks ────────────────────────────────────────────────────────
# Everything above this line can pass on a box that cannot reach ARC. These
# cannot.
if [ "$LIVE" = 1 ]; then
    step "live ARC reachability"
    if "$PY" - <<'PYEOF'
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(os.environ["CCARC3_REPO_TOOLS"]).resolve()))
from athanor.ccarc3.client import HIDE_BASELINES_ENV, list_games

# **Withholding is a property of the SOLVER's environment, not of this process.**
# The first version of this check installed the strip and then asserted that
# `list_games()` came back with empty `baseline_actions` *here*. It does not, and
# it should not: the driver reads published baselines deliberately -- shortest
# first, with baseline total standing in for wall time. What the design actually
# promises is three separate things, and only the third is testable without
# opening a scorecard:
#
#   1. `build_without_baselines` strips the numbers out of the workspace files;
#   2. the solver gets `ARC_API_KEY` withheld and `CCARC3_PROXY_URL` instead;
#   3. the solver's env carries CCARC3_HIDE_BASELINES=1, so if it calls the API
#      itself the baselines come back empty.
#
# (1) and (2) are covered by `assert_installed()` in the step above and by the
# suite. (3) is the one that depends on this machine's network reaching the real
# endpoint, so it is checked here, against live data -- and checked in both
# directions, because "empty" proves nothing if the endpoint returns nothing.
try:
    os.environ.pop(HIDE_BASELINES_ENV, None)
    visible = list_games(api_key=os.environ["ARC_API_KEY"])
except Exception as e:
    print(f"   FAIL ARC unreachable: {type(e).__name__}: {e}")
    sys.exit(1)
ids = [g.game_id for g in visible]
if not ids:
    print("   FAIL ARC answered with zero games")
    sys.exit(1)
print(f"   ok   ARC answered: {len(ids)} games, e.g. {', '.join(ids[:3])}")

with_b = [g.game_id for g in visible if g.baseline_actions]
if not with_b:
    print("   warn the endpoint returned no baselines at all, so the withholding "
          "check below cannot tell 'hidden' from 'nothing to hide'")
    sys.exit(0)

os.environ[HIDE_BASELINES_ENV] = "1"
hidden = list_games(api_key=os.environ["ARC_API_KEY"])
leaked = [g.game_id for g in hidden if g.baseline_actions]
if leaked:
    print(f"   FAIL {len(leaked)} games still carry baseline_actions under "
          f"{HIDE_BASELINES_ENV}=1, e.g. {', '.join(leaked[:3])} -- a solver on "
          f"this box could read the numbers it is scored against")
    sys.exit(1)
print(f"   ok   withholding verified on live data: {len(with_b)} games carry "
      f"baselines, 0 do under {HIDE_BASELINES_ENV}=1")
sys.exit(0)
PYEOF
    then :; else die "the API key does not work from this machine"; fi
else
    step "live ARC reachability"; warn "skipped (--no-live)"
fi

# ── 9. the quota reader ───────────────────────────────────────────────────────
# supervisor.sh gates every launch on this. If it reports nothing, `util_now`
# returns empty, the `[ -n "$u" ]` guard skips the whole stop/start block, and
# the supervisor neither stops at the ceiling nor restarts after a reset. On a
# fresh box it legitimately has no readings yet -- that is a warn, not a failure,
# because the first solver run creates them.
step "quota reader"
QOUT="$(CCARC3_SCRATCH="$SCRATCH" bash "$REPO/tools/quota.sh" 2>&1)"
if [ -n "$QOUT" ]; then
    printf '%s\n' "$QOUT" | sed 's/^/        /'
    case "$QOUT" in
        *five_hour*|*seven_day*) ok "quota.sh reports at least one window";;
        *) warn "quota.sh ran but named no window -- expected on a box that has never run a solver";;
    esac
else
    warn "quota.sh printed nothing -- expected on a fresh scratch dir with no solver streams yet"
fi

# ── 10. the test suite ────────────────────────────────────────────────────────
step "test suite"
# **The exit code, not the summary text.** This grepped the tail for "failed"
# and a fully green run printed `1563 passed, 2 skipped, 1 xfailed` -- so
# "xfailed" matched "failed" and a clean suite was reported as broken. Reading a
# proxy for the result when the result itself is right there, which is the exact
# habit the rest of this script is written against.
TLOG="$(mktemp)"
"$PY" -m pytest -q -x "$REPO/tests" > "$TLOG" 2>&1
TRC=$?
tail -5 "$TLOG" | sed 's/^/        /'
rm -f "$TLOG"
case "$TRC" in
    0) ok "tests green";;
    5) warn "pytest collected no tests -- check the path";;
    *) die "pytest exited $TRC -- tests are not green on this machine";;
esac

# ── what to run, and what NOT to ──────────────────────────────────────────────
step "summary"
if [ "$FAILED" = 0 ]; then
    printf '   \033[32mThis box can run a sweep.\033[0m\n'
else
    printf '   \033[31mSomething above failed. Fix it before launching a sweep.\033[0m\n'
fi
cat <<EOF

   Put this in your shell profile (or a run script):

       export CCARC3_SCRATCH="$SCRATCH"
       export CODEX_BINARY="$CLI"

   Launch a sweep -- the supervisor is the quota duty-cycle, the driver is the
   sweep itself. The brake file is read every 10s, so 0 stops new launches
   without killing an in-flight solver:

       echo 1 > "$SCRATCH/concurrency"
       cd "$REPO"
       CCARC3_SCRATCH="$SCRATCH" nohup bash tools/supervisor.sh \\
           >> "$SCRATCH/supervisor.log" 2>&1 &

   For a sweep whose scorecard is the submission, give it its own directory --
   the driver SKIPS any game that already has a clean_result.json:

       CCARC3_SWEEP_DIR=clean_rollouts_submission ...

   Do NOT start these here. They exist to survive a container replacement, and
   this machine does not have one:

       tools/rehydrate_box.sh        restores a box from the remote
       tools/card_vault.py           encrypts the scorecard against a rollback
       tools/preserve_evidence.sh    pushes every ${CCARC3_PRESERVE_TICK:-300}s to a PUBLIC repo

   Evidence pushing is still worth having, just not at panic cadence. If you
   want it, slow it right down:

       CCARC3_PRESERVE_TICK=1800 CCARC3_SCRATCH="$SCRATCH" \\
           nohup bash tools/preserve_evidence.sh >> "$SCRATCH/preserve.log" 2>&1 &

   Last thing, and it is the one that actually bites: stop the machine sleeping.
   A suspended desktop kills an in-flight solver exactly like a container
   replacement does, and the driver discards the attempt either way.

       linux:  sudo systemctl mask sleep.target suspend.target hibernate.target
       macos:  sudo pmset -a disablesleep 1     (or: caffeinate -dimsu -w \$\$)

EOF
exit "$FAILED"
