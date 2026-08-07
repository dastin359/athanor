"""Paired regression arm: the same games, with per-level baselines withheld.

**Why.** `baseline_actions` is not in ARC's published `/api/games` schema — the
docs list `game_id` and `title`; the live server also returns `tags` and
`baseline_actions`. This harness leans on that undocumented field hard: 22
doctrine references, all of §6, the 1.0x pace warning inside `client.status()`
(called 256 times across six runs), `client.pace()`, and the workspace
`CLAUDE.md`, which hands the array to the solver as a table row. If the
semi-private set withholds it, every baseline-derived signal goes quiet at once
and the public-set figure would be measuring a solver with information the real
evaluation does not supply.

**Design.** Paired arm over **every scored game**, so each is its own control at
its own recorded RHAE — twelve at 1.000 or near it, plus `tn36` at 0.449, which
makes that one a gain arm rather than a regression arm. A fresh game would
confound "the information mattered" with "that game was hard". Same model, same
effort, same action cap, same doctrine except the parts unusable without the
numbers.

**One variable.** Only the per-level array is withheld:

- `session.py` gets `baseline_actions=()`. No client change is needed —
  `baseline_here` returns None, so `status()` drops its pace line and `pace()`
  returns nothing. The signal disappears exactly as it would if ARC stopped
  sending the field.
- `CLAUDE.md` loses the baseline table row and the paragraphs that read it.
- `DOCTRINE.md` loses §6 and §6a, which instruct the solver to steer by a number
  it can no longer see.

Deliberately **kept**: §0 (how scoring works) and the action cap. Cutting §0
would ablate "does the solver know the objective", a different variable; and an
agent is terminated at a budget on the real benchmark whether or not it knows the
human medians, so hiding the cap would change the task rather than the signal.

**Power, stated up front.** Thirteen paired games, twelve of which the control
won. Under the null, expect 12/12 again; McNemar on the discordant pairs means
3 losses reaches p=0.125 and 5 reaches p=0.031 — so this detects a moderate-to-
large effect and nothing subtler. Report it paired, game by game, never as two
pooled rates: the pairing is the entire design.

**Quota, not wall clock, is the binding constraint.** The window was 0.91 with
~19h to reset when this was written, and the guard stops at 0.95. Expect the arm
to bank two or three games, pause, and resume after the reset. It is idempotent,
so that costs only the relaunch.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/home/user/athanor/src")

from athanor.ccarc3 import Ccarc3Config, arc_proxy, list_games, run_game
from athanor.ccarc3.client import HIDE_BASELINES_ENV
from athanor.ccarc3 import session as sess

SP = pathlib.Path(
    "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
RUNS = SP / "ablate_nobaseline"
BUDGET_MULTIPLE = 2.0          # identical to batch 3, so the arms are comparable

# **Every scored game**, per the operator: "rerun all scored games without
# baseline info provided". Read from results.jsonl rather than hardcoded, so
# games finishing after this was written are included automatically.
#
# Ordered cheapest-first. Quota is the binding constraint, not wall clock: the
# guard stops the arm at 0.95 of the seven-day window, so cheap games first
# maximises how many pairs are banked before it does. The arm is idempotent, so
# it resumes across the reset.
def scored_games():
    # **Controls are the non-ablated rows only.** `snapshot_results.py` appends
    # this arm's own results to the same ledger with `batch="ablate_nobaseline"`.
    # Keyed by game_id alone, the arm's row overwrites the control it is supposed
    # to be compared against -- so every pair reads as a tie and the progress line
    # "same games with baselines: N/N" is the arm grading itself. Filtering by
    # batch is what makes the pairing real; it also keeps a game this arm ran but
    # nobody ever ran *with* baselines out of the paired set, where it would
    # contribute a fabricated control.
    ledger = SP / "results.jsonl"
    latest = {}
    if ledger.exists():
        for line in ledger.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("batch") == "ablate_nobaseline":
                continue
            if row.get("game_id") and row.get("rhae") is not None:
                latest[row["game_id"]] = row
    # A game with no recorded cost sorts last rather than first: unknown is not
    # cheap, and putting it early would spend the window on a guess.
    reruns = [
        g for g, _ in sorted(
            latest.items(), key=lambda kv: kv[1].get("cost_usd") or float("inf")
        )
    ]
    # **The untouched environments come after the reruns, deliberately.** A rerun
    # is a paired observation -- it has a control at a known RHAE, so it tests
    # whether withholding baselines costs anything. An untouched game has no
    # control and can only extend coverage. Under a quota ceiling the paired ones
    # are worth strictly more, so they go first; the rest follow cheapest-baseline
    # first once the science is banked.
    played = set(reruns)
    fresh = sorted(
        (g for g in list_games() if g.game_id not in played),
        key=lambda g: g.baseline_total,
    )
    return reruns + [g.game_id for g in fresh], latest


# **Not at module scope.** `scored_games()` calls `list_games()`, which is a live
# `GET /api/games`, so computing the queue at import made importing this module a
# network call -- and made it fail outright without a key. `main()`'s docstring
# has always said importing must not run the arm; building the queue at import is
# the same mistake one step earlier, and it is why nothing ever imported this
# module under test. The strip that decides whether the arm measures anything
# went untested for its whole life because you could not import it to test it.


def strip_baselines(root: pathlib.Path) -> None:
    """Remove every per-level baseline the solver can see. Idempotent."""
    sp = root / "session.py"
    text = sp.read_text(encoding="utf-8")
    stripped = re.sub(r"baseline_actions=\([^)]*\),", "baseline_actions=(),", text)
    if stripped == text and "baseline_actions=()" not in text:
        raise RuntimeError(f"{sp}: baseline_actions not found; template changed?")
    sp.write_text(stripped, encoding="utf-8")

    # **Drop whole paragraphs, not lines.** The filter used to be line-wise, and
    # on hard-wrapped prose that is not a redaction, it is a shredder. The
    # baseline paragraph in `CLAUDE.md` wraps across four lines, two of which
    # happen to contain the word; deleting those two left the other two standing
    # as a pair of orphaned half-sentences:
    #
    #     must also discover them, hence the larger budget. But if you are several times
    #     wrong — go re-explore rather than grind.
    #
    # That is what all eight clean rollouts read during orientation — incoherent,
    # and still asserting an allowance the harness spent four commits removing.
    # A paragraph is the smallest unit of prose that survives having a sentence
    # taken out of it, so a paragraph is the unit.
    #
    # Table rows and list items are their own paragraphs for this purpose: they
    # are self-contained, and taking one out leaves the rest of the table intact.
    #
    # **Code blocks stay line-wise, and that is not an inconsistency.** A fenced
    # block is one paragraph by this rule, so paragraph-dropping it would take
    # the whole `client.reset()` / `client.act()` example out over a single
    # trailing comment. Code has no wrapped sentences to shred, so the objection
    # above does not apply inside a fence — each line is already self-contained.
    cm = root / "CLAUDE.md"
    kept, para, fenced = [], [], False

    def flush() -> None:
        if para and not any("baseline" in line.lower() for line in para):
            kept.extend(para)
        para.clear()

    for line in cm.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            flush()
            fenced = not fenced
            kept.append(line)
            continue
        if fenced:
            if "baseline" not in line.lower():
                kept.append(line)
            continue
        if not line.strip() or line.lstrip().startswith(("|", "- ", "* ", "#")):
            flush()
            if "baseline" not in line.lower():
                kept.append(line)
            continue
        para.append(line)
    flush()
    # Collapse the runs of blank lines the removals leave behind.
    tidy: list[str] = []
    for line in kept:
        if not line.strip() and (not tidy or not tidy[-1].strip()):
            continue
        tidy.append(line)
    cm.write_text("\n".join(tidy).rstrip() + "\n", encoding="utf-8")

    # **Turn the conditionals into statements, because the condition is always
    # true here.** The doctrine has to serve a run that CAN see the baselines, so
    # it phrases the replay rule as "if you cannot compute `raw`, replay anyway",
    # offers `arc.score_run(transitions, baselines)` for the case where you can,
    # and says "if you do not know the baselines, you still know the cap". In a
    # stripped workspace every one of those antecedents holds unconditionally --
    # `baseline_actions` is `()` and there is no path to a median.
    #
    # Leaving them as conditionals is not neutral. It tells the solver that
    # computing `raw` is a thing that might be possible, which invites it to
    # reach for whatever looks closest: `tu93` estimated `raw` from an in-game
    # drain bar and concluded a replay would not help (it was worth +0.1202), and
    # `bp35` read the harness's own `cap 1.000` as its score and stopped (it was
    # worth +0.2748). A branch whose true arm is unreachable should not be
    # written as a branch.
    #
    # Asserted rather than best-effort: if a doctrine edit breaks a match, this
    # raises instead of silently shipping the conditional again.
    doc_text = (root / "DOCTRINE.md").read_text(encoding="utf-8")
    already_stripped = True
    for old, new_text in (
        ("**If you cannot compute `raw`, replay anyway.** Without the per-level baselines\n"
         "you cannot tell whether",
         "**You cannot compute `raw` on this run, so replay at the winning frame.**\n"
         "There are no per-level baselines here and no way to obtain them, so you\n"
         "cannot tell whether"),
        ("If you *do* have the per-level baselines, `arc.score_run(client.transitions(),\n"
         "baselines)` gives `raw`, `cap` and the breakdown. If you do not, you are in the\n"
         "case above and the decision does not need them.",
         "There is no call that will give you `raw`. `arc.score_run` needs the\n"
         "per-level baselines as an argument and you do not have them, so do not go\n"
         "looking for a substitute: an in-game meter is not the human median, and\n"
         "neither is `cap`. The decision above does not need `raw` at all."),
        ("**If you do not know the baselines, you still know the cap.**",
         "**You do not know the baselines. You do know the cap.**"),
    ):
        if new_text in doc_text:
            continue                      # already rewritten; the strip is idempotent
        already_stripped = False
        if old not in doc_text:
            raise RuntimeError(
                f"DOCTRINE.md no longer contains the conditional this rewrites:\n{old[:80]}..."
            )
        doc_text = doc_text.replace(old, new_text, 1)
    (root / "DOCTRINE.md").write_text(doc_text, encoding="utf-8")

    # **Fenced passages, for the arithmetic that is scattered rather than
    # sectioned.** The three rewrites above are sentence surgery, and sentence
    # surgery only reaches the sentences you thought of: they turned the `raw`
    # conditionals into statements and left two whole passages standing that a
    # baseline-free solver cannot use -- the "judge it by arithmetic" bullets and
    # the binding-term table -- both of which lead with *a replay gains you
    # nothing*. A solver that cannot evaluate the antecedent reads the
    # conclusion, which is the failure `bp35` made and the fix was supposed to
    # prevent.
    #
    # So the doctrine now fences those passages itself and this removes whatever
    # is inside the markers. Rewording a fenced passage cannot reopen the leak,
    # and adding a new one costs a pair of comment lines rather than a patch
    # here. HTML comments are invisible in rendered Markdown, so the
    # baseline-visible copy reads as continuous prose.
    doc = root / "DOCTRINE.md"
    lines = doc.read_text(encoding="utf-8").splitlines()
    out, inside, fenced = [], False, 0
    for line in lines:
        if line.strip() == "<!-- BASELINE-ONLY -->":
            inside = True
            continue
        if line.strip() == "<!-- /BASELINE-ONLY -->":
            if not inside:
                raise RuntimeError("DOCTRINE.md: closing BASELINE-ONLY with no opener")
            inside = False
            fenced += 1
            continue
        if not inside:
            out.append(line)
    if inside:
        raise RuntimeError("DOCTRINE.md: unclosed BASELINE-ONLY fence")
    # **Count, do not merely require one.** `if not fenced` passes when one of
    # the two fence pairs is lost to an edit, shipping the passage it guarded --
    # and the passages are the "a replay gains you nothing" bullets, which are
    # the exact thing a baseline-free solver must not read.
    EXPECTED_FENCES = 2
    if fenced and fenced != EXPECTED_FENCES:
        raise RuntimeError(
            f"DOCTRINE.md has {fenced} BASELINE-ONLY fence(s), expected "
            f"{EXPECTED_FENCES}. One was lost to an edit and the passage it "
            f"guarded just shipped."
        )
    if not fenced and not already_stripped:
        # Not a no-op: a fence silently lost to an edit puts the "a replay gains
        # you nothing" bullets back in front of a solver that cannot evaluate
        # them. Raise rather than ship the workspace.
        raise RuntimeError(
            "DOCTRINE.md has no BASELINE-ONLY fences -- the doctrine lost them "
            "in an edit, and the passages they guard are baseline-only"
        )
    # Two blank lines where a fenced block was is a paragraph break plus a gap;
    # collapse so the seam is not visible as an unexplained hole.
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    doc.write_text(text.rstrip("\n") + "\n", encoding="utf-8")

    # §6 and §6a tell the solver to steer by a number it can no longer read, and
    # §6's worked examples contain real per-level medians.
    #
    # **This was the one doctrine edit that asserted nothing.** Every other one
    # here raises when its anchor moves: the three `raw` rewrites raise "no
    # longer contains the conditional", the fences raise "has no BASELINE-ONLY
    # fences". This was a bare literal match on `## 6. `, so renumbering the
    # doctrine -- an ordinary edit, and one made more likely by the fact that the
    # stripped copy already jumps 5b to 7 -- silently turns it into a no-op and
    # ships the medians. The post-strip scan does not cover it either: that looks
    # for `baseline_actions` followed by a container, and §6 writes its numbers
    # as prose in a table.
    lines = doc.read_text(encoding="utf-8").splitlines()
    out, skipping, dropped = [], False, 0
    for line in lines:
        if re.match(r"^## 6\. ", line):
            skipping = True
        if skipping and re.match(r"^## (?!6)", line):
            skipping = False
        if skipping:
            dropped += 1
        else:
            out.append(line)
    if not dropped and not already_stripped:
        raise RuntimeError(
            "DOCTRINE.md has no '## 6. ' section to remove. If the doctrine was "
            "renumbered, this strip just shipped the pacing section and its "
            "per-level medians to a baseline-free solver."
        )
    doc.write_text("\n".join(out) + "\n", encoding="utf-8")

    # **meta.json carries the whole GameInfo, including baseline_actions.**
    # `build_workspace` writes it and nothing here used to touch it, so twelve
    # runs of this arm shipped the array in a file the solver reads during
    # orientation. All twelve had it in context; `tu93` went further and called
    # `arc.score_run(ts, bl)` with it. The arm measured nothing it claimed to.
    meta = root / "meta.json"
    if meta.exists():
        blob = json.loads(meta.read_text(encoding="utf-8"))
        blob.pop("baseline_actions", None)
        # **And the budget, because it is the baseline total times five.**
        # Removing the per-level array while leaving `action_budget` hands back
        # exactly what was removed: the solver caught doing this said so in as
        # many words -- "they total 518 across six levels, and I have a budget
        # of 2590, which is 5 times the baseline". 2590/5 = 518. A strip that
        # leaves a trivially invertible function of the secret has not stripped
        # anything. The cap still reaches the client through the environment,
        # where nothing the solver reads by default will show it.
        blob.pop("action_budget", None)
        meta.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    # **Scan the whole workspace, not the files this function edited.**
    # The old check looked at exactly `sp`, `cm` and `doc` -- the three it had
    # just rewritten -- so it could only ever confirm its own edits. A leak check
    # scoped to what you changed proves nothing about what you shipped. Any file
    # the solver can open is in scope, so every file is.
    # Match the *values*, not the identifier: `baseline_actions=()` is the
    # blanked form this function produces and must not trip its own check. So
    # require a container that actually opens onto a digit --
    # `baseline_actions=(22,` and `"baseline_actions": [22,` both match, `=()`
    # does not.
    numbers = re.compile(r"""baseline_actions["'\s:=]*[(\[]\s*\d|baseline actions per level""")
    leaked = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".md", ".json", ".txt", ".yaml", ".yml"}
        and p.name not in {"trace.jsonl", "stream.jsonl"}
        and numbers.search(p.read_text(encoding="utf-8", errors="ignore"))
    )
    if leaked:
        raise RuntimeError(f"baselines still reachable in {leaked}")


_original_build = sess.build_workspace


def _actions_already_spent(root: pathlib.Path) -> int:
    """What a resumed game has already charged, per the client's own checkpoint."""
    state = root / "trace.state.json"
    if not state.exists():
        return 0
    try:
        return int(json.loads(state.read_text(encoding="utf-8")).get("actions_used", 0))
    except (OSError, ValueError, TypeError):
        return 0


# One shim per game, keyed by id. A game keeps its proxy for the whole run --
# the ARC session pinned inside it is what makes the scorecard readable -- and
# gives it back when the driver releases it.
_proxies: dict[str, "arc_proxy.Proxy"] = {}
_proxies_lock = threading.Lock()


def proxy_for(game_id: str) -> "arc_proxy.Proxy":
    with _proxies_lock:
        proxy = _proxies.get(game_id)
        if proxy is None:
            proxy = _proxies[game_id] = arc_proxy.Proxy()
            print(f"arc_proxy[{game_id.split('-')[0]}]: {proxy.url}", flush=True)
        return proxy


def release_proxy(game_id: str) -> None:
    """Free a finished game's shim. Ports are not infinite and neither is memory."""
    with _proxies_lock:
        proxy = _proxies.pop(game_id, None)
    if proxy is not None:
        proxy.shutdown()


def build_without_baselines(config, info=None, *, arc_root=None):
    proxy = proxy_for(config.game_id)
    ws = _original_build(config, info, arc_root=arc_root or proxy.url)
    strip_baselines(ws.root)

    # **The cap is the baseline total times `budget_multiple`, so a solver that
    # can read its cap can read the secret.** Stripping `action_budget` from
    # `meta.json` did not finish the job: `build_workspace` also exports
    # `CCARC3_MAX_ACTIONS` to the child, and the workspace `session.py` reads it
    # from there. The child can read its own environment as easily as its own
    # files, and `budget_multiple: float = 5.0` is a default sitting in this
    # package's source on the solver's `PYTHONPATH` -- so cap/5 recovers the
    # total exactly. Enforce it in the proxy instead, where the number lives in
    # a process the solver does not run.
    budget = ws.info.suggested_budget(ws.config.budget_multiple)
    proxy.set_budget(budget, used=_actions_already_spent(ws.root))
    ws.env.pop("CCARC3_MAX_ACTIONS", None)

    # The key would defeat everything above it. `GET /api/games` returns
    # `baseline_actions` for all 25 environments, and `install()` has pointed the
    # client at the proxy, which refuses that path and holds the credential.
    if "ARC_API_KEY" in ws.env or "ARCPRIZE_API_KEY" in ws.env:
        raise RuntimeError(
            "the solver's environment still holds an ARC key, so /api/games is "
            "one urllib call away. install() must set CCARC3_PROXY_URL before "
            "any workspace is built."
        )
    # **Close the channel no file edit can reach.** The solver holds an API key
    # and imports the package, so `arc.list_games()` would hand back
    # `baseline_actions` for all 25 environments however thoroughly the
    # workspace is sanitised. This is set on the *child's* environment only --
    # this process keeps the real numbers, which it needs to build the queue and
    # to score.
    #
    # Verified unused for the first twelve runs, so it changes no result already
    # banked; it stops the thirteenth from being the one that quietly does it.
    ws.env[HIDE_BASELINES_ENV] = "1"
    return ws


def batch6_running() -> bool:
    for d in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            argv = (d / "cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue
        if any(a.endswith("/batch6.py") for a in argv):
            return True
    return False


# **Runs concurrently with batch 3 by default**, on the operator's instruction:
# "you can launch the first non-baseline rollout once you confirm everything is
# ready. No need to wait for the current s5i5 to finish." That doubles the quota
# burn rate, which is why `quota_guard.sh` was extended to stop *both* runners at
# 0.95 rather than only batch 3. Set ABL_WAIT=1 to serialise instead.
if os.environ.get("ABL_WAIT") == "1":
    waited = 0
    while batch6_running():
        print(f"waiting for batch 3 to stop... ({waited}s)", flush=True)
        time.sleep(60)
        waited += 60
        if waited > 10800:
            print("batch6 still running after 3h; starting anyway", flush=True)
            break

def _install_patch() -> None:
    """Redirect workspace building through the strip. Called from `main()` only.

    **Not at module scope, and that is load-bearing.** `faithful_arm.py` imports
    this module for `strip_baselines`, then installs its *own* builder and keeps
    `_original_build = sess.build_workspace` as the thing to wrap. If importing
    here had already swapped in `build_without_baselines`, that capture would
    grab the patched function and every faithful-arm workspace would be stripped
    twice — once correctly, once against a template that no longer matches.

    Import side effects that rewrite another module's globals are exactly the
    kind of thing that is invisible until it produces a wrong number.
    """
    import athanor.ccarc3 as pkg

    sess.build_workspace = build_without_baselines
    pkg.build_workspace = build_without_baselines


def install() -> None:
    """Install the baseline strip. **Call this; importing does not.**

    `_install_patch()` is deliberately not a module-scope side effect, and that
    is still right -- see its docstring. But two drivers carried the line

        import ablate_baselines as ab   # installs the baseline strip

    and that comment was false. Importing installs nothing, so every run launched
    through `tools/rerun_losses.py` and `tools/clean_rollouts.py` shipped a
    workspace with the real per-level baselines and the action budget in
    `meta.json`. Three re-runs and eight rollouts were contaminated before a
    solver's own reasoning gave it away, quoting the numbers back out of
    `meta.json`.

    So the install is a function you call, with a name that says what it does,
    and `assert_installed()` below makes forgetting it fatal instead of silent.

    **It also starts the key shim**, which had been finished, tested and never
    switched on. `arc_proxy` exists precisely because file-level sanitation
    cannot stop a solver that holds `ARC_API_KEY` from calling `/api/games` and
    reading the medians for all 25 environments -- two runs did exactly that.
    Nothing anywhere set `CCARC3_PROXY_URL`, so the branch in `build_workspace`
    that drops the key never once ran, and every game in this project was played
    by a solver holding the credential. The blocker was real when it was written
    (the shim dropped session cookies, which breaks every game it touches) and
    has since been fixed and covered end to end; what was left was the wiring.
    """
    _install_patch()
    # Proves a shim can start -- the key is present and a port is bindable --
    # before any game is launched, and leaves `CCARC3_PROXY_URL` set so
    # `assert_installed` and any single-game caller still find one. Per-game
    # shims are created on demand by `proxy_for`.
    if not os.environ.get("CCARC3_PROXY_URL"):
        probe = arc_proxy.Proxy()
        os.environ["CCARC3_PROXY_URL"] = probe.url
        print(f"arc_proxy: {probe.url} (key withheld from solvers, cap enforced here)")
        # **And shut it down again.** It exists to prove a shim can bind before
        # any game is launched, and a listener kept past that is an uncapped
        # route to ARC -- `Proxy()` starts with `MAX_ACTIONS = 0`, so anything
        # reaching it spends without limit. Its URL was in `os.environ`, and
        # `build_workspace` copies the environment wholesale, so every solver
        # was handed the address of a proxy that would not have counted its
        # actions. `wa30` printed it while probing for its cap and never used
        # it; the next one might have.
        probe.shutdown()


def assert_installed() -> None:
    """Abort unless workspace building actually goes through the strip.

    The failure this exists for is not "the strip is broken" -- the strip raises
    loudly if a single file still holds a baseline. It is "the strip was never
    wired in", which produces a *successful* run against a contaminated
    workspace. That is invisible in every log and every result file, and it is
    only detectable afterwards by reading what the solver said.
    """
    if sess.build_workspace is not build_without_baselines:
        raise RuntimeError(
            "baseline strip is NOT installed: sess.build_workspace is "
            f"{getattr(sess.build_workspace, '__name__', sess.build_workspace)!r}. "
            "Call ablate_baselines.install() before running any game -- importing "
            "this module does not install it."
        )
    if not os.environ.get("CCARC3_PROXY_URL"):
        raise RuntimeError(
            "CCARC3_PROXY_URL is unset, so build_workspace will leave ARC_API_KEY "
            "in the solver's environment and /api/games will hand back the "
            "baselines for all 25 games. Call ablate_baselines.install()."
        )


def main() -> None:
    """Run the arm. Importing this module must not do that.

    Without the guard, `import ablate_baselines` — which `faithful_arm.py` does,
    to reuse `strip_baselines()` rather than keep a second copy that can drift —
    plays twenty-five games. It also made a startup crash in this module fail
    every importer, which is how the `CONTROL[g]` KeyError was found: by an
    import, not by the arm.

    The monkey-patching above stays at module scope on purpose. It is what makes
    `strip_baselines` meaningful, and an importer that wants the function but not
    the patch should take the function alone.
    """
    install()
    assert_installed()
    GAMES, CONTROL = scored_games()
    infos = {g.game_id: g for g in list_games()}
    results = []
    # `CONTROL.get`, not `CONTROL[...]`. The queue is no longer "scored games": it is
    # the paired ones *plus* every untouched environment, and those have no control
    # by definition. Indexing directly raised KeyError on `bp35` before a single game
    # ran -- and since the runner dies at import, the supervisor would have seen no
    # runner, restarted it, and crash-looped through the whole quota window.
    #
    # Latent until the CONTROL map was correctly narrowed to non-ablation rows; that
    # fix turned a bug that could not fire into one that fired every time.
    _paired = [g for g in GAMES if g in CONTROL]
    print(f"BASELINE-FREE ARM: {len(GAMES)} games — {len(_paired)} paired against a "
          f"control ({sum(1 for g in _paired if CONTROL[g].get('won'))} of those were wins), "
          f"{len(GAMES) - len(_paired)} untouched environments with no control",
          flush=True)

    for i, game_id in enumerate(GAMES, 1):
        done = RUNS / game_id / "result.json"
        if done.exists():
            try:
                prior = json.loads(done.read_text())
            except json.JSONDecodeError:
                prior = None
            if prior and not prior.get("error"):
                print(f"[{i}/{len(GAMES)}] {game_id}: already finished, skipping", flush=True)
                results.append(prior)
                continue

        # The last direct lookup in the play loop, and the queue is half-derived from
        # a ledger on disk: a game_id in results.jsonl that ARC has since dropped from
        # the public set would not be in `infos`. Safe today (all 25 resolve), guarded
        # anyway because the cost is asymmetric — a KeyError here kills the process,
        # the supervisor restarts it, and it dies on the same game forever. Skipping
        # one stale id costs one game.
        info = infos.get(game_id)
        if info is None:
            print(f"[{i}/{len(GAMES)}] {game_id}: not in list_games() any more, skipping",
                  flush=True)
            continue
        print(f"[{i}/{len(GAMES)}] {game_id} levels={info.levels} "
              f"(baselines withheld; cap {info.suggested_budget(BUDGET_MULTIPLE)})", flush=True)
        try:
            out = run_game(
                Ccarc3Config(game_id, out_dir=RUNS, budget_multiple=BUDGET_MULTIPLE,
                             wall_clock_timeout_s=7200.0),
                info,
            )
            print("   ", json.dumps(out), flush=True)
            results.append(out)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not end the arm
            print(f"    FAILED {type(exc).__name__}: {exc}", flush=True)
            results.append({"game_id": game_id, "error": str(exc)})

        done_ids = [r["game_id"] for r in results if r.get("game_id")]
        won = sum(1 for r in results if r.get("won"))
        ctrl = sum(1 for g in done_ids if CONTROL.get(g, {}).get("won"))
        print(f"    ==> baseline-free {won}/{len(results)} won | "
              f"same games with baselines: {ctrl}/{len(done_ids)}", flush=True)

    print("\n=== arm complete ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
