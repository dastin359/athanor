"""Regression tests for tools/rehydrate_box.sh (step 1 + symlinks) and
tools/box_fingerprint.sh (exit status).

All three fixes under test are "a check that reported by not running":

  (a) step 1 of rehydrate_box.sh printed "already current" for four different
      outcomes -- genuinely current, recovered-from-rollback, ff-only REFUSED
      (HEAD diverged), and every fetch retry failed. Only the first is good news.
  (b) box_fingerprint.sh ended with `tail -5 "$LOG"`, so its exit status meant
      "the log is readable"; a failed commit and four failed pushes both exited
      0 and rehydrate_box.sh printed "fingerprint logged".
  (c) rehydrate_box.sh must symlink (not copy) all five scratchpad entry points.

Conventions these tests are deliberate about, because the mistakes are easy:

  * The scripts are read into memory at import time, by absolute path, BEFORE
    any subprocess chdir's into a sandbox clone -- so a test can never end up
    exercising a stale copy of the file it is testing.
  * Every test asserts the check it names actually produced output, not merely
    that some exit code came back: an early-out returns tidy exit codes too.
  * Both directions are asserted -- the outcome that must be reported, and the
    outcomes that must NOT be claimed alongside it.
  * The subprocess environment is built from scratch. Nothing is inherited from
    the launching shell (no ARC_API_KEY, no git identity, no ambient GIT_*).
  * No network: origins are local directories, and "unreachable" is a path that
    does not exist.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


# --- locate and read the scripts up front, absolute, before anything chdir's --
def _find_tools() -> Path:
    """Directory holding the scripts under test.

    Normally tests/../tools. A worktree checked out at an older commit does not
    carry them, so the canonical checkout is the fallback -- never a relative
    path, and never resolved after a chdir.
    """
    candidates = [
        Path(__file__).resolve().parents[1] / "tools",
        Path("/home/user/athanor/tools"),
    ]
    for cand in candidates:
        if (cand / "rehydrate_box.sh").is_file() and (cand / "box_fingerprint.sh").is_file():
            return cand
    raise AssertionError(f"cannot find rehydrate_box.sh / box_fingerprint.sh in {candidates}")


TOOLS = _find_tools()
REHYDRATE = TOOLS / "rehydrate_box.sh"
FINGERPRINT = TOOLS / "box_fingerprint.sh"

REHYDRATE_TEXT = REHYDRATE.read_text()
FINGERPRINT_TEXT = FINGERPRINT.read_text()

BRANCH = "sandbox-branch"

# The four mutually exclusive verdicts step 1 has to distinguish.
VERDICTS = {
    "already current": "already current",
    "recovered from rollback": "recovered from rollback",
    "FF-ONLY REFUSED": "FF-ONLY REFUSED",
    "FETCH FAILED": "FETCH FAILED",
}


# --------------------------------------------------------------------------
# extracting a block of the script
# --------------------------------------------------------------------------
def _block(text: str, start_pat: str, end_pat: str, *, include_end: bool = False) -> str:
    """Slice one contiguous run of lines out of a shell script.

    Fails loudly if the anchors stop matching, so a rewrite of the script shows
    up as a broken test rather than as a test that quietly checks nothing.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if re.search(start_pat, line)]
    assert len(starts) == 1, f"anchor {start_pat!r} matched {len(starts)} lines"
    start = starts[0]
    ends = [j for j in range(start + 1, len(lines)) if re.search(end_pat, lines[j])]
    assert ends, f"end anchor {end_pat!r} never matched after line {start}"
    end = ends[0] + 1 if include_end else ends[0]
    block = "\n".join(lines[start:end]).strip("\n")
    assert block, "extracted an empty block"
    return block


STEP1 = _block(REHYDRATE_TEXT, r"^fetched=0$", r"^# 2\.")
STEP2 = _block(REHYDRATE_TEXT, r'^mkdir -p "\$SP"$', r"^# 3\+4\.")
FINGERPRINT_CALL = _block(
    REHYDRATE_TEXT,
    r'^if bash "\$REPO/tools/box_fingerprint\.sh"',
    r"^fi$",
    include_end=True,
)


def test_extracted_blocks_are_the_code_under_test():
    """Guard the anchors themselves: a mis-slice would make everything pass."""
    assert "git fetch origin" in STEP1
    assert "git merge --ff-only" in STEP1
    assert "ln -sfn" in STEP2
    assert "box_fingerprint.sh" in FINGERPRINT_CALL
    # step 1 must not chdir; the tests run it inside a sandbox repo.
    assert not re.search(r"^\s*cd ", STEP1, re.M)


# --------------------------------------------------------------------------
# environment / sandbox helpers
# --------------------------------------------------------------------------
def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return home


def _clean_env(home: Path, extra: dict | None = None) -> dict:
    """A subprocess environment built from nothing.

    Ambient state is how shell tests start passing for the wrong reason: an
    inherited git identity, a global gitconfig, a credential helper. None of it
    is inherited here, and ARC_API_KEY is neither passed nor needed.
    """
    gitconfig = home / "gitconfig"
    gitconfig.touch(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "America/Los_Angeles",
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/true",
    }
    if extra:
        env.update(extra)
    return env


def _git(env: dict, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-c", "user.name=Sandbox", "-c", "user.email=sandbox@example.invalid", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"git {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


def _commit(env: dict, repo: Path, name: str) -> str:
    (repo / name).write_text(name + "\n")
    _git(env, repo, "add", name)
    _git(env, repo, "commit", "-q", "-m", name)
    return _git(env, repo, "rev-parse", "--short", "HEAD").stdout.strip()


def _sleep_shim(home: Path) -> Path:
    """A `sleep` that returns instantly.

    Both scripts back off between retries (0+2+4+8+16s in step 1, 2+4+8+16s in
    the push loop). The retry *loop* is what the tests exercise; the wall clock
    is not, and a 30-second test gets deleted.
    """
    bindir = home / "shimbin"
    bindir.mkdir(exist_ok=True)
    shim = bindir / "sleep"
    shim.write_text("#!/bin/sh\nexit 0\n")
    shim.chmod(0o755)
    return bindir


def _make_origin_and_box(tmp_path: Path, env: dict):
    """A bare origin on branch BRANCH plus a clone of it."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(env, seed, "init", "-q", "-b", BRANCH)
    _commit(env, seed, "A")
    _git(env, tmp_path, "clone", "-q", "--bare", str(seed), str(origin))

    box = tmp_path / "box"
    _git(env, tmp_path, "clone", "-q", "--branch", BRANCH, str(origin), str(box))
    _git(env, box, "config", "user.name", "Sandbox")
    _git(env, box, "config", "user.email", "sandbox@example.invalid")
    return origin, box


def _run_step1(box: Path, env: dict) -> subprocess.CompletedProcess:
    """Run rehydrate_box.sh's step 1, verbatim, inside a sandbox repo."""
    script = "set -uo pipefail\nsleep() { :; }\n" + STEP1 + "\n"
    return subprocess.run(
        ["bash", "-c", script],
        cwd=str(box),
        env={**env, "BRANCH": BRANCH},
        capture_output=True,
        text=True,
        timeout=120,
    )


def _verdicts(out: str) -> set:
    return {name for name, needle in VERDICTS.items() if needle in out}


def _assert_step1_ran(proc: subprocess.CompletedProcess) -> str:
    """Prove the block reached its reporting stage.

    A fixture that makes the script bail early hands back whatever exit code the
    test expected while never running the check. Step 1 always narrates, so an
    absent "repo" line means the test proved nothing.
    """
    out = proc.stdout
    assert any(line.startswith("repo") for line in out.splitlines()), (
        f"step 1 printed no 'repo' line -- it never reached the check.\n"
        f"stdout={out!r}\nstderr={proc.stderr!r}"
    )
    return out


# --------------------------------------------------------------------------
# scenarios for (a)
# --------------------------------------------------------------------------
def _scenario_already_current(tmp_path: Path, env: dict) -> Path:
    _origin, box = _make_origin_and_box(tmp_path, env)
    return box


def _scenario_recovered(tmp_path: Path, env: dict) -> Path:
    origin, box = _make_origin_and_box(tmp_path, env)
    # origin moves ahead; the box is stuck at the image commit (a rollback).
    work = tmp_path / "work"
    _git(env, tmp_path, "clone", "-q", "--branch", BRANCH, str(origin), str(work))
    _commit(env, work, "B")
    _git(env, work, "push", "-q", "origin", BRANCH)
    return box


def _scenario_diverged(tmp_path: Path, env: dict) -> Path:
    box = _scenario_recovered(tmp_path, env)
    # ...and the box has an unpushed local commit, so the merge cannot ff.
    _commit(env, box, "local-only")
    return box


def _scenario_fetch_failed(tmp_path: Path, env: dict) -> Path:
    _origin, box = _make_origin_and_box(tmp_path, env)
    # Origin unreachable, but origin/BRANCH still resolves from the clone --
    # exactly the state where `merge --ff-only` succeeds as a no-op and the old
    # code concluded "already current" from an unchanged HEAD.
    _git(env, box, "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"))
    return box


SCENARIOS = {
    "already current": _scenario_already_current,
    "recovered from rollback": _scenario_recovered,
    "FF-ONLY REFUSED": _scenario_diverged,
    "FETCH FAILED": _scenario_fetch_failed,
}


def test_step1_reports_already_current_only_when_it_is(tmp_path):
    env = _clean_env(_home(tmp_path))
    box = _scenario_already_current(tmp_path, env)
    out = _assert_step1_ran(_run_step1(box, env))

    assert _verdicts(out) == {"already current"}, out
    head = _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip()
    assert head in out


def test_step1_names_both_heads_when_it_recovers_from_a_rollback(tmp_path):
    env = _clean_env(_home(tmp_path))
    box = _scenario_recovered(tmp_path, env)
    before = _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip()
    out = _assert_step1_ran(_run_step1(box, env))
    after = _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip()

    assert _verdicts(out) == {"recovered from rollback"}, out
    assert before != after, "fixture did not actually roll the box back"
    assert before in out and after in out, out
    # the recovery is real, not merely narrated
    assert (box / "B").exists()


def test_step1_refuses_and_says_so_when_head_has_diverged(tmp_path):
    env = _clean_env(_home(tmp_path))
    box = _scenario_diverged(tmp_path, env)
    before = _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip()
    out = _assert_step1_ran(_run_step1(box, env))

    assert _verdicts(out) == {"FF-ONLY REFUSED"}, out
    assert "diverged" in out
    assert before in out
    # the false report this replaced
    assert "already current" not in out
    # HEAD is untouched: the script says it will not resolve this, so it must not
    assert _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip() == before


def test_step1_reports_fetch_failure_instead_of_claiming_current(tmp_path):
    env = _clean_env(_home(tmp_path))
    box = _scenario_fetch_failed(tmp_path, env)
    out = _assert_step1_ran(_run_step1(box, env))

    # HEAD really is unchanged here -- that is the trap. An unchanged HEAD after
    # a failed fetch is not evidence of being current.
    assert _verdicts(out) == {"FETCH FAILED"}, out
    assert "already current" not in out
    assert "5 tries" in out, "the retry loop should report how many tries it made"
    assert "NOT known current" in out


def test_step1_four_outcomes_stay_four_distinct_reports(tmp_path):
    """The regression itself: all four used to print the same line."""
    seen = {}
    for label, build in SCENARIOS.items():
        sub = tmp_path / re.sub(r"[^a-z]+", "_", label.lower())
        sub.mkdir()
        env = _clean_env(_home(sub))
        box = build(sub, env)
        out = _assert_step1_ran(_run_step1(box, env))
        got = _verdicts(out)
        assert got == {label}, f"{label!r} scenario reported {got or 'nothing'}:\n{out}"
        # normalise away the shas so distinctness cannot come from hashes alone
        seen[label] = re.sub(r"\b[0-9a-f]{7,40}\b", "<sha>", out)

    assert len(set(seen.values())) == 4, seen


# --------------------------------------------------------------------------
# (b) box_fingerprint.sh exit status
# --------------------------------------------------------------------------
def _sandbox_fingerprint(dest: Path, repo: Path, scratch: Path) -> Path:
    """Copy box_fingerprint.sh with only its config lines re-pointed.

    The script hardcodes REPO=/home/user/athanor, so it cannot be aimed at a
    sandbox by environment alone. Only the assignments are rewritten -- every
    line of the logic under test is carried over byte for byte, and that is
    asserted here rather than assumed.
    """
    subs = [
        (r"^REPO=.*$", f'REPO="{repo}"'),
        (r"^BRANCH=.*$", f'BRANCH="{BRANCH}"'),
        (r"^SP=.*$", f'SP="{scratch}"'),
    ]
    lines = FINGERPRINT_TEXT.splitlines()
    out_lines = list(lines)
    changed = []
    for pat, repl in subs:
        hits = [i for i, line in enumerate(lines) if re.match(pat, line)]
        assert len(hits) == 1, f"{pat!r} matched {len(hits)} lines in box_fingerprint.sh"
        out_lines[hits[0]] = repl
        changed.append(hits[0])

    # Everything except those assignments is untouched.
    for i, (old, new) in enumerate(zip(lines, out_lines)):
        if i not in changed:
            assert old == new
    assert "/home/user/athanor" not in "\n".join(out_lines), "copy still points at the real repo"
    # the log path must still hang off $REPO, or the sandbox would be writing
    # into the real evidence file
    assert any(line.startswith("LOG=") and "$REPO" in line for line in out_lines)

    path = dest / "box_fingerprint.sh"
    path.write_text("\n".join(out_lines) + "\n")
    path.chmod(0o755)
    return path


def _run_fingerprint(script: Path, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _fingerprint_sandbox(tmp_path: Path, *, reachable_origin: bool, can_commit: bool = True):
    home = _home(tmp_path)
    env = _clean_env(home)
    env["PATH"] = f"{_sleep_shim(home)}:{env['PATH']}"  # no 30s of backoff
    origin, box = _make_origin_and_box(tmp_path, env)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    if not reachable_origin:
        _git(env, box, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    if not can_commit:
        # no identity anywhere, and git is forbidden from guessing one
        _git(env, box, "config", "--unset", "user.name")
        _git(env, box, "config", "--unset", "user.email")
        _git(env, box, "config", "user.useConfigOnly", "true")
    script = _sandbox_fingerprint(tmp_path, box, scratch)
    return env, origin, box, script


def _log_rows(box: Path) -> list:
    log = box / "evidence" / "box_fingerprint.tsv"
    if not log.exists():
        return []
    return [ln for ln in log.read_text().splitlines() if ln and not ln.startswith("utc\t")]


def test_fingerprint_exits_zero_when_the_row_reaches_origin(tmp_path):
    env, origin, box, script = _fingerprint_sandbox(tmp_path, reachable_origin=True)
    proc = _run_fingerprint(script, tmp_path, env)

    assert len(_log_rows(box)) == 1, "the check never ran: no fingerprint row was written"
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "PUSH FAILED" not in proc.stderr
    # and the row is genuinely on origin, which is what the 0 claims
    remote = subprocess.run(
        ["git", "-C", str(origin), "show", f"{BRANCH}:evidence/box_fingerprint.tsv"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert remote.returncode == 0, remote.stderr
    assert len([ln for ln in remote.stdout.splitlines() if ln and not ln.startswith("utc\t")]) == 1


def test_fingerprint_exits_nonzero_when_every_push_fails(tmp_path):
    """The fix: the status must mean "the row reached origin", not "tail worked"."""
    env, _origin, box, script = _fingerprint_sandbox(tmp_path, reachable_origin=False)
    proc = _run_fingerprint(script, tmp_path, env)

    rows = _log_rows(box)
    assert len(rows) == 1, "the check never ran: no fingerprint row was written"
    # It got all the way to the end -- `tail -5` ran and succeeded, and the old
    # code handed back that 0.
    assert rows[0] in proc.stdout, f"tail never printed the row: {proc.stdout!r}"
    assert proc.returncode != 0, (
        "a fingerprint that never reached origin exited 0 "
        f"(stdout={proc.stdout!r} stderr={proc.stderr!r})"
    )
    assert "PUSH FAILED" in proc.stderr
    # the commit itself did happen; only the push failed
    subject = _git(env, box, "log", "-1", "--pretty=%s").stdout.strip()
    assert subject.startswith("evidence: box fingerprint"), subject


def test_fingerprint_exits_nonzero_when_the_commit_fails(tmp_path):
    env, _origin, box, script = _fingerprint_sandbox(
        tmp_path, reachable_origin=True, can_commit=False
    )
    proc = _run_fingerprint(script, tmp_path, env)

    assert len(_log_rows(box)) == 1, "the check never ran: no fingerprint row was written"
    assert "COMMIT FAILED" in proc.stderr, proc.stderr
    assert proc.returncode != 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    # nothing was committed, so the row is still sitting in the working tree
    dirty = subprocess.run(
        ["git", "-C", str(box), "status", "--porcelain"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    assert dirty != ""


@pytest.mark.parametrize(
    "exit_code, expected, forbidden",
    [(0, "fingerprint logged", "NOT recorded"), (1, "NOT recorded", "fingerprint logged")],
)
def test_rehydrate_relays_the_fingerprint_exit_status(tmp_path, exit_code, expected, forbidden):
    """Both directions: rehydrate must branch on the status, not on `&&` luck."""
    fake_repo = tmp_path / "fakerepo"
    (fake_repo / "tools").mkdir(parents=True)
    stub = fake_repo / "tools" / "box_fingerprint.sh"
    # `tail -5` succeeds, exactly as the real script's last command used to;
    # only the explicit exit status distinguishes the two runs.
    stub.write_text(f"#!/bin/bash\ntail -5 /dev/null\nexit {exit_code}\n")
    stub.chmod(0o755)

    script = "set -uo pipefail\n" + FINGERPRINT_CALL + "\n"
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(tmp_path),
        env={**_clean_env(_home(tmp_path)), "REPO": str(fake_repo)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.stdout.strip(), "the fingerprint step printed nothing at all"
    assert expected in proc.stdout, proc.stdout
    assert forbidden not in proc.stdout, proc.stdout


# --------------------------------------------------------------------------
# (c) the scratchpad symlinks
# --------------------------------------------------------------------------
EXPECTED_LINKS = {
    "refresh_audit.sh": "tools/refresh_audit.sh",
    "heartbeat.sh": "tools/heartbeat.sh",
    "quota.sh": "tools/quota.sh",
    "AUTOPILOT.md": "docs/ccarc3_autopilot.md",
    "MEMORY.md": "docs/ccarc3_memory.md",
}


def _fake_repo(tmp_path: Path, body: str = "fresh\n") -> Path:
    fake_repo = tmp_path / "fakerepo"
    for rel in EXPECTED_LINKS.values():
        target = fake_repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return fake_repo


def _run_step2(tmp_path: Path, fake_repo: Path, scratch: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", "set -uo pipefail\n" + STEP2 + "\n"],
        cwd=str(tmp_path),
        env={**_clean_env(_home(tmp_path)), "REPO": str(fake_repo), "SP": str(scratch)},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_rehydrate_links_all_five_scratchpad_entry_points(tmp_path):
    fake_repo = _fake_repo(tmp_path)
    scratch = tmp_path / "scratch"  # deliberately absent: the step must mkdir it
    proc = _run_step2(tmp_path, fake_repo, scratch)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"

    for name, rel in EXPECTED_LINKS.items():
        link = scratch / name
        assert link.is_symlink(), f"{name} is missing, or is a copy rather than a symlink"
        assert link.resolve() == (fake_repo / rel).resolve(), f"{name} -> {os.readlink(link)}"


def test_rehydrate_relinks_over_a_stale_copy(tmp_path):
    """The bug this replaced: a *copy* in the scratchpad, five days stale."""
    fake_repo = _fake_repo(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in EXPECTED_LINKS:
        (scratch / name).write_text("stale\n")

    proc = _run_step2(tmp_path, fake_repo, scratch)
    assert proc.returncode == 0, proc.stderr
    for name in EXPECTED_LINKS:
        assert (scratch / name).is_symlink(), f"{name} stayed a stale copy"
        assert (scratch / name).read_text() == "fresh\n"


def test_rehydrate_source_still_uses_symlinks_for_all_five():
    """Static backstop, so a future `cp` cannot creep back in."""
    linked = re.findall(r"^ln -sfn\s+(\S+)\s+(\S+)", REHYDRATE_TEXT, re.M)
    looped = re.findall(r"^for tool in (.+); do$", REHYDRATE_TEXT, re.M)
    names = {dst.rsplit("/", 1)[-1].rstrip('"') for _src, dst in linked}
    for group in looped:
        names.update(group.split())
    missing = set(EXPECTED_LINKS) - names
    assert not missing, f"no ln -sfn for: {sorted(missing)}"
    assert not re.search(r"^\s*cp .*\$SP", REHYDRATE_TEXT, re.M), "copies drift; use symlinks"


# ==========================================================================
# Gaps found by mutation-testing this file, closed.
# ==========================================================================

def test_step1_names_the_heads_in_the_direction_it_moved(tmp_path: Path):
    """`$before -> $after`, not the reverse.

    The recovery line is how an operator reads which way the tree moved after a
    rollback. The old assertion only required both hashes to appear somewhere,
    so swapping them read identically — a line that says the tree went from the
    NEW head to the OLD one describes the opposite of what happened.
    """
    env = _clean_env(_home(tmp_path))
    origin, box = _make_origin_and_box(tmp_path, env)

    # origin moves ahead; the box is rewound, so step 1 must fast-forward it.
    work = tmp_path / "work"
    _git(env, tmp_path, "clone", "-q", "--branch", BRANCH, str(origin), str(work))
    _git(env, work, "config", "user.name", "W")
    _git(env, work, "config", "user.email", "w@example.invalid")
    after_sha = _commit(env, work, "B")
    _git(env, work, "push", "-q", "origin", BRANCH)
    before = _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip()

    out = _assert_step1_ran(_run_step1(box, env))
    after = _git(env, box, "rev-parse", "--short", "HEAD").stdout.strip()
    assert before != after, "the sandbox did not actually fast-forward"
    line = [l for l in out.splitlines() if "recovered from rollback" in l]
    assert line, out
    assert f"{before} -> {after}" in line[0], (
        f"the recovery line names the heads in the wrong direction: {line[0]!r}"
    )


def test_step1_tells_the_operator_what_to_do_about_a_divergence(tmp_path: Path):
    """The refusal has to be actionable.

    A bare "FF-ONLY REFUSED" tells an unattended operator that something is
    wrong and nothing about what to do; the guidance lines are the difference
    between a report and an alarm. Deleting them left the old test green.
    """
    env = _clean_env(_home(tmp_path))
    origin, box = _make_origin_and_box(tmp_path, env)

    work = tmp_path / "work"
    _git(env, tmp_path, "clone", "-q", "--branch", BRANCH, str(origin), str(work))
    _git(env, work, "config", "user.name", "W")
    _git(env, work, "config", "user.email", "w@example.invalid")
    _commit(env, work, "theirs")
    _git(env, work, "push", "-q", "origin", BRANCH)
    _commit(env, box, "ours")                      # both sides now have unique commits

    out = _assert_step1_ran(_run_step1(box, env))
    assert "FF-ONLY REFUSED" in out, out
    assert "unpushed local commits" in out, (
        f"the refusal does not say WHY the tree diverged:\n{out}"
    )
    assert "rebase or push by hand" in out, (
        f"the refusal does not say what to do about it:\n{out}"
    )
