"""Regression tests for the git semantics `tools/preserve_evidence.sh` depends on.

Four fixes landed in that daemon and none of them had a test. Each one is a
guard that reported by not running:

(a) The commit carried no pathspec while `key_is_clean` scanned `-- evidence`
    only, so anything another process had staged rode into an
    "evidence: preserve" commit without the key check ever reading it.
(b) `git push origin "$BRANCH"` pushes the ref NAMED $BRANCH, not HEAD, and
    exits 0 with "Everything up-to-date" when that ref has not moved -- so a
    daemon on the wrong branch logged success every cycle and preserved nothing.
(c) The backlog drain measures `origin/$BRANCH..$BRANCH`, which must be
    local-only: a drain that needed the network could not run on the cycle where
    the network is what failed.
(d) The drain runs BEFORE the change gate, and `$n` is assigned inside the gate.
    Under `set -u` a `$n` in the drain aborts the daemon on its first
    post-restart cycle -- the container-rollback recovery it exists for.

The git behaviours are proved against throwaway repos built with `git init` in
tmp_path (never the real repo, and never the network -- remotes here are local
bare directories). Both directions are asserted everywhere: the failure mode is
demonstrated to actually fail before the fixed form is asserted to pass.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

# Resolved once, at import, from an absolute path -- never relative to a cwd
# that a test may later have changed into a sandbox clone.
_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "tools" / "preserve_evidence.sh",
    Path("/home/user/athanor/tools/preserve_evidence.sh"),
)
SCRIPT_PATH = next((p for p in _CANDIDATES if p.is_file()), None)
if SCRIPT_PATH is None:  # pragma: no cover - environment problem, not a failure mode
    raise RuntimeError(
        "preserve_evidence.sh not found at any of: "
        + ", ".join(str(p) for p in _CANDIDATES)
    )
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
SCRIPT_LINES = SCRIPT.splitlines()

#: The same file with every full-line comment blanked out, line numbering
#: preserved. Every static assertion below runs against this, not against
#: SCRIPT: this script documents its own bugs in prose, so `git add evidence`,
#: `HEAD:$BRANCH` and `$n` all appear in comments describing the *old* broken
#: form. Searching the raw text finds the description and calls it the code.
CODE_LINES = ["" if ln.lstrip().startswith("#") else ln for ln in SCRIPT_LINES]
CODE = "\n".join(CODE_LINES)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def git_env(home: Path) -> dict:
    """A hermetic git environment.

    Nothing here is inherited from the launching shell: identity, config and
    signing are all pinned, so these tests cannot pass or fail depending on who
    ran them. The real repo signs commits through an external ssh signer; a test
    that picked that up would be testing the signer.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GIT_", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                             "https_proxy", "http_proxy", "all_proxy"))
    }
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=str(home / ".gitconfig-none"),
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="/bin/false",
        LC_ALL="C",
    )
    return env


def git(repo: Path, *args: str, env: dict, check: bool = True):
    proc = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def out(proc) -> str:
    return proc.stdout.strip()


BRANCH = "codexarc3"


@pytest.fixture
def sandbox(tmp_path: Path):
    """A throwaway work repo on $BRANCH with one commit, plus a bare remote."""
    home = tmp_path / "home"
    home.mkdir()
    env = git_env(home)

    bare = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(bare)],
        env=env, check=True, capture_output=True,
    )

    repo = tmp_path / "work"
    repo.mkdir()
    git(repo, "init", "-q", "-b", BRANCH, env=env)
    (repo / "evidence").mkdir()
    (repo / "evidence" / "seed.txt").write_text("seed\n", encoding="utf-8")
    git(repo, "add", "evidence", env=env)
    git(repo, "commit", "-q", "-m", "seed", env=env)
    git(repo, "remote", "add", "origin", str(bare), env=env)
    git(repo, "push", "-q", "origin", BRANCH, env=env)
    return repo, bare, env


def bare_rev(bare: Path, ref: str, env: dict) -> str:
    return out(
        subprocess.run(
            ["git", "--git-dir", str(bare), "rev-parse", ref],
            env=env, capture_output=True, text=True, check=True,
        )
    )


def script_branch_name() -> str:
    """The branch the script actually resolves, not the text of its assignment.

    This used to regex the value out from between the quotes. Once BRANCH became
    `${CCARC3_BRANCH:-...}` that returned the expansion verbatim -- a "branch
    name" of `${CCARC3_BRANCH:-claude/...}`. Evaluating the real line in a real
    shell is both correct and immune to the next change of shape.

    CCARC3_BRANCH is stripped so this reports the default, which is what the
    sandbox checks out.
    """
    m = re.search(r"^BRANCH=.*$", SCRIPT, re.M)
    assert m, "preserve_evidence.sh no longer defines BRANCH= at top level"
    env = {k: v for k, v in os.environ.items() if k != "CCARC3_BRANCH"}
    return subprocess.run(
        ["bash", "-c", m.group(0) + '\nprintf "%s" "$BRANCH"'],
        capture_output=True, text=True, check=True, env=env,
    ).stdout.strip()


def change_gate_bounds():
    """Line indices [start, end] of the `git status --porcelain evidence` gate."""
    start = None
    for i, line in enumerate(CODE_LINES):
        if re.match(r'^\s*if \[ -n "\$\(git status --porcelain evidence', line):
            start = i
            break
    assert start is not None, (
        'the change gate `if [ -n "$(git status --porcelain evidence ...)" ]` is gone'
    )
    indent = len(CODE_LINES[start]) - len(CODE_LINES[start].lstrip())
    closer = " " * indent + "fi"
    end = None
    for j in range(start + 1, len(CODE_LINES)):
        if CODE_LINES[j].rstrip() == closer:
            end = j
            break
    assert end is not None, "could not find the change gate's closing `fi`"
    return start, end


#: `$n` / `${n}` / `${n:-...}` but not `$name`, `$now`, `$nothing`.
N_REF = re.compile(r"\$n(?![A-Za-z0-9_])|\$\{n(?![A-Za-z0-9_])")


# --------------------------------------------------------------------------
# (a) the commit pathspec
# --------------------------------------------------------------------------


def test_pathspec_less_commit_swallows_a_foreign_staged_file(sandbox):
    """The defect, reproduced: `git commit -m MSG` commits the WHOLE index.

    This is the negative half. Without it, the next test proves only that git
    can commit a subdirectory, not that the missing pathspec was a real bug.
    """
    repo, _bare, env = sandbox
    (repo / "evidence" / "run.json.gz").write_text("evidence payload\n", encoding="utf-8")
    (repo / "secrets.env").write_text("ARC_API_KEY=sk-live-DEADBEEF\n", encoding="utf-8")
    git(repo, "add", "evidence", "secrets.env", env=env)

    # What key_is_clean would have inspected: evidence only.
    guard_scope = out(
        git(repo, "diff", "--cached", "--name-only", "--", "evidence", env=env)
    ).split()
    assert guard_scope == ["evidence/run.json.gz"]
    assert "secrets.env" not in guard_scope

    git(repo, "commit", "-q", "-m", "evidence: preserve ccarc3 run artifacts", env=env)

    committed = out(git(repo, "show", "--name-only", "--format=", "HEAD", env=env)).split()
    assert "secrets.env" in committed, (
        "expected the pathspec-less commit to swallow the foreign file; git "
        f"committed only {committed}"
    )
    assert sorted(committed) == ["evidence/run.json.gz", "secrets.env"]


def test_commit_with_evidence_pathspec_leaves_a_foreign_staged_file_staged(sandbox):
    """The fix: `git commit -m MSG -- evidence` commits evidence/ and nothing else.

    The foreign file must stay in the index -- staged, uncommitted -- because
    that is what makes the daemon's blast radius exactly the set `key_is_clean`
    inspects.
    """
    repo, _bare, env = sandbox
    (repo / "evidence" / "run.json.gz").write_text("evidence payload\n", encoding="utf-8")
    (repo / "secrets.env").write_text("ARC_API_KEY=sk-live-DEADBEEF\n", encoding="utf-8")
    git(repo, "add", "evidence", "secrets.env", env=env)

    staged_all = out(git(repo, "diff", "--cached", "--name-only", env=env)).split()
    staged_evidence = out(
        git(repo, "diff", "--cached", "--name-only", "--", "evidence", env=env)
    ).split()
    assert "secrets.env" in staged_all
    assert "secrets.env" not in staged_evidence, (
        "the key guard's scope must NOT include the foreign file -- that "
        "asymmetry is the whole bug"
    )
    # The daemon's own count must be the pathspec'd one, not len(staged_all).
    assert len(staged_evidence) == 1 and len(staged_all) == 2

    git(
        repo,
        "commit",
        "-q",
        "-m",
        "evidence: preserve ccarc3 run artifacts (1 files)",
        "--",
        "evidence",
        env=env,
    )

    committed = out(git(repo, "show", "--name-only", "--format=", "HEAD", env=env)).split()
    assert committed == ["evidence/run.json.gz"], f"commit carried {committed}"
    assert "secrets.env" not in committed

    still_staged = out(git(repo, "diff", "--cached", "--name-only", env=env)).split()
    assert still_staged == ["secrets.env"], (
        f"the foreign file must remain staged and untouched; index holds {still_staged}"
    )
    # It reached no commit at all, not merely not this one.
    history = git(repo, "log", "--all", "--format=", "--name-only", env=env).stdout
    assert "secrets.env" not in history


def test_script_commits_and_counts_the_same_set_the_key_guard_scans():
    """Static: the commit's scope and `key_is_clean`'s scope are one set."""
    assert CODE.count("git commit") == 1, "more than one commit site; re-check this test"
    i = CODE.index("git commit")
    commit_cmd = CODE[i : CODE.index("\n", CODE.index("|| {", i))]
    assert "-- evidence" in commit_cmd, (
        "git commit lost its `-- evidence` pathspec; it will commit whatever "
        "another process happened to stage"
    )

    m = re.search(r"^\s*n=\$\(git diff --cached --name-only ([^)]*)\)", CODE, re.M)
    assert m, "the staged-file count is gone or changed shape"
    assert "-- evidence" in m.group(1), "n= counts the whole index again, not just evidence"

    body = re.search(r"^key_is_clean\(\) \{\n(.*?)^\}$", CODE, re.M | re.S)
    assert body, "key_is_clean() is gone"
    assert 'git -C "$REPO" diff --cached --name-only -- evidence' in body.group(1), (
        "key_is_clean no longer scans the staged evidence pathspec"
    )
    assert CODE.index("git add evidence") < CODE.index("if key_is_clean"), (
        "the key check must run AFTER staging, or it inspects an empty index"
    )


# --------------------------------------------------------------------------
# (b) push pushes the named ref, not HEAD
# --------------------------------------------------------------------------


def test_push_of_a_named_branch_ignores_HEAD_and_reports_success(sandbox):
    """The failure mode, against a local bare remote.

    On the wrong branch, `git push origin $BRANCH` exits 0 saying "Everything
    up-to-date" while the commit just made on HEAD goes nowhere.
    """
    repo, bare, env = sandbox
    git(repo, "checkout", "-q", "-b", "main", env=env)
    (repo / "evidence" / "on_main.txt").write_text("wrong branch\n", encoding="utf-8")
    git(repo, "add", "evidence", env=env)
    git(repo, "commit", "-q", "-m", "evidence: preserve (on main)", env=env)
    head_sha = out(git(repo, "rev-parse", "HEAD", env=env))
    branch_sha = out(git(repo, "rev-parse", BRANCH, env=env))
    assert head_sha != branch_sha

    proc = git(repo, "push", "origin", BRANCH, env=env, check=False)

    assert proc.returncode == 0, "the push was supposed to succeed trivially"
    assert "Everything up-to-date" in (proc.stdout + proc.stderr), (
        "expected the up-to-date message that made this look healthy; got "
        f"{proc.stdout!r} / {proc.stderr!r}"
    )
    assert bare_rev(bare, f"refs/heads/{BRANCH}", env) == branch_sha, (
        "the named ref should not have moved"
    )
    missing = subprocess.run(
        ["git", "--git-dir", str(bare), "cat-file", "-e", head_sha],
        env=env, capture_output=True, text=True,
    )
    assert missing.returncode != 0, "the wrong-branch commit reached the remote after all"

    # Positive direction: on the right branch the same command does publish.
    git(repo, "checkout", "-q", BRANCH, env=env)
    (repo / "evidence" / "on_branch.txt").write_text("right branch\n", encoding="utf-8")
    git(repo, "add", "evidence", env=env)
    git(repo, "commit", "-q", "-m", "evidence: preserve (on branch)", env=env)
    good_sha = out(git(repo, "rev-parse", "HEAD", env=env))
    git(repo, "push", "-q", "origin", BRANCH, env=env)
    assert bare_rev(bare, f"refs/heads/{BRANCH}", env) == good_sha


def _branch_guard_snippet() -> str:
    """The HEAD-vs-BRANCH guard, lifted verbatim out of the script."""
    start = None
    for i, line in enumerate(CODE_LINES):
        if line.lstrip().startswith('cur="$(git symbolic-ref'):
            start = i
            break
    assert start is not None, "the `cur=$(git symbolic-ref ...)` guard is gone"
    indent = len(CODE_LINES[start]) - len(CODE_LINES[start].lstrip())
    end = None
    for j in range(start + 1, len(CODE_LINES)):
        if CODE_LINES[j].rstrip() == " " * indent + "fi":
            end = j
            break
    assert end is not None, "could not find the guard's closing `fi`"
    block = "\n".join(
        line[indent:] if line.startswith(" " * indent) else line
        for line in CODE_LINES[start : end + 1]
    )
    assert "REFUSING" in block, "the guard no longer refuses; it only computes"
    # The daemon's loop control has no meaning outside the loop.
    return block.replace('sleep "$TICK"', ":").replace("continue", "exit 7")


def test_branch_guard_refuses_on_the_wrong_branch_and_passes_on_the_right_one(sandbox):
    """The guard itself, run against a real repo on each branch in turn.

    Asserting the exit code alone would be satisfied by any early-out, so both
    directions assert on the guard's own output: the REFUSING line when it
    refuses, and a marker past the guard when it does not.
    """
    repo, _bare, env = sandbox
    snippet = _branch_guard_snippet()
    program = (
        "set -u\n"
        f"BRANCH={script_branch_name()!r}\n"
        "TICK=0\n"
        'log() { echo "GUARD: $*"; }\n'
        f"cd {str(repo)!r} || exit 99\n"
        f"{snippet}\n"
        "echo GUARD_PASSED\n"
    )

    def run():
        return subprocess.run(["bash", "-c", program], env=env, capture_output=True, text=True)

    # Right branch -> falls through.
    assert out(git(repo, "symbolic-ref", "--short", "HEAD", env=env)) == script_branch_name()
    ok = run()
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "GUARD_PASSED" in ok.stdout, "the guard refused on its own branch"
    assert "REFUSING" not in ok.stdout

    # Wrong branch -> refuses, by name, and does not fall through.
    git(repo, "checkout", "-q", "-b", "main", env=env)
    bad = run()
    assert bad.returncode == 7, f"guard did not take the refuse path: {bad.stdout}{bad.stderr}"
    assert "REFUSING" in bad.stdout, f"the guard produced no refusal line: {bad.stdout!r}"
    assert "main" in bad.stdout, "the refusal must name the branch actually checked out"
    assert "GUARD_PASSED" not in bad.stdout

    # Detached HEAD -> also refuses: `|| echo DETACHED` must not equal $BRANCH.
    sha = out(git(repo, "rev-parse", "HEAD", env=env))
    git(repo, "checkout", "-q", "--detach", sha, env=env)
    det = run()
    assert det.returncode == 7, det.stdout + det.stderr
    assert "DETACHED" in det.stdout, f"detached HEAD was not reported: {det.stdout!r}"
    assert "GUARD_PASSED" not in det.stdout


def test_script_asserts_HEAD_equals_BRANCH_before_committing():
    """Static: the symbolic-ref check exists, and precedes staging and pushing."""
    assert "git symbolic-ref --quiet --short HEAD" in CODE, (
        "no symbolic-ref check -- the daemon can commit onto whatever branch is "
        "checked out while pushing an untouched named ref"
    )
    assert re.search(r'if \[ "\$cur" != "\$BRANCH" \]', CODE), (
        "HEAD is no longer compared against $BRANCH"
    )
    guard_at = CODE.index("git symbolic-ref --quiet --short HEAD")
    assert guard_at < CODE.index("git add evidence") < CODE.index("git commit")
    assert guard_at < CODE.index("git push")
    # Not silently redirected: `HEAD:$BRANCH` would publish whatever is checked out.
    assert "HEAD:$BRANCH" not in CODE
    assert re.search(r'git push[^\n]*origin "\$BRANCH"', CODE), (
        "the push no longer names $BRANCH explicitly"
    )


# --------------------------------------------------------------------------
# (c) the backlog measurement is local-only
# --------------------------------------------------------------------------


def test_rev_list_count_against_origin_is_local_only(sandbox):
    """`git rev-list --count origin/B..B` reads the tracking ref, not the remote.

    Proved by breaking the remote first: with origin pointing at a path that
    does not exist, push and ls-remote fail while rev-list still answers.
    """
    repo, _bare, env = sandbox
    for i in (1, 2):
        (repo / "evidence" / f"c{i}.txt").write_text(f"{i}\n", encoding="utf-8")
        git(repo, "add", "evidence", env=env)
        git(repo, "commit", "-q", "-m", f"evidence: preserve {i}", env=env)

    git(repo, "remote", "set-url", "origin", "/nonexistent/definitely-not-a-repo.git", env=env)

    # The remote really is unreachable -- otherwise the next assertion is empty.
    assert git(repo, "ls-remote", "origin", env=env, check=False).returncode != 0
    assert git(repo, "push", "origin", BRANCH, env=env, check=False).returncode != 0

    proc = git(repo, "rev-list", "--count", f"origin/{BRANCH}..{BRANCH}", env=env, check=False)
    assert proc.returncode == 0, f"rev-list needed the network: {proc.stderr}"
    assert out(proc) == "2", f"expected 2 unpushed commits, got {out(proc)!r}"

    # Nothing unpushed reads as 0, not as an error.
    git(repo, "update-ref", f"refs/remotes/origin/{BRANCH}", "HEAD", env=env)
    zero = git(repo, "rev-list", "--count", f"origin/{BRANCH}..{BRANCH}", env=env, check=False)
    assert zero.returncode == 0 and out(zero) == "0"


def test_rev_list_fails_without_a_tracking_ref_so_the_fallback_is_load_bearing(tmp_path: Path):
    """The other direction: no origin/B at all makes rev-list exit nonzero.

    That is why the script writes `|| echo 0` -- otherwise `$ahead` would be
    empty on a fresh post-rollback checkout that has not fetched yet.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = git_env(home)
    repo = tmp_path / "lonely"
    repo.mkdir()
    git(repo, "init", "-q", "-b", BRANCH, env=env)
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    git(repo, "add", "f.txt", env=env)
    git(repo, "commit", "-q", "-m", "only commit", env=env)

    proc = git(repo, "rev-list", "--count", f"origin/{BRANCH}..{BRANCH}", env=env, check=False)
    assert proc.returncode != 0, "expected failure with no remote-tracking ref"

    fallback = subprocess.run(
        [
            "bash",
            "-c",
            f"set -u; cd {str(repo)!r} && "
            f'a="$(git rev-list --count "origin/{BRANCH}..{BRANCH}" 2>/dev/null || echo 0)"; '
            'echo "ahead=$a"; [ "${a:-0}" -gt 0 ] 2>/dev/null && echo DRAINING || echo IDLE',
        ],
        env=env, capture_output=True, text=True,
    )
    assert "ahead=0" in fallback.stdout, fallback.stdout
    assert "IDLE" in fallback.stdout, fallback.stdout


def test_script_measures_the_backlog_against_the_pushed_ref():
    """Static: the drain compares origin/$BRANCH..$BRANCH, and precedes the gate."""
    m = re.search(r'ahead="\$\(git rev-list --count "([^"]+)"(.*?)\)"', CODE, re.S)
    assert m, "the backlog measurement is gone or changed shape"
    assert m.group(1) == "origin/$BRANCH..$BRANCH", (
        f"the backlog is measured against {m.group(1)!r}, not the ref that gets pushed"
    )
    assert "|| echo 0" in m.group(2), "the rev-list fallback is gone"
    assert "fetch" not in m.group(0), "the backlog measurement must not touch the network"

    gate_start, _gate_end = change_gate_bounds()
    ahead_line = CODE[: m.start()].count("\n")
    assert ahead_line < gate_start, (
        "the drain moved back inside the change gate -- a push that failed on the "
        "last cycle carrying new evidence would never be retried"
    )


# --------------------------------------------------------------------------
# (d) no $n outside the change gate, because the drain runs first under set -u
# --------------------------------------------------------------------------


def test_unbound_n_under_set_u_kills_the_shell():
    """The consequence, demonstrated: this is why (d) is not a style rule."""
    proc = subprocess.run(
        ["bash", "-c", 'set -u; echo "count is $n"; echo SURVIVED'],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"},
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "SURVIVED" not in proc.stdout
    assert "unbound variable" in proc.stderr


def test_no_reference_to_n_outside_the_change_gate():
    """Static: `$n` appears only inside the gate that assigns it."""
    assert re.search(r"^set -u\s*$", CODE, re.M), (
        "the script no longer runs under `set -u`; this whole class of bug changes shape"
    )
    gate_start, gate_end = change_gate_bounds()
    gate_lines = CODE_LINES[gate_start : gate_end + 1]

    # The detector must be able to see a real use, or its silence means nothing.
    assert any(N_REF.search(ln) for ln in gate_lines), (
        "found no $n inside the gate -- the detector is not detecting"
    )
    assert any("n=$(" in ln for ln in gate_lines), "n is no longer assigned inside the gate"

    offenders = [
        (i + 1, ln)
        for i, ln in enumerate(CODE_LINES)
        if not (gate_start <= i <= gate_end) and N_REF.search(ln)
    ]
    assert not offenders, (
        "`$n` is referenced outside the change gate, where it is unbound under "
        "`set -u` -- the daemon aborts on any cycle that skips the gate:\n"
        + "\n".join(f"  line {no}: {ln.strip()}" for no, ln in offenders)
    )


def test_the_drain_reports_its_own_count_not_n():
    """The drain's log lines must report `$ahead`, the value it actually measured."""
    gate_start, _ = change_gate_bounds()
    drain_msgs = [
        ln
        for ln in CODE_LINES[:gate_start]
        if re.search(r"\blog\s+\"", ln)
    ]
    assert any("$ahead" in ln for ln in drain_msgs), (
        "the drain reports no count at all; it used to borrow $n from the gate"
    )
    assert not any(N_REF.search(ln) for ln in drain_msgs)


# ==========================================================================
# Gap found by mutation-testing this file, closed.
# ==========================================================================

def test_the_key_refusal_unstages_only_evidence(sandbox):
    """**A refusal must not destroy an index it does not own.**

    When `key_is_clean` refuses, the script unstages what it staged so the next
    cycle's `git add` cannot sweep the offending files into a passing commit.
    That line is `git reset -q -- evidence`, and dropping the pathspec left every
    test here green — while a bare `git reset` wipes the WHOLE index. This daemon
    shares its repo with a session that stages files continuously and ticks every
    300s, so the blast radius of that mutation is another process's staged work,
    discarded by a guard that was only supposed to decline a commit.

    Same shape as the defect this file was written for: the guard's scope and the
    command's scope have to be the same set.
    """
    repo, _bare, env = sandbox
    (repo / "evidence" / "new.gz").write_text("x", encoding="utf-8")
    (repo / "unrelated.txt").write_text("someone else's work\n", encoding="utf-8")
    git(repo, "add", "evidence", "unrelated.txt", env=env)
    assert "unrelated.txt" in out(git(repo, "diff", "--cached", "--name-only", env=env))

    # Exactly what the refusal branch runs.
    git(repo, "reset", "-q", "--", "evidence", env=env)

    staged = out(git(repo, "diff", "--cached", "--name-only", env=env)).split()
    assert staged == ["unrelated.txt"], (
        f"the refusal must unstage evidence and nothing else; index now {staged}"
    )


def test_the_script_keeps_the_pathspec_on_its_unstage(sandbox):
    """The behaviour above only holds if the script still spells it that way.

    Pinned separately because the sandbox test exercises git's semantics, not
    the script's text, and the mutation that survived was in the text.
    """
    repo, _bare, env = sandbox
    m = re.search(r"^\s*git reset\b[^\n]*$", SCRIPT, re.M)
    assert m, "the key-refusal branch no longer unstages at all"
    assert "-- evidence" in m.group(0), (
        f"`git reset` lost its pathspec and would wipe the whole index: {m.group(0)!r}"
    )

    # And prove that is not a distinction without a difference.
    (repo / "evidence" / "new.gz").write_text("x", encoding="utf-8")
    (repo / "unrelated.txt").write_text("work\n", encoding="utf-8")
    git(repo, "add", "evidence", "unrelated.txt", env=env)
    git(repo, "reset", "-q", env=env)                      # the mutant's form
    assert out(git(repo, "diff", "--cached", "--name-only", env=env)) == "", (
        "a pathspec-less reset should empty the index — if it does not, this "
        "test is not demonstrating the risk it claims"
    )
