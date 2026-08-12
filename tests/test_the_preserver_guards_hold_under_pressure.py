"""`preserve_evidence.sh` commits and pushes to a public remote every 300s.

Nine of seventeen mutants against it survived. The file is unusually well
commented — nearly every guard records the incident that produced it — and the
comments were doing the work the tests were not. Each hole below is a guard
whose *reason* is written down at length and whose *behaviour* nothing checked.

The two that would publish a secret:

* **`key_is_clean` returning clean when `ARC_API_KEY` is unset.** The guard
  cannot run without the key, so the only safe answer is to refuse. Returning 0
  turns "I could not check" into "I checked and it is fine" — and this same
  guard has already been caught once inspecting an empty index.
* **`key_is_clean` resolving staged paths against the cwd instead of `$REPO`.**
  The file list comes from `git -C "$REPO"` and is repo-relative. Where the cwd
  differs, `[ -f "$f" ]` is false for every entry, the loop skips them all,
  `hits` stays empty and the function returns CLEAN. Verbatim the failure its own
  comment is about, one cause along.

And the ones that lose the evidence rather than leak it: never retrying a failed
push, retrying only once, going silent when the backlog is stuck, committing the
whole index instead of the `evidence` pathspec, and keeping the proxy value
inherited at launch on a box where the port moves.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRESERVER = REPO / "tools" / "preserve_evidence.sh"


def _fn(name: str) -> str:
    """Extract a function, whether it spans lines or sits on one.

    `refresh_proxy` is a one-liner, so searching for a lone closing brace on its
    own line finds the *next* function's — or nothing. An extractor that quietly
    grabs the wrong text is the same hazard as one that grabs none.
    """
    body = PRESERVER.read_text()
    start = body.index(f"{name}() {{")
    first_line_end = body.index("\n", start)
    first_line = body[start:first_line_end]
    if first_line.rstrip().endswith("}"):
        return first_line
    out = body[start:body.index("\n}\n", start) + 3]
    assert out.count("{") >= 1, f"extraction of {name} looks empty"
    return out


def _bash(script: str, cwd: Path, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], cwd=cwd, capture_output=True,
                          text=True, timeout=90, env={**os.environ, **env})


@pytest.fixture()
def repo(tmp_path):
    """A tiny git repo with one staged evidence file."""
    root = tmp_path / "repo"
    (root / "evidence").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    return root


def _key_check(repo: Path, contents: str, *, key: str | None, cwd: Path) -> str:
    (repo / "evidence" / "trace.txt").write_text(contents)
    subprocess.run(["git", "add", "evidence"], cwd=repo, check=True)
    env = {"REPO": str(repo)}
    if key is not None:
        env["ARC_API_KEY"] = key
    else:
        env["ARC_API_KEY"] = ""
    script = "\n".join([
        'log() { echo "$*"; }',
        f'REPO="{repo}"',
        _fn("key_is_clean"),
        "key_is_clean && echo VERDICT=CLEAN || echo VERDICT=REFUSED",
    ])
    return _bash(script, cwd, **env).stdout


# --------------------------------------------------------------------------- #
# the API-key guard
# --------------------------------------------------------------------------- #


def test_a_staged_key_is_refused(repo, tmp_path):
    out = _key_check(repo, "token=SECRETKEY123", key="SECRETKEY123", cwd=tmp_path)
    assert "VERDICT=REFUSED" in out, out


def test_a_clean_file_commits(repo, tmp_path):
    out = _key_check(repo, "nothing to see", key="SECRETKEY123", cwd=tmp_path)
    assert "VERDICT=CLEAN" in out, out


def test_no_key_means_refuse_not_pass(repo, tmp_path):
    """"I could not check" must not read as "I checked and it is fine"."""
    out = _key_check(repo, "token=SECRETKEY123", key=None, cwd=tmp_path)
    assert "VERDICT=REFUSED" in out, out
    assert "cannot run" in out


def test_the_guard_reads_files_from_the_repo_not_the_cwd(repo, tmp_path):
    """Run it from a directory where the repo-relative paths do not resolve.

    This is the failure the guard's own comment is about, one cause along: every
    `[ -f ]` false, every entry skipped, `hits` empty, verdict CLEAN.
    """
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    out = _key_check(repo, "token=SECRETKEY123", key="SECRETKEY123", cwd=elsewhere)
    assert "VERDICT=REFUSED" in out, (
        f"the key was missed because the cwd was not the repo: {out!r}"
    )


def test_a_gzipped_file_is_read_too(repo, tmp_path):
    import gzip
    (repo / "evidence" / "trace.gz").write_bytes(gzip.compress(b"token=SECRETKEY123"))
    subprocess.run(["git", "add", "evidence"], cwd=repo, check=True)
    script = "\n".join(['log() { echo "$*"; }', f'REPO="{repo}"', _fn("key_is_clean"),
                        "key_is_clean && echo VERDICT=CLEAN || echo VERDICT=REFUSED"])
    out = _bash(script, tmp_path, REPO=str(repo), ARC_API_KEY="SECRETKEY123").stdout
    assert "VERDICT=REFUSED" in out, out


# --------------------------------------------------------------------------- #
# the push backlog — the container-rollback recovery
# --------------------------------------------------------------------------- #


def _backlog_block() -> str:
    body = PRESERVER.read_text()
    start = body.index('    ahead="$(git rev-list --count')
    # The block ends at the LAST of the nested `fi`s, not the first.
    end = body.index("\n        fi\n    fi\n", start) + len("\n        fi\n    fi\n")
    block = body[start:end]
    assert "BACKLOG STUCK" in block and "drained" in block, "extraction truncated"
    return block


def test_a_failed_push_is_retried_more_than_once(tmp_path):
    """One attempt is not a retry. The comment calls this the recovery for a
    push that failed on the cycle carrying a sweep's final result.json."""
    # **Count into a file, not onto stderr.** The real line is
    # `git push -q origin "$BRANCH" 2>/dev/null`, so a stub that announces
    # itself on stderr is swallowed by the very redirection under test — the
    # first draft read zero attempts from a block that had made four.
    tally = tmp_path / "attempts"
    script = "\n".join([
        'log() { echo "$*"; }',
        f'git() {{ if [ "$1" = "rev-list" ]; then echo 3; else echo x >> "{tally}"; return 1; fi; }}',
        'sleep() { :; }',
        'BRANCH=b; HTTPS_PROXY=x',
        _backlog_block(),
    ])
    _bash(script, tmp_path)
    attempts = len(tally.read_text().split()) if tally.exists() else 0
    assert attempts >= 4, f"the push was attempted {attempts} time(s), expected 4"


def test_a_stuck_backlog_is_reported(tmp_path):
    script = "\n".join([
        'log() { echo "$*"; }',
        'git() { if [ "$1" = "rev-list" ]; then echo 3; else return 1; fi; }',
        'sleep() { :; }',
        'BRANCH=b; HTTPS_PROXY=x',
        _backlog_block(),
    ])
    assert "BACKLOG STUCK" in _bash(script, tmp_path).stdout


def test_a_drained_backlog_is_reported_and_not_retried_forever(tmp_path):
    script = "\n".join([
        'log() { echo "$*"; }',
        'git() { if [ "$1" = "rev-list" ]; then echo 2; else return 0; fi; }',
        'sleep() { :; }',
        'BRANCH=b; HTTPS_PROXY=x',
        _backlog_block(),
    ])
    out = _bash(script, tmp_path).stdout
    assert "drained a backlog of 2" in out
    assert "BACKLOG STUCK" not in out


def test_no_backlog_means_no_push_and_no_noise(tmp_path):
    """The gate must stay quiet on an ordinary cycle, or the line that matters
    is buried among the ones that do not."""
    script = "\n".join([
        'log() { echo "$*"; }',
        'git() { if [ "$1" = "rev-list" ]; then echo 0; else echo "PUSH" >&2; fi; }',
        'sleep() { :; }',
        'BRANCH=b; HTTPS_PROXY=x',
        _backlog_block(),
    ])
    out = _bash(script, tmp_path)
    assert out.stdout.strip() == "", out.stdout
    assert "PUSH" not in out.stderr


# --------------------------------------------------------------------------- #
# scope, flags, and the self-healing proxy
# --------------------------------------------------------------------------- #


def test_the_commit_is_scoped_to_evidence():
    """`git commit` without the pathspec would carry whatever another process
    had staged, under an "evidence: preserve" message, past a guard that only
    scans `-- evidence`."""
    body = PRESERVER.read_text()
    assert 'git commit -q -m "evidence: preserve' in body
    assert "git commit -q -a" not in body, "the commit lost its pathspec"
    assert body.count('-- evidence') >= 2, (
        "the guard's scope and the commit's scope must be the same set"
    )


def test_the_proxy_is_re_read_every_cycle():
    """A daemon keeps the port it inherited at launch; this one ran for two
    hours pushing into a dead proxy before the re-read was added."""
    fn = _fn("refresh_proxy")
    assert "proxy_env" in fn and "." in fn, "refresh_proxy no longer sources the file"


def _once_parse(arg: str | None) -> str:
    body = PRESERVER.read_text()
    start = body.index("ONCE=0")
    end = body.index("\n", body.index("--once", start)) + 1
    snippet = body[start:end]
    assert "ONCE=1" in snippet
    argv = [] if arg is None else [arg]
    out = subprocess.run(["bash", "-c", f"set -u\n{snippet}\necho ONCE=$ONCE", "sh", *argv],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.parametrize("arg", ["--help", "-o", "once", "--onc", ""])
def test_a_stray_argument_is_not_treated_as_once(arg):
    assert _once_parse(arg) == "ONCE=0", f"{arg!r} was accepted as --once"


def test_the_real_once_flag_is_accepted():
    assert _once_parse("--once") == "ONCE=1"


def test_once_is_referenced_defensively_under_set_u():
    """Bare `$ONCE` kills the block under `set -u` when a test extracts it as a
    standalone slice. Learned twice in one night on two different scripts."""
    body = PRESERVER.read_text()
    assert 'log "preserving evidence' in body
    line = next(l for l in body.splitlines() if 'log "preserving evidence' in l)
    assert "${ONCE:-0}" in line, f"bare ONCE reference: {line!r}"
