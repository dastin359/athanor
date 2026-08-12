"""No tracked file may carry a session cookie value.

`evidence/` is pushed to GitHub on a five-minute cycle. `trace.state.json`
persists the client's cookie jar so a run can rebind its scorecard after a
restart, and a `GAMESESSION` value is a live credential for the ARC backend
exactly as the API key is.

**79 tracked files carried one**, every one committed and pushed by
`preserve_evidence.sh` under a message reading "preserve ccarc3 run artifacts".
The daemon had a pre-commit secret check the whole time — it looked for the API
key and nothing else, and a guard that names one secret is not a guard against
secrets.

Two checks here, and they are deliberately different in kind:

* a **repo-state** check, which fails if any tracked file carries a value. It
  guards the artifact rather than the code that writes it, which is the only
  thing that catches a leak arriving by some path nobody has thought of.
* a **behavioural** check on `cookies_are_clean`, so the daemon's refusal is
  held to actually refusing.

The scan is **structural, not value-matching**. Matching secrets by value has
failed twice in this project — the space is dense enough that a scan either
collides constantly or gets narrowed until it catches nothing. What identifies a
cookie is the shape it is written in.
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRESERVER = REPO / "tools" / "preserve_evidence.sh"

_COOKIE_NAME = re.compile(rb"(GAMESESSION|AWSALBAPP[-_A-Za-z0-9]*)")
# **The charset has to cover base64, not just hex** -- for the tokens whose
# first characters are not all alphanumeric. The `AWSALBAPP-*` stickiness token
# is ~144 characters of base64 containing `/` and `+`; the narrow class still
# matched the one that leaked on 2026-08-11, because that one opens with a run
# of `A` and reaches the length floor long before its first `/`. A token that
# opens `x+9/...` does not, and the class is widened for that case rather than
# for the observed one. Written this way because the first version of this
# comment asserted the stronger claim and a surviving mutant disproved it.
_LONG_VALUE = re.compile(rb"""["']?(?:value|=)["']?\s*[:=]\s*["']([A-Za-z0-9%_\-.+/=]{20,})["']""")


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    return [REPO / rel for rel in out.stdout.split()]


def _body(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        try:
            return gzip.decompress(raw)
        except OSError:
            return raw
    return raw


def test_no_tracked_file_carries_a_session_cookie_value():
    """A cookie NAME is fine — the source discusses these by name. A name
    followed by a long value is a credential."""
    offenders = []
    for path in _tracked():
        if not path.is_file():
            continue
        try:
            body = _body(path)
        except OSError:
            continue
        for match in _COOKIE_NAME.finditer(body):
            if _LONG_VALUE.search(body[match.start():match.start() + 220]):
                offenders.append(str(path.relative_to(REPO)))
                break
    assert not offenders, (
        f"{len(offenders)} tracked file(s) carry a session cookie value: "
        f"{offenders[:5]}"
    )


def test_talking_about_a_cookie_by_name_is_still_allowed():
    """The positive control. `client.py` explains the stickiness cookies in
    prose; a check that banned the name would force that explanation out, and an
    unexplained mechanism is how the leak happened in the first place."""
    named = [p for p in _tracked()
             if p.is_file() and p.suffix == ".py" and _COOKIE_NAME.search(_body(p))]
    assert named, "expected the source to discuss the cookies by name"


# --------------------------------------------------------------------------- #
# the daemon's own refusal
# --------------------------------------------------------------------------- #


def _shell_slice(marker: str, *, expect: tuple[str, ...] = ()) -> str:
    """A function definition lifted out of the daemon, checked for truncation."""
    text = PRESERVER.read_text()
    assert text.count(marker) == 1, f"{marker!r} is not unique in the daemon"
    body = text[text.index(marker):]
    body = body[:body.index("\n}\n") + 3]
    for token in expect:
        assert token in body, f"extraction of {marker!r} looks truncated: no {token!r}"
    return body


def _cookie_patterns() -> str:
    """The `_COOKIE_*` assignments the guard and the stripper both read.

    **Extracted, and asserted non-empty.** These moved to module scope when the
    guard grew past a single `case`, and a slice that took only the function body
    left them unbound -- which is not a broken test, it is a test that quietly
    inverts the thing under test. `grep -E -e ""` matches every line, so an
    unbound pattern makes the guard refuse EVERYTHING: every "must refuse" case
    passes, the suite goes mostly green, and the daemon it is standing in for
    would preserve nothing at all, every cycle, forever.
    """
    text = PRESERVER.read_text()
    lines = [ln for ln in text.splitlines() if re.match(r"^_COOKIE_[A-Z]+=", ln)]
    assert len(lines) >= 3, f"expected the three cookie patterns, found {len(lines)}"
    for ln in lines:
        name, _, value = ln.partition("=")
        assert len(value.strip("'\"")) > 10, f"{name} looks empty: {ln!r}"
    return "\n".join(lines)


def _cookies_are_clean_body() -> str:
    return _cookie_patterns() + "\n" + _shell_slice(
        "cookies_are_clean() {", expect=("REFUSING", "git -C"))


def _run_guard(tmp_path: Path, payload: str, *, gz: bool = False,
               name: str = "trace.state.json") -> subprocess.CompletedProcess:
    """Stage one evidence file with the given contents and ask the guard.

    ``gz`` matters more than it looks: every one of the 79 leaked files was
    gzipped, so a guard that reads only plaintext would have caught none of
    them — and a fixture that stages only plaintext cannot tell the difference.
    """
    repo = tmp_path / "repo"
    (repo / "evidence").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    if gz:
        target = repo / "evidence" / (name + ".gz")
        target.write_bytes(gzip.compress(payload.encode()))
    else:
        target = repo / "evidence" / name
        target.write_text(payload)
    subprocess.run(["git", "add", "evidence"], cwd=repo, check=True)

    script = "\n".join([
        "log() { echo \"$*\"; }",
        f'REPO="{repo}"',
        _cookies_are_clean_body(),
        "cookies_are_clean && echo VERDICT=CLEAN || echo VERDICT=REFUSED",
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)


def test_the_daemon_refuses_to_commit_a_cookie_value(tmp_path):
    payload = '{"cookies": [{"name": "GAMESESSION", "value": "' + "a" * 64 + '"}]}'
    out = _run_guard(tmp_path, payload)
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr
    assert "REFUSING TO COMMIT" in out.stdout


def test_the_daemon_looks_inside_a_gzipped_trace(tmp_path):
    """The shape the real leak had. `trace.state.json` is stored gzipped, so a
    guard reading only plaintext would have passed all 79 files."""
    payload = '{"cookies": [{"name": "GAMESESSION", "value": "' + "b" * 64 + '"}]}'
    out = _run_guard(tmp_path, payload, gz=True)
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr


def test_a_redacted_gzipped_trace_still_commits(tmp_path):
    payload = '{"cookies": [{"name": "GAMESESSION", "value": ""}]}'
    out = _run_guard(tmp_path, payload, gz=True)
    assert "VERDICT=CLEAN" in out.stdout, out.stdout + out.stderr


def test_a_redacted_jar_still_commits(tmp_path):
    """The fix must keep the file, not ban the field — the jar's shape is part
    of what makes a trace readable."""
    payload = '{"cookies": [{"name": "GAMESESSION", "value": "", "redacted": "removed"}]}'
    out = _run_guard(tmp_path, payload)
    assert "VERDICT=CLEAN" in out.stdout, out.stdout + out.stderr


def test_an_ordinary_trace_still_commits(tmp_path):
    out = _run_guard(tmp_path, '{"level": 3, "actions_used": 120}')
    assert "VERDICT=CLEAN" in out.stdout, out.stdout + out.stderr


def test_the_guard_is_wired_into_the_commit_path():
    """A guard nobody calls is the shape this leak already had once."""
    text = PRESERVER.read_text()
    assert "if key_is_clean && cookies_are_clean; then" in text, (
        "cookies_are_clean is defined but not consulted before committing"
    )


# --------------------------------------------------------------------------- #
# the shapes that actually leaked, and the ones the first two guards could not see
# --------------------------------------------------------------------------- #

_REAL = "3f3" + "0123456789abcdef" * 3 + "9ab"          # 64 chars, as GAMESESSION is
_STICKY = "AAAAAAAAAACRtm2O8M3z6q8a8ONY/dP6bTXpOb2bhzS9vq6n3X+8pY3Xlctmb5TAfXzg="
# Same length and alphabet, but the `+` and `/` fall inside the first sixteen
# characters -- so a charset that omits them cannot reach the floor and the scan
# misses it entirely. `_STICKY` cannot show that: it opens with eleven `A`s.
_STICKY_AWKWARD = "a+9/Kk2k/8Lm+Qw3z6q8a8ONYdP6bTXpOb2bhzS9vq6n3X8pY3Xlctmb5TAfXzg="


def test_the_daemon_refuses_a_jar_written_as_a_python_repr(tmp_path):
    """The shape of the leak found on 2026-08-11, in the file that was pushed.

    A solver tool result printed the client's state dict, so the jar arrived as
    `'cookies': [{'name': 'GAMESESSION', 'value': '...'}]` -- single quotes. The
    guard tested `*'"cookies"'*'"value"'*`, matched nothing, `continue`d, and
    reported the file clean. It did not fail; it declined to run.
    """
    payload = "{'cookies': [{'name': 'GAMESESSION', 'value': '%s'}]}" % _REAL
    out = _run_guard(tmp_path, payload, gz=True, name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr


def test_the_daemon_refuses_a_jar_nested_inside_a_json_string(tmp_path):
    """A stream line is JSONL whose `content` is a STRING, so a jar quoted inside
    it arrives backslash-escaped and `["']value["']` cannot match `\"value\"`."""
    payload = ('{"type":"user","message":{"content":[{"content":'
               '"state = {\\"cookies\\": [{\\"name\\": \\"GAMESESSION\\", '
               '\\"value\\": \\"%s\\"}]}"}]}}' % _REAL)
    out = _run_guard(tmp_path, payload, gz=True, name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr


def test_the_daemon_refuses_a_raw_set_cookie_header(tmp_path):
    """`GAMESESSION=<value>` has no `value` key to anchor on. A header echoed
    into a log is a credential exactly as a jar is."""
    out = _run_guard(tmp_path, 'Set-Cookie: GAMESESSION=%s; Path=/' % _REAL,
                     name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr


def test_the_daemon_refuses_a_stickiness_token_on_its_own(tmp_path):
    """base64, with `/` and `+`. Neither the daemon's old `[^"]` nor the
    repo-state scan's `[A-Za-z0-9%_.-]` could match one, so a leak of the cookie
    that actually binds the card -- and without which it cannot be CLOSED --
    would have gone unseen by both."""
    payload = "{'cookies': [{'name': 'AWSALBAPP-0', 'value': '%s'}]}" % _STICKY
    out = _run_guard(tmp_path, payload, gz=True, name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr


def test_the_backends_own_tombstone_is_not_a_secret(tmp_path):
    """`_remove_` is what the ARC backend sends to delete a cookie. Refusing over
    it costs the whole cycle's evidence for a value that is public by design."""
    payload = "{'cookies': [{'name': 'AWSALBAPP-1', 'value': '_remove_'}]}"
    out = _run_guard(tmp_path, payload, gz=True, name="stream.jsonl")
    assert "VERDICT=CLEAN" in out.stdout, out.stdout + out.stderr


def test_the_stripper_can_clear_the_refusal_it_causes(tmp_path):
    """**The deadlock property, and it is not hypothetical.**

    The guard refuses; `redact_cookies` is what makes the file committable again.
    If the two disagree the daemon does not leak -- it stops preserving
    ANYTHING, every cycle, forever, which for a sweep whose disk is not durable
    is the more expensive failure of the two.

    The first redaction marker was `SESSION-COOKIE-REDACTED`: 23 characters of
    exactly the charset the guard calls credential-shaped, so a redacted file
    went on being refused and the stripper could never clear it. The marker now
    opens with `[`, which is outside the charset by construction rather than by
    being on an exception list.
    """
    dirty = ("{'game_id': 'lf52', 'cookies': [{'name': 'GAMESESSION', 'value': '%s', "
             "'domain': '127.0.0.1'}, {'name': 'AWSALBAPP-0', 'value': '%s'}, "
             "{'name': 'AWSALBAPP-1', 'value': '_remove_'}]}" % (_REAL, _STICKY))

    src = tmp_path / "stream.jsonl"
    src.write_text(dirty)
    out_file = tmp_path / "clean.jsonl"
    script = "\n".join([
        _cookie_patterns(),
        _shell_slice("redact_cookies() {", expect=("sed", "value")),
        f'redact_cookies "{src}" "{out_file}"',
    ])
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr

    cleaned = out_file.read_text()
    assert _REAL not in cleaned and _STICKY not in cleaned, "a value survived redaction"
    assert "GAMESESSION" in cleaned and "cookies" in cleaned, (
        "the names and the structure must stay -- a vanished field is "
        "indistinguishable from a run that never had one")
    assert "_remove_" in cleaned, "the tombstone is not a secret and must not be touched"

    verdict = _run_guard(tmp_path / "guard", cleaned, gz=True, name="stream.jsonl")
    assert "VERDICT=CLEAN" in verdict.stdout, (
        "the stripper's own output is still refused — the daemon would preserve "
        "nothing, forever:\n" + verdict.stdout + verdict.stderr)


def test_the_stripper_is_wired_into_the_copy_path():
    """A stripper nobody calls is the shape this leak already had once."""
    text = PRESERVER.read_text()
    assert "redact_cookies " in text.split("redact_cookies() {")[1], (
        "redact_cookies is defined but never invoked")
    body = _shell_slice("gz_atomic() {", expect=("gzip",))
    assert "redact_cookies" in body, "gz_atomic does not run the text-level pass"
    assert "_COOKIE_KEY" in body, (
        "gz_atomic must decide with the same patterns the guard refuses with")


def test_a_token_whose_base64_starts_awkwardly_is_still_caught(tmp_path):
    """The case that makes the widened charset load-bearing.

    A surviving mutant is what produced this test: narrowing the value class back
    to `[A-Za-z0-9%_=.-]` changed nothing, because every sample here opened with
    a long alphanumeric run. A token with `+` and `/` in its first characters is
    the one the class actually protects, and nothing was asserting it.
    """
    payload = "{'cookies': [{'name': 'AWSALBAPP-0', 'value': '%s'}]}" % _STICKY_AWKWARD
    out = _run_guard(tmp_path, payload, gz=True, name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, out.stdout + out.stderr


def test_a_jar_with_an_unfamiliar_cookie_name_is_still_caught(tmp_path):
    """What the `cookies`-key pattern is actually for, in both quotings.

    Every other case here is caught by the NAME-anchored pattern, so two mutants
    survived: making the key pattern double-quote-only, and removing its
    tolerance for backslash-escaped quotes, changed nothing that was asserted.
    Both are real regressions — the key pattern is the only one that fires when
    the backend introduces a cookie this code has never heard of, which is the
    one case where knowing the name in advance is impossible.
    """
    unknown = "ARCSESSION2"
    assert unknown not in PRESERVER.read_text(), (
        "pick a name the NAME-anchored pattern does not know, or this asserts nothing")

    repr_jar = "{'cookies': [{'name': '%s', 'value': '%s'}]}" % (unknown, _REAL)
    out = _run_guard(tmp_path / "repr", repr_jar, gz=True, name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, "single-quoted jar:\n" + out.stdout + out.stderr

    # Built with `json.dumps` rather than hand-escaped. The hand-written version
    # emitted TWO backslashes per quote, which real JSON never produces -- and the
    # key-anchored pattern allows exactly one, so the fixture was asserting
    # against a shape that does not occur while appearing to cover the real one.
    escaped = json.dumps(
        {"content": '{"cookies": [{"name": "%s", "value": "%s"}]}' % (unknown, _REAL)})
    out = _run_guard(tmp_path / "esc", escaped, gz=True, name="stream.jsonl")
    assert "VERDICT=REFUSED" in out.stdout, "escaped jar:\n" + out.stdout + out.stderr
