"""The two path detectors must keep working when the repo moves.

Both hard-coded ``/home/user/athanor`` -- this container's clone location. A
fresh container, or the same repo checked out by a different account, gets a
different one, and the two failures point in OPPOSITE directions:

* ``proofread_trace``'s "outside the workspace" rule stops matching the repo,
  so the proofread pass returns clean while the route it exists to catch is
  open. Under-reporting: it fails silently and looks like a pass.
* ``scoring._BENIGN_PATH_PREFIXES`` stops matching the solver's real
  interpreter, so the path the harness *instructs* the solver to run is flagged
  as contamination on every run. Over-reporting: it buries real findings.

Neither is visible from this container, where the literal happens to be right.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))


def _reach_rule():
    import proofread_trace as pt

    for value in vars(pt).values():
        if not isinstance(value, tuple):
            continue
        for item in value:
            if (isinstance(item, tuple) and len(item) == 2
                    and item[1] == "outside the workspace"):
                return item[0]
    raise AssertionError("proofread_trace no longer defines an 'outside the "
                         "workspace' rule; this test needs updating")


def test_reach_rule_matches_the_repo_it_actually_lives_in() -> None:
    """Not 'matches /home/user/athanor' -- that passes on a hardcoded rule too."""
    rule = _reach_rule()
    assert rule.search(f"cat {REPO}/tools/clean_rollouts.py")
    assert rule.search(f"sed -n 1,5p {REPO}/src/athanor/ccarc3/client.py")


def test_reach_rule_is_not_a_frozen_literal(tmp_path) -> None:
    """The load-bearing half -- and the first draft of it did not work.

    That draft asserted ``re.escape(str(REPO)) in rule.pattern``. On this
    container ``REPO`` IS ``/home/user/athanor`` and ``re.escape`` leaves it
    unchanged, so a rule frozen at that literal contains it and the assertion
    passed. The test agreed with the truth only in the environment it was
    written in, which is the very defect it was written to catch; the mutant
    survived the whole file.

    Copying the module to a new root and importing it THERE is the only form
    that separates them: a frozen rule still names ``/home/user/athanor``,
    while a derived one names the copy.
    """
    dest_tools = tmp_path / "relocated" / "tools"
    dest_tools.mkdir(parents=True)
    subprocess.run(
        ["cp", str(REPO / "tools" / "proofread_trace.py"), str(dest_tools)],
        check=True, capture_output=True,
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(dest_tools)!r}); "
        "import proofread_trace as pt\n"
        "for value in vars(pt).values():\n"
        "    if not isinstance(value, tuple):\n"
        "        continue\n"
        "    for item in value:\n"
        "        if (isinstance(item, tuple) and len(item) == 2\n"
        "                and item[1] == 'outside the workspace'):\n"
        "            print(item[0].pattern)\n"
    )
    out = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    pattern = out.stdout.strip()
    assert pattern, "the relocated module defines no 'outside the workspace' rule"

    relocated_repo = str(tmp_path / "relocated")
    assert re.escape(relocated_repo) in pattern, (
        f"a copy living at {relocated_repo} still guards {pattern!r} -- the "
        f"reach rule is a frozen literal and stops matching once the repo moves"
    )
    assert re.escape(str(REPO)) not in pattern, (
        f"the relocated copy still names the original repo in {pattern!r}"
    )


def test_reach_rule_still_catches_the_other_routes() -> None:
    rule = _reach_rule()
    assert rule.search("git log --oneline")
    assert rule.search("ls site-packages/")
    assert not rule.search("ls explore/ && cat notes.md")


def test_benign_prefixes_track_the_running_interpreter() -> None:
    from athanor.cc_harness.scoring import _BENIGN_PATH_PREFIXES

    assert any(sys.prefix.rstrip("/") + "/" == p for p in _BENIGN_PATH_PREFIXES), (
        f"the running interpreter's prefix {sys.prefix!r} is not benign -- every "
        f"solver command invoking it would be flagged as reaching outside its "
        f"workspace. Got {_BENIGN_PATH_PREFIXES!r}"
    )


def test_benign_prefixes_track_the_installed_package() -> None:
    import athanor
    from athanor.cc_harness.scoring import _BENIGN_PATH_PREFIXES

    package_root = str(pathlib.Path(athanor.__file__).resolve().parents[1]) + "/"
    assert package_root in _BENIGN_PATH_PREFIXES, (
        f"the directory athanor was imported from ({package_root!r}) is not "
        f"benign. Got {_BENIGN_PATH_PREFIXES!r}"
    )


def test_benign_prefixes_do_not_whitelist_the_world() -> None:
    """Deriving must not degrade into '/' or an empty prefix.

    A prefix of ``/`` makes every absolute path benign and the contamination
    scan reports clean on everything -- a check that passes by not running.
    """
    from athanor.cc_harness.scoring import _BENIGN_PATH_PREFIXES

    for prefix in _BENIGN_PATH_PREFIXES:
        assert prefix not in ("", "/"), f"prefix {prefix!r} whitelists everything"
        assert prefix.startswith("/") and len(prefix) > 2, f"suspect prefix {prefix!r}"


def test_the_interpreter_is_benign_under_a_relocated_package(tmp_path) -> None:
    """The real test of derivation: run from a COPY of the tree at a new path.

    Every case above runs where the literal is still correct, so a frozen list
    passes all of them. This one copies the package somewhere else and imports
    it there -- if the prefixes were literals, the relocated package root would
    be absent from them.
    """
    dest = tmp_path / "relocated"
    (dest / "athanor").mkdir(parents=True)
    subprocess.run(
        ["cp", "-r", str(REPO / "src" / "athanor"), str(dest)],
        check=True, capture_output=True,
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(dest)!r}); "
        "from athanor.cc_harness.scoring import _BENIGN_PATH_PREFIXES as B; "
        "print('\\n'.join(B))"
    )
    out = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    prefixes = out.stdout.split()
    assert str(dest) + "/" in prefixes, (
        f"a package imported from {dest} did not mark that directory benign: "
        f"{prefixes!r} -- the prefixes are frozen literals"
    )
