"""A repo that only works from its own root is a repo that works on one box.

Found on 2026-08-12 while checking whether CCARC3 could be stood up on hardware
we own rather than a container replaced every ~6h. Eight test modules reached
`tools/` with ``sys.path.insert(0, "tools")`` -- a path resolved against the
*process cwd*, not the repository. Invoked from ``/`` the suite did not fail a
test; it failed to COLLECT, on
``ModuleNotFoundError: No module named 'proofread_trace'``.

This is the project's recurring defect class one more time: a value that agrees
with the truth in the only environment anyone exercises it in. Every by-hand run
of pytest happens from the repo root, so the relative path was indistinguishable
from a correct one until the day something ran from somewhere else. "The tests
are green" was, strictly, a claim about the operator's working directory.

Two tests here, and the split is deliberate:

* the static one names the defect, and would have caught it at review time;
* the dynamic one is the only one that could have caught it at all, because a
  static scan can be evaded by any spelling nobody thought to look for.

The dynamic one is the point. It runs the collector from a directory that is not
the repo, which is precisely the environment where the original bug lives.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _relative_path_inserts(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every ``sys.path.insert``/``append`` given a relative string constant.

    Parsed, not grepped. Several modules embed solver-facing source as string
    literals containing ``sys.path.insert(0, '..')`` -- a grep flags those and a
    parse does not, because in the AST they are strings, not calls.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"insert", "append"}:
            continue
        # sys.path.insert(...) -- the receiver must be `sys.path`
        recv = node.func.value
        if not (isinstance(recv, ast.Attribute) and recv.attr == "path"
                and isinstance(recv.value, ast.Name) and recv.value.id == "sys"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if not arg.value.startswith("/"):
                    out.append((node.lineno, arg.value))
    return out


def _python_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for sub in ("tests", "tools"):
        files += sorted(p for p in (REPO / sub).rglob("*.py")
                        if "__pycache__" not in p.parts)
    files += sorted(p for p in (REPO / "src").rglob("*.py")
                    if "__pycache__" not in p.parts)
    return files


def test_no_module_reaches_a_sibling_directory_through_the_cwd():
    offenders: list[str] = []
    for f in _python_files():
        for lineno, value in _relative_path_inserts(f):
            offenders.append(f"{f.relative_to(REPO)}:{lineno}  sys.path <- {value!r}")
    assert not offenders, (
        "these resolve against the process cwd, so the module they import is "
        "found only when the interpreter happens to be started from the repo "
        "root:\n  " + "\n  ".join(offenders)
        + "\nUse a path derived from __file__ instead."
    )


def test_the_scan_can_actually_see_the_defect(tmp_path):
    """The static check above must fail on the thing it claims to catch.

    Without this, a scan that silently matched nothing -- a typo'd attribute
    name, a walk that never recursed -- would report a clean repo forever. That
    is the failure mode this whole file is about: passing by not running.
    """
    bad = tmp_path / "offender.py"
    bad.write_text('import sys\nsys.path.insert(0, "tools")\n', encoding="utf-8")
    assert _relative_path_inserts(bad) == [(2, "tools")]

    # ...and must NOT fire on the same text quoted inside a string, which is how
    # several solver-facing fixtures legitimately spell it.
    quoted = tmp_path / "quoted.py"
    quoted.write_text('SNIPPET = "sys.path.insert(0, \'..\')"\n', encoding="utf-8")
    assert _relative_path_inserts(quoted) == []

    # An absolute insert is fine, however it is spelled.
    good = tmp_path / "good.py"
    good.write_text('import sys\nsys.path.insert(0, "/opt/x")\n', encoding="utf-8")
    assert _relative_path_inserts(good) == []


# ── the second spelling ──────────────────────────────────────────────────────
# The `sys.path` scan above was written first and was not enough. Running the
# full suite from `/` turned up two more failures in a different shape:
#
#     _shell_slice(pathlib.Path("tools/preserve_evidence.sh"), ...)
#
# Not a path *entry* — a path *literal*, opened directly. Same root cause, and
# invisible to a scan that only knows about `sys.path`. Worth stating plainly:
# the static check found the class it was written for and missed the sibling
# case sitting three files away, while the dynamic drill found both. A scan is a
# cheap early warning, never the evidence.
#
# Worst of the three was `test_a_real_multi_play_attempt_still_corroborates`,
# which guarded its relative path with `if not exists(): return`. From any cwd
# but the repo root it asserted nothing and reported green — the control test
# for the whole corroboration check, quietly not running.
_REPO_SUBDIRS = ("tools/", "src/", "tests/", "docs/", "evidence/")
_PATH_CTORS = {"open", "Path"}          # open(...), Path(...), pathlib.Path(...)


def _is_path_ctor(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in _PATH_CTORS
    if isinstance(func, ast.Attribute):
        return func.attr in _PATH_CTORS
    return False


def _relative_repo_paths(path: pathlib.Path) -> list[tuple[int, str]]:
    """Repo-relative literals handed straight to ``open()``/``Path()``.

    Deliberately narrow. A bare literal like ``'tests/test_x.py::test_broken'``
    is a *label* in several fixtures and resolving it would be meaningless, so
    only literals in constructor position count — which is exactly the form that
    reads a file. Broadening this to every string that looks like a path flags
    ~90 sites, nearly all of them fine, and a check nobody can keep green is a
    check that gets deleted.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_path_ctor(node.func) and node.args:
            first = node.args[0]
            if (isinstance(first, ast.Constant) and isinstance(first.value, str)
                    and first.value.startswith(_REPO_SUBDIRS)):
                out.append((node.lineno, first.value))
    return out


def test_no_module_opens_a_repo_file_through_the_cwd():
    offenders: list[str] = []
    for f in _python_files():
        for lineno, value in _relative_repo_paths(f):
            offenders.append(f"{f.relative_to(REPO)}:{lineno}  opens {value!r}")
    assert not offenders, (
        "these resolve against the process cwd, so the file is found only when "
        "the interpreter happens to be started from the repo root:\n  "
        + "\n  ".join(offenders)
        + "\nDerive the path from __file__ instead."
    )


def test_the_path_scan_can_actually_see_the_defect(tmp_path):
    bad = tmp_path / "offender.py"
    bad.write_text('import pathlib\np = pathlib.Path("tools/x.sh")\n', encoding="utf-8")
    assert _relative_repo_paths(bad) == [(2, "tools/x.sh")]

    # A label, not a path. Must not fire -- several fixtures carry these.
    label = tmp_path / "label.py"
    label.write_text('NODEID = "tests/test_x.py::test_broken"\n', encoding="utf-8")
    assert _relative_repo_paths(label) == []

    # Repo-derived, however spelled. Must not fire.
    good = tmp_path / "good.py"
    good.write_text('import pathlib\np = pathlib.Path(__file__).parent / "tools/x.sh"\n',
                    encoding="utf-8")
    assert _relative_repo_paths(good) == []


@pytest.mark.slow
def test_the_suite_collects_from_a_foreign_directory(tmp_path):
    """Collection from somewhere that is not the repo. The real check.

    ``--collect-only`` rather than a full run: collection is where the original
    bug landed (imports happen at module scope), it exercises every test module,
    and it costs seconds instead of minutes. A failure here reports the module
    that could not be imported.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", str(REPO / "tests")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"the suite cannot even be COLLECTED from {tmp_path}, so 'the tests pass' "
        f"is a statement about the operator's working directory:\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
