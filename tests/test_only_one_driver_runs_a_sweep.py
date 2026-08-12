"""A second driver must be a no-op, not a second copy of the whole sweep.

The supervisor starts a driver whenever it sees none running, so any gap between
stopping one and starting the next lets its ten-minute cycle fire into that gap.
Measured 2026-08-11 with a ~30-second gap: two drivers, ten solvers, all five
in-flight games being played twice simultaneously.

The card tolerates it -- a duplicate play is just a play and the environment
keeps its best -- but it doubles the burn, and cleaning up is genuinely awkward:
solvers are `setsid` into their own process groups, so killing a driver's group
does not reach its solvers, and each one has to be matched back to its parent by
walking /proc.

`flock`, not a pidfile: a pidfile written by a driver that is SIGKILLed outlives
it and locks the sweep out permanently, whereas the kernel drops an flock however
the holder dies.
"""
from __future__ import annotations

import multiprocessing
import sys

import pytest

sys.path.insert(0, "tools")


def _grab_driver_lock(path, out):
    """Top-level so macOS's spawn multiprocessing context can pickle it."""
    import clean_rollouts as cr
    cr.SP, cr.OUT = path, path / "sweep"
    out.put(cr._only_driver() is not None)
    # process exits here; the kernel must drop the lock


@pytest.fixture
def driver(monkeypatch, tmp_path):
    import clean_rollouts as cr
    monkeypatch.setattr(cr, "SP", tmp_path)
    monkeypatch.setattr(cr, "OUT", tmp_path / "sweep")
    (tmp_path / "sweep").mkdir()
    return cr


def test_the_first_driver_takes_the_lock(driver):
    assert driver._only_driver() is not None


def test_a_second_driver_is_refused(driver):
    first = driver._only_driver()
    assert first is not None
    assert driver._only_driver() is None, (
        "a second driver started; both would play every outstanding game at once"
    )


def test_the_lock_is_released_when_the_holder_dies(driver, tmp_path):
    """The whole reason for flock over a pidfile."""
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_grab_driver_lock, args=(tmp_path, q))
    proc.start()
    assert q.get(timeout=30) is True
    proc.join(timeout=30)
    assert proc.exitcode == 0

    assert driver._only_driver() is not None, (
        "the lock outlived its holder; a driver killed hard would lock the sweep "
        "out permanently, which is exactly the pidfile failure flock avoids"
    )
