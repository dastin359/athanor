"""The rollout queue must run in queue order, not in whatever order threads wake.

`clean_rollouts.py` gives every game a thread up front and bounds concurrency
with a condition variable, so the limit can be retuned mid-run. The cost of that
design is that all games arrive at the gate simultaneously, and `notify_all`
wakes every waiter at once: Python guarantees no ordering across
`Condition.wait`, and a timed wait means a thread can wake with no notification
at all. Whichever waiter the scheduler picks takes the slot.

That made the queue's shortest-first ordering decorative, and it was caught in
production on 2026-08-07. The queue was `cd82, r11l, sc25, su15, lp85, tr87,
...`; `cd82` and `r11l` took the opening slots, and when `r11l` finished the
freed slot went to **`tr87`** -- past three games ahead of it, which had no
workspace at all.

Ordering is not cosmetic here. It exists so a window that ends early has banked
the achievable games rather than spending itself on a long one, and a racy queue
inverts exactly that. The same failure, from a fixed order rather than a racy
one, once turned a three-game experiment into a one-game one in
`rerun_losses.py`.

Nothing in any log showed it. The driver reported each game starting, correctly,
in an order nobody had a reason to check against the queue.
"""
from __future__ import annotations

import pathlib
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))


@pytest.fixture
def driver(monkeypatch):
    """Import the driver without letting it start a proxy or call the API.

    `install_strip()` lives in `main()` precisely so this is possible -- the
    module imports clean, which is what makes the admission logic testable.
    """
    import clean_rollouts as cr  # noqa: PLC0415

    monkeypatch.setattr(cr, "_running", 0, raising=False)
    monkeypatch.setattr(cr, "_last_limit", 0, raising=False)
    monkeypatch.setattr(cr, "_waiting", set(), raising=False)
    return cr


def test_slots_are_granted_in_queue_order_not_wake_order(driver, monkeypatch):
    """Twenty games, two slots: the order they start must be the queue order.

    Threads are started deliberately *backwards* and with a stagger, so the
    scheduler is handed every opportunity to grant a late game an early slot.
    Under the old `if _running < limit` the last-submitted game could take the
    first free slot; under rank-ordered admission it cannot.
    """
    monkeypatch.setattr(driver, "concurrency", lambda: 2)
    games = [f"g{i:02d}" for i in range(20)]
    started: list[str] = []
    lock = threading.Lock()

    def worker(gid: str, rank: int) -> None:
        driver._take_slot(gid, rank)
        try:
            with lock:
                started.append(gid)
            time.sleep(0.02)
        finally:
            driver._free_slot()

    driver._register_queue(len(games))    # what one_pass does before submitting
    threads = [threading.Thread(target=worker, args=(g, i))
               for i, g in enumerate(games)]
    for t in reversed(threads):          # last game gets the head start
        t.start()
        time.sleep(0.001)
    for t in threads:
        t.join(timeout=30)

    assert started == games, (
        "slots were granted out of queue order; shortest-first is not in force"
    )


def test_the_limit_is_still_honoured_while_ordering(driver, monkeypatch):
    """Rank ordering must not let more than `concurrency()` games run at once.

    The ordering check above would also pass for a broken gate that admitted
    everyone in order and bounded nothing, which is a worse bug than the one
    being fixed.
    """
    monkeypatch.setattr(driver, "concurrency", lambda: 3)
    peak = 0
    live = 0
    lock = threading.Lock()

    def worker(rank: int) -> None:
        nonlocal peak, live
        driver._take_slot(f"g{rank}", rank)
        try:
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
        finally:
            with lock:
                live -= 1
            driver._free_slot()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert peak == 3, f"concurrency limit not honoured: peak {peak}"


def test_a_raised_limit_is_picked_up_without_a_restart(driver, monkeypatch):
    """The whole reason for the condition variable: retune a 15-hour queue live.

    `$SP/concurrency` is edited by a human, so nothing signals the condition.
    The timed wait is what makes a raise take effect, and rank ordering must not
    have broken it -- a waiter that is not the minimum still has to re-check the
    limit rather than sleep until someone finishes.
    """
    limit = 1
    monkeypatch.setattr(driver, "concurrency", lambda: limit)
    live = 0
    peak = 0
    lock = threading.Lock()

    def worker(rank: int) -> None:
        nonlocal live, peak
        driver._take_slot(f"g{rank}", rank)
        try:
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.5)
        finally:
            with lock:
                live -= 1
            driver._free_slot()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    limit = 3                            # the live retune, mid-flight
    for t in threads:
        t.join(timeout=40)

    assert peak >= 2, (
        "a raised limit never took effect; the timed wait is what picks it up "
        "and nothing else signals the condition"
    )


def test_an_abort_stops_the_games_that_have_not_started(driver, monkeypatch):
    """The breaker exists to stop the queue burning, and it could not.

    `_run_one` raises SystemExit when the ARC endpoint is unreachable, because
    every remaining game would fail identically. The raise unwinds one worker
    thread and reaches none of the others: `one_pass` only learns of it through
    `as_completed`, so on 2026-08-07 seventeen futures failed in milliseconds and
    **thirteen games started and failed before the main thread saw the first
    one**. The abort that exists to stop the queue arrived last.

    An exception cannot reach a sibling. A flag can, so `_take_slot` and
    `_guarded` check one.
    """
    monkeypatch.setattr(driver, "concurrency", lambda: 1)
    driver._aborted.clear()
    ran: list[int] = []
    lock = threading.Lock()

    def worker(rank: int) -> None:
        try:
            driver._take_slot(f"g{rank}", rank)
        except driver._Aborted:
            return
        try:
            if driver._aborted.is_set():
                return
            with lock:
                ran.append(rank)
            if rank == 0:                     # the endpoint dies on the first game
                driver._aborted.set()
                with driver._slots:
                    driver._slots.notify_all()
        finally:
            driver._free_slot()

    driver._register_queue(10)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    driver._aborted.clear()

    assert ran == [0], f"games ran after the abort: {ran}"
