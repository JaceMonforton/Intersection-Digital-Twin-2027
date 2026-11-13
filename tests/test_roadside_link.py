"""Unit tests for how the ML car gets its view of the world. No CARLA needed."""
import json
import socket
import time

from roadside_link import (
    STALE_AFTER,
    RoadsideReceiver,
    RoadsideSender,
    RoadsideView,
    TickAck,
    TickGate,
    TruthView,
)

ADDRESS = ("127.0.0.1", 47899)


def wait_for(receiver, test=lambda m: m is not None):
    deadline = time.time() + 2.0
    while time.time() < deadline:
        message = receiver.poll()
        if test(message):
            return message
        time.sleep(0.01)
    raise AssertionError("no message arrived")


def test_tracks_reach_the_receiver():
    receiver = RoadsideReceiver(ADDRESS)
    try:
        RoadsideSender(ADDRESS).send(12.5, [(1.0, 2.0, 3.0, 4.0, 90.0)], [(5.0, 6.0, 0.5, 0.0)])
        message = wait_for(receiver)
        assert message == {"t": 12.5, "cars": [[1.0, 2.0, 3.0, 4.0, 90.0]], "walkers": [[5.0, 6.0, 0.5, 0.0]]}
    finally:
        receiver.close()


def test_the_receiver_keeps_only_the_newest_and_remembers_it():
    receiver = RoadsideReceiver(ADDRESS)
    try:
        sender = RoadsideSender(ADDRESS)
        sender.send(1.0, [], [])
        sender.send(2.0, [], [])
        assert wait_for(receiver, lambda m: m is not None and m["t"] == 2.0)["t"] == 2.0
        assert receiver.poll()["t"] == 2.0
    finally:
        receiver.close()


def test_a_bad_packet_doesnt_crash_the_receiver():
    receiver = RoadsideReceiver(ADDRESS)
    try:
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        raw.sendto(b"not json", ADDRESS)
        time.sleep(0.05)
        assert receiver.poll() is None
    finally:
        receiver.close()


class Stub:
    def __init__(self, message):
        self.message = message

    def poll(self):
        return self.message


def test_roadside_view_passes_on_fresh_tracks():
    view = RoadsideView(Stub({"t": 10.0, "cars": [[1, 2, 3, 4, 5]], "walkers": [[6, 7, 8, 9]]}))
    assert view.see(None, {}, {}, 10.1) == ([(1, 2, 3, 4, 5)], [(6, 7, 8, 9)], True)


def test_roadside_view_reports_old_or_missing_tracks_as_not_fresh():
    assert RoadsideView(Stub(None)).see(None, {}, {}, 10.0) == ([], [], False)
    old = {"t": 10.0, "cars": [[1, 2, 3, 4, 5]], "walkers": []}
    assert RoadsideView(Stub(old)).see(None, {}, {}, 10.0 + STALE_AFTER + 0.1) == ([], [], False)


def test_truth_view_gives_everything():
    assert TruthView().see(None, {1: (0, 0, 0, 0, 0)}, {2: (1, 1, 0, 0)}, 0.0) == ([(0, 0, 0, 0, 0)], [(1, 1, 0, 0)], True)


def test_messages_are_valid_json_with_no_infinity():
    # Cars with no known heading are given one by the processor, so everything here is a plain number
    receiver = RoadsideReceiver(ADDRESS)
    try:
        RoadsideSender(ADDRESS).send(1.0, [(0.123456, 0.0, 0.0, 0.0, 0.0)], [])
        assert wait_for(receiver)["cars"][0][0] == 0.12
        assert json.dumps(receiver.latest)
    finally:
        receiver.close()


def test_a_view_leaves_a_shared_receiver_open_for_the_next_car():
    receiver = RoadsideReceiver(ADDRESS)
    try:
        RoadsideView(receiver).close()
        RoadsideSender(ADDRESS).send(1.0, [], [])
        assert wait_for(receiver) is not None
    finally:
        receiver.close()


LOCKSTEP = ("127.0.0.1", 47898)


def test_the_tick_gate_waits_for_the_driver_only_while_one_is_acking():
    gate, ack = TickGate(LOCKSTEP, wait=0.3), TickAck(LOCKSTEP)
    try:
        started = time.monotonic()
        gate.wait(1)  # no driver yet: setup ticks mustn't stall
        assert time.monotonic() - started < 0.1 and gate.timeouts == 0
        ack.send(1)
        gate.wait(1)  # the driver has acted on frame 1, so the next tick may go
        assert gate.engaged and gate.timeouts == 0
        started = time.monotonic()
        gate.wait(2)  # the driver is acking but hasn't done frame 2: wait, then give up and stop waiting
        assert 0.25 < time.monotonic() - started < 1.0 and gate.timeouts == 1 and not gate.engaged
        started = time.monotonic()
        gate.wait(3)
        assert time.monotonic() - started < 0.1 and gate.timeouts == 1
        ack.send(4)
        ack.release()  # the driver has left its loop: don't wait for it
        started = time.monotonic()
        gate.wait(5)
        assert time.monotonic() - started < 0.1 and gate.timeouts == 1 and not gate.engaged and gate.acked == 4
    finally:
        gate.close()
        ack.close()
