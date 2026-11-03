"""The agent's reading of the SafetyGate. Skipped where carla or pygame isn't installed."""
import pytest

pytest.importorskip("carla")
pytest.importorskip("pygame")

from ditto_carla_agent import parse_emergency_brake


def gate(value):
    return {"features": {"SafetyGate": {"properties": {"emergency_brake": value}}}}


def test_brake_engaged():
    assert parse_emergency_brake(gate(True)) is True


@pytest.mark.parametrize("thing", [gate(False), gate(None), gate("true"), {"features": {}}, {}])
def test_anything_else_is_not_a_brake(thing):
    assert parse_emergency_brake(thing) is False
