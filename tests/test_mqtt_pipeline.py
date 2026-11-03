"""
End-to-end safety override benchmark against the running Docker stack:
    vehicle telemetry -> MQTT -> Ditto -> safety engine -> Ditto -> MQTT -> motor cut

Skipped when the stack isn't up, as on CI. Re-provisions the twin, so it resets Thing state.

    docker compose -f docker/docker-compose.yml up -d
    pytest tests/test_mqtt_pipeline.py -v -s
"""
import json
import statistics
import threading
import time

import paho.mqtt.client as mqtt
import pytest
import requests
import setup_entities
from central_gpu_processor import COLLISION_RADIUS, TTC_BRAKE, DittoLink, SafetyEngine
from physical_ml_subscriber import SafetyGateSubscriber, SimMotors

BROKER = "localhost"
ROUNDS = 20
LATENCY_TARGET_MS = 100.0  # REQ-SYS-002
SETTLE_TIMEOUT = 5.0
TELEMETRY_RATE = 20  # Hz


class TimingMotors(SimMotors):
    """Timestamps stops, so a benchmark can see the moment the cut happens."""

    def __init__(self):
        super().__init__()
        self.last_stop = 0.0
        self.stopped = threading.Event()

    def stop(self):
        super().stop()
        self.last_stop = time.perf_counter()
        self.stopped.set()


def wait_until(condition, timeout=SETTLE_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    try:
        s.get(f"{setup_entities.DITTO_API}/things", timeout=3).raise_for_status()
        probe = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        probe.connect(BROKER, 1883, 5)
        probe.disconnect()
    except (requests.RequestException, OSError) as e:
        pytest.skip(f"Docker stack not reachable: {e}")
    setup_entities.provision(s)
    return s


@pytest.fixture
def gate(session):
    # The brake latches, so an earlier test may have left it engaged
    setup_entities.put(session, setup_entities.SAFETY_GATE_URL, setup_entities.gate_released(), "release")
    motors = TimingMotors()
    subscriber = SafetyGateSubscriber(motors, BROKER)
    subscriber.start()
    assert wait_until(lambda: not subscriber.held), "retained release never reached the vehicle"
    yield subscriber
    subscriber.stop()


@pytest.fixture
def publisher(session):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, 1883, 30)
    client.loop_start()
    yield client
    client.loop_stop()
    client.disconnect()


def publish_pose(client, name, x, y, vx, vy):
    client.publish(f"telemetry/{name}", json.dumps({
        "topic": f"org.example/{name}/things/twin/commands/merge",
        "headers": {"content-type": "application/merge-patch+json", "response-required": False},
        "path": "/features",
        "value": {
            "location": {"properties": {"x": x, "y": y, "z": 0.0}},
            "velocity": {"properties": {"vx": vx, "vy": vy}},
        },
    }))


def test_override_latency_from_ditto_write_to_motor_cut(session, gate):
    samples = []
    for _ in range(ROUNDS):
        setup_entities.put(session, setup_entities.SAFETY_GATE_URL, setup_entities.gate_released(), "release")
        assert wait_until(lambda: not gate.held)
        gate.motors.stopped.clear()

        start = time.perf_counter()
        session.put(setup_entities.SAFETY_GATE_URL, json={"emergency_brake": True}, timeout=2).raise_for_status()
        assert gate.motors.stopped.wait(SETTLE_TIMEOUT), "brake never reached the motors"
        samples.append((gate.motors.last_stop - start) * 1000.0)

    samples.sort()
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(f"\n[BENCH] write -> motor cut over {ROUNDS} rounds: "
          f"p50={statistics.median(samples):.1f} ms p95={p95:.1f} ms max={samples[-1]:.1f} ms")
    assert p95 < LATENCY_TARGET_MS


def run_scenario(session, gate, publisher, ml_start, human_start, ml_v, human_v, duration):
    """Drives both vehicles over MQTT with the engine live. Returns the elapsed scenario time
    at the motor cut and the state of the engine's last evaluation, or None if no cut happened."""
    setup_entities.put(session, setup_entities.SAFETY_GATE_URL, setup_entities.gate_released(), "release")
    assert wait_until(lambda: not gate.held)
    gate.motors.stopped.clear()

    ditto = DittoLink()
    ditto.start()
    assert ditto.subscribed.wait(SETTLE_TIMEOUT)
    engine = SafetyEngine(ditto)

    start = time.time()
    cut_at = None
    next_publish = start
    try:
        while time.time() - start < duration:
            now = time.time()
            elapsed = now - start
            if now >= next_publish:
                publish_pose(publisher, "vehicle_ml", ml_start + ml_v * elapsed, 0.0, ml_v, 0.0)
                publish_pose(publisher, "human_vehicle", human_start + human_v * elapsed, 0.0, human_v, 0.0)
                next_publish += 1.0 / TELEMETRY_RATE
            engine.step(now)
            if gate.motors.stopped.is_set():
                cut_at = time.time() - start
                break
            time.sleep(0.005)
    finally:
        ditto.stop()
    return cut_at


def test_collision_course_cuts_motors_before_impact(session, gate, publisher):
    # Head-on at a 10 m/s closing speed, starting 30 m apart
    separation, closing = 30.0, 10.0
    impact_at = (separation - COLLISION_RADIUS) / closing
    trigger_at = impact_at - TTC_BRAKE

    cut_at = run_scenario(session, gate, publisher, 0.0, separation, 5.0, -5.0, impact_at)

    assert cut_at is not None, "motors were never cut"
    margin = impact_at - cut_at
    print(f"\n[BENCH] TTC threshold crossed at {trigger_at:.2f} s, motors cut at {cut_at:.2f} s, "
          f"{margin:.2f} s before impact at {impact_at:.2f} s")
    assert margin > 0
    assert cut_at >= trigger_at - 0.2  # and not spuriously early


def test_diverging_vehicles_do_not_brake(session, gate, publisher):
    assert run_scenario(session, gate, publisher, 0.0, 30.0, -5.0, 5.0, 2.0) is None


def test_rebooted_vehicle_learns_a_latched_brake_from_the_retained_message(session):
    session.put(setup_entities.SAFETY_GATE_URL, json={"emergency_brake": True}, timeout=2).raise_for_status()
    time.sleep(0.5)

    subscriber = SafetyGateSubscriber(SimMotors(), BROKER)
    subscriber.start()
    try:
        assert wait_until(lambda: subscriber.reason == "emergency_brake == True")
        assert subscriber.held
    finally:
        subscriber.stop()
        setup_entities.put(session, setup_entities.SAFETY_GATE_URL, setup_entities.gate_released(), "release")
