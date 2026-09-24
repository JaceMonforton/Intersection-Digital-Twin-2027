import json
import pytest

# --- Helper Logic to Test ---

def parse_vehicle_telemetry(payload_json: str) -> dict:
    """Parses incoming MQTT JSON payload from physical vehicles."""
    data = json.loads(payload_json)
    
    # Required keys for state synchronization
    if "vehicle_id" not in data or "pose" not in data:
        raise ValueError("Missing essential telemetry fields")
        
    return {
        "id": data["vehicle_id"],
        "x": float(data["pose"]["x"]),
        "y": float(data["pose"]["y"]),
        "yaw": float(data["pose"].get("yaw", 0.0)),
        "speed": float(data.get("speed", 0.0))
    }


def convert_lab_to_carla_coords(x_phys: float, y_phys: float, scale_factor: float = 10.0) -> tuple:
    """Transforms 1:10 scaled physical lab coordinates to CARLA world space."""
    return round(x_phys * scale_factor, 2), round(y_phys * scale_factor, 2)


def calculate_time_to_collision(distance: float, relative_speed: float) -> float:
    """Computes simple TTC safety metric in seconds."""
    if relative_speed <= 0:
        return float('inf')  # Moving apart or stationary relative to each other
    return round(distance / relative_speed, 2)


# --- Pytest Test Suite ---

def test_valid_telemetry_parsing():
    valid_json = '{"vehicle_id": "car_01", "pose": {"x": 1.2, "y": 3.4, "yaw": 90.0}, "speed": 0.5}'
    parsed = parse_vehicle_telemetry(valid_json)
    
    assert parsed["id"] == "car_01"
    assert parsed["x"] == 1.2
    assert parsed["y"] == 3.4
    assert parsed["speed"] == 0.5


def test_missing_field_raises_error():
    invalid_json = '{"vehicle_id": "car_01"}'  # Missing 'pose' field
    
    with pytest.raises(ValueError, match="Missing essential telemetry fields"):
        parse_vehicle_telemetry(invalid_json)


def test_coordinate_scaling():
    phys_x, phys_y = 1.25, 2.50
    carla_x, carla_y = convert_lab_to_carla_coords(phys_x, phys_y, scale_factor=10.0)
    
    assert carla_x == 12.50
    assert carla_y == 25.00


def test_time_to_collision_safety_metric():
    # Car 10 meters away closing at 2 m/s -> TTC = 5 seconds
    ttc = calculate_time_to_collision(distance=10.0, relative_speed=2.0)
    assert ttc == 5.0

    # Cars moving apart (negative speed) -> Infinite TTC
    safe_ttc = calculate_time_to_collision(distance=10.0, relative_speed=-1.0)
    assert safe_ttc == float('inf')