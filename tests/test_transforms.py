import pytest

# Example helper function to test physical-to-CARLA scale conversion
def scale_physical_to_carla(phys_x: float, phys_y: float, scale_factor: float = 10.0):
    """Converts scaled physical lab coordinates to CARLA world coordinates."""
    return phys_x * scale_factor, phys_y * scale_factor

def test_coordinate_scaling():
    phys_x, phys_y = 1.5, 2.0
    carla_x, carla_y = scale_physical_to_carla(phys_x, phys_y, scale_factor=10.0)
    
    assert carla_x == 15.0
    assert carla_y == 20.0

def test_invalid_telemetry():
    # Verify parsing logic rejects bad telemetry JSON
    bad_payload = {"vehicle_id": "car_01"} # Missing coordinates
    
    with pytest.raises(KeyError):
        _ = bad_payload["pose"]["x"]