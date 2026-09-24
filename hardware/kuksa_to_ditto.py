import json
import time
import paho.mqtt.client as mqtt
from kuksa_client.grpc import VSSClient, Datapoint

# Configuration
KUKSA_HOST = "localhost" # KUKSA Databroker running on vehicle
KUKSA_PORT = 55555
MQTT_BROKER = "192.168.1.100" # Central IP running Mosquitto / Eclipse Ditto
MQTT_PORT = 1883
VEHICLE_ID = "org.eclipse.ditto:vehicle_01"

# Setup MQTT Client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Connect to KUKSA Databroker via gRPC
with VSSClient(KUKSA_HOST, KUKSA_PORT) as kuksa:
    print(f"Connected to KUKSA Databroker on vehicle. Exporting to Ditto...")
    
    # Subscribe to target VSS signal paths
    vss_paths = [
        "Vehicle.CurrentLocation.Latitude", # Used here for local map X
        "Vehicle.CurrentLocation.Longitude", # Used here for local map Y
        "Vehicle.CurrentLocation.Heading",
        "Vehicle.Speed"
    ]
    
    for updates in kuksa.subscribe_current_values(vss_paths):
        # Extract VSS values safely
        x = updates.get("Vehicle.CurrentLocation.Latitude", Datapoint(0.0)).value
        y = updates.get("Vehicle.CurrentLocation.Longitude", Datapoint(0.0)).value
        heading = updates.get("Vehicle.CurrentLocation.Heading", Datapoint(0.0)).value
        speed = updates.get("Vehicle.Speed", Datapoint(0.0)).value

        # Construct Eclipse Ditto Protocol JSON payload
        ditto_payload = {
            "topic": f"org.eclipse.ditto/vehicle_01/things/twin/commands/modify",
            "headers": {"content-type": "application/json"},
            "path": "/features/kinematics/properties",
            "value": {
                "x": float(x),
                "y": float(y),
                "heading": float(heading),
                "speed": float(speed)
            }
        }

        # Publish update to Eclipse Ditto Ingestion Topic
        mqtt_client.publish(
            f"telemetry/vehicle_01", 
            json.dumps(ditto_payload)
        )
        time.sleep(0.05) # 20 Hz update rate