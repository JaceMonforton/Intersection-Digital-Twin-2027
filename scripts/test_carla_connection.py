import carla
import time

def main():
    # 1. Connect to the CARLA UE4 Server running locally on port 2000
    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # 10 second connection timeout

    # 2. Get the simulation world instance
    world = client.get_world()
    print(f"Connected to CARLA Map: {world.get_map().name}")

    # 3. Retrieve Blueprint Library to spawn actors
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

    # 4. Pick a spawn location from the map definition
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0] if spawn_points else carla.Transform()

    # 5. Spawn the physical representation actor
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        print(f"Successfully spawned vehicle (ID: {vehicle.id}) at {spawn_point.location}")
        
        # Keep alive for 5 seconds then clean up
        time.sleep(5.0)
        vehicle.destroy()
        print("Vehicle destroyed. Connection test complete.")
    else:
        print("Failed to spawn vehicle at target transform.")

if __name__ == '__main__':
    main()