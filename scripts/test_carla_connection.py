import time
import carla
def main():
    # Connect to CARLA Server on default port 2000
    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # 10 second timeout limit

    # Retrieve current world instance
    world = client.get_world()
    print(f"Connected to world map: {world.get_map().name}")

    # Set Synchronous Mode (Prevents lag/stutter by syncing step ticks)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS fixed time-step
    world.apply_settings(settings)

    try:
        # Spawn a vehicle from blueprint library
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Pick first available spawn point in map
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned Digital Twin Vehicle (ID: {vehicle.id}) at {spawn_point.location}")

        # Simulation loop tick
        for step in range(100):
            world.tick()  # Step the simulator explicitly
            location = vehicle.get_location()
            print(f"Step {step}: Vehicle Position -> X: {location.x:.2f}, Y: {location.y:.2f}")
            time.sleep(0.05)

    finally:
        # Clean up actor and revert settings on exit
        if 'vehicle' in locals():
            vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Cleaned up simulation actors.")

if __name__ == '__main__':
    main()