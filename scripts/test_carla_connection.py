import time
import carla

def main():
    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print(f"Connected to world map: {world.get_map().name}")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    vehicle = None  # Safe initialization

    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned Digital Twin Vehicle (ID: {vehicle.id}) at {spawn_point.location}")

        for step in range(100):
            world.tick()
            location = vehicle.get_location()
            print(f"Step {step:03d}: Vehicle Position -> X: {location.x:.2f}, Y: {location.y:.2f}")
            time.sleep(0.05)

    finally:
        if vehicle is not None:
            vehicle.destroy()
            print("Destroyed vehicle actor.")
            
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Restored asynchronous world settings.")

if __name__ == '__main__':
    main()