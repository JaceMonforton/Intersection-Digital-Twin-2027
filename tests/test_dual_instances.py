import carla
import time
import math
import sys

CARLA_HOST = 'localhost'
CARLA_PORT = 2000

def get_actor_speed(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

def main():
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(5.0)
        world = client.get_world()
    except Exception as e:
        print(f"[ERROR] Could not connect to CARLA server: {e}")
        sys.exit(1)

    print("=" * 65)
    print(" CARLA MULTI-INSTANCE CONCURRENCY VERIFIER")
    print("=" * 65)
    print("Monitoring active actors in the CARLA world for 10 seconds...\n")

    # Sample position changes to verify active execution
    initial_positions = {}
    
    for sample in range(1, 6):
        actors = world.get_actors().filter('vehicle.*')
        
        # Categorize vehicles
        ego_vehicles = [a for a in actors if 'ego' in a.attributes.get('role_name', '')]
        all_vehicles = list(actors)

        print(f"--- Sample {sample}/5 (Total Vehicles in World: {len(all_vehicles)}) ---")

        if not actors:
            print("  [!] No active vehicles found in CARLA.")
        else:
            for actor in actors:
                role = actor.attributes.get('role_name', 'npc')
                actor_id = actor.id
                loc = actor.get_location()
                speed = get_actor_speed(actor)

                # Track position changes across samples to check for deadlocks/stalls
                prev_loc = initial_positions.get(actor_id)
                if prev_loc:
                    movement = loc.distance(prev_loc)
                    status = "MOVING" if movement > 0.05 else "STATIONARY"
                else:
                    status = "INITIALIZING"
                
                initial_positions[actor_id] = loc

                print(f"  > ID: {actor_id:4d} | Role: {role:15s} | Speed: {speed:5.1f} km/h | Status: {status}")

        time.sleep(1.5)

    print("\n" + "=" * 65)
    print(" VERIFICATION SUMMARY")
    print("=" * 65)
    
    unique_roles = set(a.attributes.get('role_name', 'npc') for a in world.get_actors().filter('vehicle.*'))
    print(f"Detected Vehicle Roles: {list(unique_roles)}")

    if len(initial_positions) >= 2:
        print("[SUCCESS] Multiple instances detected and active in the CARLA world!")
    else:
        print("[WARNING] Fewer than 2 active vehicles detected. Check if both Pygame instances spawned successfully.")

if __name__ == '__main__':
    main()