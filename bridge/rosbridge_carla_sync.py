import math
import time
import carla
import roslibpy


class ROSBridgeCarlaSync:
    def __init__(self, rosbridge_host='localhost', rosbridge_port=9090, scale_factor=10.0):
        self.scale_factor = scale_factor
        
        # 1. Connect to CARLA
        print("Connecting to CARLA Server...")
        self.carla_client = carla.Client('localhost', 2000)
        self.carla_client.set_timeout(10.0)
        self.world = self.carla_client.get_world()
        
        # 2. Spawn Digital Twin Vehicle Actor
        bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.actor = self.world.spawn_actor(bp, spawn_point)
        print(f"Spawned CARLA actor (ID: {self.actor.id})")

        # 3. Connect to ROSBridge WebSocket Server
        print(f"Connecting to ROSBridge at ws://{rosbridge_host}:{rosbridge_port}...")
        self.ros_client = roslibpy.Ros(host=rosbridge_host, port=rosbridge_port)
        
        # 4. Subscribe to ROS Topic
        self.pose_listener = roslibpy.Topic(
            self.ros_client, 
            '/vehicle/pose', 
            'geometry_msgs/PoseStamped'
        )

    def ros_pose_callback(self, message):
        """Processes incoming ROS PoseStamped message and updates CARLA vehicle."""
        position = message['pose']['position']
        orientation = message['pose']['orientation']

        # Extract ROS ENU coordinates
        ros_x = position['x']
        ros_y = position['y']
        
        # Convert Quaternion to Yaw angle (radians)
        qx = orientation['x']
        qy = orientation['y']
        qz = orientation['z']
        qw = orientation['w']
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw_rad)

        # Coordinate Frame Transformation (ROS ENU -> CARLA Left-Handed)
        carla_x = ros_x * self.scale_factor
        carla_y = -ros_y * self.scale_factor
        carla_yaw = -yaw_deg

        # Apply spatial transform in CARLA
        new_transform = carla.Transform(
            carla.Location(x=carla_x, y=carla_y, z=0.5),
            carla.Rotation(pitch=0.0, yaw=carla_yaw, roll=0.0)
        )
        self.actor.set_transform(new_transform)

    def start(self):
        self.ros_client.run()
        self.pose_listener.subscribe(self.ros_pose_callback)
        print("ROSBridge <-> CARLA synchronization node active.")

    def stop(self):
        self.pose_listener.unsubscribe()
        self.ros_client.terminate()
        if self.actor:
            self.actor.destroy()
            print("Cleaned up CARLA actor.")


if __name__ == '__main__':
    sync_node = ROSBridgeCarlaSync()
    try:
        sync_node.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sync_node.stop()