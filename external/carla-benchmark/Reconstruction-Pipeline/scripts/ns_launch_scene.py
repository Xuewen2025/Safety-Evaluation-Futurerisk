import carla
import json
import math
import os
import sys
import argparse
from settings import SIMULATION_DIR
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

class BlindAgent(BasicAgent):
    """
    A subclass of BasicAgent that:
    1. Ignores obstacles and traffic lights.
    2. Forces the vehicle to exact target speed (ignoring acceleration/turning physics).
    """
    def run_step(self):

        control = self._local_planner.run_step(debug=False)
        
        v_ms = self._target_speed / 3.6

        transform = self._vehicle.get_transform()
        fwd = transform.get_forward_vector()
        
        current_vel = self._vehicle.get_velocity()

        self._vehicle.set_target_velocity(carla.Vector3D(
            x=fwd.x * v_ms,
            y=fwd.y * v_ms,
            z=current_vel.z 
        ))
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        
        return control

class WayPointWrapper:
    """
    A dummy class that mimics a carla.Waypoint.
    BasicAgent expects objects that have specific attributes like .transform, .road_id, etc.
    """
    def __init__(self, location, rotation=carla.Rotation(0,0,0)):
        self.transform = carla.Transform(location, rotation)
        self.location = location 
        
        # --- ADD THESE DUMMY ATTRIBUTES ---
        self.road_id = 0       # Default dummy ID
        self.section_id = 0    # Default dummy section
        self.lane_id = 0       # Default dummy lane
        self.s = 0.0           # Default distance along road
        self.is_junction = False 
        self.lane_width = 3.5  # Standard width

def convert_points_to_global_plan(point_list):
    """
    Converts a list of (x, y) tuples into the format required by BasicAgent:
    [(WayPointWrapper, RoadOption), ...]
    """
    plan = []
    for x, y in point_list:
        loc = carla.Location(x=x, y=y, z=0.5)
        
        fake_wp = WayPointWrapper(loc)
        
        plan.append((fake_wp, RoadOption.LANEFOLLOW))
    return plan

def run_simulation(json_path):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    # --- Simulation Settings ---
    SIM_FPS = 20.0
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)
    
    vehicles = []

    try:
        # 1. Load Data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded scenario from {json_path}")
        
        # 2. Extract Vehicle Data
        v1_data = data['vehicle1']
        v2_data = data['vehicle2']
        
        bp1 = blueprint_library.filter('vehicle.tesla.model3')[0]
        bp1.set_attribute('color', '0, 255, 0')
        start1 = v1_data['start_position']
        v1_transform = carla.Transform(
            carla.Location(x=start1['x'], y=start1['y'], z=start1['z']),
            carla.Rotation(yaw=v1_data['yaw'])
        )
        
        bp2 = blueprint_library.filter('vehicle.tesla.model3')[0]
        bp2.set_attribute('color', '255, 0, 0')
        start2 = v2_data['start_position']
        v2_transform = carla.Transform(
            carla.Location(x=start2['x'], y=start2['y'], z=start2['z']),
            carla.Rotation(yaw=v2_data['yaw'])
        )

        v1_speed = v1_data['speed_kmh']
        v2_speed = v2_data['speed_kmh']

        v1_trajectory = v1_data['trajectory']
        v2_trajectory = v2_data['trajectory']

        v1_route = [tuple(point) for point in v1_trajectory]
        v2_route = [tuple(point) for point in v2_trajectory]

        vehicle1 = world.try_spawn_actor(bp1, v1_transform)
        vehicle2 = world.try_spawn_actor(bp2, v2_transform)
        
        if not vehicle1 or not vehicle2:
            print("Error: Could not spawn vehicles. Check for collisions at spawn points.")
            return

        vehicles.extend([vehicle1, vehicle2])
        print("Vehicles spawned successfully.")

        v1_ms = v1_speed / 3.6
        v2_ms = v2_speed / 3.6
        
        # Vehicle 1 Start Velocity
        yaw1_rad = math.radians(v1_transform.rotation.yaw)
        vehicle1.set_target_velocity(carla.Vector3D(
            x=v1_ms * math.cos(yaw1_rad),
            y=v1_ms * math.sin(yaw1_rad),
            z=0
        ))

        # Vehicle 2 Start Velocity
        yaw2_rad = math.radians(v2_transform.rotation.yaw)
        vehicle2.set_target_velocity(carla.Vector3D(
            x=v2_ms * math.cos(yaw2_rad),
            y=v2_ms * math.sin(yaw2_rad),
            z=0
        ))

        crash_event_triggered = False
        veh1_route = None
        veh2_route = None

        def on_collision(event):
            nonlocal crash_event_triggered 
            if crash_event_triggered:
                return
            other_actor = event.other_actor
            
            if other_actor.id == vehicle2.id:
                print(f"\n!!! CRASH Between Two Vehicles DETECTED!!!")
                crash_event_triggered = True
            else:
                pass

        collision_bp = blueprint_library.find('sensor.other.collision')
        col_sensor1 = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle1)
        col_sensor1.listen(lambda event: on_collision(event))
        vehicles.append(col_sensor1)

        agent1 = BlindAgent(vehicle1, target_speed = v1_speed)
        veh1_route = convert_points_to_global_plan(v1_route)
        agent1.set_global_plan(veh1_route)
        
        agent2 = BlindAgent(vehicle2, target_speed = v2_speed)
        veh2_route = convert_points_to_global_plan(v2_route)
        agent2.set_global_plan(veh2_route)


        print("Starting simulation loop...")
        MAX_TICK_COUNT = 800000
        
        for i in range(MAX_TICK_COUNT):
            world.tick()

            if crash_event_triggered:
                control_stop = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
                vehicle1.apply_control(control_stop)
                vehicle2.apply_control(control_stop)
                

            else:
                if agent1.done() and agent2.done():
                    print("Both agents reached destination.")
                    break
                if not agent1.done():
                    vehicle1.apply_control(agent1.run_step())
                if not agent2.done():
                    vehicle2.apply_control(agent2.run_step())


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if client and 'settings' in locals():
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
        print("Cleaning up actors...")

        sensors = [x for x in vehicles if 'sensor' in x.type_id]
        cars = [x for x in vehicles if 'vehicle' in x.type_id]

        for sensor in sensors:
            if sensor.is_alive:
                sensor.stop()  
                sensor.destroy() 
        
        if cars:
            client.apply_batch([carla.command.DestroyActor(x) for x in cars])
            
        print("Scenario finished safely.")

# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spawn vehicles for a specific crash scenario.")
    
    parser.add_argument("crash_id", type=int, nargs='?', help="The ID of the crash scenario to simulate.")

    args = parser.parse_args()

    CRASH_ID = args.crash_id

    simulation_path = os.path.join(SIMULATION_DIR, f"simulation_{CRASH_ID}.json")

    if args.crash_id is None:
        print("Error: Crash ID is missing. Please provide a crash ID (e.g., python script.py 510163).")
        sys.exit(1)
    else:
        run_simulation(simulation_path)