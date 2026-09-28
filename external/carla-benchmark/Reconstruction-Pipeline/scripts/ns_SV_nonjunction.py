import carla
import math
import random
import sys
import os
import time
import json
from ns_check_points import check_and_get_direction
import ns_GT_nonjunction
import ns_validation_manager

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

def readScene(pointsPath):

    with open(pointsPath, 'r') as f:
        data = json.load(f)

    # Extract Crash Location
    crash_x = data["crash_location"]["x"]
    crash_y = data["crash_location"]["y"]

    # Extract Vehicle 1 details
    veh1_road_id = data["vehicle_1"]["road_id"]
    veh1_x = data["vehicle_1"]["position"]["x"]
    veh1_y = data["vehicle_1"]["position"]["y"]

    # Extract Vehicle 2 details
    veh2_road_id = data["vehicle_2"]["road_id"]
    veh2_x = data["vehicle_2"]["position"]["x"]
    veh2_y = data["vehicle_2"]["position"]["y"]

    return (crash_x, crash_y, 
            veh1_road_id, veh1_x, veh1_y, 
            veh2_road_id, veh2_x, veh2_y)

def get_movement_info(line):
    """
    Checks if a line contains the target movement keywords.
    """
    keywords = ["turning left", "turning right", "going straight"]
    lower_line = line.lower()
    
    for key in keywords:
        if key in lower_line:
            return key
    return None

def readSummary(reportPath):
    with open(reportPath, 'r') as file:
        for line in file:
            line = line.strip()
            
            if not line: continue 

            elif "vehicle 1" in line.lower():
                current_vehicle = 1
            elif "vehicle 2" in line.lower():
                current_vehicle = 2

            elif "impact point" in line.lower():
                if current_vehicle == 1 and not any(char.isdigit() for char in line.lower()):
                    veh1_impact = -1
                elif current_vehicle == 2 and not any(char.isdigit() for char in line.lower()):
                    veh2_impact = -1
                else:
                    impact_point = int(line.split(":")[1].strip().split(" ")[0].strip())
                    if current_vehicle == 1:
                        veh1_impact = impact_point
                    elif current_vehicle == 2:
                        veh2_impact = impact_point

            elif "speed:" in line.lower():
                if current_vehicle == 1 and "reported" in line.lower():
                    veh1_speed = -1
                elif current_vehicle == 1 and "stopped" in line.lower():
                    veh1_speed = 0
                elif current_vehicle == 2 and "reported" in line.lower():
                    veh2_speed = -1
                elif current_vehicle == 2 and "stopped" in line.lower():
                    veh2_speed = 0
                else:
                    speed = float(line.split(":")[1].strip().split(" ")[0].strip())
                    if current_vehicle == 1:
                        veh1_speed = speed
                    elif current_vehicle == 2:
                        veh2_speed = speed
            
            if "p_crash" in line.lower() or "pcrash" in line.lower():
                direction = get_movement_info(line)
                if current_vehicle == 1:
                        v1_direction = direction 
                if current_vehicle == 2:
                        v2_direction = direction

    return veh1_impact, veh1_speed, v1_direction, veh2_impact, veh2_speed, v2_direction

def draw_point(world, point_x, point_y):
    world.debug.draw_box(
        carla.BoundingBox(carla.Location(x=point_x, y=point_y, z=0.5), carla.Vector3D(0.5, 0.5, 5.0)),
        carla.Rotation(0, 0, 0),
        0.1,  # Line thickness
        carla.Color(0, 0, 255, 255), 
        10.0 # Life time in seconds
    )

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

def spawn_vehicle(xodrPath, pointsPath, reportPath, trajectory_path, simulation_path):

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicles = []
    SIM_FPS = 20.0  
    MAX_TICK_COUNT = 800000
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    crash_x, crash_y, veh1_road_id, veh1_x, veh1_y, veh2_road_id, veh2_x, veh2_y = readScene(pointsPath) 
    crash_P = (crash_x, crash_y)
    crash_location = carla.Location(x=crash_x, y=crash_y, z=0.1)

    draw_point(world, crash_x, crash_y)

    veh1_impact_point, veh1_speed, veh1_direction, veh2_impact_point, veh2_speed, veh2_direction = readSummary(reportPath)
    
    ###### Spawning Vehicle 1 ######
    print("vehicle 1")
    veh1_spawn = (veh1_x, veh1_y)
    _, veh1_x, veh1_y, _, hdg_1 = check_and_get_direction(veh1_spawn, crash_P, xodrPath, snap=True)
    veh1_angle = math.degrees(math.atan2(-math.sin(hdg_1), math.cos(hdg_1)))
    print("spawn Points: ", veh1_x, veh1_y, veh1_angle)
    veh1_spawn_z = 0.5 # Slightly above ground to avoid collision with ground
    veh1_yaw = veh1_angle
    vehicle_bp1 = random.choice(blueprint_library.filter('vehicle.tesla.model3')) 
    vehicle_bp1.set_attribute('color', '0, 255, 0') 
    location1 = carla.Location(x=veh1_x, y=veh1_y, z=veh1_spawn_z)
    rotation1 = carla.Rotation(pitch=0.0, yaw=veh1_yaw, roll=0.0)
    veh1_spawn_transform = carla.Transform(location1, rotation1)

    #### Spawning Vehicle 2 #########
    print("vehicle 2")
    veh2_spawn = (veh2_x, veh2_y)
    _, veh2_x, veh2_y, _, hdg_2 = check_and_get_direction(veh2_spawn, crash_P, xodrPath, snap=True)
    veh2_angle = math.degrees(math.atan2(-math.sin(hdg_2), math.cos(hdg_2)))
    print("spawn Points: ", veh2_x, veh2_y, veh2_angle)
    veh2_spawn_z = 0.5 # Slightly above ground to avoid collision with ground
    veh2_yaw = veh2_angle
    vehicle_bp2 = random.choice(blueprint_library.filter('vehicle.tesla.model3')) 
    vehicle_bp2.set_attribute('color', '255, 0, 0') 
    location2 = carla.Location(x=veh2_x, y=veh2_y, z=veh2_spawn_z)
    rotation2 = carla.Rotation(pitch=0.0, yaw=veh2_yaw, roll=0.0)
    veh2_spawn_transform = carla.Transform(location2, rotation2)

    v1_P = (veh1_x, veh1_y)
    v2_P = (veh2_x, veh2_y)
    veh1_speed, veh2_speed = ns_GT_nonjunction.calculate_synchronized_speeds(xodrPath, crash_P, 
                                                                         v1_P, veh1_road_id, veh1_speed, v2_P,
                                                                         veh2_road_id, veh2_speed)

    # --- Spawn the Vehicles ---
    try:
        vehicle1 = world.try_spawn_actor(vehicle_bp1, veh1_spawn_transform)
        vehicle2 = world.try_spawn_actor(vehicle_bp2, veh2_spawn_transform)

        if not vehicle1:
            print("Failed to spawn 1st vehicle. Check coordinates/occupancy.")

        if not vehicle2:
            print("Failed to spawn 2nd vehicle. Check coordinates/occupancy.")
        
        if vehicle1 and vehicle2:
            print("Successfully spawned to vehicles.")
            vehicles.extend([vehicle1, vehicle2])

            v1_ms = veh1_speed / 3.6
            v2_ms = veh2_speed / 3.6
            
            # Vehicle 1 Start Velocity
            yaw1_rad = math.radians(veh1_spawn_transform.rotation.yaw)
            vehicle1.set_target_velocity(carla.Vector3D(
                x=v1_ms * math.cos(yaw1_rad),
                y=v1_ms * math.sin(yaw1_rad),
                z=0
            ))

            # Vehicle 2 Start Velocity
            yaw2_rad = math.radians(veh2_spawn_transform.rotation.yaw)
            vehicle2.set_target_velocity(carla.Vector3D(
                x=v2_ms * math.cos(yaw2_rad),
                y=v2_ms * math.sin(yaw2_rad),
                z=0
            ))

            # print("veh 1 loc: ", location1)
            # print("veh 2 loc: ", location2)
            # print("crash loc: ", crash_location)

            validator = ns_validation_manager.ValidationManager(
                crash_location=(crash_x, crash_y)   
            )

            crash_event_triggered = False
            veh1_route = None
            veh2_route = None
            veh1_route_points = None
            veh2_route_points = None

            def on_collision(event):
                nonlocal crash_event_triggered 
                if crash_event_triggered:
                    return
                other_actor = event.other_actor
            
                if other_actor.id == vehicle2.id:
                    print(f"\n!!! CRASH Between Two Vehicles DETECTED!!!")
                    crash_event_triggered = True
                
                    current_sim_time = world.get_snapshot().timestamp.elapsed_seconds
                    validator.update_arrival_times(current_sim_time, vehicle1.get_location(), vehicle2.get_location())
                    validator.register_crash(vehicle1, vehicle2, veh1_route_points, veh2_route_points, 
                                             veh1_direction, veh2_direction, 
                                             veh1_impact_point, veh2_impact_point)
                else:
                    pass

            collision_bp = blueprint_library.find('sensor.other.collision')
            col_sensor1 = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle1)
            col_sensor1.listen(lambda event: on_collision(event))
            vehicles.append(col_sensor1)

            
            agent1 = BlindAgent(vehicle1, target_speed = veh1_speed)

            veh1_route_points = ns_GT_nonjunction.route_generator(xodrPath, trajectory_path, crash_P, veh1_x, veh1_y, veh1_road_id,
                                                           fileopen = "w")
            if veh1_route_points:
                veh1_route = convert_points_to_global_plan(veh1_route_points)
                agent1.set_global_plan(veh1_route)
            
            agent2 = BlindAgent(vehicle2, target_speed = veh2_speed)

            veh2_route_points = ns_GT_nonjunction.route_generator(xodrPath, trajectory_path, crash_P, veh2_x, veh2_y, veh2_road_id,
                                                           fileopen = "a")
            if veh2_route_points:
                veh2_route = convert_points_to_global_plan(veh2_route_points)
                agent2.set_global_plan(veh2_route)

            """ Save Simulation info in a json """
            simulation_data = {
                "vehicle1": {
                    "start_position": {"x": veh1_x, "y": veh1_y, "z": veh1_spawn_z},
                    "speed_kmh": veh1_speed,  
                    "yaw": veh1_yaw,
                    "trajectory": veh1_route_points if veh1_route_points else []
                },
                "vehicle2": {
                    "start_position": {"x": veh2_x, "y": veh2_y, "z": veh2_spawn_z},
                    "speed_kmh": veh2_speed,  
                    "yaw": veh2_yaw,
                    "trajectory": veh2_route_points if veh2_route_points else []
                },
                "crash_point": {"x": crash_x, "y": crash_y}
            }
            with open(simulation_path, "w") as json_file:
                json.dump(simulation_data, json_file, indent=4)


            for i in range(MAX_TICK_COUNT):
                world.tick()
                current_sim_time = world.get_snapshot().timestamp.elapsed_seconds
                validator.update_arrival_times(current_sim_time, vehicle1.get_location(), vehicle2.get_location())

                if crash_event_triggered:
                    control_stop = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
                    vehicle1.apply_control(control_stop)
                    vehicle2.apply_control(control_stop)

                else:
                    if agent1.done() and agent2.done():
                        break
                    if not agent1.done():
                        vehicle1.apply_control(agent1.run_step())
                    if not agent2.done():
                        vehicle2.apply_control(agent2.run_step())

                
            
    except Exception as e:
        print(f"An error occurred while running the simulation: {e}")
            
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

    


if __name__ == '__main__':

    try:
        spawn_vehicle()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)
    