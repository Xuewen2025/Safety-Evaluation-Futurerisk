"""Logging variant of ns_launch_scene_bm.py (TRACE authors' own benchmark
replay script) -- identical spawn/agent/collision-sensor logic, but records
per-tick ego/agent pose+velocity and the exact collision tick to a JSON file
instead of only printing to console, so the real CARLA physics replay can
be parsed into this project's RawTrajectoryInstance format (replacing the
kinematic-reconstruction approximation in futurerisk/adapters/trace.py with
ground-truth CARLA output + an independent collision-sensor label).

Only the additions are: (1) OUTPUT_DIR / per-tick recording, (2) an early
stop shortly after collision instead of running to MAX_TICK_COUNT, (3) a
bounded overall tick cap so a scenario that neither collides nor completes
its route doesn't hang the batch run. Everything else (BlindAgent,
WayPointWrapper, spawn logic, collision-sensor wiring) is copied verbatim
from the original so the replay is faithful to what the TRACE authors
themselves intended to run.
"""
import carla
import json
import math
import os
import sys
import argparse
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

SIMULATION_DIR = "Benchmark/simulations"
OUTPUT_DIR = "Benchmark/replay"

POST_COLLISION_TICKS = 20      # ~1s tail at 20Hz, matches trace.py's post_impact_tail_s
MAX_TICK_COUNT = 3000           # ~150s sim time safety cap (route-completion / never-collide case)


class BlindAgent(BasicAgent):
    def run_step(self):
        control = self._local_planner.run_step(debug=False)
        v_ms = self._target_speed / 3.6
        transform = self._vehicle.get_transform()
        fwd = transform.get_forward_vector()
        current_vel = self._vehicle.get_velocity()
        self._vehicle.set_target_velocity(carla.Vector3D(
            x=fwd.x * v_ms, y=fwd.y * v_ms, z=current_vel.z,
        ))
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        return control


class WayPointWrapper:
    def __init__(self, location, rotation=carla.Rotation(0, 0, 0)):
        self.transform = carla.Transform(location, rotation)
        self.location = location
        self.road_id = 0
        self.section_id = 0
        self.lane_id = 0
        self.s = 0.0
        self.is_junction = False
        self.lane_width = 3.5


def convert_points_to_global_plan(point_list):
    plan = []
    for x, y in point_list:
        loc = carla.Location(x=x, y=y, z=0.5)
        fake_wp = WayPointWrapper(loc)
        plan.append((fake_wp, RoadOption.LANEFOLLOW))
    return plan


def _pose_row(vehicle):
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    return {
        "x": t.location.x, "y": t.location.y, "z": t.location.z,
        "yaw_deg": t.rotation.yaw,
        "vx": v.x, "vy": v.y, "vz": v.z,
    }


def run_simulation(json_path, crash_id, sim_fps: float = 20.0):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / sim_fps
    world.apply_settings(settings)

    vehicles = []
    log = {"crash_id": crash_id, "sim_fps": sim_fps, "collision_tick": None,
           "vehicle1": [], "vehicle2": [], "status": "unknown"}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded scenario from {json_path}")

        v1_data = data['vehicle1']
        v2_data = data['vehicle2']

        bp1 = blueprint_library.filter('vehicle.tesla.model3')[0]
        bp1.set_attribute('color', '0, 255, 0')
        start1 = v1_data['start_position']
        v1_transform = carla.Transform(
            carla.Location(x=start1['x'], y=start1['y'], z=start1['z']),
            carla.Rotation(yaw=v1_data['yaw']),
        )

        bp2 = blueprint_library.filter('vehicle.tesla.model3')[0]
        bp2.set_attribute('color', '255, 0, 0')
        start2 = v2_data['start_position']
        v2_transform = carla.Transform(
            carla.Location(x=start2['x'], y=start2['y'], z=start2['z']),
            carla.Rotation(yaw=v2_data['yaw']),
        )

        v1_speed = v1_data['speed_kmh']
        v2_speed = v2_data['speed_kmh']
        v1_route = [tuple(p) for p in v1_data['trajectory']]
        v2_route = [tuple(p) for p in v2_data['trajectory']]

        vehicle1 = world.try_spawn_actor(bp1, v1_transform)
        vehicle2 = world.try_spawn_actor(bp2, v2_transform)
        if not vehicle1 or not vehicle2:
            log["status"] = "spawn_failed"
            print("Error: Could not spawn vehicles. Check for collisions at spawn points.")
            return log

        vehicles.extend([vehicle1, vehicle2])
        print("Vehicles spawned successfully.")

        for veh, speed, transform in ((vehicle1, v1_speed, v1_transform), (vehicle2, v2_speed, v2_transform)):
            v_ms = speed / 3.6
            yaw_rad = math.radians(transform.rotation.yaw)
            veh.set_target_velocity(carla.Vector3D(x=v_ms * math.cos(yaw_rad), y=v_ms * math.sin(yaw_rad), z=0))

        state = {"crash_tick": None, "ticks_since_crash": 0}

        def on_collision(event):
            if state["crash_tick"] is not None:
                return
            if event.other_actor.id == vehicle2.id:
                print("\n!!! CRASH Between Two Vehicles DETECTED !!!")
                state["crash_tick"] = -1  # marker: set on next tick index below

        collision_bp = blueprint_library.find('sensor.other.collision')
        col_sensor1 = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle1)
        col_sensor1.listen(on_collision)
        vehicles.append(col_sensor1)

        agent1 = BlindAgent(vehicle1, target_speed=v1_speed)
        agent1.set_global_plan(convert_points_to_global_plan(v1_route))
        agent2 = BlindAgent(vehicle2, target_speed=v2_speed)
        agent2.set_global_plan(convert_points_to_global_plan(v2_route))

        print("Starting simulation loop...")
        for i in range(MAX_TICK_COUNT):
            world.tick()

            if state["crash_tick"] == -1:
                state["crash_tick"] = i
                log["collision_tick"] = i

            log["vehicle1"].append(_pose_row(vehicle1))
            log["vehicle2"].append(_pose_row(vehicle2))

            if state["crash_tick"] is not None:
                control_stop = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
                vehicle1.apply_control(control_stop)
                vehicle2.apply_control(control_stop)
                state["ticks_since_crash"] += 1
                if state["ticks_since_crash"] > POST_COLLISION_TICKS:
                    log["status"] = "collided"
                    break
            else:
                if agent1.done() and agent2.done():
                    print("Both agents reached destination without colliding.")
                    log["status"] = "no_collision_route_completed"
                    break
                if not agent1.done():
                    vehicle1.apply_control(agent1.run_step())
                if not agent2.done():
                    vehicle2.apply_control(agent2.run_step())
        else:
            log["status"] = "no_collision_tick_cap_reached"

    except Exception as e:
        log["status"] = f"error: {e}"
        print(f"An error occurred: {e}")

    finally:
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass
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

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a TRACE benchmark scenario in real CARLA physics, logging per-tick state.")
    parser.add_argument("crash_id", type=int, nargs='?', help="The ID of the crash scenario to simulate.")
    args = parser.parse_args()
    if args.crash_id is None:
        print("Error: Crash ID is missing.")
        sys.exit(1)

    simulation_path = os.path.join(SIMULATION_DIR, f"simulation_{args.crash_id}.json")
    result = run_simulation(simulation_path, args.crash_id)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"replay_{args.crash_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"Wrote {out_path} (status={result['status']}, n_ticks={len(result['vehicle1'])}, collision_tick={result['collision_tick']})")
