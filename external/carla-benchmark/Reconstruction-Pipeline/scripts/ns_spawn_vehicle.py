import os
import sys
import ns_SV_nonjunction
import ns_SV_junction
import argparse
from settings import CLIPPED_MERGED_XODR_DIR
from settings import SCENE_POINTS_DIR
from settings import MODIFIED_SUMMARY_DIR
from settings import SIMULATION_DIR
from settings import ROUTES_DIR
from settings import TRAJECTORY_DIR

def spawn_vehicle(crash_id):

    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    os.makedirs(SIMULATION_DIR, exist_ok=True)

    xodr_path = os.path.join(CLIPPED_MERGED_XODR_DIR, f"map_{crash_id}.xodr")
    points_path = os.path.join(SCENE_POINTS_DIR, f"scenepoints_{crash_id}.json")
    summary_path = os.path.join(MODIFIED_SUMMARY_DIR, f"summary_{crash_id}.txt")
    trajectory_path = os.path.join(TRAJECTORY_DIR, f"trajectory_{crash_id}.txt")
    simulation_path = os.path.join(SIMULATION_DIR, f"simulation_{crash_id}.json")

    with open(summary_path, "r") as file:
        lines = file.readlines()

    intersection =  True

    for line in lines:
        if  "Collision Place" in line:
            if line.split(":")[-1].strip().lower() == "not an intersection":
                intersection = False
    
    if intersection:
        route_path = os.path.join(ROUTES_DIR, f"route_{crash_id}.json")
        ns_SV_junction.spawn_vehicle(xodr_path, points_path, route_path, summary_path, trajectory_path, simulation_path)
    else:
        ns_SV_nonjunction.spawn_vehicle(xodr_path, points_path, summary_path, trajectory_path, simulation_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Spawn vehicles for a specific crash scenario.")
    
    parser.add_argument("crash_id", type=int, nargs='?', help="The ID of the crash scenario to simulate.")

    args = parser.parse_args()

    if args.crash_id is None:
        print("Error: Crash ID is missing. Please provide a crash ID (e.g., python script.py 510163).")
        sys.exit(1)
    else:
        spawn_vehicle(args.crash_id)
   
    