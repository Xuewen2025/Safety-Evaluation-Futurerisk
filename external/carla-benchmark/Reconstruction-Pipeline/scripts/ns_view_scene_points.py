import carla
import json
import sys
import argparse
import os
from settings import SCENE_POINTS_DIR

client = carla.Client('localhost', 2000)
world = client.get_world()

parser = argparse.ArgumentParser(description="Spawn vehicles for a specific crash scenario.")
    
parser.add_argument("crash_id", type=int, nargs='?', help="The ID of the crash scenario to simulate.")

args = parser.parse_args()

if args.crash_id is None:
    print("Error: Crash ID is missing. Please provide a crash ID (e.g., python script.py 510163).")
    sys.exit(1)

crash_id = args.crash_id

json_file_path = os.path.join(SCENE_POINTS_DIR, f"scenepoints_{crash_id}.json")

with open(json_file_path, 'r') as f:
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

# Optional: Print to verify
print(f"Crash Location: ({crash_x}, {crash_y})")
print(f"Vehicle 1: Road {veh1_road_id}, Pos ({veh1_x}, {veh1_y})")
print(f"Vehicle 2: Road {veh2_road_id}, Pos ({veh2_x}, {veh2_y})")


world.debug.draw_box(
    carla.BoundingBox(carla.Location(x=crash_x, y=crash_y, z=0.5), carla.Vector3D(0.2, 0.2, 5.0)),
    carla.Rotation(0, 0, 0),
    0.1,  # Line thickness
    carla.Color(255, 0, 0, 255), 
    120.0 # Life time in seconds
)
world.debug.draw_string(
    # Location: Use the same location as the box (or slightly offset it)
    location=carla.Location(x=crash_x, y=crash_y, z=0.5 + 5.5), 
    text=str(1), 
    draw_shadow=True, 
    color=carla.Color(255, 255, 255, 255), 
    life_time=120.0
)

world.debug.draw_box(
    carla.BoundingBox(carla.Location(x=veh1_x, y=veh1_y, z=0.5), carla.Vector3D(0.2, 0.2, 5.0)),
    carla.Rotation(0, 0, 0),
    0.1,  # Line thickness
    carla.Color(255, 0, 0, 255), 
    120.0 # Life time in seconds
)
world.debug.draw_string(
    # Location: Use the same location as the box (or slightly offset it)
    location=carla.Location(x=veh1_x, y=veh1_y, z=0.5 + 5.5), 
    text=str(2), 
    draw_shadow=True, 
    color=carla.Color(255, 255, 255, 255), 
    life_time=120.0
)

world.debug.draw_box(
    carla.BoundingBox(carla.Location(x=veh2_x, y=veh2_y, z=0.5), carla.Vector3D(0.2, 0.2, 5.0)),
    carla.Rotation(0, 0, 0),
    0.1,  # Line thickness
    carla.Color(255, 0, 0, 255), 
    120.0 # Life time in seconds
)
world.debug.draw_string(
    # Location: Use the same location as the box (or slightly offset it)
    location=carla.Location(x=veh2_x, y=veh2_y, z=0.5 + 5.5), 
    text=str(3), 
    draw_shadow=True, 
    color=carla.Color(255, 255, 255, 255), 
    life_time=120.0
)
