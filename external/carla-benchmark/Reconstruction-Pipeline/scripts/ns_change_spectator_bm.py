import carla
import argparse
import sys
import os


client = carla.Client('localhost', 2000)
world = client.get_world()

parser = argparse.ArgumentParser(description="Spawn vehicles for a specific crash scenario.")
    
parser.add_argument("crash_id", type=int, nargs='?', help="The ID of the crash scenario to simulate.")

args = parser.parse_args()

if args.crash_id is None:
    print("Error: Crash ID is missing. Please provide a crash ID (e.g., python script.py 510163).")
    sys.exit(1)

crash_id = args.crash_id

carla_x = 0.0
carla_y = 0.0

# Teleport the spectator
spectator = world.get_spectator()

spectator_transform = carla.Transform(
    # 1. Set Z to 500 meters
    carla.Location(x=carla_x, y=carla_y, z=150.0), 
    
    # 2. Set Pitch to -90 to look straight down
    carla.Rotation(pitch=-90, yaw=0, roll=0) 
)

spectator.set_transform(spectator_transform)