""" Convert OpenStreetMap file to OpenDRIVE file. """

import argparse
import glob
import os
import sys
from settings import EDITED_OSM_DIR
from settings import CLIPPED_HEADERS_DIR
from settings import CLIPPED_XODR_DIR

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def convert(args, osm_file, xodr_file, header_file):
    # Read the .osm data
    with open(osm_file, mode="r", encoding="utf-8") as osmFile:
        osm_data = osmFile.read()

    # Define the desired settings
    settings = carla.Osm2OdrSettings()

    # Set OSM road types to export to OpenDRIVE
    settings.set_osm_way_types([
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential"
    ])
    settings.default_lane_width = args.lane_width
    settings.generate_traffic_lights = True
    settings.all_junctions_with_traffic_lights = args.all_junctions_lights
    settings.center_map = args.center_map

    # Convert to .xodr
    xodr_data = carla.Osm2Odr.convert(osm_data, settings)

    with open(header_file, "r", encoding="utf-8") as hfile:
        header = hfile.read()

    start_index = xodr_data.find("<header")
    end_index = xodr_data.find("</header>") + len("</header>")
 
    if start_index != -1 and end_index != -1:
        new_xodr_data = xodr_data[:start_index] + header + xodr_data[end_index:]   

    with open(xodr_file, "w", encoding="utf-8") as xodrFile:
        xodrFile.write(new_xodr_data)
        print(f"content written to {xodrFile}")


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description="Spawn vehicles for a specific crash scenario.")
    argparser.add_argument(
        '--lane-width',
        default=6.0,
        help='width of each road lane in meters')
    argparser.add_argument(
        '--traffic-lights',
        action='store_true',
        help='enable traffic light generation from OSM data')
    argparser.add_argument(
        '--all-junctions-lights',
        action='store_true',
        help='set traffic lights for all junctions')
    argparser.add_argument(
        '--center-map',
        action='store_true',
        help='set center of map to the origin coordinates')


    args = argparser.parse_args()

    os.makedirs(CLIPPED_XODR_DIR, exist_ok=True)

    converted_files = 0

    for filename in os.listdir(EDITED_OSM_DIR):
        if filename.endswith(".osm"):
            try:
                crash_id = filename.split("_")[1]
                print(f"{crash_id} started conversion\n")
                osm_path = os.path.join(EDITED_OSM_DIR, filename)
                xodr_path = os.path.join(CLIPPED_XODR_DIR, f"map_{crash_id}.xodr")
                header_path = os.path.join(CLIPPED_HEADERS_DIR, f"map_{crash_id}_header.txt")
                convert(args, osm_path, xodr_path, header_path)
                
                converted_files += 1
            except Exception as e:
                print('\nAn error has occurred in conversion.', e)

    print("Converted files: ", converted_files)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)
