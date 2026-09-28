import math
import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer, Geod
from ns_geo_to_carla import ProjectionMapper, get_proj_string_from_xodr
from ns_check_points import check_and_get_direction
import os
import shutil
from settings import CLIPPED_XODR_DIR
from settings import SUMMARY_DIR
from settings import SCALING_PASSED_DIR
from settings import REAL_POINTS_DIR
from settings import CONVERTED_POINTS_DIR


def verify_map_scale(xodr_path, real_path, converted_path, center_lat, center_lon, actual_dist_meters):
    """
    Verifies if the OpenDRIVE map projection preserves real-world distances
    using the ProjectionMapper class.
    """
    
    if not os.path.exists(xodr_path):
        print(f"Error: File not found: {xodr_path}")
        return

    proj_string = get_proj_string_from_xodr(xodr_path)
    if not proj_string:
        print("Error: Could not extract geoReference string from XODR.")
        return

    mapper = ProjectionMapper(proj_string)
    geod = Geod(ellps='WGS84')
    
    lon_n, lat_n, _ = geod.fwd(center_lon, center_lat, 0, actual_dist_meters)
    lon_e, lat_e, _ = geod.fwd(center_lon, center_lat, 90, actual_dist_meters)
    lon_s, lat_s, _ = geod.fwd(center_lon, center_lat, 180, actual_dist_meters)
    lon_w, lat_w, _ = geod.fwd(center_lon, center_lat, 270, actual_dist_meters)

    with open(real_path, "w", encoding="utf-8") as file:
        file.write(f"{center_lat}, {center_lon}\n")
        file.write(f"{lat_n}, {lon_n}\n")
        file.write(f"{lat_e}, {lon_e}\n")
        file.write(f"{lat_s}, {lon_s}\n")
        file.write(f"{lat_w}, {lon_w}\n")

    test_points = {
        "North": (lat_n, lon_n),
        "East":  (lat_e, lon_e),
        "South": (lat_s, lon_s),
        "West":  (lat_w, lon_w)
    }

    center_x, center_y = mapper.latlon_to_carla(center_lat, center_lon)

    max_error = 0.0

    file = open(converted_path, "w")
    file.write(f"{center_x}, {center_y}\n")

    for direction, (lat, lon) in test_points.items():
        pt_x, pt_y = mapper.latlon_to_carla(lat, lon)
        file.write(f"{pt_x}, {pt_y}\n")
        
        carla_distance = math.sqrt((pt_x - center_x)**2 + (pt_y - center_y)**2)
        error = abs(carla_distance - actual_dist_meters)
        error_percent = (error / actual_dist_meters) * 100
        if error > max_error:
            max_error = error

    file.close()

    if max_error >= 0.5:
        return max_error, f"Actual Distance: {actual_dist_meters}. Found Distance: {carla_distance}"
    
    return max_error, f"Actual Distance: {actual_dist_meters}. Found Distance: {carla_distance}"

def verify_CP_on_road(converted_path, xodr_path):

    with open(converted_path, "r", encoding="utf-8") as file:
        line = file.readline()
        crash_x = float(line.split(",")[0].strip())
        crash_y = float(line.split(",")[1].strip())

    crash_P = (crash_x, crash_y)
    on_road, _, _, distance, _ = check_and_get_direction(crash_P, None, xodr_path, snap = False)

    if not on_road:
        return on_road, f"Crash Point is {distance} meter off the road."

    return on_road, f"Crash Point is on the road."


def check_alignment_and_on_road():

    ACTUAL_DISTANCE_METERS = 100
    
    
    os.makedirs(SCALING_PASSED_DIR, exist_ok=True)
    os.makedirs(REAL_POINTS_DIR, exist_ok=True)
    os.makedirs(CONVERTED_POINTS_DIR, exist_ok=True)

    crash_lat = 0.0
    crash_lon = 0.0

    total_map_processed = 0
    failed_scaling = 0
    failed_CP_off_road = 0
    copied_count = 0
    
    for filename in os.listdir(CLIPPED_XODR_DIR):
        if filename.endswith(".xodr"):
            try:
                xodr_path = os.path.join(CLIPPED_XODR_DIR, filename)

                crash_id = filename.split('_')[-1].replace('.xodr', '')

                summary_path = os.path.join(SUMMARY_DIR, f"summary_{crash_id}.txt")
                with open(summary_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                for line in lines:
                    if "Latitude" in line:
                        crash_lat = float(line.split(":")[-1].strip())
                    if "Longitude" in line:
                        crash_lon = float(line.split(":")[-1].strip())


                realpoints_path = os.path.join(REAL_POINTS_DIR, f"realpoints_{crash_id}.txt")
                convertedpoints_path = os.path.join(CONVERTED_POINTS_DIR, f"convertedpoints_{crash_id}.txt")
                
                error, reason = verify_map_scale(xodr_path, realpoints_path, convertedpoints_path, crash_lat, crash_lon, ACTUAL_DISTANCE_METERS)

                if error < 0.5: 
                    print(f"✅ Map Scaling {filename}: Passed")
                    on_road, reason = verify_CP_on_road(convertedpoints_path, xodr_path)

                    if on_road:
                        print(f"✅ Crash Point on the road. {filename}: Passed")
                        dest_path = os.path.join(SCALING_PASSED_DIR, filename)
                        shutil.copy2(xodr_path, dest_path)
                        copied_count += 1
                    else:
                        failed_CP_off_road += 1
                        print(f"❌ Crash Point off the road. {filename}: Failed")
                        print(f"Reason: {reason}")
                else:
                    failed_scaling += 1
                    print(f"❌ Map Scaling {filename}: Failed")
                    print(f"Reason: {reason}")

                total_map_processed += 1

            except Exception as e:
                print('\nAn error has occurred in conversion.', e)


    print(f"Total Map Processed: {total_map_processed}")
    print(f"Failed in scaling checking: {failed_scaling}")
    print(f"Failed by CP off the road: {failed_CP_off_road}")
    print(f"Passing {copied_count} xodr maps for merging.")

if __name__ == "__main__":
    
    check_alignment_and_on_road()
    