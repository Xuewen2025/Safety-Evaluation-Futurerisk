import requests
import sys
import os
import math
import argparse
import time
from settings import SUMMARY_DIR
from settings import MAPS_OSM_DIR
from settings import MAP_BBOX_DIR

RADIUS_METER = 300  # Radius in meters

# Earth's radius in kilometers
# We use this to convert distance (km) into degrees of latitude/longitude.
EARTH_RADIUS_KM = 6371.0

def calculate_bbox(center_lat, center_lon, radius_meter):
    """
    Calculates the bounding box (N, S, E, W) in degrees 
    around a center point for a given radius.
    
    Args:
        center_lat (float): Latitude of the center point.
        center_lon (float): Longitude of the center point.
        radius_km (float): The distance (radius) from the center in kilometers.
        
    Returns:
        tuple: (north, south, west, east) coordinates for the bounding box.
    """
    # 1 degree of latitude is approximately 111 km.
    # The latitude change is mostly constant.
    radius_km = radius_meter/ 1000.0
    lat_delta = radius_km / 111.0

    north = center_lat + lat_delta
    south = center_lat - lat_delta

    # 1 degree of longitude is approximately 111 km * cos(latitude).
    # We use the center latitude for the approximation.
    # Convert latitude to radians for the cos function.
    cos_lat = math.cos(math.radians(center_lat))
    
    # Handle division by zero at the poles (though unlikely for typical maps)
    if cos_lat == 0:
        lon_delta = 360.0 # If at the pole, any longitude works, but Overpass requires a box.
    else:
        lon_delta = radius_km / (111.0 * cos_lat)

    west = center_lon - lon_delta
    east = center_lon + lon_delta
    
    return (north, south, west, east)


def extract_osm_from_coords(north, south, west, east):
    """
    Downloads OpenStreetMap data using the Overpass API for a given bounding box.
    (This function remains largely the same, just accepting the calculated BBOX)
    """
    try:
        # Define the Overpass API endpoint.
        overpass_url = "https://overpass.kumi.systems/api/interpreter"

        # The query to send to the Overpass API. It specifies the bounding box
        # and asks for all nodes, ways, and relations within that box.
        overpass_query = f"""
            [out:xml][timeout:1500];
            (
              node({south},{west},{north},{east});
              way({south},{west},{north},{east});
              relation({south},{west},{north},{east});
            );
            (._;>;);
            out;
        """

        # Print the query to be sent for debugging purposes.
        print("\n--- Sending Query, Please Wait ---")
        # print("Bounding Box: N={:.6f}, S={:.6f}, E={:.6f}, W={:.6f}".format(north, south, east, west))

        # Send the POST request to the API with the query.
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Check if the response content is not empty.
        if not response.content:
            print("Error: The API returned an empty response.")
            return

        else:
            return response.content

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while connecting to the Overpass API: {e}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():

    if not os.path.isdir(SUMMARY_DIR):
        print(f"Error: Directory '{SUMMARY_DIR}' not found.")
        sys.exit(1)
    
    os.makedirs(MAPS_OSM_DIR, exist_ok=True)
    os.makedirs(MAP_BBOX_DIR, exist_ok=True)

    files_downloaded = 0

    for filename in os.listdir(SUMMARY_DIR):
        try:
            print(f"Downloading Map for {filename}...")

            crash_number = filename.split("_")[1].split(".")[0]

            with open(os.path.join(SUMMARY_DIR, filename)) as file:
                lines = file.readlines()

                for line in lines:
                    if "Latitude" in line:
                        center_lat = float(line.split(":")[1].strip())
                    if "Longitude" in line:
                        center_lon = float(line.split(":")[1].strip())

            
            # Calculate the Bounding Box
            north, south, west, east = calculate_bbox(center_lat, center_lon, RADIUS_METER)
            with open(os.path.join(MAP_BBOX_DIR, f"bbox_{crash_number}.txt"), "w") as bbox_file:
                bbox_file.write(f"max lat: {north}\n")
                bbox_file.write(f"max long: {east}\n")
                bbox_file.write(f"min lat: {south}\n")
                bbox_file.write(f"min long: {west}\n")
            
            # Download the map
            map_content = extract_osm_from_coords(north, south, west, east)

            # Write the content to the specified output file.
            output_file = os.path.join(MAPS_OSM_DIR, "map_" + str(crash_number) + ".osm")
            
            if map_content:
                with open(output_file, "wb") as f:
                    f.write(map_content)
        
                print(f"Successfully downloaded OSM data to '{output_file}'")

                files_downloaded += 1

        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(10)

    print("Maps Downloaded: ", files_downloaded)

if __name__ == "__main__":

    main()