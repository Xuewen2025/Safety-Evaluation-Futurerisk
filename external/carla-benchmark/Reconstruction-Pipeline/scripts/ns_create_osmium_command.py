import os

from settings import ELEVANTION_PASSED_DIR
from settings import MAP_BBOX_DIR
from settings import OSMIUM_COMMAND_FILE
from settings import MAPS_OSM_DIR
from settings import CLIPPED_OSM_DIR

with open(OSMIUM_COMMAND_FILE, "w", encoding="UTF-8") as c_file:
    file_count = 0
    for filename in os.listdir(ELEVANTION_PASSED_DIR):
        try:
            print(f"Processing {filename}.")
            crash_id = filename.split('_')[1].split(".")[0]
            with open(os.path.join(MAP_BBOX_DIR, f"bbox_{crash_id}.txt")) as file:
                    max_lat = file.readline().split(':')[1].strip()
                    max_lon = file.readline().split(':')[1].strip()
                    min_lat = file.readline().split(':')[1].strip()
                    min_lon = file.readline().split(':')[1].strip()

                    command = "osmium extract -b " + min_lon + "," + min_lat + "," + max_lon + "," + max_lat + " "
                    command +=  os.path.join(MAPS_OSM_DIR, f"map_{crash_id}.osm") + " -o " + os.path.join(CLIPPED_OSM_DIR, f"map_{crash_id}.osm")
                    c_file.write(crash_id)
                    c_file.write("\n")
                    c_file.write(command)
                    c_file.write("\n\n")
            file_count += 1
        except Exception as e:
             print(e)
print(f"Processed {file_count} files.")