import xml.etree.ElementTree as ET
import math
import os
import shutil
from settings import MAPS_OSM_DIR
from settings import SUMMARY_DIR
from settings import ELEVANTION_PASSED_DIR
SEARCH_THRESHOLD = 100.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points 
    on the Earth's surface in meters.
    """
    R = 6371000  # Earth radius in meters
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def is_safe_elevation(osm_file_path, crash_lat, crash_lon, safety_threshold_meters):
    """
    Parses an OSM file and checks if any elevated roads exist within the 
    safety threshold distance of the crash point.
    
    Returns:
        (bool, str): (True, "Reason") if Safe (Keep Map)
                     (False, "Reason") if Unsafe (Discard Map)
    """
    try:
        tree = ET.parse(osm_file_path)
        root = tree.getroot()
        
        nodes = {}
        for node in root.findall('node'):
            n_id = node.get('id')
            try:
                lat = float(node.get('lat'))
                lon = float(node.get('lon'))
                nodes[n_id] = (lat, lon)
            except (ValueError, TypeError):
                continue

        for way in root.findall('way'):
            tags = {}
            for tag in way.findall('tag'):
                tags[tag.get('k')] = tag.get('v')

            is_elevated = False
            
            if 'layer' in tags:
                try:
                    if int(tags['layer']) != 0:
                        is_elevated = True
                except ValueError:
                    pass
            
            if 'bridge' in tags and tags['bridge'] not in ['no', 'false', '0']:
                is_elevated = True

            if is_elevated:
                road_nodes = way.findall('nd')
                
                for nd in road_nodes:
                    ref_id = nd.get('ref')
                    if ref_id in nodes:
                        node_lat, node_lon = nodes[ref_id]
                        
                        dist = haversine_distance(crash_lat, crash_lon, node_lat, node_lon)
                        
                        if dist < safety_threshold_meters:
                            way_id = way.get('id')
                            return False, f"Elevated road (ID: {way_id}) found {dist:.1f}m from crash point."

        return True, "Safe: No elevated roads found within threshold."

    except ET.ParseError:
        return False, "Error: Invalid XML format"
    except FileNotFoundError:
        return False, "Error: File not found"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_elevation():
    
    os.makedirs(ELEVANTION_PASSED_DIR, exist_ok=True)

    crash_lat = 0.0
    crash_lon = 0.0

    total_map_count = 0
    failed_elevation = 0
    copied_count = 0

    for filename in os.listdir(MAPS_OSM_DIR):
        if filename.endswith(".osm"):
            try:
                osm_path = os.path.join(MAPS_OSM_DIR, filename)

                crash_id = filename.split('_')[-1].replace('.osm', '')
                summary_path = os.path.join(SUMMARY_DIR, f"summary_{crash_id}.txt")
                with open(summary_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                for line in lines:
                    if "Latitude" in line:
                        crash_lat = float(line.split(":")[-1].strip())
                    if "Longitude" in line:
                        crash_lon = float(line.split(":")[-1].strip())
                

                is_safe, reason = is_safe_elevation(osm_path, crash_lat, crash_lon, SEARCH_THRESHOLD)
                
                if is_safe:
                    print(f"✅ Elevation {filename}: Passed.")
                    dest_path = os.path.join(ELEVANTION_PASSED_DIR, filename)
                    shutil.copy2(osm_path, dest_path)
                    copied_count += 1

                else:
                    failed_elevation += 1
                    print(f"❌ Elevation {filename}: Failed.")
                    print(f"Reason: {reason}")

                
            except Exception as e:
                print('\nAn error has occurred in conversion.', e)

            total_map_count += 1

    print(f"Total maps processed: {total_map_count}")
    print(f"Failed in elevation checking: {failed_elevation}")
    print(f"Passing {copied_count} osm maps for clipping.")

if __name__ == "__main__":

    check_elevation()