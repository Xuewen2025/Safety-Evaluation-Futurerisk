import os
import json
import re
from ns_ER_junction import RoadExtractionJunction
from ns_ER_nonjunction import RoadExtractionNonJunction
from settings import SEGMENTS_DIR
from settings import SEGMENT_BBOX_DIR
from settings import ROUTES_DIR
from settings import CLIPPED_MERGED_XODR_DIR
from settings import CONVERTED_POINTS_DIR
from settings import MODIFIED_SUMMARY_DIR

SEARCH_DIST_METER = 100.0

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

def ExtractRoad():
     
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    os.makedirs(SEGMENT_BBOX_DIR, exist_ok=True)
    os.makedirs(ROUTES_DIR, exist_ok=True)

    for filename in os.listdir(CLIPPED_MERGED_XODR_DIR):
        intersection = True
        if filename.endswith(".xodr"):
            try:
                print(f"Processing {filename}")
                xodr_path = os.path.join(CLIPPED_MERGED_XODR_DIR, filename)
                with open(xodr_path, 'r') as file:
                    xodr_content = file.read()

                crash_id = filename.split('_')[-1].replace('.xodr', '')
                converted_points_path = os.path.join(CONVERTED_POINTS_DIR, f"convertedpoints_{crash_id}.txt")
                with open(converted_points_path, "r") as file:
                    line = file.readline()
                    crash_point_x = float(line.split(",")[0].strip())
                    crash_point_y = float(line.split(",")[1].strip())

                summary_path = os.path.join(MODIFIED_SUMMARY_DIR, f"summary_{crash_id}.txt")
                with open(summary_path, "r", encoding='utf-8') as file:
                    lines = file.readlines()

                for line in lines:
                    if  "Collision Place" in line:
                        if line.split(":")[-1].strip().lower() == "not an intersection":
                            intersection = False

                if intersection:
                    for line in lines:
                        line = line.strip()
                        if "Vehicle 1:" in line:
                            current_vehicle = 1
                        elif "Vehicle 2:" in line:
                            current_vehicle = 2
                        
                        if "p_crash" in line.lower() or "pcrash" in line.lower():
                            direction = get_movement_info(line)
                            if direction is not None and current_vehicle == 1:
                                    v1_direction = direction 
                            if direction is not None and current_vehicle == 2:
                                    v2_direction = direction

                    vehicles_to_process = {
                        "V1": v1_direction,   
                        "V2": v2_direction     
                    }

                    checker = RoadExtractionJunction(xodr_content)
                
                    results = {}

                    for vehicle_id, description in vehicles_to_process.items():
                        
                        connected_road_segments = checker.check_point_on_road(crash_point_x, crash_point_y, SEARCH_DIST_METER, description)
                        
                        results[vehicle_id] = connected_road_segments

                    with open (os.path.join(SEGMENTS_DIR, f"segments_{crash_id}.txt"), "w") as file:
                        for vehicle_id, all_connected_road_segments in results.items():
                            for road in all_connected_road_segments:
                                segments = road['connected_road_segments']
                                for segment in segments:
                                    for x, y in segment:
                                        file.write(f"{x}, {y}\n")

                    with open (os.path.join(SEGMENTS_DIR, f"segments_junc_{crash_id}.txt"), "w") as file:
                        for vehicle_id, all_connected_road_segments in results.items():
                            for road in all_connected_road_segments:
                                for x,y in road['junction_road_segment']:
                                    file.write(f"{x}, {y}\n") 

                    output_data = []
                    seen_junction_road_ids = set()
                    seen_vehicle = set()
                    for vehicle_id, all_connected_road_segments in results.items():
                        for road in all_connected_road_segments:
                            current_id = road['junction_road_id']
                            if current_id not in seen_junction_road_ids and vehicle_id not in seen_vehicle:
                                route_entry = {
                                    "vehicle id": vehicle_id,   # have to fix this line
                                    "junction_id": road['junction_id'],
                                    "junction_road_id": current_id,
                                    "connected_road_id": road['connected_road_id']
                                }
                                output_data.append(route_entry)
                                seen_junction_road_ids.add(current_id)
                                seen_vehicle.add(vehicle_id)

                    with open (os.path.join(ROUTES_DIR, f"route_{crash_id}.json"), "w") as file:
                        json.dump(output_data, file, indent=4)

                    with open(os.path.join(SEGMENT_BBOX_DIR, f"bbox_{crash_id}.txt"), "w") as file:
                        seen_junction_road_ids = set()
                        seen_vehicle = set()
                        output_data = []

                        for vehicle_id, all_connected_road_segments in results.items():
                            for road in all_connected_road_segments:
                                current_id = road['junction_road_id']

                                if current_id not in seen_junction_road_ids and vehicle_id not in seen_vehicle:
                                    junction_bbox = [list(pt) for pt in road['junction_road_segment']]
                                    
                                    pred_road = road['connected_road_id']
                                    
                                    formatted_segments = []
                                    segments = road['connected_road_segments']
                                    
                                    for i, segment in enumerate(segments):
                                        formatted_segments.append({
                                            "bbox_id": i + 1,
                                            "coordinates": [list(pt) for pt in segment]
                                        })
                                
                                    junction_entry = {
                                        "vehicle id": vehicle_id,
                                        "junction_road_id": current_id,
                                        "junction_bounding_box": junction_bbox,
                                        "connected_road_id": pred_road,
                                        "connected_segments": formatted_segments
                                    }
                                    
                                    output_data.append(junction_entry)
                                    seen_junction_road_ids.add(current_id)
                                    seen_vehicle.add(vehicle_id)

                        json_str = json.dumps(output_data, indent=4)
                        compact_json_str = re.sub(
                            r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]', 
                            r'[\1, \2]', 
                            json_str
                        )

                        file.write(compact_json_str)

                    print("Everything written successfully.")

                else:
                    checker = RoadExtractionNonJunction(xodr_content)
                    road_segments = checker.check_point_on_road(crash_point_x, crash_point_y, SEARCH_DIST_METER)
                
                    with open (os.path.join(SEGMENTS_DIR, f"segments_{crash_id}.txt"), "w") as file:
                        segments = road_segments['road_segments']
                        for segment in segments:
                            left_segment = segment[0]
                            if left_segment:
                                for x, y in left_segment:
                                    file.write(f"{x}, {y}\n")
                            right_segment = segment[1]
                            if right_segment:
                                for x, y in right_segment:
                                    file.write(f"{x}, {y}\n")

                    with open(os.path.join(SEGMENT_BBOX_DIR, f"bbox_{crash_id}.txt"), "w") as file:
                        road_id = road_segments['road_id']
                        segments = road_segments['road_segments']
                        
                        data_structure = {
                            "road_id": road_id,
                            "bounding_boxes": []
                        }

                        for i, segment in enumerate(segments):
                            left_segment = segment[0]
                            right_segment = segment[1]
                            
                            bbox_data = {
                                "bbox_id": i + 1,
                                "left_lane_segment": [list(pt) for pt in left_segment] if left_segment is not None else [],
                                "right_lane_segment": [list(pt) for pt in right_segment] if right_segment is not None else []
                            }
                            
                            data_structure["bounding_boxes"].append(bbox_data)

                        json_str = json.dumps(data_structure, indent=4)

                        compact_json_str = re.sub(
                            r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]', 
                            r'[\1, \2]', 
                            json_str
                        )

                        file.write(compact_json_str)
                        
                    print("Everything written successfully.")

            except Exception as e:
                print('\nAn error has occurred in conversion.', e)



if __name__ == "__main__":
    ExtractRoad()
    