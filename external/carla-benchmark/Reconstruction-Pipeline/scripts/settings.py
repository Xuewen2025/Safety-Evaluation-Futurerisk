import os


# secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# directories
REPORTS_DIR = "data/crashes/reports"
SUMMARY_DIR = "data/crashes/summary"
MODIFIED_SUMMARY_DIR = "data/crashes/modified_summary"

#osmium_command
OSMIUM_COMMAND_FILE = "data/maps/command_osmium.txt"

# maps
MAPS_OSM_DIR            = "data/maps/osm"
MAP_BBOX_DIR            = "data/maps/map_bbox"
ELEVANTION_PASSED_DIR   = "data/maps/mv_elevation_passed"
CLIPPED_OSM_DIR         = "data/maps/clipped_osm"
EDITED_OSM_DIR          = "data/maps/clipped_edited_osm"
CLIPPED_HEADERS_DIR     = "data/maps/clipped_headers"
CLIPPED_XODR_DIR        = "data/maps/clipped_xodr"
SCALING_PASSED_DIR      = "data/maps/mv_scaling_passed"
CLIPPED_MERGED_XODR_DIR = "data/maps/clipped_merged_xodr"

# points
REAL_POINTS_DIR      = "data/points/real_points"
CONVERTED_POINTS_DIR = "data/points/converted_points"
SEGMENTS_DIR         = "data/points/segments"
ROUTES_DIR           = "data/points/routes"
SEGMENT_BBOX_DIR     = "data/points/segment_bbox"
SCENE_POINTS_DIR     = "data/points/scene_points"
TRAJECTORY_DIR       = "data/points/trajectory"

# llm
PROMPTS_DIR     = "data/llm/prompts"
REASONINGS_DIR  = "data/llm/reasoning"

# output
SIMULATION_DIR = "data/simulations"