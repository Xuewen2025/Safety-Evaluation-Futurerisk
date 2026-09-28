#!/bin/bash

echo "<< formatting xml reports."
python3.8 ns_format_xml_reports.py

echo ">> extracting summary if the number of vehicles involved in the crash is 2."
python3.8 ns_extract_crash_summary.py

echo ">> Checking summary if the information is complete."
python3.8 ns_check_summary.py

echo ">> Modifying the summary crash location to carla coordinates."
python3.8 ns_modify_summary.py

echo ">> Downloading OSM maps from OpenStreetMap."
python3.8 ns_extract_osm_maps.py

echo ">> Checking Elevation in the maps."
python3.8 ns_MV_elevation.py

echo ">> Creating osmium command."
python3.8 ns_create_osmium_command.py

CLIPPED_OSM_DIR=$(python3.8 -c "import sys, os; sys.path.append(os.getcwd()); from settings import CLIPPED_OSM_DIR; print(CLIPPED_OSM_DIR)")
mkdir "$CLIPPED_OSM_DIR"

OSMIUM_COMMAND_FILE=$(python3.8 -c "import sys, os; sys.path.append(os.getcwd()); from settings import OSMIUM_COMMAND_FILE; print(OSMIUM_COMMAND_FILE)")
echo ">> Clipping the maps"
grep "^osmium extract" "$OSMIUM_COMMAND_FILE" | while read -r line ; do echo "Running: $line"; $line ; done

echo ">> Editing OSM files."
python3.8 ns_edit_osm.py

echo ">> Converting to XODR."
python3.8 ns_osm_to_xodr.py

echo ">> Checking Proportion of the maps after conversion."
python3.8 ns_MV_scaling.py

echo ">> Reorganizing the XODR files."
python3.8 ns_reorganize_xodr.py

echo ">> Extracting road segments."
python3.8 ns_extract_road.py

echo ">> invoking LLM to get the initial positions of the vehicles."
python3.8 ns_invoke_LLM.py

