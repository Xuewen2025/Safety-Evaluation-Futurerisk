import numpy as np
from pyproj import CRS, Transformer
import xml.etree.ElementTree as ET
import argparse
import os
from settings import CLIPPED_OSM_DIR
from settings import EDITED_OSM_DIR
from settings import SUMMARY_DIR
from settings import CLIPPED_HEADERS_DIR

regionSpecificScaleFactor = 1.0

def modify_file(osm_input_path, summary_path, osm_edited_path, header_path):

    with open(summary_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if "Latitude" in line:
            fixed_center_latitude = float(line.split(" ")[-1].strip())
        if "Longitude" in line:
            fixed_center_longitude = float(line.split(" ")[-1].strip())

    tree = ET.parse(osm_input_path)
    root = tree.getroot()

    # print(f"Processing map with FIXED ANCHOR: {fixed_center_latitude}, {fixed_center_longitude}")

    crs_4326  = CRS.from_epsg(4326) # WGS84
    uproj_string = "+proj=tmerc +lat_0={0} +lon_0={1} +x_0=0 +y_0=0 +k_0={2} +ellps=GRS80 +units=m".format(
        fixed_center_latitude, fixed_center_longitude, regionSpecificScaleFactor
    )
    uproj = CRS.from_proj4(uproj_string)
    transformer = Transformer.from_crs(crs_4326, uproj)
    
    for entity in root:
        if entity.tag == "node":
            real_lat = float(entity.attrib["lat"])
            real_lon = float(entity.attrib["lon"])
            
            realX, realY = next(transformer.itransform([(real_lat, real_lon)]))
            
            fakeLat = realY / 111136.
            fakeLon = realX / (111320. * np.cos(fakeLat * 2.0 * np.pi / 360.))
            
            entity.attrib["lat"] = str(fakeLat)
            entity.attrib["lon"] = str(fakeLon)

    with open(osm_edited_path, 'w') as f:
        tree.write(f, encoding='unicode')

    print(f"Done. OSM map written to {osm_edited_path}")

    north = 5000
    south = -5000
    east = 5000
    west = -5000

    stringToPutAsODriveHeader = """<header revMajor="1" revMinor="4" name="" version="1" date="2019-02-18T13:36:12" north="{0}" south="{1}" east="{2}" west="{3}">
        <geoReference><![CDATA[+proj=tmerc +lat_0={4} +lon_0={5} +x_0=0 +y_0=0 +k_0={6} +ellps=GRS80 +units=m +no_defs]]></geoReference>
        </header>""".format(north, abs(south), east, abs(west), fixed_center_latitude, fixed_center_longitude, regionSpecificScaleFactor)

    with open(header_path, 'w') as f:
        f.write(stringToPutAsODriveHeader)
    
    print(f"Done. Header written to {header_path}")


def main():
    os.makedirs(EDITED_OSM_DIR, exist_ok=True)
    os.makedirs(CLIPPED_HEADERS_DIR, exist_ok=True)
    edited_count = 0

    for filename in os.listdir(CLIPPED_OSM_DIR):
        if filename.endswith(".osm"):
            try:
                osm_input_path = os.path.join(CLIPPED_OSM_DIR, filename)

                summary_file = filename.replace("map", "summary")
                summary_file = summary_file.replace("osm", "txt")
                summary_path = os.path.join(SUMMARY_DIR, summary_file)

                osm_edited_file = filename.replace(".osm", "_edited.osm")
                osm_edited_path = os.path.join(EDITED_OSM_DIR, osm_edited_file)

                header_file = filename.replace(".osm", "_header.txt")
                header_path = os.path.join(CLIPPED_HEADERS_DIR, header_file)
            
                modify_file(osm_input_path, summary_path, osm_edited_path, header_path)
                edited_count += 1
    
            except:
                print(f'\nAn error has occurred in conversion for file {filename}.')
    print(f"Edited {edited_count} files.")

if __name__ == "__main__":
    main()

    