import pyproj
from typing import Tuple
import os
import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer
from settings import REAL_POINTS_DIR
from settings import CONVERTED_POINTS_DIR
from settings import CLIPPED_MERGED_XODR_DIR


class ProjectionMapper:

    def __init__(self, proj_string):
        # if not proj_string:
        #     raise ValueError("PROJ string cannot be empty. Please extract it from the XODR file.")
            
        # self.proj_wgs84 = pyproj.Proj(proj='latlong', ellps='GRS80')
        
        # self.proj_local = pyproj.Proj(proj_string)

        self.crs_4326 = CRS.from_epsg(4326)
        self.map_proj = CRS.from_proj4(proj_string)
        self.transformer = Transformer.from_crs(self.crs_4326, self.map_proj, always_xy=True)

    def latlon_to_carla(self, latitude, longitude):

        # projected_x, projected_y = pyproj.transform(self.proj_wgs84, self.proj_local, longitude, latitude)

        # carla_y = -projected_y
        # carla_x = projected_x
        
        # return carla_x, carla_y
    
        x, y = self.transformer.transform(longitude, latitude)
        return x, -y

def get_proj_string_from_xodr(xodr_path):
    """
    Parses the .xodr file to find the <geoReference> tag content.
    """
    try:
        tree = ET.parse(xodr_path)
        root = tree.getroot()
        
        header = root.find('header')
        if header is not None:
            georef = header.find('geoReference')
            if georef is not None and georef.text:
                return georef.text.strip()
    except Exception as e:
        print(f"Error reading XODR {xodr_path}: {e}")
    
    return None

if __name__ == '__main__':

    os.makedirs(CONVERTED_POINTS_DIR, exist_ok=True)

    for filename in os.listdir(REAL_POINTS_DIR):
        if filename.endswith(".txt"):
            try:
                crash_id = filename.split('_')[-1].replace('.txt', '')
                real_points_path = os.path.join(REAL_POINTS_DIR, filename)
                
                xodr_path = os.path.join(CLIPPED_MERGED_XODR_DIR, f"map_{crash_id}.xodr")
                
                if not os.path.exists(xodr_path):
                    print(f"SKIPPING {filename}: No matching .xodr map found for ID {crash_id}")
                    continue
                proj_string = get_proj_string_from_xodr(xodr_path)
                if not proj_string:
                    print(f"SKIPPING {filename}: Could not extract geoReference from {xodr_path}")
                    continue

                mapper = ProjectionMapper(proj_string)

                # 5. Process the points
                converted_filename = filename.replace('real_points', 'converted_points')
                converted_path = os.path.join(CONVERTED_POINTS_DIR, converted_filename)

                print(f"Processing {filename} using map {os.path.basename(xodr_path)}...")
                
                with open(real_points_path, 'r') as f_in, open(converted_path, 'w') as f_out:
                    for line in f_in:
                        lat_str, lon_str = line.strip().split(',')

                        lat = float(lat_str)
                        lon = float(lon_str)

                        carla_x, carla_y = mapper.latlon_to_carla(lat, lon)
                        
                        f_out.write(f"{carla_x:.3f}, {carla_y:.3f}\n")

            except Exception as e:
                print(f"Error processing file {real_points_path}: {e}")
    