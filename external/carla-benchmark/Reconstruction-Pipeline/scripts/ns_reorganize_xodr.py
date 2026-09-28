import xml.etree.ElementTree as ET
import xml.dom.minidom
import sys
import argparse
import re
import os
import glob
from settings import SCALING_PASSED_DIR
from settings import CLIPPED_MERGED_XODR_DIR

class OpenDRIVEModifier:
    """
    Handles parsing, merging, and writing OpenDRIVE road networks.
    Specifically designed to merge a dual-carriageway setup (where two separate 
    roads share a centerline) into a single, two-way road (Host).
    """

    def __init__(self, xml_data):
        self.tree = ET.ElementTree(ET.fromstring(xml_data))
        self.root = self.tree.getroot()

        self.road_map = {}

        geo_ref_element = self.root.find('./header/geoReference')
        if geo_ref_element is not None and geo_ref_element.text is not None:
             self.geo_reference_content = geo_ref_element.text.strip()
             geo_ref_element.text = None 
        else:
             self.geo_reference_content = None

    def _prettify_xml(self, element):
        """Returns a pretty-printed XML string for a given element."""
        rough_string = ET.tostring(element, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="    ")

    def _get_road_element(self, road_id):
        """Finds a road element by its ID."""
        return self.root.find(f"./road[@id='{road_id}']")
    
    def update_lane_change_permissions(self):
        """
        Iterates through EVERY lane in the entire map.
        Finds the <roadMark> element (or creates it if missing) and sets
        the 'laneChange' attribute to 'both'.
        """
        
        all_lanes = self.root.findall(".//lane")
        
        for lane in all_lanes:
            lane_id = lane.get('id')
            
            road_marks = lane.findall('roadMark')
            
            if road_marks:
                for rm in road_marks:
                    rm.set('laneChange', 'both')

    def merge_roads(self, host_id, donor_id):
        """
            host_id (str): The ID of the road element that receives the new lanes (e.g., '170').
            donor_id (str): The ID of the road element that contains the lanes to be moved (e.g., '171').
        """
        host_road = self._get_road_element(host_id)
        donor_road = self._get_road_element(donor_id)

        if not host_road or not donor_road:
            print(f"Error: Could not find road elements for Host ID {host_id} or Donor ID {donor_id}.")
            return False

        if donor_road not in self.root.findall('road'):
             print(f"Skipping merge: Donor road {donor_id} has already been removed.")
             return False

        host_lanes_section = host_road.find('./lanes/laneSection')
        if host_lanes_section is None:
             print("Error: Host road is missing a <laneSection>.")
             return False

        donor_right_lanes = donor_road.findall('./lanes/laneSection/right/lane')
        if not donor_right_lanes:
            print("  Warning: Donor road has no right lanes defined to migrate.")
        
        host_left_element = host_lanes_section.find('left')
        if host_left_element is None:
            host_left_element = ET.SubElement(host_lanes_section, 'left')
        
        for lane_to_copy in donor_right_lanes:
            new_lane = ET.fromstring(ET.tostring(lane_to_copy))
            
            old_id = int(new_lane.get('id'))
            new_id = str(abs(old_id))
            new_lane.set('id', new_id)

            host_left_element.append(new_lane)

        host_center_element = host_lanes_section.find('center')
        host_right_element = host_lanes_section.find('right')

        host_lanes_section[:] = [] 

        if host_left_element is not None:
             host_lanes_section.append(host_left_element)
        if host_center_element is not None:
             host_lanes_section.append(host_center_element)
        if host_right_element is not None:
             host_lanes_section.append(host_right_element)

        self.road_map[donor_id] = host_id

        self.root.remove(donor_road)

        return True

    def update_removed_road_references(self):
        """
        Updates all junction road and lane references from Donor IDs to Host IDs,
        and flips the lane ID sign where necessary to point to the new Host Left Lane.
        This function handles both global road links and junction connections.
        """
        if not self.road_map:
            return

        
        for road in self.root.findall('road'):
            road_id = road.get('id')
            is_junction_road = road.get('junction') != '-1'

            link = road.find('link')
            if link is not None:
                
                predecessor_was_donor = False
                successor_was_donor = False
                
                for link_tag in link.findall('*'): 
                    element_id = link_tag.get('elementId')
                    
                    if element_id in self.road_map:
                        
                        link_tag.set('elementId', self.road_map[element_id])
                        
                        if link_tag.tag == 'predecessor':
                            predecessor_was_donor = True
                        elif link_tag.tag == 'successor':
                            successor_was_donor = True

                if is_junction_road and (predecessor_was_donor or successor_was_donor):
                    lane_sections = road.findall('./lanes/laneSection')
                    for section in lane_sections:
                        for lane_side in section.findall('right'): 
                            for lane in lane_side.findall('lane'):
                                internal_link = lane.find('link')
                                if internal_link is not None:
                                    if predecessor_was_donor:
                                        pred_link = internal_link.find('predecessor')
                                        if pred_link is not None:
                                            old_id = pred_link.get('id')
                                            new_id = str(abs(int(old_id))) 
                                            pred_link.set('id', new_id)
                                           
                                    if successor_was_donor:
                                        succ_link = internal_link.find('successor')
                                        if succ_link is not None:
                                            old_id = succ_link.get('id')
                                            new_id = str(abs(int(old_id)))
                                            succ_link.set('id', new_id)
                                           
        junction_elements = self.root.findall('junction')
        if not junction_elements:
             return
        
        for junction in junction_elements:
            junction_id = junction.get('id')
            
            for connection in junction.findall('connection'):
                
                incoming_id = connection.get('incomingRoad')
                incoming_was_donor = False
                if incoming_id in self.road_map:
                    connection.set('incomingRoad', self.road_map[incoming_id])
                    incoming_was_donor = True

                connecting_id = connection.get('connectingRoad')
                connecting_was_donor = False
                if connecting_id in self.road_map:
                    connection.set('connectingRoad', self.road_map[connecting_id])
                    connecting_was_donor = True

                for lane_link in connection.findall('laneLink'):
                    
                    if incoming_was_donor:
                        from_id = lane_link.get('from')
                        if from_id:
                            new_from_id = str(abs(int(from_id)))
                            lane_link.set('from', new_from_id)

                    if connecting_was_donor:
                        to_id = lane_link.get('to')
                        if to_id:
                            new_to_id = str(abs(int(to_id)))
                            lane_link.set('to', new_to_id)


    def save_to_file(self, output_filepath):
        """Saves the modified XML tree to a file with proper formatting."""
        
        xml_string = self._prettify_xml(self.root)
        
        if xml_string.count("<?xml version=") > 1:
            xml_string = xml_string.replace("<?xml version=\"1.0\" ?>\n", "", 1)

        if self.geo_reference_content is not None:
            old_tag = f"<geoReference></geoReference>" 

            old_tag_structure = re.search(r'<geoReference\s*/>|<geoReference\s*>\s*</geoReference>', xml_string)
            if old_tag_structure:
                 old_tag_str = old_tag_structure.group(0)
                 new_cdata_tag = f"<geoReference><![CDATA[{self.geo_reference_content}]]></geoReference>"
                 xml_string = xml_string.replace(old_tag_str, new_cdata_tag)

        lines = xml_string.split('\n')
        cleaned_lines = [line for line in lines if line.strip() != '']
        xml_string = '\n'.join(cleaned_lines)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        print(f"Modified OpenDRIVE map saved to: {output_filepath}")

def reorganize_xodr():

    os.makedirs(CLIPPED_MERGED_XODR_DIR, exist_ok=True)

    if not os.path.isdir(SCALING_PASSED_DIR):
        print(f"ERROR: Python cannot find the directory: {os.path.abspath(SCALING_PASSED_DIR)}")
        print("Check your relative path or use an absolute path.")
        sys.exit(1)

    LENGTH_TOLERANCE = 0.01

    files_to_process = glob.glob(os.path.join(SCALING_PASSED_DIR, '*.xodr'))

    if not files_to_process:
        print(f"WARNING: Directory found, but NO .xodr files inside: {SCALING_PASSED_DIR}")
        print(f"Looking for: {os.path.join(SCALING_PASSED_DIR, '*.xodr')}")
    else:
        print(f"Found {len(files_to_process)} xodr files.")
    
    total_files_processed = 0

    for filename in os.listdir(SCALING_PASSED_DIR):
        if filename.endswith(".xodr"):
            
            input_filepath = os.path.join(SCALING_PASSED_DIR, filename)
            output_filepath = os.path.join(CLIPPED_MERGED_XODR_DIR, filename)
            
            print(f"\n--- Processing File: {filename} ---")
        
            try:
                with open(input_filepath, 'r') as f:
                    xodr_content = f.read()

                modifier = OpenDRIVEModifier(xodr_content)
                
                road_data = []
                for road in modifier.root.findall('road'):
                    if road.get('junction') == '-1':
                        try:
                            road_data.append({
                                'id': road.get('id'),
                                'name': road.get('name'),
                                'length': float(road.get('length')),
                                'used': False
                            })
                        except (TypeError, ValueError):
                            print(f"Warning: Road ID {road.get('id')} is missing a valid 'length' attribute. Skipping.")
                
                merge_tasks = []
                
                for i in range(len(road_data)):
                    host = road_data[i]
                    if host['used']:
                        continue

                    for j in range(i + 1, len(road_data)):
                        donor = road_data[j]
                        
                        if not donor['used'] and donor['name'] == host['name'] and abs(host['length'] - donor['length']) < LENGTH_TOLERANCE:
                            merge_tasks.append((host['id'], donor['id']))
                            host['used'] = True
                            donor['used'] = True
                            break
                
                success_count = 0
                
                if not merge_tasks:
                    print("No suitable road pairs found for length-based merging.")
                
                for host_id, donor_id in merge_tasks:
                    if modifier.merge_roads(host_id, donor_id):
                        success_count += 1

                if success_count > 0:
                    modifier.update_removed_road_references()

                modifier.update_lane_change_permissions()

                modifier.save_to_file(output_filepath)
                total_files_processed += 1

            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                sys.exit(1)


    print(f"\n=============================================")
    print(f"Files Processed: {total_files_processed}")
    print(f"Output Directory: {CLIPPED_MERGED_XODR_DIR}")
    print(f"=============================================")



if __name__ == "__main__":

    reorganize_xodr()