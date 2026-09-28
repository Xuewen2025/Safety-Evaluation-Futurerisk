import xml.etree.ElementTree as ET
import os
from settings import REPORTS_DIR
from settings import SUMMARY_DIR

def extract_crash_data(xml_file_path):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Find the main crash data section. The outer elements (Results, ArrayOfCaseDetail,
        # CaseDetail) do not have a namespace prefix in the provided XML.
        case_detail = root.find('Results/ArrayOfCaseDetail/CaseDetail')
        if case_detail is None:
            print("Error: Could not find CaseDetail in the XML.")
            return None

        # Find namespaced inner elements using the full URI for robustness.
        crash_result_set = case_detail.find('CrashResultSet')
        if crash_result_set is None:
            print("Error: Could not find CrashResultSet in the XML.")
            return None
        
        #date and time
        date_elem = crash_result_set.find('DAYNAME')
        date = date_elem.text if date_elem is not None else "N/A"
        month_elem = crash_result_set.find('MonthName')
        month = month_elem.text if month_elem is not None else "N/A"
        hour_elem = crash_result_set.find('HOUR')
        hour = hour_elem.text if hour_elem is not None else "N/A"
        minute_elem = crash_result_set.find('MINUTE')
        minute = minute_elem.text if minute_elem is not None else "N/A"
        
        # Name of the Road
        primary_road_elem = crash_result_set.find('TWAY_ID')
        priomary_road_name = primary_road_elem.text if primary_road_elem is not None else "N/A"
        second_road_elem = crash_result_set.find('TWAY_ID2')
        secondary_road_name = second_road_elem.text if second_road_elem is not None else "N/A"

        # 1. Number of vehicles
        vehicles_elem = crash_result_set.find('Vehicles')
        num_vehicles = len(vehicles_elem) if vehicles_elem is not None else 0
        if num_vehicles != 2:
            print("Number of vehicles involved in the crash is not two. Discrading this report...")
            return None

        # 2. Latitude and Longitude
        latitude_elem = crash_result_set.find('LATITUDE')
        longitude_elem = crash_result_set.find('LONGITUD')
        latitude = float(latitude_elem.text) if latitude_elem is not None else "N/A"
        longitude = float(longitude_elem.text) if longitude_elem is not None else "N/A"

        if latitude == "N/A" or longitude == "N/A":
            print("No crash coordinates mentioned. Discarding this report...")
            return None

        #harmful event
        harmful_event_name = crash_result_set.find('HARM_EVNAME')
        harmful_event = harmful_event_name.text if harmful_event_name is not None else "N/A"

        # 3. Manner of Collision
        man_coll_name_elem = crash_result_set.find('MAN_COLLNAME')
        manner_of_collision = man_coll_name_elem.text if man_coll_name_elem is not None else "N/A"

        collision_place_elem = crash_result_set.find('TYP_INTNAME')
        collision_place = collision_place_elem.text if collision_place_elem is not None else "N/A"
        if collision_place == "N/A":
            print("No collision place mentioned. Discarding this report...")

        #Vehicles section
        #write a code to extract the vehicle model, body-type, harmful event, manner of collision, most harmful event
        #impact point
        vehicles = []
        if num_vehicles == 2:
            for vehicle_elem in vehicles_elem.findall('Vehicle'):
                vehicle_data = {}
                model_elem = vehicle_elem.find('MAK_MODNAME')
                body_type_elem = vehicle_elem.find('BODY_TYPENAME')
                harmful_event_elem = vehicle_elem.find('HARM_EVNAME')
                manner_of_collision_elem = vehicle_elem.find('MAN_COLLNAME')
                most_harmful_event_elem = vehicle_elem.find('M_HARMNAME')
                impact_point_elem = vehicle_elem.find('IMPACT1NAME')
                speed_elem = vehicle_elem.find('TRAV_SPNAME')
                direction_elem = vehicle_elem.find('P_CRASH1NAME')
                direction_elem2 = vehicle_elem.find('P_CRASH2NAME')
                direction_elem3 = vehicle_elem.find('P_CRASH3NAME')
                direction_elem4 = vehicle_elem.find('PCRASH4NAME')
                direction_elem5 = vehicle_elem.find('PCRASH5NAME')
                accident_type_elem = vehicle_elem.find('ACC_TYPENAME')
                damages_elem = vehicle_elem.find('Damages')
                damages = []
                for damage_elem in damages_elem.findall('Damage'):
                    damage = damage_elem.find('DAMAGENAME')
                    damages.append(damage.text if damage is not None else "N/A")

                #append the data to the vehicle_data dictionary
                vehicle_data = {
                    "Model": model_elem.text if model_elem is not None else "N/A",
                    "Body Type": body_type_elem.text if body_type_elem is not None else "N/A",
                    "Harmful Event": harmful_event_elem.text if harmful_event_elem is not None else "N/A",
                    "Manner of Collision": manner_of_collision_elem.text if manner_of_collision_elem is not None else "N/A",
                    "Most Harmful Event": most_harmful_event_elem.text if most_harmful_event_elem is not None else "N/A",
                    "Impact Point": impact_point_elem.text if impact_point_elem is not None else "N/A",
                    "Speed": speed_elem.text if speed_elem is not None else "N/A",
                    "P_CRASH1": direction_elem.text if direction_elem is not None else "N/A",
                    "P_CRASH2": direction_elem2.text if direction_elem2 is not None else "N/A",
                    "P_CRASH3": direction_elem3.text if direction_elem3 is not None else "N/A",
                    "P_CRASH4": direction_elem4.text if direction_elem4 is not None else "N/A",
                    "P_CRASH5": direction_elem5.text if direction_elem5 is not None else "N/A",
                    "Accident Type": accident_type_elem.text if accident_type_elem is not None else "N/A",
                    "Damages": damages
                }
                vehicles.append(vehicle_data)
       
        # Find the events section
        c_events = crash_result_set.find('CEvents')
        
        # 5. Number of total events
        num_events = len(c_events) if c_events is not None else 0
        
        # 6. Sequence of events
        sequence_of_events = []
        if c_events:
            for event_elem in c_events.findall('CEvent'):
                soe_name_elem = event_elem.find('SOENAME')
                area_of_impact_elem1 = event_elem.find('AOI1NAME')
                area_of_impact_elem2 = event_elem.find('AOI2NAME')
                veh_num_elem = event_elem.find('VNUMBER1')
                veh_num_elem2 = event_elem.find('VNUMBER2')
                event_data = {
                    "Area of Impact 1": area_of_impact_elem1.text if area_of_impact_elem1 is not None else "N/A",
                    "Area of Impact 2": area_of_impact_elem2.text if area_of_impact_elem2 is not None else "N/A",
                    "Vehicle Number 1": veh_num_elem.text if veh_num_elem is not None else "N/A",
                    "Vehicle Number 2": veh_num_elem2.text if veh_num_elem2 is not None else "N/A",
                    "Event Name": soe_name_elem.text if soe_name_elem is not None else "N/A"
                }
                sequence_of_events.append(event_data)
                

        # Build the final dictionary
        crash_data = {
            "Number of Vehicles": num_vehicles,
            "date": date,
            "month": month,
            "hour": hour,
            "minute": minute,
            "Primary Road Name": priomary_road_name,
            "Secondary Road Name": secondary_road_name,
            "Latitude": latitude,
            "Longitude": longitude,
            "First Harmful Event": harmful_event,
            "Manner of Collision": manner_of_collision,
            "Collision Place": collision_place,
            "Number of Total Events": num_events,
            "Sequence of Events": sequence_of_events,
            "Vehicles": vehicles  
        }

        return crash_data

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: The file '{xml_file_path}' was not found.")
        return None

def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    files_processed = 0
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".xml"):
            try:
                print(f"Extracting Summary {filename}")

                crash_id = filename.split("_")[1].split(".")[0]
                
                file_path = os.path.join(REPORTS_DIR, filename)

                data = extract_crash_data(file_path)

                output_file_path = os.path.join(SUMMARY_DIR, f"summary_{crash_id}.txt")

                if data:
                    with open(output_file_path, 'w') as f:
                        f.write("--- Extracted Crash Summary ---\n")
                        f.write("Crash Details:\n")
                        f.write(f" Number of moving vehicles involved in the crash: {data['Number of Vehicles']}\n")
                        f.write(f" Date: {data['date']} {data['month']}\n")
                        f.write(f" Time: {data['hour']}:{data['minute']}\n")
                        f.write(f" Primary Road Name: {data['Primary Road Name']}\n")
                        f.write(f" Secondary Road Name: {data['Secondary Road Name']}\n")
                        f.write(f" Latitude: {data['Latitude']}\n")
                        f.write(f" Longitude: {data['Longitude']}\n")
                        f.write(f" First Harmful Event: {data['First Harmful Event']}\n")
                        f.write(f" Manner of Collision: {data['Manner of Collision']}\n")
                        f.write(f" Collision Place: {data['Collision Place']}\n\n")

                        f.write("\nSequence of Events:\n")
                        f.write(f" Number of Total Events: {data['Number of Total Events']}\n")
                        if data['Sequence of Events']:
                            for i, event in enumerate(data['Sequence of Events'], 1):
                                f.write(f" event {i}: \n")
                                for key, value in event.items():
                                    f.write(f"    {key}: {value}\n")
                                f.write("\n")
                        else:
                            f.write("  No sequence of events found.\n")

                        f.write("\nVehicles Details:\n")
                        if data['Vehicles']:
                            for i, vehicle in enumerate(data['Vehicles'], 1):
                                f.write(f" Vehicle {i}:\n")
                                for key, value in vehicle.items():
                                    f.write(f"   {key}: {value}\n")
                                f.write("\n")
                        else:
                            f.write("  No vehicle details found.\n")

                        print(f"Data successfully extracted and written to '{output_file_path}'")
                        files_processed += 1
                else:
                    print("\nFailed to extract summary.")
                    
            except Exception as e:
                print(f"An error occurred: {e}")
        
    print(f"Data successfully written to '{SUMMARY_DIR}'")
    print("Files Processed: ", files_processed)


if __name__ == "__main__":
    main()
