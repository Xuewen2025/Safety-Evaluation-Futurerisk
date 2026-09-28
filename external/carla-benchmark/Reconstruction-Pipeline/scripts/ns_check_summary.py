import os
import sys
import glob
from settings import SUMMARY_DIR

def check_movement_info(line):
    """
    Checks if a line contains the target movement keywords.
    """
    keywords = ["turning left", "turning right", "going straight"]
    lower_line = line.lower()
    
    for key in keywords:
        if key in lower_line:
            return True
    return False

def check_vehicle_info(file_path):
    """
    Parses a single crash report file to check:
    1. Intersection status (True if intersection, False if not)
    2. V1 and V2 movement info presence
    
    Returns a tuple: (is_intersection, has_v1_info, has_v2_info)
    """
    v1_has_info = False
    v2_has_info = False
    is_intersection = True # Default to True, set to False if "not at intersection" found
    
    current_vehicle = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')

            for line in lines:
                if "Collision Place:" in line:
                    val = line.split(":", 1)[1].strip().lower()
                    if "not at intersection" in val or "not an intersection" in val:
                        is_intersection = False
                    break 
            
            # --- 2. Check Movement Info ---
            if is_intersection:
                for line in lines:
                    line = line.strip()
                    
                    if "Vehicle 1:" in line:
                        current_vehicle = 1
                    elif "Vehicle 2:" in line:
                        current_vehicle = 2
                    
                    # Check P_CRASH fields
                    if "p_crash" in line.lower() or "pcrash" in line.lower():
                        if check_movement_info(line):
                            if current_vehicle == 1:
                                v1_has_info = True
                            elif current_vehicle == 2:
                                v2_has_info = True

                return is_intersection, v1_has_info, v2_has_info
            
            return is_intersection, v1_has_info, v2_has_info
                            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return (False, False, False)


def main():
    
    if not os.path.isdir(SUMMARY_DIR):
        print(f"Error: Directory '{SUMMARY_DIR}' not found.")
        sys.exit(1)

    files_processed = 0
    for filename in os.listdir(SUMMARY_DIR):
        should_delete = False
        print(f"Checking Summary {filename}.")

        summary_path = os.path.join(SUMMARY_DIR, filename)

        is_inter, v1_has_info, v2_has_info = check_vehicle_info(summary_path)

        if is_inter:
            if not (v1_has_info and v2_has_info):
                should_delete = True
            else:
                files_processed += 1
        else:
            files_processed += 1

        if should_delete:
            print("Movement info NOT found. -> DELETING.")
            try:
                os.remove(summary_path)
            except OSError as e:
                print(f"Error: {e.strerror}")

    print("Files Processed: ", files_processed)

if __name__ == "__main__":
    main()