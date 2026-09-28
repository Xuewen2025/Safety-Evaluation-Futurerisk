import os
import concurrent.futures
import re
import json
import time

import requests

from ns_check_points import check_and_get_direction

from settings import SEGMENT_BBOX_DIR
from settings import MODIFIED_SUMMARY_DIR
from settings import CLIPPED_MERGED_XODR_DIR
from settings import PROMPTS_DIR
from settings import REASONINGS_DIR
from settings import SCENE_POINTS_DIR

# --- CONCURRENCY SETTINGS ---
# Be careful with MAX_WORKERS. Too high = Rate Limit Errors (429).
# Start with 5, increase if your API quota allows.
MAX_WORKERS = 10
MAX_RETRIES = 5
MODEL_NAME = "gemini-3-flash-preview"

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")

def get_file_content(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def construct_prompt(summary_data, bbox_data):
    """
    Constructs the prompt by injecting file data into the optimized template.
    """
    prompt_template = f"""
You are an expert in accident reconstruction using CARLA.
Your task is to deduce the pre-crash positions (5 seconds prior) of two vehicles involved in a crash.

### INPUT DATA
**1. Crash Summary:**
{summary_data}

**2. Map Context (Local CARLA Coordinates):**
{bbox_data}

---

### CRITICAL TRAFFIC RULES
**Region:** United States (USA).
**Rule:** **RIGHT-HAND TRAFFIC (RHT)** is mandatory.
- Vehicles MUST drive on the right side of the road.
- When moving "backwards" from the crash point to find the start point, ensure the resulting position places the vehicle in the correct lane (the right lane relative to its forward direction).

---

### REQUIRED REASONING STEPS
Before generating the final JSON, you must strictly follow these reasoning steps. 
**Do not skip these steps**, as they ensure the coordinates are physically valid.

**Task 1: Scenario Analysis**
- Summarize the crash scenario.

**Task 2: Distance Calculation**
- Calculate the distance traveled in 5 seconds.
- Formula: Distance = Speed (m/s) * 5. (Convert MPH to m/s first: 1 MPH = 0.447 m/s).

**Task 3: Crash Location**
- Extract the crash location (x, y) explicitly from the summary.

**Task 4: Road & Bounding Box Selection**
- Identify which Road ID and Bounding Box each vehicle started in.
- **Lane Assignment Rule (Strict JSON Mapping):**
  - Do NOT calculate the "Right" side based on X/Y coordinate vectors (e.g., do not determine that East is Right).
  - Use the following mapping based on the sequence of Bounding Boxes:
      - **Direction A (Ascending IDs, e.g., Box 1 → Box 10):** The vehicle MUST be placed in `right_lane_segment`.
      - **Direction B (Descending IDs, e.g., Box 10 → Box 1):** The vehicle MUST be placed in `left_lane_segment`.
- **Constraint:** Vehicles approaching a junction usually come from different roads.
- **Constraint:** There must be a valid path from the Start Point to the Crash Point.
- **Constraint:** Putting the vehicles in the assigned bounding box has more priority than following the speed mentioned.
- **Constraint:** For both junction and non-junction crashes, the initial locations of the two vehicles cannot be the same.

**Task 5: Coordinate Calculation & Verification (CRITICAL)**
- Calculate the start position (x, y) by moving backwards from the crash site by the distance found in Task 2.
- **Constraint:** The coordinates you select must be **inside** the respective assigned bounding box.
- **Constraint:** The coordinates cannot be the four corners of the bounding boxes provided.
- **Constraint:** The coordinates must not be very close to the edge of the bounding box.
- **Constraint:** The positions of the two vehicles cannot be same.
- **VERIFICATION:** You must strictly verify your calculated point is inside the chosen bounding box.
- *Check:* Is calculated X between Box X_min and X_max? 
- *Check:* Is calculated Y between Box Y_min and Y_max?
- If the point is outside, adjust it slightly along the road geometry until it is **inside** the box.

---

### FINAL OUTPUT
After you have completed the reasoning above, output the final result in a single VALID JSON block.
The JSON must be the **last** thing in your response.

```json
{{
  "crash_location": {{ "x": float, "y": float }},
  "vehicle_1": {{
      "road_id": int,
      "position": {{ "x": float, "y": float }}
  }},
  "vehicle_2": {{
      "road_id": int,
      "position": {{ "x": float, "y": float }}
  }}
}}
```
"""
    return prompt_template

def parse_json_from_response(text):
    """Helper to safely extract JSON from LLM markdown."""
    
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index != -1 and end_index != -1:
            json_str = text[start_index:end_index]
        else:
            return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def validate_positions(data, xodr_path):
    """
    Checks if the points in the data are valid using the imported checker.
    Returns: (is_valid: bool, error_message: str)
    """
    if not data:
        return False, "Failed to parse valid JSON from response."

    errors = []

    v1_x = data['vehicle_1']['position']['x']
    v1_y = data['vehicle_1']['position']['y']
    v1_P = (v1_x, v1_y)
    is_v1_valid, _, _, _, _ = check_and_get_direction(v1_P, None, xodr_path, snap=False) 

    if not is_v1_valid:
        errors.append(f"Vehicle 1 position ({v1_x}, {v1_y}) is OUTSIDE the valid bounding boxes.")

    v2_x = data['vehicle_2']['position']['x']
    v2_y = data['vehicle_2']['position']['y']
    v2_P = (v2_x, v2_y)
    is_v2_valid, _, _, _, _ = check_and_get_direction(v2_P, None, xodr_path, snap=False)

    if not is_v2_valid:
        errors.append(f"Vehicle 2 position ({v2_x}, {v2_y}) is OUTSIDE the valid bounding boxes.")

    if errors:
        return False, " ".join(errors) + " Please recalculate coordinates that are strictly INSIDE the bounding boxes."
    
    return True, ""

def generate_content_http(conversation_history): 
    """ Manually calls the Gemini API using requests to bypass SDK version issues. """ 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }
 
    payload = {
        "contents": conversation_history,
        "generationConfig": {
            "temperature": 0.8
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:   
        raise Exception(f"API Error {response.status_code}: {response.text}")
        
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise Exception(f"Unexpected API response format: {data}")


def process_crash_with_retries(summary_text, bbox_text, xodr_path):
    """
    Pass 'client' as an argument to ensure thread safety if we decide 
    to create one client per thread later.
    """
    initial_prompt = construct_prompt(summary_text, bbox_text)

    conversation_history = [
        {"role": "user", "parts": [{"text": initial_prompt}]}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            # Call the manual HTTP function
            response_text = generate_content_http(conversation_history)
            # print(response_text)
            
            json_data = parse_json_from_response(response_text)
            
            is_valid, error_msg = validate_positions(json_data, xodr_path)
            
            if is_valid:
                return attempt + 1, response_text, json_data
            
            # Feedback loop
            conversation_history.append({"role": "model", "parts": [{"text": response_text}]})
            
            correction_prompt = f"Your previous output was incorrect.\nERROR: {error_msg}\n\nPlease strictly follow the instructions: Points must be INSIDE the bounding box. Recalculate."
            conversation_history.append({"role": "user", "parts": [{"text": correction_prompt}]})
            
            time.sleep(1) 

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(2)

    return attempt + 1, None, None

def process_single_file(filename):
    """
    Worker function that processes one file entirely.
    """
    print(f"Starting {filename}...")
    
    try:
        bbox_path = os.path.join(SEGMENT_BBOX_DIR, filename)
        bbox_text = get_file_content(bbox_path)

        crash_id = filename.split('_')[-1].replace('.txt','')
        xodr_path = os.path.join(CLIPPED_MERGED_XODR_DIR, f"map_{crash_id}.xodr")
        summary_path = os.path.join(MODIFIED_SUMMARY_DIR, f"summary_{crash_id}.txt")
        
        if not os.path.exists(summary_path):
            return f"SKIP: Summary not found for {filename}"

        summary_text = get_file_content(summary_path)
        
        initial_prompt = construct_prompt(summary_text, bbox_text)
        with open(os.path.join(PROMPTS_DIR, f"prompt_{crash_id}.txt"), "w") as f:
            f.write(initial_prompt)

        attempt, final_text, final_json = process_crash_with_retries(summary_text, bbox_text, xodr_path)
        
        if final_json:
            with open(os.path.join(REASONINGS_DIR, f"llmreasoning_{crash_id}"), "w") as f:
                f.write(final_text)
            
            with open(os.path.join(SCENE_POINTS_DIR, f"scenepoints_{crash_id}.json"), "w") as f:
                json.dump(final_json, f, indent=4)
            
            return f"SUCCESS: {filename} in {attempt} attempts."
        else:
            return f"FAILURE: {filename} after {attempt} retries."

    except Exception as e:
        return f"ERROR: {filename} - {str(e)}"

def main():
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(REASONINGS_DIR, exist_ok=True)
    os.makedirs(SCENE_POINTS_DIR, exist_ok=True)

    # Gather all files to process
    files_to_process = [f for f in os.listdir(SEGMENT_BBOX_DIR) if f.endswith(".txt")]
    
    print(f"Processing {len(files_to_process)} files with {MAX_WORKERS} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, f): f for f in files_to_process}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            print(result)

        
if __name__ == "__main__":
    main()