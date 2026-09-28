import os
from settings import SUMMARY_DIR
from settings import MODIFIED_SUMMARY_DIR

def process_crash_summaries(input_path, output_path):
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_summary = []
        lat_indent = ""
        
        for line in lines:
            stripped = line.strip()
        
            if stripped.startswith("Latitude:"):
                lat_indent = line[:line.find("Latitude")]
                continue
            elif stripped.startswith("Longitude:"):
                new_line = f"{lat_indent}Crash Location: ({0.0}, {-0.0})\n"
                new_summary.append(new_line)
            else:
                new_summary.append(line)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_summary)

def main():

    os.makedirs(MODIFIED_SUMMARY_DIR, exist_ok=True)

    files_processed = 0
    for filename in os.listdir(SUMMARY_DIR):
        try:
            if filename.endswith(".txt"): 
                input_path = os.path.join(SUMMARY_DIR, filename)
                output_path = os.path.join(MODIFIED_SUMMARY_DIR, filename)
                process_crash_summaries(input_path, output_path)
                print(f"Processed: {filename}")
                files_processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nDone! Processed {files_processed} files.")
    
if __name__ == "__main__":
    main()