import json
import os
from pprint import pprint


def txt_to_json(lines):
    json_data = []
    current_entry = None

    for line in lines:
        if line.strip():  # Make sure to skip empty lines
            parts = line.strip().split('|')
            if len(parts) > 3 and "SEG_" in parts[2]:  # Segment identifier in the third part
                if current_entry:
                    json_data.append(current_entry)  # Save the previous entry if exists
                # Parse the segment line with type definition if present
                segment_type = parts[3].split('=')[1] if '=' in parts[3] else "Unknown"
                current_entry = {
                    "start_time": parts[0],
                    "end_time": parts[1],
                    "segment": parts[2],
                    "type": segment_type,
                    "content": []
                }
            elif current_entry is not None and len(parts) > 3:
                # Append content to the current segment
                entry_content = {
                    "time_range": f"{parts[0]}|{parts[1]}",
                    "text": "|".join(parts[3:])  # Include all parts after the third as they are part of the text
                }
                current_entry["content"].append(entry_content)

    if current_entry:
        json_data.append(current_entry)  # Save the last entry

    return json_data


if __name__ == "__main__":
    file_path = '../2016_txt_data/2016-01-01_0000_US_CNN_Erin_Burnett_OutFront.txt'

    with open(file_path, 'r', encoding='utf-8') as file:
        txt_content = file.readlines()

    # pprint(txt_content)

    final_json = txt_to_json(txt_content)

    json_sample = json.dumps(final_json, indent=2) 
    pprint(json_sample)
    with open(os.path.join('jsons', 'output.json'), 'w') as f:
        f.write(json_sample)
    
