import json
import os
import re


def txt_to_json(txt_path):
    MARK = ">>"  # some speech begin with two arrows
    REGULAR = r"\d{14}\.\d{3}\|\d{14}\.\d{3}\|CC1\|"
    STORY_SEG = "|SEG_00|Type=Story start"
    SEG_COM = "|SEG_00|Type=Commercial"

    # json format: {"time": "", "location": "", "language": "", "content": []}
    json_data = {"time": "", "location": "", "language": "", "content": []}

    with open(txt_path, "r", encoding="utf-8") as file:
        txt_content = file.read()
    txt_content = re.sub(r">>>", MARK, txt_content)

    segments = txt_content.split(STORY_SEG)
    language = re.search(r"(LAN\|)(.*)", segments[0])[2]
    time_location = re.search(r"(LBT\|)(.*)", segments[0])[2]
    time = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})(.*)", time_location)[1]
    location = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})(.*)", time_location)[2]
    json_data["language"] = language
    json_data["time"] = time
    json_data["location"] = location


    for seg in segments[1:]:
        seg = seg.split(SEG_COM)[0].strip()
        parts = seg.split(MARK)
        content = []
        for part in parts:
            sentence = re.sub(REGULAR, "", part)
            sentence = sentence.strip()
            sentence = sentence.replace("\n", " ")
            sentence = sentence.lower()
            json_data["content"].append(sentence)
        # json_data["content"].append("\n\n".join(content))


    json_data = json.dumps(json_data)

    json_name = txt_path.split("/")[-1].replace(".txt", ".json")
    with open(os.path.join('jsons', json_name), 'w') as f:
        f.write(json_data)
    return json_data


if __name__ == "__main__":

    file_path = "./txt/2016-01-01_0000_US_CNN_Erin_Burnett_OutFront.txt"

    # pprint(txt_content)

    txt_to_json(txt_path=file_path)

