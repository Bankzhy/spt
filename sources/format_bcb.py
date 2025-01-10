import json
import os


def run():
    output = r"C:\worksapce\research\SPT-Code\sources\dataset\fine_tune\clone"
    code_path = r"C:\worksapce\research\SPT-Code\sources\dataset\fine_tune\clone\BCB\bigclonebenchdata"

    clone_mapping = []

    for filename in os.listdir(code_path):
        data = {}
        idx = filename.split('.')[0]

        filepath = os.path.join(code_path, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            with open(filepath, 'r', encoding="utf8") as f:
                content = f.read()
                data['idx'] = idx
                data['func'] = content
                clone_mapping.append(data)
    output = os.path.join(output, 'data.jsonl')
    with open(output, "w") as json_file:
        for js in clone_mapping:
            json_file.write(json.dumps(js))
            json_file.write("\n")
        json_file.close()

if __name__ == '__main__':
    run()