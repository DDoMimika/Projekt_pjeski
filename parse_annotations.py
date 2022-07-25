import os
import xmltodict
import json
import pathlib

path = pathlib.Path("./archive/annotations/Annotation")
folders = os.listdir(path)
data_base = {}

for folder in folders:
    files = os.listdir(path / folder)
    for file in files:
        info_dogs = []
        info = {}
        with open(path / folder / file, "r") as f:
            data = xmltodict.parse(f.read())
            if type(data["annotation"]["object"]) == list:
                for object in data["annotation"]["object"]:
                    info_dogs.append(
                        {"name": object["name"], "bndbox": object["bndbox"]}
                    )
            else:
                info_dogs = [
                    {
                        "name": data["annotation"]["object"]["name"],
                        "bndbox": data["annotation"]["object"]["bndbox"],
                    }
                ]
            data_base[file] = info_dogs
json_data_base = json.dumps(data_base, indent=4)
with open("ready_data_base.json", "w") as file_to_write:
    file_to_write.write(json_data_base)
