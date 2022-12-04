import os
import xmltodict
import json
import paths
# nie używamy aktualnie
path = paths.PATH_ANNOTATIONS
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
with open(paths.READY_DATABASE_FILENAME, "w") as f:
    f.write(json_data_base)
