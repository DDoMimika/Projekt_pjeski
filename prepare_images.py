import os
import pathlib
import json
import re
from PIL import Image

size_parametr = (100, 100)
with open("ready_data_base.json") as dog_data_base:
    dog_data_base = json.load(dog_data_base)

path = pathlib.Path("./archive/images/Images")
folders = os.listdir(path)
int_of_file_name = 0

for folder in folders:
    images = os.listdir(path / folder)
    for image in images:
        image_of_dog = Image.open(path / folder / image)
        image_of_dog_name = image.split(".")[0]
        for dog in dog_data_base[image_of_dog_name]:
            file_name = dog["name"]
            sizes = []
            for size in dog["bndbox"].values():
                size = int(size)
                sizes.append(size)
            width = sizes[2] - sizes[0]
            height = sizes[3] - sizes[1]
            if width < height:
                croped_image = image_of_dog.crop(
                    (sizes[0], sizes[1], sizes[2], sizes[1] + width)
                )
            else:
                croped_image = image_of_dog.crop(
                    (sizes[0], sizes[1], sizes[0] + height, sizes[3])
                )
            ready_image = croped_image.convert("L")
            ready_image.thumbnail(size_parametr)
            ready_image.save(
                "C:/Users/Akshei/Desktop/projekt_pjeski/ready_images/"
                + file_name
                + f"{int_of_file_name}"
                + ".jpg"
            )
            int_of_file_name += 1
    int_of_file_name = 0
