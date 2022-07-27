import os
import pathlib
import json

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
            xmin = int(dog["bndbox"]["xmin"])
            ymin = int(dog["bndbox"]["ymin"])
            xmax = int(dog["bndbox"]["xmax"])
            ymax = int(dog["bndbox"]["ymax"])
            width = xmax - xmin
            height = ymax - ymin
            if width < height:
                croped_image = image_of_dog.crop((xmin, ymin, xmax, width + ymin))
            else:
                croped_image = image_of_dog.crop((xmin, ymin, xmin + height, ymax))
            ready_image = croped_image.convert("L")
            if ready_image.width < 100 or ready_image.height < 100:
                ready_image = ready_image.resize((100, 100))
            if ready_image.width < 100 or ready_image.height < 100:
                ready_image.show()
            ready_image.thumbnail(size_parametr)
            ready_image.save(
                "C:/Users/Akshei/Desktop/projekt_pjeski/ready_images/"
                + file_name
                + f"{int_of_file_name}"
                + ".jpg"
            )
            int_of_file_name += 1
    int_of_file_name = 0
