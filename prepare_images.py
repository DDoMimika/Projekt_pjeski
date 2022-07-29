import os
import json
import paths
import pathlib

from PIL import Image
from convert_image import prepare_image

iterator_names = 0

with open(paths.READY_DATABASE_FILENAME) as dog_data_base:
    dog_data_base = json.load(dog_data_base)

for folder in os.listdir(paths.PATH_IMAGES):

    for image in os.listdir(paths.PATH_IMAGES / folder):

        image_of_dog = Image.open(paths.PATH_IMAGES / folder / image)
        image_of_dog_name = image.split(".")[0]

        for dog in dog_data_base[image_of_dog_name]:

            file_name = dog["name"]
            xmin = int(dog["bndbox"]["xmin"])
            ymin = int(dog["bndbox"]["ymin"])
            xmax = int(dog["bndbox"]["xmax"])
            ymax = int(dog["bndbox"]["ymax"])

            ready_image = prepare_image(xmin, ymin, xmax, ymax, image_of_dog)

            ready_image.save(
                str(paths.PATH_READY_IMAGES.absolute())
                + f"/{file_name}"
                + f"{iterator_names}"
                + ".jpg"
            )
            iterator_names += 1

    iterator_names = 0
