import pathlib
import os
import re
from PIL import Image
import numpy as np
from numpy import int8, rad2deg, save

path = pathlib.Path("./ready_images")
folder = os.listdir(path)
array_images = []

array_of_ansers = []


def get_name_from_filename(image):
    name = image.split(".")[0]
    name = re.search(r"[A-Za-z_]+", name).group()
    return name


def get_array_of_names():
    array_names = []
    for image in folder:
        name = get_name_from_filename(image)
        if not name in array_names:
            array_names.append(name)
    return array_names


if __name__ == "__main__":
    for image in folder:
        image_of_dog = Image.open(path / image)
        info = image_of_dog.getdata()
        info = np.reshape(np.array(info), (100, 100, 1))
        array_images.append(info)
        array_answer = [0 for i in range(120)]
        name = get_name_from_filename(image)
        array_answer[get_array_of_names().index(name)] = 1
        array_of_ansers.append(array_answer)
    np_array_images = np.array(array_images, dtype=int8)
    np_array_names = np.array(array_of_ansers)
    save("inputs.npy", np_array_images)
    save("outputs.npy", np_array_names)
