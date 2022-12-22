import os
import re
import numpy as np
import paths

from PIL import Image
from numpy import int8, save
from convert_image import IMG_SIZE


def load_image(image_path):
    dog_image = Image.open(image_path)
    pixels = dog_image.getdata()
    pixels = np.reshape(np.array(pixels), (IMG_SIZE, IMG_SIZE, 1))
    return pixels


if __name__ == "__main__":

    images = []
    answers = []

    for i, image in enumerate(os.listdir(paths.PATH_READY_IMAGES)):

        pixels = load_image(paths.PATH_READY_IMAGES / image)
        images.append(pixels)

        if image.split("_")[0].lower() =="cat":

            answers.append(0)
        else:
            answers.append(1)
    np_images = np.array(images, dtype=int8)
    np_names = np.array(answers)
    save(paths.INPUT_FILENAME, np_images)
    save(paths.OUTPUT_FILENAME, np_names)
