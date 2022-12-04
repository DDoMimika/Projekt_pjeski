import os
import paths

from PIL import Image
from convert_image import prepare_image

iterator_names = 0

for folder in os.listdir(paths.PATH_IMAGES):

    for image in os.listdir(paths.PATH_IMAGES / folder):
        try:
            image_of_dog = Image.open(paths.PATH_IMAGES / folder / image)
            image_of_dog_name = image
            ready_image = prepare_image(image_of_dog)

            ready_image.save(
                str(paths.PATH_READY_IMAGES.absolute())
                + f"/{folder}_"
                + f"{iterator_names}"
                + ".jpg"
            )
            iterator_names += 1
        except Exception as e:
            print(e)

    iterator_names = 0
