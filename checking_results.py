import random
import os
import numpy as np
import paths

from model import get_model
from PIL import Image
from prepare_outputs_and_inputs import get_array_of_names as array_names
from convert_image import IMG_SIZE
from training_model import load_model
from paths import PATH_READY_IMAGES
from prepare_outputs_and_inputs import load_image

images = os.listdir(PATH_READY_IMAGES)

model = load_model(paths.PATH_CHECKPOINT)

random_dog = images[random.randint(0, len(images))]
print(random_dog)
pixels = load_image(PATH_READY_IMAGES / random_dog)
array_of_dogs = np.array([pixels for i in range(64)])
value = np.argmax(model.predict(array_of_dogs)[0])
if value==0:
    print("cat")
else:
    print("dog")
