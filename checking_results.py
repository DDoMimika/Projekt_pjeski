import pathlib
import random
import os
import numpy as np
from importlib.resources import path
from statistics import mode
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential
from model import get_model
from PIL import Image
from prepare_outputs_and_inputs import get_array_of_names as array_names

path = pathlib.Path("./ready_images")
folder = os.listdir(path)
model = get_model()
model.load_weights("weight.ckpt")

random_dog = folder[random.randint(0, 22126)]
print(random_dog)
guess = Image.open(path / random_dog)
array_of_dogs = np.array(
    [np.reshape(np.array(guess.getdata()), (100, 100, 1)) for i in range(64)]
)
value = np.argmax(model.predict(array_of_dogs)[0])
print(array_names()[value])
