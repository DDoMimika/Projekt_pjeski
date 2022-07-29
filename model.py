from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential
from convert_image import IMG_SIZE

NUM_CLASSES = 120
POOL_SIZE = 2


def get_model():
    model = Sequential(
        [
            layers.Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
                input_shape=(IMG_SIZE, IMG_SIZE, 1),
            ),
            layers.MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES),
        ]
    )

    return model
