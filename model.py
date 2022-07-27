from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential

NUM_CLASSES = 120


def get_model():
    model = Sequential(
        [
            layers.Conv2D(
                16, 3, padding="same", activation="relu", input_shape=(100, 100, 1)
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES),
        ]
    )

    return model
