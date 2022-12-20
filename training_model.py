from keras.utils.np_utils import to_categorical
import tensorflow as tf
import json
import paths

from genericpath import exists
from pickletools import optimize
from tensorflow import keras
from keras import layers, losses
from keras.models import Sequential
from model import get_model
from prepare_dataset import prepare_dataset

EPOCHS = 20


def load_model(checkpoint):
    model = get_model()
    if tf.train.latest_checkpoint(checkpoint) != None:
        model.load_weights(tf.train.latest_checkpoint(checkpoint))
    return model


if __name__ == "__main__":
    model = load_model(paths.PATH_CHECKPOINT)

    model.summary()
    model.compile(
        optimizer="adam",
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(paths.PATH_CHECKPOINT.absolute()) + "/{loss}",
        save_weights_only=True,
        verbose=1,
    )

    train, test = prepare_dataset()
    history = model.fit(train, epochs=EPOCHS,callbacks=[cp_callback])
    acc = history.history['accuracy']
    loss = history.history['loss']
    # json_history = json.dumps(history)

    # with open(paths.HISTORY_FILENAME, "w") as f:
    #     f.write(json_history)

    model.evaluate(test)

import matplotlib.pyplot as plt

epochs = EPOCHS
plt.plot(range(epochs), loss, label='Training loss')
plt.savefig("loss.png")
plt.plot(range(epochs), acc, 'bo', label='Training acc')
plt.savefig("acc.png")
plt.show()