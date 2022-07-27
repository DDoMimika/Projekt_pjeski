from genericpath import exists
from pickletools import optimize
import tensorflow as tf
import numpy as np
import json
from model import get_model
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential
from prepare_outputs_and_inputs import get_array_of_names as array_names
import random

file_name = "weight.ckpt"
train_input = np.load("./inputs.npy")
train_output = np.load("./outputs.npy")
IMAGE_SIZE = 100
np.random.seed(67)
np.random.shuffle(train_input)
np.random.seed(67)
np.random.shuffle(train_input)
train = tf.data.Dataset.from_tensor_slices(
    (train_input[:17701], train_output[:17701])
).batch(64)
test = tf.data.Dataset.from_tensor_slices(
    (train_input[17701:], train_output[17701:])
).batch(64)
model = get_model()
model.build()
model.summary()
model.compile(
    optimizer="adam",
    loss=losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=file_name, save_weights_only=True, verbose=1
)
checkpoint_path = "./projekt_pjeski"
ckpt = tf.train.Checkpoint()
if tf.train.latest_checkpoint(checkpoint_path) != None:
    ckpt_path = tf.train.latest_checkpoint(checkpoint_path)
    ckpt.restore(ckpt_path)

history = model.fit(train, epochs=100, callbacks=[cp_callback])
json_history = json.dumps(history.history)
with open("history.json", "w") as file_to_write:
    file_to_write.write(json_history)
loss, acc = model.evaluate(test)
print(acc)
