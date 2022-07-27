from genericpath import exists
from pickletools import optimize
import tensorflow as tf
import numpy as np
import json
from model import get_model
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential

file_name = "weight.ckpt"
train_input = np.load("./inputs.npy")
train_output = np.load("./outputs.npy")
train = (
    tf.data.Dataset.from_tensor_slices((train_input[0:20000], train_output[0:20000]))
    .batch(64)
    .shuffle(64)
)
test = tf.data.Dataset.from_tensor_slices(
    (train_input[20000:], train_output[20000:])
).batch(64)
IMAGE_SIZE = 100

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
else:
    model.load_weights("weight.ckpt")

history = model.fit(train, epochs=100, callbacks=[cp_callback])
json_history = json.dumps(history.history)
with open("history.json", "w") as file_to_write:
    file_to_write.write(json_history)
loss, acc = model.evaluate(test)
print(acc)
