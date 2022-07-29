import numpy as np
import paths
import tensorflow as tf

SEED = 234
NUMBER_TRANING = 17701
BATCH = 64


def prepare_dataset():
    input_data = np.load(paths.PATH_INPUT)
    output_data = np.load(paths.PATH_OUTPUT)

    np.random.seed(SEED)
    np.random.shuffle(input_data)
    np.random.seed(SEED)
    np.random.shuffle(output_data)

    train = (
        tf.data.Dataset.from_tensor_slices(
            (input_data[:NUMBER_TRANING], output_data[:NUMBER_TRANING])
        )
        .shuffle(len(input_data))
        .batch(BATCH)
    )

    test = tf.data.Dataset.from_tensor_slices(
        (input_data[NUMBER_TRANING:], output_data[NUMBER_TRANING:])
    ).batch(BATCH)
    return train, test
