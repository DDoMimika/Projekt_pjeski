import numpy as np
import paths
import pandas as pd

SEED = 234
NUMBER_TRANING = 20000
IMAGE_SIZE = 100
CLASS_SIZE = 2

def prepare_sets():
    input_data = np.load(paths.PATH_INPUT)
    output_data = np.load(paths.PATH_OUTPUT)
    #print(output_data)

    np.random.seed(SEED)
    np.random.shuffle(input_data)
    np.random.seed(SEED)
    np.random.shuffle(output_data)

    train_input = input_data[:NUMBER_TRANING]
    train_output = output_data[:NUMBER_TRANING]

    test_input = input_data[NUMBER_TRANING:]
    test_output = output_data[NUMBER_TRANING:]

    data_sample = np.array([0 for _ in range(IMAGE_SIZE*IMAGE_SIZE)])
    train_input_out = np.array([data_sample for _ in range(len(train_input))])
    for i in range(len(train_input)):
        train_input_out[i] = train_input[i].flatten()

    test_input_out = np.array([data_sample for _ in range(len(test_input))])
    for i in range(len(test_input)):
        test_input_out[i] = test_input[i].flatten()
    return (train_input_out, train_output), (test_input_out, test_output)