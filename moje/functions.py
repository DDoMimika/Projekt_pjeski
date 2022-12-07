import numpy as np
import paths
import pandas as pd

SEED = 234
#NUMBER_TRANING = 17701
NUMBER_TRANING = 17701
IMAGE_SIZE = 60
CLASS_SIZE = 120

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
    #return (train_input_out, train_output)
    return (train_input_out, train_output), (test_input_out, test_output)

def funkcja(train_input, train_outpout):
    seria = pd.DataFrame(np.concatenate((train_input,np.array([train_outpout]).T), axis=1))
    s = seria.loc[seria[IMAGE_SIZE*IMAGE_SIZE] == 73][0].mean()
    print("srednia: ",s)
    print(seria)

# sets = prepare_sets()
# # print(sets[0][1])
# # print(sets[0][0][0])
# funkcja(sets[0][0], sets[0][1])

def values():
    par = np.load(paths.PATH_PARAMETERS)
    for p in range(len(par)):
        if par[p][0] == 'nan' or par[p][1] == 'nan':
            print(f"nan na pozycji {p}")
    print(par)

#values()