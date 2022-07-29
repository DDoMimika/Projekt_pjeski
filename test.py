import numpy as np

# train_input = np.load("./inputs.npy")
# train_output = np.load("./outputs.npy")
# IMAGE_SIZE = 100
# np.random.seed(8008)
# np.random.shuffle(train_input)
# np.random.seed(8008)
# np.random.shuffle(train_input)

a = [[i, 0, 1, 2] for i in range(100)]
b = [[i, 0, 1, 2] for i in range(100)]
a = np.array(a)
b = np.array(b)
print(a, b)
np.random.seed(8008)
np.random.shuffle(a)
np.random.seed(8008)
np.random.shuffle(b)
print(a, b)
for i in range(100):
    assert a[i][0] == b[i][0]
