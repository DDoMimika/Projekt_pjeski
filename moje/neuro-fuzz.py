import numpy as np
import paths
from fuzzy_network_class import *
from functions import *

sets = prepare_sets()


# images = np.array([[1,2],[2,3],[0.5,1],[0.125,0.25]])
# category_images = np.array([1, 2, 3, 4])
model = fuzzy_network(CLASS_SIZE,IMAGE_SIZE*IMAGE_SIZE)
print("Stworzono siec")
model.pre_train(sets[0][0],sets[0][1])
print("Przed ucznie zakończone")
model.train(sets[0][0], sets[0][1], 3)





        






    