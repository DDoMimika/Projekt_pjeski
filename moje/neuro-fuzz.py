import numpy as np
from fuzzy_network_class import *


images = np.array([[1,2],[2,3],[0.5,1],[0.125,0.25]])
category_images = np.array([1, 2, 3, 4])
model = fuzzy_network(2,len(images[0]))
model.train(images, category_images, 1)





        






    