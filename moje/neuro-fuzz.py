# warstwa 1 - pobieranie danych - l. neuronów = image_size; fuzzyfikacja - gauss funkcji - zm.: mean sigma
# war 2  - normalizacja
# war 3 - konkluzcja reguł 
# war 4 - suma konklucji

import numpy as np
from fuzzy_network_class import *


image = [1,2]
model = fuzzy_network(2,len(image))
model.work(image)





        






    