from fuzzy_network_class import *
from fuzzy_prepare_sets import *

sets = prepare_sets()

#membership_parameters_learn(sets[0][0],sets[0][1], CLASS_SIZE, IMAGE_SIZE*IMAGE_SIZE)

model = fuzzy_network(CLASS_SIZE,IMAGE_SIZE*IMAGE_SIZE)
print("Stworzono siec")
#model.conclusion_pre_learn(sets[0][0], sets[0][1])
#model.pre_learn(sets[0][0], sets[0][1], conclusion_from_file = False)
model.train(sets[0][0], sets[0][1], sets[1][0], sets[1][1], 7, mi =1)     





        






    