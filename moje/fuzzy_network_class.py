import numpy as np
from neuron_class import *

class fuzzy_network:
    def __init__(self, k, image_size):
        self.k = k
        self.fuzzyfication_layer = np.array([fuzzyfication_neuron() for _ in range(k* image_size)])
        self.firing_layer = np.array([firing_neuron() for _ in range(k)])
        self.normalization_layer = np.array([normalizaton_neuron() for _ in range(image_size)])
        self.conclusion_layer = np.array([conclusion_neuron(image_size) for _ in range(k)])
        self.result_layer = result_neuron()
        self.net_rasult = 0

    def fuzzyfication_layer_work(self,image):
        input_image= np.repeat(image, self.k)
        for i in range(len(self.fuzzyfication_layer)):
            self.fuzzyfication_layer[i].gauss_function(input_image[i])
            print(f"fuzzyfication_layer[{i}] = {self.fuzzyfication_layer[i].output}")

    def firing_layer_work(self):
        for i in range(len(self.firing_layer)):
            self.firing_layer[i].firing(self.fuzzyfication_layer, i, self.k)
            print(f"firing_layer[{i}] = {self.firing_layer[i].h}")

    def normalization_layer_work(self):
        for i in range(len(self.normalization_layer)):
            self.normalization_layer[i].normalization(self.firing_layer[i].h, self.firing_layer)
            print(f"normalization_layer[{i}] = {self.normalization_layer[i].h}")

    def conclusion_layer_work(self,image):
        for i in range(len(self.conclusion_layer)):
            self.conclusion_layer[i].conclusion(image, self.normalization_layer[i].h)
            print(f"conclusion_layer[{i}] = {self.conclusion_layer[i].y}")

    def result_layer_work(self):
        self.result_layer.outcome(self.conclusion_layer)
        self.net_rasult = self.result_layer.y
        print(f"result_layer = {self.result_layer.y}")

    def work(self, image):
        self.fuzzyfication_layer_work(image)
        self.firing_layer_work()
        self.normalization_layer_work()
        self.conclusion_layer_work(image)
        self.result_layer_work()

        return self.net_rasult