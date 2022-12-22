import random as rd
import numpy as np

class fuzzyfication_neuron:
    def __init__(self):
        self.output = 0
        self.mean = 0
        self.sigma = 1

    def gauss_function(self, x):
        self.output = np.exp(-(x-self.mean)**2/(2*self.sigma))

class firing_neuron:
    def __init__(self):
        self.h = 1

    def firing(self, layer, i, k):
        layer = layer[i::k]
        tab = np.zeros(len(layer))
        for i in range(len(layer)):
            tab[i] = layer[i].output
        self.h = np.min(tab)

class normalizaton_neuron:
    def __init__(self):
        self.h = 0
        self.s = 0

    def normalization(self, h, layer):
        s = 0
        for neuron in layer:
            s += neuron.h
        self.h = h/s

class conclusion_neuron:
    def __init__(self, data_size):
        self.a = np.array([rd.uniform(2**(-32),1) for _ in range(data_size+1)], dtype = float)  
        self.f = np.array([rd.uniform(2**(-32),1) for _ in range(data_size+1)], dtype = float)
        self.y = 0

    def conclusion(self, data, h):
        self.f[:len(self.f)-1] = np.multiply(data,h)  
        self.f[len(self.f)-1] = h
        s = self.f @ self.a
        self.y = s

class result_neuron:
    def __init__(self):
        self.y = 0

    def outcome(self, layer):
        s=0
        for neuron in layer:
            s += neuron.y
        self.y = s
        
