import random as rd
import numpy as np

class fuzzyfication_neuron:
    def __init__(self):
        self.output = 0
        self.mean = rd.random()
        self.sigma = rd.random()

    def gauss_function(self, x):
        #self.output = x
        self.output = np.exp(-((x-self.mean)/self.sigma)**2)

class firing_neuron:
    def __init__(self):
        self.h = 1

    def firing(self, layer, i, k):
        layer = layer[i::k]
        for l in layer:
            self.h *= l.output

class normalizaton_neuron:
    def __init__(self):
        self.h = 0

    def normalization(self, h, layer):
        s = 0
        for neuron in layer:
            s += neuron.h
        self.h = h/s

class conclusion_neuron:
    def __init__(self, image_size):
        self.a = np.array([rd.random() for _ in range(image_size+1)])  
        self.f = np.array([rd.random() for _ in range(image_size+1)])     #a - conclusion parameters
        self.y = 0

    def conclusion(self, image, h):
        self.f[:len(self.f)-1] = np.multiply(image,h)  #błą image*h
        self.f[len(self.f)-1] = h
        s = self.f @ self.a
        self.y = h*s

class result_neuron:
    def __init__(self):
        self.y = 0

    def outcome(self, layer):
        s=0
        for neuron in layer:
            s += neuron.y
        self.y = s
        
