import numpy as np
from neuron_class import *

class fuzzy_network:
    def __init__(self, N, data_size):
        self.N = N
        self.data_size = data_size
        self.fuzzyfication_layer = np.array([fuzzyfication_neuron() for _ in range(N* data_size)])
        self.firing_layer = np.array([firing_neuron() for _ in range(N)])
        self.normalization_layer = np.array([normalizaton_neuron() for _ in range(data_size)])
        self.conclusion_layer = np.array([conclusion_neuron(data_size) for _ in range(N)])
        self.result_layer = result_neuron()
        self.net_rasult = 0

    def fuzzyfication_layer_work(self,image):
        input_image= np.repeat(image, self.N)
        for i in range(len(self.fuzzyfication_layer)):
            self.fuzzyfication_layer[i].gauss_function(input_image[i])
            print(f"fuzzyfication_layer[{i}] = {self.fuzzyfication_layer[i].output}")

    def firing_layer_work(self):
        for i in range(len(self.firing_layer)):
            self.firing_layer[i].firing(self.fuzzyfication_layer, i, self.N)
            #print(f"firing_layer[{i}] = {self.firing_layer[i].h}")

    def normalization_layer_work(self):
        for i in range(len(self.normalization_layer)):
            self.normalization_layer[i].normalization(self.firing_layer[i].h, self.firing_layer)
            print(f"normalization_layer[{i}] = {self.normalization_layer[i].h}")

    def conclusion_layer_work(self,image):
        for i in range(len(self.conclusion_layer)):
            self.conclusion_layer[i].conclusion(image, self.normalization_layer[i].h)
            #print(f"conclusion_layer[{i}] = {self.conclusion_layer[i].y}")

    def result_layer_work(self):
        self.result_layer.outcome(self.conclusion_layer)
        self.net_rasult = self.result_layer.y
        #print(f"result_layer = {self.result_layer.y}")

    def work(self, data):
        self.fuzzyfication_layer_work(data)
        self.firing_layer_work()
        self.normalization_layer_work()
        self.conclusion_layer_work(data)
        self.result_layer_work()

        return self.net_rasult

    def train(self, train_set, train_category_set, epochs):
        C = self.N*(self.data_size+1)
        #print("C_len = ",C)
        F = np.array([0 for _ in range(C*len(train_set))], dtype=float).reshape((len(train_set), C))
        #print("F_init = ", F)
        Y = np.array([0 for _ in range(len(train_set))], dtype=float)
        #print("Y_init = ", Y)
        for _ in range(epochs):
            for i in range(len(train_set)):
                self.work(train_set[i])
                Y[i] = train_category_set[i]
                #print(f"Y_{i} = ", Y)
                for k in range(self.N):
                    print(f"f_{k} = ", self.conclusion_layer[k].f)
                    F[i][(self.data_size+1)*k:(self.data_size+1)*(k+1)]=self.conclusion_layer[k].f
                    #print(f"F_{k} = ", F)
            
            print("F = ", F)
            A = np.linalg.lstsq(F,Y,rcond=None)[0]
            print("A = ", A)

            for data in train_set:
                y = self.work(data)
                for i in range(self.data_size):
                    for k in range(self.N):
                        const_k = self.conclusion_layer[k].y - self.normalization_layer[k].h*y
                        m= self.fuzzyfication_layer[2*i +k].mean
                        self.fuzzyfication_layer[2*i +k].mean += 2*(data[i] - m)/(self.fuzzyfication_layer[2*i +k].sigma)**2
                        self.fuzzyfication_layer[2*i +k].sigma += 2*(data[i] - m)**2/(self.fuzzyfication_layer[2*i +k].sigma)**3
        


            

