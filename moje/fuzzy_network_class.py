import numpy as np
import pandas as pd
import paths
from neuron_class import *
from functions import IMAGE_SIZE, CLASS_SIZE

class fuzzy_network:
    def __init__(self, N, data_size):
        self.N = N
        self.data_size = data_size
        self.fuzzyfication_layer = np.array([fuzzyfication_neuron() for _ in range(N* data_size)])
        self.firing_layer = np.array([firing_neuron() for _ in range(N)])
        self.normalization_layer = np.array([normalizaton_neuron() for _ in range(N)])
        self.conclusion_layer = np.array([conclusion_neuron(data_size) for _ in range(N)])
        self.result_layer = result_neuron()
        self.net_rasult = 0

    def fuzzyfication_layer_work(self,data):
        input_data= np.repeat(data, self.N)
        for i in range(len(self.fuzzyfication_layer)):
            self.fuzzyfication_layer[i].gauss_function(input_data[i])
            #print(f"fuzzyfication_layer[{i}] = {self.fuzzyfication_layer[i].output}")

    def firing_layer_work(self):
        for i in range(len(self.firing_layer)):
            self.firing_layer[i].firing(self.fuzzyfication_layer, i, self.N)
            #print(f"firing_layer[{i}] = {self.firing_layer[i].h}")

    def normalization_layer_work(self):
        for i in range(len(self.normalization_layer)):
            self.normalization_layer[i].normalization(self.firing_layer[i].h, self.firing_layer)
            #print(f"normalization_layer[{i}] = {self.normalization_layer[i].h}")

    def conclusion_layer_work(self,data):
        for i in range(len(self.conclusion_layer)):
            self.conclusion_layer[i].conclusion(data, self.normalization_layer[i].h)
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

    def pre_train(self, train_input, train_outpout, from_file = False):
        if from_file:
            #f_handle = np.file(paths.PATH_PAREMETRES, 'a')
            seria = pd.DataFrame(np.concatenate((train_input,np.array([train_outpout]).T), axis=1))
            parameters = np.ones(self.N*self.data_size*2).reshape(self.N*self.data_size,2)
            for i in range(self.data_size):
                for k in range(self.N):
                    v = seria.loc[seria[self.data_size] == k][i].var()
                    m = seria.loc[seria[self.data_size] == k][i].mean()
                    parameters[k+(self.N)*i][0]=m
                    parameters[k+(self.N)*i][1]=v
                    self.fuzzyfication_layer[k+(self.N)*i].sigma = v
                    self.fuzzyfication_layer[k+(self.N)*i].mean = m

                print(f"{i} pixel nauczony")
                #print(f"m = {m}, s = {v}")
            np.save(paths.PATH_PARAMETERS, parameters)
            #f_handle.close()
        else:
            print("jestem")
            parameters = np.load(paths.PATH_PARAMETERS)
            print(parameters)
            for i in range(self.data_size):
                for k in range(self.N):
                    self.fuzzyfication_layer[k+(self.N)*i].sigma = parameters[k+(self.N)*i][1]
                    self.fuzzyfication_layer[k+(self.N)*i].mean = parameters[k+(self.N)*i][0]

                print(f"{i} pixel nauczony")
                #print(f"m = {m}, s = {v}")
                


    def learning(self, train_set, train_category_set):
        print("C")
        C = self.N*(self.data_size+1)
        print("F")
        #print("C_len = ",C)
        F = np.array([0 for _ in range(C*len(train_set))], dtype=float).reshape((len(train_set), C))
        print("Y")
        #print("F_init = ", F)
        Y = np.array([0 for _ in range(len(train_set))], dtype=float)
        #print("Y_init = ", Y)

        print("uczenie konklucji")
        for i in range(len(train_set)):
            self.work(train_set[i])
            Y[i] = train_category_set[i]
            #print(f"Y_{i} = ", Y)
            for k in range(self.N):
                #print(f"f_{k} = ", self.conclusion_layer[k].f)
                F[i][(self.data_size+1)*k:(self.data_size+1)*(k+1)]=self.conclusion_layer[k].f
                #print(f"F_{k} = ", F)

        #print("F = ", F)
        A = np.linalg.lstsq(F,Y,rcond=None)[0]
        #print("A = ", A)

        for k in range(self.N):
            #print("k = ", k)
            #print("przed: ", self.conclusion_layer[k].a)
            self.conclusion_layer[k].a = A[(self.N+1)*k:(self.N+1)*(k+1)]
            #print("po: ", self.conclusion_layer[k].a)

        print("zmiana sigm i średnich")
        for data in train_set:
            y = self.work(data)
            for i in range(self.data_size):
                for k in range(self.N):
                    const_k = self.conclusion_layer[k].y - self.normalization_layer[k].h*y
                    m= self.fuzzyfication_layer[2*i +k].mean
                    self.fuzzyfication_layer[2*i +k].mean += 2*(data[i] - m)/(self.fuzzyfication_layer[2*i +k].sigma)**2*const_k
                    self.fuzzyfication_layer[2*i +k].sigma += 2*(data[i] - m)**2/(self.fuzzyfication_layer[2*i +k].sigma)**3*const_k

    def train(self, train_set, train_category_set, test_set, test_category_set, epochs, precision = 0.1):
        for e in range(epochs):
            print("Zaczęta epoka: ", e+1)
            self.learning(train_set, train_category_set)

            print("Zaczęto accuracy")
            t = 0
            for i in range(len(test_set)):
                y = self.work(test_set[i])
                if y>= test_category_set[i]-precision and y<= test_category_set[i]+precision:
                    t+=1
            
            accuracy = t/len(test_set)
            print(f"epoch: {e+1}\n  accuracy = {accuracy*100} %")


            
        


            

