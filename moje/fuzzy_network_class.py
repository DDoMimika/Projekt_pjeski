import numpy as np
import pandas as pd
import paths
from neuron_class import *

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

    def firing_layer_work(self):
        for i in range(len(self.firing_layer)):
            self.firing_layer[i].firing(self.fuzzyfication_layer, i, self.N)

    def normalization_layer_work(self):
        for i in range(len(self.normalization_layer)):
            self.normalization_layer[i].normalization(self.firing_layer[i].h, self.firing_layer)

    def conclusion_layer_work(self,data):
        for i in range(len(self.conclusion_layer)):
            self.conclusion_layer[i].conclusion(data, self.normalization_layer[i].h)

    def result_layer_work(self):
        self.result_layer.outcome(self.conclusion_layer)
        self.net_rasult = self.result_layer.y

    def work(self, data):
        self.fuzzyfication_layer_work(data)
        self.firing_layer_work()
        self.normalization_layer_work()
        self.conclusion_layer_work(data)
        self.result_layer_work()

        return self.net_rasult

    def membership_parameters_learn(self, train_input, train_outpout,):
        seria = pd.DataFrame(np.concatenate((train_input,np.array([train_outpout]).T), axis=1))
        parameters = np.ones(self.N*self.data_size*2).reshape(self.N*self.data_size,2)
        for i in range(self.data_size):
            for k in range(self.N):
                parameters[k+(self.N)*i][1]= seria.loc[seria[self.data_size] == k][i].var()
                parameters[k+(self.N)*i][0] = seria.loc[seria[self.data_size] == k][i].mean()
            print(f"{i} pixel nauczony")
        np.save(paths.PATH_MEM_PARAMETERS, parameters) 

    def pre_learn(self, train_input, train_output, memebership_from_file = True, conclusion_from_file = True):
        if memebership_from_file:
            parameters = np.load(paths.PATH_MEM_PARAMETERS)
            for i in range(self.data_size):
                for k in range(self.N):
                    self.fuzzyfication_layer[k+(self.N)*i].sigma = parameters[k+(self.N)*i][1]
                    self.fuzzyfication_layer[k+(self.N)*i].mean = parameters[k+(self.N)*i][0]

            print("membership_parameters nauczone")

        else:
            seria = pd.DataFrame(np.concatenate((train_input,np.array([train_output]).T), axis=1))
            for i in range(self.data_size):
                for k in range(self.N):
                    self.fuzzyfication_layer[k+(self.N)*i].sigma = seria.loc[seria[self.data_size] == k][i].var()
                    self.fuzzyfication_layer[k+(self.N)*i].mean = seria.loc[seria[self.data_size] == k][i].mean()

            print("membership_parameters nauczone")

        if conclusion_from_file:
            parameters = np.load(paths.PATH_CON_PARAMETERS)
            for k in range(self.N):
                self.conclusion_layer[k].a = parameters[k]
            print("conclusion_parameters nauczone")

        else:
            C = self.N*(self.data_size+1)
            F = np.array([0 for _ in range(C*len(train_input))], dtype=float).reshape((len(train_input), C))

            for i in range(len(train_input)):
                self.work(train_input[i])
                for k in range(self.N):
                    F[i][(self.data_size+1)*k:(self.data_size+1)*(k+1)]=self.conclusion_layer[k].f

            A = np.linalg.lstsq(F,train_output,rcond=None)[0]
            parameters = np.zeros(C).reshape(self.N, self.data_size+1)
            for k in range(self.N):
                parameters[k] = A[(self.data_size+1)*k:(self.data_size+1)*(k+1)]
                self.conclusion_layer[k].a = A[(self.data_size+1)*k:(self.data_size+1)*(k+1)]

            np.save(paths.PATH_CON_PARAMETERS, parameters)
            print("conclusion_parameters nauczone")
     
    def accuracy(self, test_input, test_output, precision):
        t = 0
        # print(len(test_input))
        for i in range(len(test_input)): 
            # if i %500 == 0: print(i)
            y = self.work(test_input[i])
            d=abs(test_output[i]-y)
            if d<precision:
                    t+=1
        print(f"  accuracy = {t/len(test_input)*100 } %")
        print(f"  error = {d**2/2}")

    def mi_der_mean(self, m, s, x):
        y = -(x-m)/s
        return y

    def mi_der_sigma(self, m, s, x):
        y = (x-m)**2/(2*s**2)
        return y
            
    def train(self, train_input, train_output, test_input, test_output, epochs, precision = 0.5, mi =0.5):
        self.pre_learn(train_input, train_output)   # zczytywanie danych z pliku
        print("Pre_learn")
        self.accuracy(test_input, test_output, precision)
       
        C = self.N*(self.data_size+1)
        F = np.array([0 for _ in range(C*len(train_input))], dtype=float).reshape((len(train_input), C))
        
        for e in range(epochs):
            print("Epoka: ", e+1)
            er = 0
            for n in range(len(train_input)): 
                y = self.work(train_input[n])
                er = (y-train_output[n])**2/2
                # UCZENIE WSTECZNĄ PROPAGACJĄ BŁĘDU WYNIKAJĄCE Z T-NORMY ALGEBRAICZNEJ
                p = train_output[n]
                for i in range(self.data_size):
                    m = self.fuzzyfication_layer[self.N*i +p].mean
                    s = self.fuzzyfication_layer[self.N*i +p].sigma
                    self.fuzzyfication_layer[self.N*i +p].mean -= mi*(y - train_output[n])*(self.conclusion_layer[p].y-y*self.normalization_layer[p].h)*self.mi_der_mean(m, s, train_input[n][i])
                    self.fuzzyfication_layer[self.N*i +p].sigma -= mi*(y - train_output[n])*(self.conclusion_layer[p].y-y*self.normalization_layer[p].h)*self.mi_der_sigma(m, s, train_input[n][i])
            print("  error = ", er)  
            # for n in range(57):
            #     self.work(train_input[n])
            #     for k in range(self.N):
            #         F[n][(self.data_size+1)*k:(self.data_size+1)*(k+1)]=self.conclusion_layer[k].f.copy()

            # A = np.linalg.lstsq(F,train_output,rcond=None)[0]

            # for k in range(self.N):
            #     self.conclusion_layer[k].a = A[(self.data_size+1)*k:(self.data_size+1)*(k+1)].copy()

            self.accuracy(test_input, test_output, precision) 
        

            
        


            

