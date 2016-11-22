# -*- coding: utf-8 -*-
import random
from operator import add, mul, sub
from math import exp, ceil, log
import numpy as np
from skimage.io import imread
from skimage.color import rgb2grey
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec

class LambdaRankEstimator:
    def __init__(self) :
        pass
    def init(self, count_input, count_layers=1, act_funcs=None, der_act_func=None,
             count_neurals_layer=None, learning_rate=0.1, shuffle = True) :
        defalt_count_neural = 10
        if count_neurals_layer is None:
            count_neurals_layer = []
            for l in range(count_layers):
                count_neurals_layer.append(defalt_count_neural)
        if act_funcs is not None :
            self.act_funcs = act_funcs
            self.der_act_func = der_act_func
        else :   
            self.act_funcs = []
            self.der_act_func = []
            for l in range(count_layers):
                self.act_funcs.append(logistic_activation_1) # max_0)
                self.der_act_func.append(der_logistic_activation_1) # der_max_0
        self.learning_rate = learning_rate
        self.count_output = 1
        self.shuffle = shuffle
        count_neurals_layer.append(self.count_output)
        self.act_funcs.append(equal_func)
        self.der_act_func.append(equal_func)
        #self.function_error = mean_square
        #self.dE_dz_last = der_mean_square - надо заменить!!!!!!!!!!!!!!!
        self.create_network(count_input, count_layers + 1, count_neurals_layer)
        self.initialize()

    def fit(self, data, add_step = 3, add_iteration=100, max_epoche=3000,
            coeff_R1=0, coeff_R2=0) :
        self.coeff_R1 = coeff_R1
        self.begin_coeff_R1 = coeff_R1
        self.coeff_R2 = coeff_R2
        self.begin_coeff_R2 = coeff_R2
        self.validation_error = []

        if self.shuffle:
            indexs = random.sample(range(0, data.shape[0]), data.shape[0])
            data_x = data[indexs]
            #y_true = y_true[indexs]
        
        count_valid = 4 * data.shape[0] / 10
        data_valid = data[:count_valid]
        data_train = data[count_valid:]

        best_error = 1e100
        best_weight = None
        number_epoche = None
        cur_step = 0
        current_epoche = 0
        while True :
            #epoche
            indexs = random.sample(range(0, data_train.shape[0]), data_train.shape[0])
            data_train = data_train[indexs]
            epoche_error = 0
            for iteration in range(data_train.shape[0]):#range(self.batch_count) :
                x = data_train[iteration][:, :-1]
                y_true = data_train[iteration][:, -1]
                y_pred = self.forward_propogation(x)

                """
                self.init_add_weight()
                error = 0
                for j in range(size_batch) :
                    n = iteration * size_batch + j
                    answer = self.forward_propogation(data_train[n])
                    self.back_propogation(data_train[n], answer, y_train[n])
                    error = error + self.function_error(y_train[n], answer)
                self.add_mean_gradient(size_batch)
                epoche_error = epoche_error + error
                """
            """
            epoche_error = epoche_error/ (self.batch_count * size_batch)
            print "Epoche error", epoche_error, 
            
            error = 0
            answer = self._predict_(data_x_valid)
            for i, y in enumerate(answer):
                error = error + self.function_error(y_valid[i], y)
            error = error / answer.shape[0]            
            print "Validation error", error
            
            if error <= best_error:
              #  print "New best error", error
                best_error = error
                best_weight = self.get_copy_weight(self.weight_network)
                self.validation_error.append(error)
                number_epoche = current_epoche
                
            
            if  number_epoche + add_iteration <= current_epoche:# and cur_step < add_step:
                cur_step = cur_step + 1                
                self.weight_network = self.get_copy_weight(best_weight)
                self.learning_rate = self.learning_rate / 2.0
                number_epoche = current_epoche
                print "New learning rate", self.learning_rate
                if cur_step > add_step:
                    self.count_epoche_for_learn = current_epoche - 1
                    break
            
            self.coeff_R1 = self.begin_coeff_R1 / ((current_epoche) ** 3 + 1)
            self.coeff_R2 = self.begin_coeff_R2 / ((current_epoche) ** 3 + 1)
            for i, x in enumerate(self.KL_probability):
                if x is not None:
                    prob, coeff = self.KL_begin_probability[i]
                    self.KL_probability[i] = (prob, coeff / ((current_epoche) ** 3 + 1))

            current_epoche = current_epoche + 1    
            if current_epoche > max_epoche:
                self.weight_network = self.get_copy_weight(best_weight)
                self.count_epoche_for_learn = current_epoche - 1
                break
            """


        
    def get_copy_weight(self, weights) :
        new_weight = []
        for w in weights:
            new_weight.append(w.copy())
        return new_weight
        
    def _predict_(self, data) :
        y = np.empty((data.shape[0], self.count_output))
        for i, x in enumerate(data):
            y[i, :] = self.forward_propogation(x)
        return y
        
    def predict(self, data):
        y_answer = self._predict_(data)
        if self.isClassification:
            y_pred = np.empty(y_answer.shape[0])
            for i, y in enumerate(y_answer):            
                y_pred[i] = self.uniq_label[y.argmax()]
            return y_pred        
        return y_answer
    
    def init_add_weight(self) :
        self.add_weight = []
        for layer in self.weight_network:
            self.add_weight.append(np.zeros(layer.shape))

    def create_network(self, count_input, count_layers, count_neurals_layer) :
        self.weight_network = []
        self.KL_probability = []
        self.KL_begin_probability = []
        count_neurals_layer = count_neurals_layer[:]
        count_neurals_layer.insert(0, count_input)    
        for l in range(1, count_layers + 1) :
            self.weight_network.append(np.zeros((count_neurals_layer[l], count_neurals_layer[l-1])))
            self.KL_probability.append(None)
            self.KL_begin_probability.append(None)

    def initialize(self, mean=0, var=1.0/800) :
        for l, layer in enumerate(self.weight_network) :
            var = 1.0 / layer.shape[1] #/ 10.0 #* 25
            weights = [[random.gauss(mean, var) for i in range(layer.shape[1])] for j in range(layer.shape[0])]
            self.weight_network[l] = np.array(weights)

    def set_KL_probality(self, layer, prob, coeff):
        self.KL_probability[layer] = (prob, coeff)
        self.KL_begin_probability[layer] = (prob, coeff)
        
    def forward_propogation(self, data) :
        cur_x = data
        self.x = []
        self.dy_dz = []
        self.x.append(cur_x)
        for l, weight_matrix in enumerate(self.weight_network):
            z = np.dot(weight_matrix, cur_x)
            func = (self.act_funcs)[l]
            der_func = self.der_act_func[l]
            cur_x = func(z)
            self.x.append(cur_x)
            self.dy_dz.append(der_func(z))
        """
        if self.isClassification :
            cur_x = self.soft_max(cur_x)
            self.x.pop()
            self.x.append(cur_x)
        """
        return cur_x
        
    def back_propogation(self, data, answer, true_y) :
        dE_dz = self.dE_dz_last(true_y, answer)        
        l = len(self.add_weight)
        prev_x = self.x[l - 1]
        dE_dz = dE_dz.reshape((dE_dz.shape[0], 1))
        prev_x = prev_x.reshape((1, prev_x.shape[0]))
        self.add_weight[l - 1] = self.add_weight[l - 1]  + np.dot(dE_dz, prev_x
                ) + self.coeff_R1 * sign(self.weight_network[l - 1]
                ) + self.coeff_R2 * self.weight_network[l - 1]
        for i in range(l - 2, -1, -1):
            tmp1_matrix = np.dot(self.weight_network[i+1].T , dE_dz)
            prev_x = self.x[i]
            prev_x = prev_x.reshape((1, prev_x.shape[0]))
            dy_dz = self.dy_dz[i]
            dy_dz = dy_dz.reshape((dy_dz.shape[0], 1))
            tmp2_matrix = np.dot(dy_dz, prev_x)
            add_weight = tmp2_matrix * tmp1_matrix  
            add_R1 = self.coeff_R1 * sign(self.weight_network[i]) 
            add_R2 = self.coeff_R2 * self.weight_network[i]         
            if self.KL_probability[i] is not None:
                prob, coeff = self.KL_probability[i]
                add_dE_dz = - dy_dz * (log(prob) / self.weight_network[i].shape[0])
                add_KL_reg = np.dot(add_dE_dz, prev_x) * coeff
            else:
                add_dE_dz = 0
                add_KL_reg = 0
            add_weight = add_weight + add_R1 + add_R2 + add_KL_reg       
            
            self.add_weight[i] = self.add_weight[i]  + add_weight
            dE_dz = tmp1_matrix * dy_dz + add_dE_dz
            
    def add_mean_gradient(self, count_iteration):
        for i, layer in enumerate(self.add_weight) : 
           # print "layer" , i
           # print "w", self.weight_network[i][:3, :3]
           # print "add", (- self.learning_rate * layer / count_iteration)[:3, :3]
            self.weight_network[i] = self.weight_network[i] - self.learning_rate * layer / count_iteration
      
    def print_weight(self):
        for l, layer in enumerate(self.weight_network) :   
            print("LAYER:", l, "shape", layer.shape)
            for n, neuron in enumerate(layer):
                print("neuron:", n)
                for weight in neuron:
                    print(weight)
                print('')

def sign(data):
    return (data > 0) * 1.0  - (data < 0) * 1.0

def mean_square(y, answer):
    return sum((y - answer) ** 2) / 2.0
def der_mean_square(y, answer):
    return answer - y
    
def entropy(y, answer) :
    return sum(-y * np.log(answer + 1e-20))
def der_soft_max(y, answer):
    return answer - y
    
def equal_func(z) :
    return z  
def one_func (z) :
    return 1.0;
    
def logistic_activation_1(z) :
    return logistic_activation_a(1.0, z)
def der_logistic_activation_1(z) :
    return der_logistic_activation_a(1.0, z)
    
def max_0(z) :
    return (z > 0) * z
def der_max_0(z):
    return (z > 0) * 1.0

                
def logistic_activation_a(a, z) :
    return 1.0 / (1 + np.exp(-a * z))
def der_logistic_activation_a(a, z):
    return a*logistic_activation_a(a, z)*(1.0 - logistic_activation_a(a, z))
    
def load_data() :
    path = './big_alphabet_29x29/mutant-'
    count_char = 25
    count_example = 8
    image = rgb2grey(imread('./big_alphabet_29x29/mutant-0-0-0.bmp'))
    size = image.shape[0] * image.shape[1]
    data_x = np.zeros((count_char * count_example, size))
    y = np.zeros(count_char * count_example)    

        
    for char in range(count_char) :
        for i in range(count_example):
            path_img = path + str(char) + '-' + str(i) + '-0.bmp'
            data_x[char * count_example + i, :] = rgb2grey(imread(path_img)).reshape(size)
            y[char * count_example + i] = char
    data_x =  data_x - 0.5
  #  data_x = data_x / np.max(np.abs(data_x))
    return data_x, y 
    
def acurancy(y_pred, y_true):
    return float(sum(y_pred == y_true)) / y_pred.shape[0]

if __name__ == '__main__':
    trainPath = "../data/train.data.cvs"
    rowData = DataFrame.from_csv(trainPath, index_col=False).as_matrix()
    data = DataFrame.from_csv(trainPath)
    queries = rowData[:, -1]
    uniq_queries = np.unique(queries)
    data = []
    for q in uniq_queries:
        data.append(rowData[queries == q][:, :-1])

"""
def MAIN_CHECK_NEURAL_NETWORK() :     
    data, y = load_data()
    count_input = data.shape[1]
    count_output = np.unique(y).shape[0]
    
    indexs = random.sample(range(0, data.shape[0]), data.shape[0])
    data= data[indexs]
    y = y[indexs]
    
    count_test=  5 * data.shape[0] / 10
    data_x_vtest = data[:count_test]
    y_test= y[:count_test]
    data_train = data[count_test:]
    y_train = y[count_test:]
    
    count_layers0 = 2
    count_neurals_layer0 = [25, 25]
    network0 = neuralNetwork()    
    network0.init(count_input=count_input, count_output=count_output, count_layers=count_layers0,
        count_neurals_layer=count_neurals_layer0, batch_count=1, learning_rate=0.3, shuffle=False)
    network0.fit(data_train, y_train, add_step=7, add_iteration=500, max_epoche=20000)
    print acurancy(network0.predict(data_x_vtest), y_test)
    
   
    count_layers = 1
    count_neurals_layer = [25]
    network1 = neuralNetwork()    
    network1.init(count_input=count_input, count_output=count_output, count_layers=count_layers,
        count_neurals_layer=count_neurals_layer, batch_count=1, learning_rate=0.3, shuffle=False)
    network1.fit(data_train, y_train)
    
    count_layers2 = 0
    count_neurals_layer2 = []
    network2 = neuralNetwork()
    network2.init(count_input=count_input, count_output=count_output, count_layers=count_layers2,
        count_neurals_layer=count_neurals_layer2, batch_count=1, learning_rate=0.3, shuffle=False)
    network2.fit(data_train, y_train)
    
    print acurancy(network0.predict(data_x_vtest), y_test)
    print acurancy(network1.predict(data_x_vtest), y_test)
    print acurancy(network2.predict(data_x_vtest), y_test)
 
def show_image_digits(data, pred_data):
    for i, image in enumerate(data): 
        plt.figure(figsize=(5, 5))
        plt.subplot(121)
        plt.imshow(image.reshape((29, 29)), cmap='Greys_r')
        plt.subplot(122)
        plt.imshow(pred_data[i].reshape((29, 29)), cmap='Greys_r')
        plt.show()
        
def MAIN_AUTOENCODER():
    data, y = load_data()
    count_input = data.shape[1]
    count_output = data.shape[1]
    
    indexs = random.sample(range(0, data.shape[0]), data.shape[0])
    data= data[indexs]
    y = y[indexs]
    
    count_test=  3 * data.shape[0] / 10
    data_x_test = data[:count_test]
    data_train = data[count_test:]
   
    count_layers = 1
    count_neurals_layer = [25]
    network1 = neuralNetwork()    
    network1.init(count_input=count_input, count_output=count_output, count_layers=count_layers,
        count_neurals_layer=count_neurals_layer, batch_count=3, learning_rate=0.03, shuffle=False, isClassification=False)
    network1.set_KL_probality(0, 0.05, 1e-1) #1e-4
    network1.fit(data_train, data_train, max_epoche=40000, coeff_R2=5e-4, coeff_R1=1e-4)
    
    pred_data = network1.predict(data_x_test) + 0.5
    pred_data = (pred_data > 0.5) * 1.0
    show_image_digits(data_x_test, pred_data)
        
def MAIN_EXAMPLE_REGULARIZATION():
    data, y = load_data()
    count_input = data.shape[1]
    count_output = data.shape[1]
    
    indexs = random.sample(range(0, data.shape[0]), data.shape[0])
    data= data[indexs]
    y = y[indexs]
    
    count_test=  3 * data.shape[0] / 10
    data_test = data[:count_test]
    y_test = y[:count_test]
    data_train = data[count_test:]
    y_train = y[count_test:]
    
    count_output = 25
    count_layers = 1
    count_neurals_layer = [25]
    network1 = neuralNetwork()    
    network1.init(count_input=count_input, count_output=count_output, count_layers=count_layers,
        count_neurals_layer=count_neurals_layer, batch_count=3, learning_rate=0.3, shuffle=False)
    network1.fit(data_train, y_train, coeff_R2=5e-5, coeff_R1=1e-5)
    
    count_layers = 1
    count_neurals_layer = [25]
    network2 = neuralNetwork()    
    network2.init(count_input=count_input, count_output=count_output, count_layers=count_layers,
        count_neurals_layer=count_neurals_layer, batch_count=3, learning_rate=0.3, shuffle=False)
    network2.fit(data_train, y_train)
    
    
    print acurancy(network1.predict(data_test), y_test), "count epoche", network1.count_epoche_for_learn
    print acurancy(network2.predict(data_test), y_test), "count epoche", network2.count_epoche_for_learn

    plt.plot(range(len(network1.validation_error)), network1.validation_error, color='r')
    plt.plot(range(len(network2.validation_error)), network2.validation_error, color='b')
    plt.legend(['With Reg', 'Without Reg'], loc='top right')
    plt.show()
"""



    
    
    
