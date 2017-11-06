import random
import math
from math import exp
import sys

import numpy

class DE:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, pop_size):
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))  # array size rows = # hidden nodes, cols = # if inputs
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.pop_size = pop_size
        self.population = []  # for GA, contains weights
        self.population_error = numpy.ones(shape=(pop_size, 1))
        self.threshold = 100
        self.num_of_hidden = num_of_hidden
        self.activation_type = "s"

    def feed_DE(self, row, individual):
        self.v_1 = self.input_values[row]                                     #index is input value row, self.population[individual][0] is correct way to access pop weights
        #print(self.v_1)
        # print(self.v_1)
        temp_node_val = self.population[individual][0].dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.population[individual][1].dot(self.hidden_node_val)
        #print(self.output)
        return self.output

    def initialize_de(self):
        self.population = []
        for i in range(self.pop_size):
            id_array = numpy.identity(len(self.w_1), dtype=float)  # create an identity matrix for setting rand values to wieghts
            for j in range(len(id_array)):  # apply rand values to actual values in identity
                for k in range(len(id_array)):
                    id_array[j][k] = random.random()
            for j in range(len(self.w_1)):
                for k in range(len(self.w_1[0])):
                    self.w_1[j][k] = random.random()
            temp_1 = id_array.dot(self.w_1)  # multiply rand identity to weight matrix 1
            temp_2 = self.w_2.dot(id_array)
            self.population.append([temp_1, temp_2])

    def calc_fit(self, individual, data_row, output):
        #print(int(self.expected[data_row][0]))
        self.population_error[individual] += int(self.expected[data_row][0]) - output

    def evolve(self):
        graded = []
        for i in range(len(self.population_error)):
            add_to_graded = (self.population_error[i], i)
            graded.append(add_to_graded)
        graded.sort()
        print(graded)
        # parents = graded[:(int(len(graded) * retain))]
        # diversify(rate, retain, graded)
        # mutate(parents, mutatation)
        # return crossover(population, parents)

    def activation(self, value):
        if (self.activation_type == "s"):                           #for activation in ff
            return self.sigmoid(value)                              #use "s" for sigmoid, set in __init__ file
        else:
            print("Activation Function Mismatch")

    def sigmoid(self, value):
        return 1.0 / (1.0 + exp(-value))