import random
import math
from math import exp

import numpy

class FF:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, activation_type):
        self.input_values = numpy.array(inputs, dtype=float)
        self.hidden_node_val = numpy.zeros(shape=(num_of_hidden))
        self.output = numpy.zeros(shape=(num_of_outs))
        self.expected = numpy.array(expected, dtype=float)
        self.w_1 = numpy.zeros(shape=(num_of_hidden, len(inputs[0])))             #array size rows = # hidden nodes, cols = # if inputs
        self.w_2 = numpy.zeros(shape=(num_of_outs, num_of_hidden))                #array size rows = # output nodes, cols = # hidden nodes
        self.error_1 = numpy.zeros(shape=(num_of_outs))
        self.error_2 = numpy.zeros(shape=(num_of_hidden))
        self.learn_rate = .00001            #.00001 for backprop works, not higher
        self.activation_type = activation_type

    def feed_forward(self, index):
        temp_node_val = self.input_values[index].dot(self.w_1.T)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.hidden_node_val.dot(self.w_2.T)
        return self.output

################BACKPROP######################

    def backprop(self, index):
        self.update_output_error(index)
        self.update_w_2()
        self.update_hidden_error()
        self.update_w_1(index)

    def update_output_error(self, index):
        unprocessed_error = int(self.expected[index][0]) - self.output
        self.error_1 = unprocessed_error * self.linear_derivative(self.output)

    def update_w_2(self):
        unprocessed_update = self.hidden_node_val * self.error_1
        weight_update = self.learn_rate * unprocessed_update
        self.w_2 += weight_update

    def update_hidden_error(self):
        unprocessed_error = self.w_2 * self.error_1
        self.error_2 = unprocessed_error * self.transfer_derivative(self.hidden_node_val)

    def update_w_1(self, index):
        unprocessed_update = numpy.zeros(shape=(len(self.error_2[0]), len(self.input_values[index])))
        for i in range(len(self.error_2[0])):
            for j in range(len(self.input_values[index])):
                unprocessed_update[i][j] = self.input_values[index][j] * self.error_2[0][i]
        weight_update = self.learn_rate * unprocessed_update
        self.w_1 += weight_update

    def linear_derivative(self, output):
        return 1

    def transfer_derivative(self, input):
        input2 = numpy.copy(input)
        for i in range(len(input2)):
            input2[i] = input2[i] * (1.0 - input2[i])
        return input2

    def sigmoid(self, value):
        return 1.0 / (1.0 + exp(-value))

################FUNCTIONS######################

    def initialize(self):
        for i in range(len(self.w_1)):                   #columns
            for j in range(len(self.w_1[0])):                  #rows
                self.w_1[i][j] = random.random()
        for i in range(len(self.w_2)):                   #columns
            for j in range(len(self.w_2[0])):                  #rows
                self.w_2[i][j] = random.random()

    def activation(self, value):
        if (self.activation_type == "s"):
            return self.sigmoid(value)
        else:
            print("Activation Function Mismatch")
