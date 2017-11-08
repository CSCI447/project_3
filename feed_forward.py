import random
import math
from math import exp

import numpy

class FF:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, activation_type):

        self.input_values = numpy.array(inputs, dtype=float)                    #init variables for bp
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(inputs, dtype=float)
        self.test_outputs = numpy.array(expected, dtype=float)
        self.hidden_node_val = numpy.zeros(shape=(num_of_hidden, 1))
        self.output = numpy.zeros(shape=(num_of_outs, 1))
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))
        self.error_1 = numpy.zeros(shape=(num_of_outs, 1))
        self.error_2 = numpy.zeros(shape=(num_of_hidden, 1))
        self.learn_rate = .00001                                                #.00001 for backprop, not higher
        self.activation_type = activation_type
        self.v_1 = numpy.ones(shape=(len(self.input_values[0]), 1))

    def update_in_out(self, inputs, expected, test_in, test_expected):          #used for crossvalidation to update the used train and test data
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(test_in, dtype=float)
        self.test_outputs = numpy.array(test_expected, dtype=float)

    def feed_forward(self, index):                                              #runs data through the newtwork with the current weight configurations
        self.v_1 = self.input_values[index]
        temp_node_val = self.v_1.dot(self.w_1.T)                                #process through first weights=
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])                #activation function
        self.hidden_node_val = temp_node_val
        self.output = self.hidden_node_val.dot(self.w_2.T)                      #process through next weights
        return self.output                                                      #return output

    def feed_forward_test(self, index):                                         #just like last function but for testing varibales spec
        self.v_1 = self.test_inputs[index]
        temp_node_val = self.v_1.dot(self.w_1.T)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.hidden_node_val.dot(self.w_2.T)
        return self.output

    def backprop(self, index):                                                      #helper function to run backprop functions
        self.update_output_error(index)
        self.update_w_2()
        self.update_hidden_error()
        self.update_w_1(index)

    def update_output_error(self, index):                                                   #update the error of the output nodes
        unprocessed_error = int(self.expected[index][0]) - self.output
        self.error_1 = unprocessed_error * self.linear_derivative(self.output)              #(actual value - output) * linear derivative

    def update_w_2(self):                                                                   #update weights between hidden layer nad output layer
        weight_update = self.hidden_node_val * self.error_1 * self.learn_rate
        self.w_2 += weight_update

    def update_hidden_error(self):                                                          #update the error of the hidden layer nodes
        unprocessed_error = self.w_2 * self.error_1                                         #weights between hidden and output nodes * error of output nodes
        self.error_2 = unprocessed_error * self.transfer_derivative(self.hidden_node_val)   #values * transfer derivative

    def update_w_1(self, index):                                                            #update weights between input layer and hidden layer
        weight_update = self.v_1 * self.error_2.T * self.learn_rate
        self.w_1 += weight_update

    def linear_derivative(self, output):                                                    #helper function for linear derivative, although obviously not that helpful
        return 1

    def transfer_derivative(self, input):                                                   #transfer derivative function
        input2 = numpy.copy(input)
        for i in range(len(input2)):
            input2[i] = input2[i] * (1.0 - input2[i])
        return input2

    def sigmoid(self, value):                                                               #sigmoid function
        return 1.0 / (1.0 + exp(-value))

    def initialize(self):
        id_array = numpy.identity(len(self.w_1), dtype=float)       #create an identity matrix for setting rand values to wieghts
        for i in range(len(id_array)):                              #apply rand values to actual values in identity
            id_array[i][i] = random.random()
        self.w_1 = id_array.dot(self.w_1)                           #multiply rand identity to weight matrix 1
        self.w_2 = self.w_2.dot(id_array)                           #multiply rand identity to weight matrix 2

    def activation(self, value):
        if (self.activation_type == "s"):                           #for activation in ff
            return self.sigmoid(value)                              #use "s" for sigmoid, set in __init__ file
        else:
            print("Activation Function Mismatch")                   #error check
