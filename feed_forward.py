import random
import math
from math import exp

import numpy

class FF:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, activation_type, pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate):

        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
################BACKPROP_VARIABLES######################
        self.hidden_node_val = numpy.zeros(shape=(num_of_hidden, 1))
        self.output = numpy.zeros(shape=(num_of_outs, 1))
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))             #array size rows = # hidden nodes, cols = # if inputs
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))                #array size rows = # output nodes, cols = # hidden nodes
        self.error_1 = numpy.zeros(shape=(num_of_outs, 1))
        self.error_2 = numpy.zeros(shape=(num_of_hidden, 1))
        self.learn_rate = .00001            #.00001 for backprop works, not higher
        self.activation_type = activation_type
        self.v_1 = numpy.ones(shape=(len(self.input_values[0]), 1))
################GA_VARIABLES######################
        self.pop_size = pop_size
        self.num_tournament_participants = num_tournament_participants
        self.num_tournament_victors = num_tournament_victors
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []                                    #for GA, contains weights
        self.population_error = []                         #for GA
        #self.threshold = threshold

    def feed_forward(self, index):
        self.v_1 = self.input_values[index]
        temp_node_val = self.v_1.dot(self.w_1.T)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.hidden_node_val.dot(self.w_2.T)
        return self.output

    def feed_GA(self, index):
        self.v_1 = self.input_values[index]
        temp_node_val = self.v_1.dot(self.w_1.T)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.hidden_node_val.dot(self.w_2.T)
        return self.output

################GA######################

    def genetic_alg(self, index, output):
        self.calc_fit(index, output)
        self.tournament()
        # self.crossover()
        # self.mutate()
        # self.winner()

    def calc_fit(self, index, output):
        self.population_error.append(int(self.expected[index][0]) - output)

    def tournament(self):
        particpants_chrome = []                  #random.sample(self.population, self.num_tournament_participants)
        particpants_error = []
        return_victors = []
        p = list(range(0, self.pop_size))
        part_indices = random.sample(p, self.num_tournament_participants)
        for i in range(self.num_tournament_participants):
            temp_chrome = (self.population[part_indices[i]], part_indices[i])
            particpants_chrome.append(temp_chrome)
            temp_error = self.population_error[part_indices[i]], part_indices[i]
            particpants_error.append(temp_error)
        sorted_error = sorted(particpants_error, key=lambda student: student[0])            #I think this is working, maybe not tho
        #print(sorted_error[0][1])
        for i in range(self.num_tournament_victors):
            return_victors.append(sorted_error[i][1])
        self.mate(return_victors)

    def mate(self, return_victors):
        children = []
        for p1 in return_victors:
            for p2 in return_victors:
                if not (p1 == p2):
                    children.append(self.crossover(p1, p2))
        # print(self.population)
        # print(children)
        self.population = children

    def crossover(self, p1, p2):
        p1 = self.population[p1]
        p2 = self.population[p2]
        current = 0
        child = []
        for i in range(len(p1)):
            if (random.random() < self.crossover_rate):
                current = (current+1)%2
            if (current == 0):
                child.append(p1[i])
            else:
                child.append(p2[i])
        return child


    def mutate(self):
        #mutation_array = numpy.identity(len(self.w_1), dtype=float)
        for j in range(len(self.population)):
            for i in range(len(self.population[0])):
                mutation_array = numpy.identity(len(self.w_2), dtype=float)
                for k in range(len(self.w_2)):
                    if random.random() < self.mutation_rate:
                        mutation_array[k][k] = random.random() - .5
                self.population[j][i] * mutation_array
        #print(self.population)

    def winner(self, expectedOutputArray):
        sum_error = 0
        for i in range(len(self.input_values)):
            output = self.feed_GA(i)
            sum_error += (int(expectedOutputArray[i][0]) - output)
        #print("Sum Error: %f" % sum_error)

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
        #unprocessed_update = self.hidden_node_val * self.error_1
        weight_update = self.hidden_node_val * self.error_1 * self.learn_rate
        self.w_2 += weight_update

    def update_hidden_error(self):
        unprocessed_error = self.w_2 * self.error_1
        self.error_2 = unprocessed_error * self.transfer_derivative(self.hidden_node_val)

    def update_w_1(self, index):
        weight_update = self.v_1 * self.error_2.T * self.learn_rate
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

    def initialize_ga(self):
        self.population = []
        for i in range(self.pop_size):
            id_array = numpy.identity(len(self.w_1), dtype=float)  # create an identity matrix for setting rand values to wieghts
            for i in range(len(id_array)):  # apply rand values to actual values in identity
                id_array[i][i] = random.random()
            self.w_1 = id_array.dot(self.w_1)  # multiply rand identity to weight matrix 1
            self.w_2 = self.w_2.dot(id_array)
            self.population.append([self.w_1, self.w_2])

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
