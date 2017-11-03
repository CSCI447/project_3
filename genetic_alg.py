import random
import math
from math import exp

import numpy

class GA:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate):
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))             #array size rows = # hidden nodes, cols = # if inputs
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))                #array size rows = # output nodes, cols = # hidden nodes
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.pop_size = pop_size
        self.num_tournament_participants = num_tournament_participants
        self.num_tournament_victors = num_tournament_victors
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []  # for GA, contains weights
        self.population_error = []  # for GA
        # self.threshold = threshold
        self.activation_type = "s"

    def feed_GA(self, index):                           #index is
        self.v_1 = self.input_values[index]
        temp_node_val = self.v_1.dot(self.w_1.T)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.hidden_node_val.dot(self.w_2.T)
        return self.output

    def genetic_alg(self, index, output):
        self.calc_fit(index, output)
        self.tournament()
        # self.crossover()
        # self.mutate()
        # self.winner()

    def calc_fit(self, index, output):
        #print(self.population_error())
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
        #it = 0
        for i in range(len(self.input_values)):
            output = self.feed_GA(i)
            sum_error += (int(expectedOutputArray[i][0]) - output)
            #it += 1
            #print("Error = %f, Iteration = %d" % (sum_error, it))

    def initialize_ga(self):
        self.population = []
        for i in range(self.pop_size):
            id_array = numpy.identity(len(self.w_1), dtype=float)  # create an identity matrix for setting rand values to wieghts
            for i in range(len(id_array)):  # apply rand values to actual values in identity
                id_array[i][i] = random.random()
            self.w_1 = id_array.dot(self.w_1)  # multiply rand identity to weight matrix 1
            self.w_2 = self.w_2.dot(id_array)
            self.population.append([self.w_1, self.w_2])
    def activation(self, value):
        if (self.activation_type == "s"):                           #for activation in ff
            return self.sigmoid(value)                              #use "s" for sigmoid, set in __init__ file
        else:
            print("Activation Function Mismatch")

    def sigmoid(self, value):
        return 1.0 / (1.0 + exp(-value))