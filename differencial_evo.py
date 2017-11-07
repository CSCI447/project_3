import random
import math
from math import exp
import sys

import numpy

class DE:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, pop_size, mutation_rate, crossover_rate):
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))  # array size rows = # hidden nodes, cols = # if inputs
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(expected, dtype=float)
        self.test_outputs = numpy.array(expected, dtype=float)
        self.pop_size = pop_size
        self.population = []  # for GA, contains weights
        self.population_error = numpy.ones(shape=(pop_size, 1))
        self.child_error = numpy.ones(shape=(3, 1))
        self.threshold = .001
        self.num_of_hidden = num_of_hidden
        self.activation_type = "s"
        self.trial_vector_size = 3
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.all_runs = []

    def update_in_out(self, inputs, expected, test_in, test_expected):
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(test_in, dtype=float)
        self.test_outputs = numpy.array(test_expected, dtype=float)

    def feed_DE(self, row, individual):
        self.v_1 = self.input_values[row]                                     #index is input value row, self.population[individual][0] is correct way to access pop weights
        feed_w1 = numpy.array(self.population[individual][0])
        feed_w2 = numpy.array(self.population[individual][1])
        temp_node_val = feed_w1.dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = feed_w2.dot(self.hidden_node_val)
        return self.output

    def feed_DE_children(self, row, child):
        self.v_1 = self.input_values[row]                                     #index is input value row, self.population[individual][0] is correct way to access pop weights
        child_w1 = numpy.array(child[0])
        child_w2 = numpy.array(child[1])
        temp_node_val = child_w1.dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        hidden_val = temp_node_val
        output = child_w2.dot(hidden_val)
        return output

    def feed_testing_DE(self, row, individual):
        self.v_1 = self.test_inputs[row]                                    #index is input value row, self.population[individual][0] is correct way to access pop weights
        # print(self.population[individual][1])
        # print(individual)
        # sys.exit()
        test_w1 = numpy.array(self.population[individual][0])
        test_w2 = numpy.array(self.population[individual][1])
        #temp_node_val = self.population[individual][0].dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        temp_node_val = test_w1.dot(self.v_1)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        #self.output = self.population[individual][1].dot(self.hidden_node_val)
        self.output = test_w2.dot(self.hidden_node_val)
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
        self.population_error[individual] += int(self.expected[data_row][0]) - output

    def calc_fit_children(self, individual, data_row, output):
        self.child_error[individual] += int(self.expected[data_row][0]) - output

    def evolve(self):
        for index, individual in enumerate(self.population):
            children = []
            p = list(range(0, self.pop_size -1))
            trial_indices = random.sample(p, self.trial_vector_size)
            for i in range(len(trial_indices)):                                  #make sure no asex reproduction, works!
                if (trial_indices[i] == index):
                    trial_indices[i] += 1
            children = self.mate(index, trial_indices)
            output = numpy.ones(shape=(self.pop_size, 1))
            for j, child in enumerate(children):
                for i in range(len(self.input_values)):
                    output[j] = self.feed_DE_children(i, child)
                    self.calc_fit_children(j, i, output[j])
                if (self.child_error[j] < self.population_error[index]):
                    # print("touched")
                    self.population[index] = child

    def mate(self, p1, trial_vector):
        children = []
        for p2 in trial_vector:
            children.append(self.crossover(p1, p2))
        return children

    def crossover(self, p1, p2):
        # print(p1)
        # print(p2)
        p1 = self.population[p1]
        p2 = self.population[p2]
        current = 0
        weight_1 = []
        weight_2 = []
        child = []
        for i in range(len(p1[0])):
            if (random.random() < self.crossover_rate):
                current = (current+1)%2
            if (current == 0):
                weight_1.append(p1[0][i])
            else:
                weight_1.append(p2[0][i])
        for i in range(len(p1[1])):
            if (random.random() < self.crossover_rate):
                current = (current+1)%2
            if (current == 0):
                weight_2.append(p1[1][i])
            else:
                weight_2.append(p2[1][i])
        child.append(weight_1)
        child.append(weight_2)
        return child

    def mutate(self):
        for j in range(len(self.population)):
            for i in range(len(self.population[0])):
                if random.random() < (self.mutation_rate + 1):
                    #print(self.population[j][i])
                    self.population[j][i] += (random.random() * (self.population_error[j]*0.001)/self.pop_size)

    def winner(self, epoch):
        print("Error = %f" % (self.population_error[0]))
        sum_error = 0
        it = 0
        best_chrome = []
        for individual in range(self.pop_size):
            if ((self.population_error[individual] < self.threshold) and (self.population_error[individual] > -(self.threshold))):
                chrome = []
                for i in range(self.num_of_hidden):
                    for j in range(len(self.w_1[0])):
                        chrome.append(self.population[individual][0][i][j])
                for i in range(len(self.w_2)):
                    for j in range(self.num_of_hidden):
                        chrome.append(self.population[individual][1][i][j])
                print("The Winning Chromosome is Individual %d on Epoch %d" % (individual, epoch))
                print("Error = %f" % self.population_error[individual])
                print(chrome)
                sys.exit()
            build_chrome = (self.population_error[individual], individual)
            best_chrome.append(build_chrome)
        best_chrome = sorted(best_chrome, key=lambda student: student[0])
        return_chrome = self.population_error[best_chrome[0][1]], self.population[best_chrome[0][1]]
        self.all_runs.append(return_chrome)

    def sort_all_runs(self):
        self.all_runs = sorted(self.all_runs, key=lambda student: student[0])
        return self.all_runs[0][1]

    def zero_error(self):
        self.population_error = numpy.ones(shape=(self.pop_size, 1))

    def activation(self, value):
        if (self.activation_type == "s"):                           #for activation in ff
            return self.sigmoid(value)                              #use "s" for sigmoid, set in __init__ file
        else:
            print("Activation Function Mismatch")

    def sigmoid(self, value):
        return 1.0 / (1.0 + exp(-value))