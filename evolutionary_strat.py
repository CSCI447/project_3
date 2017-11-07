import random
import math
from math import exp
import sys

import numpy

class ES:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, pop_size, mutation_rate, crossover_rate):
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))             #array size rows = # hidden nodes, cols = # if inputs
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))                #array size rows = # output nodes, cols = # hidden nodes
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(expected, dtype=float)
        self.test_outputs = numpy.array(expected, dtype=float)
        self.pop_size = pop_size
        self.mutation_rate = 1
        self.crossover_rate = crossover_rate
        self.population = []                                                    # for GA, contains weights
        self.population_error = numpy.ones(shape=(pop_size, 1))                                              # for GA
        self.child_error = numpy.ones(shape=(120, 1))
        self.all_errors = []
        self.threshold = 1
        self.activation_type = "s"
        self.num_of_hidden = num_of_hidden
        self.all_runs = []
        self.lambda_factor = 6
        self.children = []

    def update_in_out(self, inputs, expected, test_in, test_expected):
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(test_in, dtype=float)
        self.test_outputs = numpy.array(test_expected, dtype=float)

    def feed_ES(self, row, individual):
        self.v_1 = self.input_values[row]                                     #index is input value row, self.population[individual][0] is correct way to access pop weights
        temp_node_val = self.population[individual][0].dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.population[individual][1].dot(self.hidden_node_val)
        return self.output

    def feed_testing_ES(self, row, individual):
        self.v_1 = self.test_inputs[row]                                     #index is input value row, self.population[individual][0] is correct way to access pop weights
        temp_node_val = self.population[individual][0].dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.population[individual][1].dot(self.hidden_node_val)
        return self.output

    def feed_ES_children(self, row, child):
        self.v_1 = self.input_values[row]                                     #index is input value row, self.population[individual][0] is correct way to access pop weights
        child_w1 = numpy.array(child[0])
        child_w2 = numpy.array(child[1])
        temp_node_val = child_w1.dot(self.v_1)            #temp node is transition between v_1 and hidden_valuse.  This step multiplies the values v_1 * connect weights
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        hidden_val = temp_node_val
        output = child_w2.dot(hidden_val)
        return output

    def zero_error(self):
        self.population_error = numpy.ones(shape=(self.pop_size, 1))
        self.child_error = numpy.ones(shape=(120, 1))
        self.all_errors = []

    def calc_fit(self, individual, data_row, output):
        self.population_error[individual] += int(self.expected[data_row][0]) - output

    def calc_fit_children(self, individual, data_row, output):
        self.child_error[individual] += int(self.expected[data_row][0]) - output

    ################################ES METHODS########################################

    def evolve(self):
        children = []
        for index, individual in enumerate(self.population):
            p = list(range(0, self.pop_size-1))
            selected_individuals = random.sample(p, self.lambda_factor)
            for i in range(len(selected_individuals)):                                  #make sure no asex reproduction, works!
                if selected_individuals[i] == index:
                    selected_individuals[i] += 1
            self.mate(index, selected_individuals, children)
        self.children = self.mutate_children(children)
        output = numpy.ones(shape=(120, 1))
        for j, child in enumerate(self.children):
            for i in range(len(self.input_values)):
                output[j] = self.feed_ES_children(i, child)
                self.calc_fit_children(j, i, output[j])

        for i in range(len(self.population_error)):
            temp_pop_error_index = (self.population_error[i], i)
            self.all_errors.append(temp_pop_error_index)
        for i in range(len(self.child_error)):
            temp_child_error_index = (self.child_error[i], i)
            self.all_errors.append(temp_child_error_index)
        self.all_errors = sorted(self.all_errors, key=lambda student: student[0])

        for i in range(len(self.population)):
            if (self.all_errors[i][1] < 20):
                self.population[i] = self.population[self.all_errors[i][1]]
            else:
                self.population[i] = self.children[self.all_errors[i][1]-20]

    def mate(self, p1, selected_individuals, children):
        for p2 in selected_individuals:
            children.append(self.crossover(p1, p2))
        return children

    def crossover(self, p1, p2):
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

    def mutate_children(self, children):
        for j in range(len(children)):
            for i in range(len(children[0])):
                if random.random() < self.mutation_rate:
                    children[j][i] += (random.random() * (self.population_error[0]*0.001)/self.pop_size)
        return children

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

    def initialize_ES(self):
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

    def activation(self, value):
        if (self.activation_type == "s"):                           #for activation in ff
            return self.sigmoid(value)                              #use "s" for sigmoid, set in __init__ file
        else:
            print("Activation Function Mismatch")

    def sigmoid(self, value):
        return 1.0 / (1.0 + exp(-value))