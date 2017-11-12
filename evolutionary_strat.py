import random
import math
from math import exp
import sys

import numpy

class ES:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, pop_size, mutation_rate, crossover_rate):
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))             #ES init variables
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(expected, dtype=float)
        self.test_outputs = numpy.array(expected, dtype=float)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.population_error = numpy.ones(shape=(pop_size, 1))
        self.child_error = numpy.ones(shape=(120, 1))
        self.all_errors = []
        self.threshold = 1
        self.momentum = 0.005
        self.activation_type = "s"
        self.num_of_hidden = num_of_hidden
        self.all_runs = []
        self.lambda_factor = 6
        self.children = []

    def update_in_out(self, inputs, expected, test_in, test_expected):          #this is for updating the training and testing datasets for each epoc
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(test_in, dtype=float)
        self.test_outputs = numpy.array(test_expected, dtype=float)

    def feed_ES(self, row, individual):                                             #for passing datapoints through network with current weight valuse
        self.v_1 = self.input_values[row]
        feed_w1 = numpy.array(self.population[individual][0])
        feed_w2 = numpy.array(self.population[individual][1])
        temp_node_val = feed_w1.dot(self.v_1)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])            #weights * values with activation funciton
        self.hidden_node_val = temp_node_val                                #save to hidden nodes
        self.output = feed_w2.dot(self.hidden_node_val)                     #process next weights between hidden and output layers
        return self.output

    def feed_testing_ES(self, row, individual):
        self.v_1 = self.test_inputs[row]
        test_w1 = numpy.array(self.population[individual][0])
        test_w2 = numpy.array(self.population[individual][1])                       #same as last fucntion but for testing data
        temp_node_val = test_w1.dot(self.v_1)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = test_w2.dot(self.hidden_node_val)
        return self.output

    def feed_ES_children(self, row, child):                                     #same as functions above but for feeding children through the network
        self.v_1 = self.input_values[row]
        child_w1 = numpy.array(child[0])
        child_w2 = numpy.array(child[1])
        temp_node_val = child_w1.dot(self.v_1)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        hidden_val = temp_node_val
        output = child_w2.dot(hidden_val)
        return output

    def zero_error(self):                                                       #used to reset the population error
        self.population_error = numpy.ones(shape=(self.pop_size, 1))
        self.child_error = numpy.ones(shape=(120, 1))
        self.all_errors = []

    def calc_fit(self, individual, data_row, output):                                            #calculating fitness of the whole population.  This is unique to genetic algorithms
        self.population_error[individual] += int(self.expected[data_row][0]) - output

    def calc_fit_children(self, individual, data_row, output):                                  #same as above but for the children
        self.child_error[individual] += int(self.expected[data_row][0]) - output

    ################################ES METHODS########################################

    def evolve(self):                                                           #for handling ES evolution
        children = []
        for index, individual in enumerate(self.population):                    #this is used to generate children that will be mutated and evaluated for replacement in population
            p = list(range(0, self.pop_size-1))
            selected_individuals = random.sample(p, self.lambda_factor)
            for i in range(len(selected_individuals)):                          #make sure no asex reproduction
                if selected_individuals[i] == index:
                    selected_individuals[i] += 1                                #If same person is chosen to mate, choose next in pop
            self.mate(index, selected_individuals, children)                    #mate all individuals
        self.children = self.mutate_children(children)                          #take all the children and mutate them
        output = numpy.ones(shape=(120, 1))                                     #reset the output array
        for j, child in enumerate(self.children):                               #this is to generate the fitness of all the children
            for i in range(len(self.input_values)):
                output[j] = self.feed_ES_children(i, child)
                self.calc_fit_children(j, i, output[j])

        for i in range(len(self.population_error)):                                     #some for loops to generate the correct information for comparing the best in parents and children for reproduction
            temp_pop_error_index = (self.population_error[i], i)
            self.all_errors.append(temp_pop_error_index)
        for i in range(len(self.child_error)):
            temp_child_error_index = (self.child_error[i], i)
            self.all_errors.append(temp_child_error_index)
        self.all_errors = sorted(self.all_errors, key=lambda student: student[0])       #sort to find best

        for i in range(len(self.population)):                                           #create the new population based on what children or parents are most fit
            if (self.all_errors[i][1] < 20):
                self.population[i] = self.population[self.all_errors[i][1]]
            else:
                self.population[i] = self.children[self.all_errors[i][1]-self.pop_size]

    def mate(self, p1, selected_individuals, children):                                 #this is for mating p1 with the selected individuals
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
        for i in range(len(p1[0])):                                #this is for creating the child of the weight matrix 0 based on the crossover rate
            if (random.random() < self.crossover_rate):
                current = (current+1)%2
            if (current == 0):
                weight_1.append(p1[0][i])
            else:
                weight_1.append(p2[0][i])
        for i in range(len(p1[1])):                                #this is for creating the child of the weight matrix 1 based on the crossover rate
            if (random.random() < self.crossover_rate):
                current = (current+1)%2
            if (current == 0):
                weight_2.append(p1[1][i])
            else:
                weight_2.append(p2[1][i])
        child.append(weight_1)                                      #creating the actual child, it is a combo of weight matrix 0 and 1
        child.append(weight_2)
        return child

    def mutate_children(self, children):
        for j in range(len(children)):
            for i in range(len(children[0])):                       #iterate to each gene and mutate based on mutation rate
                if random.random() < self.mutation_rate:
                    children[j][i] += (random.random() * (self.population_error[0]*self.momentum)/self.pop_size)
        return children

    def winner(self, epoch):
        print("Error = %f" % (self.population_error[0]))
        sum_error = 0
        it = 0
        best_chrome = []
        for individual in range(self.pop_size):                         #this is to check if a chomosome in the population has achieved an error less than the threshold
            if ((self.population_error[individual] < self.threshold) and (self.population_error[individual] > -(self.threshold))):
                chrome = []
                for i in range(self.num_of_hidden):                                 #these two for loops create a chomosome for printing out of the weight matrices used
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
            best_chrome.append(build_chrome)                                        #best chrome, return chrome, and self.all_runs help to keep track of and find the best chomosome if max genrations is reached
        best_chrome = sorted(best_chrome, key=lambda student: student[0])
        return_chrome = self.population_error[best_chrome[0][1]], self.population[best_chrome[0][1]]
        self.all_runs.append(return_chrome)

    def sort_all_runs(self):                                                       #method to allow the init file to know the best chomosome after max generations
        self.all_runs = sorted(self.all_runs, key=lambda student: student[0])
        return self.all_runs[0][1]

    def initialize_ES(self):
        self.population = []
        for i in range(self.pop_size):
            id_array = numpy.identity(len(self.w_1), dtype=float)                   # create an identity matrix for setting rand values to wieghts
            for j in range(len(id_array)):                                          # apply rand values to actual values in identity
                for k in range(len(id_array)):
                    id_array[j][k] = random.random()
            for j in range(len(self.w_1)):
                for k in range(len(self.w_1[0])):
                    self.w_1[j][k] = random.random()
            temp_1 = id_array.dot(self.w_1)                                         # multiply rand identity to weight matrix 1
            temp_2 = self.w_2.dot(id_array)
            self.population.append([temp_1, temp_2])

    def activation(self, value):
        if (self.activation_type == "s"):                                           #for activation in ff
            return self.sigmoid(value)                                              #use "s" for sigmoid, set in __init__ file
        else:
            print("Activation Function Mismatch")

    def sigmoid(self, value):
        value = value * 0.1
        return 1.0 / (1.0 + exp(-value))


