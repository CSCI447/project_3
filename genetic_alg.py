import random
import math
from math import exp
import sys

import numpy

class GA:
    def __init__(self, inputs, expected, num_of_hidden, num_of_outs, pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate):
        self.w_1 = numpy.ones(shape=(num_of_hidden, len(inputs[0])))                                #init varaible of GA
        self.w_2 = numpy.ones(shape=(num_of_outs, num_of_hidden))
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(expected, dtype=float)
        self.test_outputs = numpy.array(expected, dtype=float)
        self.pop_size = pop_size
        self.num_tournament_participants = num_tournament_participants
        self.num_tournament_victors = num_tournament_victors
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.population_error = numpy.ones(shape=(pop_size, 1))
        self.threshold = 1
        self.activation_type = "s"
        self.num_of_hidden = num_of_hidden
        self.all_runs = []

    def update_in_out(self, inputs, expected, test_in, test_expected):          #this is for updating the training and testing datasets for each epoc
        self.input_values = numpy.array(inputs, dtype=float)
        self.expected = numpy.array(expected, dtype=float)
        self.test_inputs = numpy.array(test_in, dtype=float)
        self.test_outputs = numpy.array(test_expected, dtype=float)

    def feed_GA(self, row, individual):                                                 #for passing datapoints through network with current weight valuse
        self.v_1 = self.input_values[row]
        temp_node_val = self.population[individual][0].dot(self.v_1)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])                        #weights * values with activation funciton
        self.hidden_node_val = temp_node_val                                            #save to hidden nodes
        self.output = self.population[individual][1].dot(self.hidden_node_val)          #process next weights between hidden and output layers
        return self.output

    def feed_testing_GA(self, row, individual):                                         #same as last fucntion but for testing data
        self.v_1 = self.test_inputs[row]
        temp_node_val = self.population[individual][0].dot(self.v_1)
        for i in range(len(temp_node_val)):
            temp_node_val[i] = self.activation(temp_node_val[i])
        self.hidden_node_val = temp_node_val
        self.output = self.population[individual][1].dot(self.hidden_node_val)
        return self.output

    def zero_error(self):
        self.population_error = numpy.ones(shape=(self.pop_size, 1))                    #used to reset the population error

    def calc_fit(self, individual, data_row, output):                                   #calculating fitness of the whole population.  This is unique to genetic algorithms
        self.population_error[individual] += int(self.expected[data_row][0]) - output

    def tournament(self):                                                               #used to select parents and breed
        participants_chrome = []                                                        #init variables
        participants_error = []
        return_victors = []
        p = list(range(0, self.pop_size))                                               #list of numbers 0 to pop_size
        part_indices = random.sample(p, self.num_tournament_participants)               #generate num_tournament_participants number of reference indices
        for i in range(self.num_tournament_participants):
            temp_chrome = (self.population[part_indices[i]], part_indices[i])           #get weights for a chromesome
            participants_chrome.append(temp_chrome)                                     #adding paticipants
            temp_error = self.population_error[part_indices[i]], part_indices[i]
            participants_error.append(temp_error)                                       #keeping track of errors
        sorted_error = sorted(participants_error, key=lambda student: student[0])       #sort to find the best "tournament"
        for i in range(self.num_tournament_victors):
            return_victors.append(sorted_error[i][1])
        self.mate(return_victors)                                                       #mate the tournament victors

    def mate(self, return_victors):
        children = []
        for p1 in return_victors:                                           #iterate through victors
            for p2 in return_victors:                                       #make sure no asex breeding
                if not (p1 == p2):
                    children.append(self.crossover(p1, p2))                 #send to crossover to make the babies
        for i in range(len(self.population[0])):
            for j in range(len(self.population[0][0])):
                self.population[j][i][0] = children[j][i][0]                #set children to be parents in next gen

    def crossover(self, p1, p2):
        p1 = self.population[p1]
        p2 = self.population[p2]
        current = 0
        weight_1 = []
        weight_2 = []
        child = []
        for i in range(len(p1[0])):                                         #iterate through each gene in partents, make child based on crossover rate, this is for weights between input and hidden layer
            if (random.random() < self.crossover_rate):
                current = (current+1)%2
            if (current == 0):
                weight_1.append(p1[0][i])
            else:
                weight_1.append(p2[0][i])
        for i in range(len(p1[1])):                                         #doing the same but for weights between hidden and output
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
        for j in range(len(self.population)):                               #iterate to each gene and mutate based on mutation rate
            for i in range(len(self.population[0])):
                if random.random() < self.mutation_rate:
                    self.population[j][i] += (random.random() * (self.population_error[j]*0.001)/self.pop_size)

    def winner(self, epoch):
        print("Error = %f" % (self.population_error[0]))
        sum_error = 0
        it = 0
        best_chrome = []
        for individual in range(self.pop_size):             #this is to check if a chomosome in the population has achieved an error less than the threshold
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

    def sort_all_runs(self):
        self.all_runs = sorted(self.all_runs, key=lambda student: student[0])       #this funciton is used to determine the best run in the whole group
        return self.all_runs[0][1]

    def initialize_ga(self):                                                        #initialized all weights for first generation
        self.population = []
        for i in range(self.pop_size):
            id_array = numpy.identity(len(self.w_1), dtype=float)                   #create an identity matrix for setting rand values to wieghts
            for j in range(len(self.w_1)):
                for k in range(len(self.w_1[0])):
                    self.w_1[j][k] = random.random()
            temp_1 = id_array.dot(self.w_1)                                         #multiplying by the identity was a previous idea that doesn't do much now but maintains the for of the matricy i want
            temp_2 = self.w_2.dot(id_array)
            self.population.append([temp_1, temp_2])

    def activation(self, value):                                                    #helper function for any activation we want
        if (self.activation_type == "s"):
            return self.sigmoid(value)
        else:
            print("Activation Function Mismatch")

    def sigmoid(self, value):                                                       #sigmoid used as activation
        return 1.0 / (1.0 + exp(-value))

