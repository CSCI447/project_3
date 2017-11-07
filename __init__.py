from feed_forward import *
from genetic_alg import *
from differencial_evo import *
from evolutionary_strat import *
import csv
import codecs
import time


neural_net = "ES"                           #"BP" or "GA" or "DE" or "ES";  This decides what network to run
inputArray = []                             #Generate initial input array
expectedOutputArray = []                    #Generate expected output array
cross_valid_fold = 8                        #used to determine how many folds in crossvalidation

with codecs.open('Data_old_fortest/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:              #pull in input data from csv
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        inputArray.append(row)

cross_variable = 0
with codecs.open('Data_old_fortest/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:         #pull in varifaction data from csv
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)

t0 = time.time()
len_in_outs = len(inputArray)

if (neural_net == "ES"):
    hidden_nodes_amount = 20
    output_nodes_amount = 1
    max_generations = 1000
    pop_size = 20
    run_condition = "epocs"
    output = numpy.ones(shape=(pop_size, 1))
    mutation_rate = .25
    crossover_rate = .75
    cross_valid_variable = 0

    if(run_condition == "epocs"):
        ES = ES(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, pop_size, mutation_rate, crossover_rate)
        ES.initialize_ES()
        for j in range(max_generations):
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            ES.zero_error()
            for i in range(len_in_outs - int((len_in_outs / cross_valid_fold))):  # manual crossvalidation
                train_inputs.append(inputArray[cross_valid_variable % len_in_outs])
                train_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            for i in range(int((len_in_outs / cross_valid_fold))):
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            ES.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)
            for individual in range(pop_size):
                for i in range(len(train_inputs)):
                    output[individual] = ES.feed_ES(i, individual)
                    ES.calc_fit(individual, i, output[individual])
            ES.evolve()
            ES.winner(j)
            test_final = 0

            for i in range(len(test_inputs)):
                output[i] = ES.feed_testing_ES(i, 0)
                test_final += int(test_outputs[i][0]) - output[i]
            test_final = test_final * test_final
            test_final = math.sqrt(test_final)
            print("Test Sum Error = %f" % test_final)
            print("Generation %d" % j)

            #############################DONE CONDITION#######################################

        best_chrome = ES.sort_all_runs()
        final_chrome = []
        for i in range(hidden_nodes_amount):
            for j in range(len(inputArray_ga[0][0])):
                # print(self.population[individual][0][j][i])
                final_chrome.append(best_chrome[0][i][j])
        for i in range(len(expectedOutputArray_ga[0])):
            for j in range(hidden_nodes_amount):
                final_chrome.append(best_chrome[1][i][j])
        print("best chrome")
        print(final_chrome)
        t1 = time.time()
        runtime = t1 - t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))


elif (neural_net == "DE"):
    hidden_nodes_amount = 20
    output_nodes_amount = 1
    max_generations = 1000
    pop_size = 20
    run_condition = "epocs"
    output = numpy.ones(shape=(pop_size, 1))
    mutation_rate = .25
    crossover_rate = .75
    cross_valid_variable = 0

    if(run_condition == "epocs"):
        DE = DE(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, pop_size, mutation_rate, crossover_rate)
        DE.initialize_de()
        for j in range(max_generations):
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            DE.zero_error()
            for i in range(len_in_outs - int((len_in_outs / cross_valid_fold))):  # manual crossvalidation
                train_inputs.append(inputArray[cross_valid_variable % len_in_outs])
                train_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            for i in range(int((len_in_outs / cross_valid_fold))):
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            DE.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)
            for individual in range(pop_size):
                for i in range(len(train_inputs)):
                    output[individual] = DE.feed_DE(i, individual)
                    DE.calc_fit(individual, i, output[individual])
            DE.evolve()
            DE.mutate()
            DE.winner(j)
            test_final = 0

            for i in range(len(test_inputs)):
                output[i] = DE.feed_testing_DE(i, 0)
                test_final += int(test_outputs[i][0]) - output[i]
            test_final = test_final * test_final
            test_final = math.sqrt(test_final)
            print("Test Sum Error = %f" % test_final)
            print("Generation %d" % j)

            #############################DONE CONDITION#######################################

        best_chrome = DE.sort_all_runs()
        final_chrome = []
        for i in range(hidden_nodes_amount):
            for j in range(len(inputArray_ga[0][0])):
                # print(self.population[individual][0][j][i])
                final_chrome.append(best_chrome[0][i][j])
        for i in range(len(expectedOutputArray_ga[0])):
            for j in range(hidden_nodes_amount):
                final_chrome.append(best_chrome[1][i][j])
        print("best chrome")
        print(final_chrome)
        t1 = time.time()
        runtime = t1 - t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))


elif (neural_net == "GA"):
    hidden_nodes_amount = 20
    output_nodes_amount = 1
    activation_type = "s"
    cross_valid_variable = 0
    max_generations = 1000
    pop_size = 50
    num_tournament_participants = 10
    num_tournament_victors = 5
    mutation_rate = .25
    crossover_rate = .75
    output = numpy.ones(shape=(pop_size, 1))
    run_condition = "epocs"              #"test" or "epocs"
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    # for j in range(1):                #for total epocs
    if (run_condition == "epocs"):
        GA = GA(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate)
        GA.initialize_ga()
        for j in range(max_generations):
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            GA.zero_error()
            for i in range(len_in_outs - int((len_in_outs/cross_valid_fold))):                      #manual crossvalidation
                train_inputs.append(inputArray[cross_valid_variable%len_in_outs])
                train_outputs.append(expectedOutputArray[cross_valid_variable%len_in_outs])
                cross_valid_variable += 1
            for i in range(int((len_in_outs/cross_valid_fold))):
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            GA.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)
            for individual in range(pop_size):
                for i in range(len(train_inputs)):
                    output[individual] = GA.feed_GA(i, individual)
                    GA.calc_fit(individual, i, output[individual])
            GA.tournament()
            GA.mutate()
            GA.winner(j)
            test_final = 0

            for i in range(len(test_inputs)):
                output[i] = GA.feed_testing_GA(i, 0)
                test_final += int(test_outputs[i][0]) - output[i]
            test_final = test_final * test_final
            test_final = math.sqrt(test_final)
            print("Test Sum Error = %f" % test_final)

            print("Generation %d" % j)

#############################DONE CONDITION#######################################

        best_chrome = GA.sort_all_runs()
        final_chrome = []
        for i in range(hidden_nodes_amount):
            for j in range(len(inputArray_ga[0][0])):
                final_chrome.append(best_chrome[0][i][j])
        for i in range(len(expectedOutputArray_ga[0])):
            for j in range(hidden_nodes_amount):
                final_chrome.append(best_chrome[1][i][j])
        print("best chrome")
        print(final_chrome)
        t1 = time.time()
        runtime = t1-t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))

if (neural_net == "BP"):
    hidden_layer_amount = 0
    hidden_nodes_amount = 20
    output_nodes_amount = 1
    epocs = 1000
    activation_type = "s"               #set: "s" for sig,
    converge_test = "epoc"          #set "epoc", "sum_error", "optimize" or "test"
    sum_error = 1                       #dont change
    opt_hidden = 0                      #dont change
    opt_epocs = 10000                  #dont change
    opt_runtime = 10000                 #dont change
    opt_test_start = 720
    opt_test_end = 740
    opt_test_interval = 1
    cross_valid_variable = 0

    if (converge_test == "epoc"):
        FF = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)
        FF.initialize()
        for j in range(epocs):
            sum_error = 0
            sum_error_test = 0
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            for i in range(len_in_outs - int((len_in_outs/cross_valid_fold))):                      #manual crossvalidation
                train_inputs.append(inputArray[cross_valid_variable%len_in_outs])
                train_outputs.append(expectedOutputArray[cross_valid_variable%len_in_outs])
                cross_valid_variable += 1
            for i in range(int((len_in_outs/cross_valid_fold))):
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            FF.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)
            for i in range(len(train_inputs)):
                output = FF.feed_forward(i)
                sum_error += (int(train_outputs[i][0]) - output)
                FF.backprop(i)
            for i in range(len(test_inputs)):
                output_test = FF.feed_forward_test(i)
                sum_error_test += (int(test_outputs[i][0]) - output_test)
            sum_error_test = sum_error_test * sum_error_test
            sum_error_test = math.sqrt(sum_error_test)
            print('Generation=%d, Training Error=%.3f, Testing Error=%.3f' % (j, sum_error, sum_error_test))
        print("Convergence test unrecognized")

    t1 = time.time()
    runtime = t1-t0

