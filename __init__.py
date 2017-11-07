from feed_forward import *
from genetic_alg import *
from differencial_evo import *
import csv
import codecs
import time


neural_net = "DE"                           #"backprop" or "GA" or "DE"
inputArray = []
expectedOutputArray = []
inputArray_ga = []
inputArray_de = []
expectedOutputArray_ga = []
expectedOutputArray_de = []
cross_valid_fold = 8

with codecs.open('Data_old_fortest/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        inputArray.append(row)
        inputArray_ga.append(row)
        inputArray_de.append(row)

cross_variable = 0
with codecs.open('Data_old_fortest/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)
        expectedOutputArray_ga.append(row)
        expectedOutputArray_de.append(row)

t0 = time.time()
len_in_outs = len(inputArray)


if (neural_net == "DE"):
    hidden_nodes_amount = 20
    output_nodes_amount = 1
    max_generations = 100
    pop_size = 20
    run_condition = "epocs"
    output = numpy.ones(shape=(pop_size, 1))
    mutation_rate = .25
    crossover_rate = .75
    cross_valid_variable = 0

    if(run_condition == "epocs"):
        DE = DE(inputArray_de, expectedOutputArray_de, hidden_nodes_amount, output_nodes_amount, pop_size, mutation_rate, crossover_rate)
        DE.initialize_de()
        for j in range(max_generations):
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            DE.zero_error()
            for i in range(len_in_outs - int((len_in_outs / cross_valid_fold))):  # manual crossvalidation
                train_inputs.append(inputArray_ga[cross_valid_variable % len_in_outs])
                train_outputs.append(expectedOutputArray_ga[cross_valid_variable % len_in_outs])
                cross_valid_variable += 1
            for i in range(int((len_in_outs / cross_valid_fold))):
                test_inputs.append(inputArray_ga[cross_valid_variable % len_in_outs])
                test_outputs.append(expectedOutputArray_ga[cross_valid_variable % len_in_outs])
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
    max_generations = 500
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
        GA = GA(inputArray_ga, expectedOutputArray_ga, hidden_nodes_amount, output_nodes_amount, pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate)
        GA.initialize_ga()
        for j in range(max_generations):
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            GA.zero_error()
            for i in range(len_in_outs - int((len_in_outs/cross_valid_fold))):                      #manual crossvalidation
                train_inputs.append(inputArray_ga[cross_valid_variable%len_in_outs])
                train_outputs.append(expectedOutputArray_ga[cross_valid_variable%len_in_outs])
                cross_valid_variable += 1
            for i in range(int((len_in_outs/cross_valid_fold))):
                test_inputs.append(inputArray_ga[cross_valid_variable % len_in_outs])
                test_outputs.append(expectedOutputArray_ga[cross_valid_variable % len_in_outs])
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
            print("Test Sum Error = %f" % test_final)

            print("Generation %d" % j)

#############################DONE CONDITION#######################################

        best_chrome = GA.sort_all_runs()
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
        runtime = t1-t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))

if (neural_net == "backprop"):
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

    if (converge_test == "epoc"):
        feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)
        feedforward.initialize()
        for j in range(epocs):
            sum_error = 0
            for i in range(len(inputArray)):
                output = feedforward.feed_forward(i)
                sum_error += (int(expectedOutputArray[i][0]) - output)
                feedforward.backprop(i)
            print('epoc=%d, training error=%.3f' % (j, sum_error))
    elif (converge_test == "sum_error"):
        feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)
        feedforward.initialize()
        epocs = 0
        while (sum_error > .1):
            sum_error = 0
            for i in range(len(inputArray)):
                output = feedforward.feed_forward(i)
                sum_error += (int(expectedOutputArray[i][0]) - output)
                feedforward.backprop(i)
            print('epoc=%d, training error=%.3f' % (epocs, sum_error))
            epocs += 1
    elif (converge_test == "optimize"):
        for a in range(opt_test_start, opt_test_end, opt_test_interval):
            t_opt0 = time.time()
            hidden_nodes_amount = a + 1
            epocs = 0
            sum_error = 10
            feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)
            feedforward.initialize()
            print("New Test: Hidden Nodes = %d" %a)
            while (sum_error > 1):
                sum_error = 0
                for i in range(len(inputArray)):
                    output = feedforward.feed_forward(i)
                    sum_error += (int(expectedOutputArray[i][0]) - output)
                    feedforward.backprop(i)
                epocs += 1
            t_opt1 = time.time()
            runtime = t_opt1 - t_opt0
            if (runtime < opt_runtime):
                opt_runtime = runtime
                opt_epocs = epocs
                opt_hidden = a
            print("Runtime = %3f, Epocs = %d" % (runtime, epocs))
    elif (converge_test == "test"):
        feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)
        feedforward.initialize()
        feedforward.feed_forward(0)           #testing inits
        feedforward.backprop(0)               #testing inits
    else:
        print("Convergence test unrecognized")

    t1 = time.time()
    runtime = t1-t0

    if (converge_test == "optimize"):
        print("Optimal: Runtime = %3f, Hidden Nodes = %d, Epocs = %d" % (opt_runtime, opt_hidden, opt_epocs))
    print("Runtime = %f, Error = %f" % (runtime, sum_error))
