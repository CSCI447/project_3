from feed_forward import *
import csv
import codecs
import time

neural_net = "GA"
#inputArray = numpy.zeros(shape=(8, 1))
#expectedOutputArray = numpy.zeros(shape=(8, 1))
inputArray = []
expectedOutputArray = []
cross_variable = 0
with codecs.open('Data_old_fortest/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        #print(cross_variable)
        #numpy.append(inputArray[cross_variable % 8], row)
        inputArray.append(row)
        cross_variable += 1

cross_variable = 0
with codecs.open('Data_old_fortest/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        #numpy.append(expectedOutputArray[cross_variable % 8], row)
        expectedOutputArray.append(row)
        cross_variable += 1
#print(inputArray)
t0 = time.time()

if (neural_net == "GA"):
    hidden_nodes_amount = 100               #not used
    output_nodes_amount = 1                 #not used
    activation_type = "s"                   #not used
    max_generations = 1000
    pop_size = 20
    num_tournament_participants = 10
    num_tournament_victors = 5
    mutation_rate = .25
    crossover_rate = .75
    threshold = .1
    output = []

    # for j in range(1):                #for total epocs
    GA = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type,  pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate)
    GA.initialize_ga()
    for j in range(max_generations):
        for i in range(len(inputArray)):
            for individual in range(pop_size):
                output.append(GA.feed_GA(i))
                GA.calc_fit(i, output[individual])
            GA.tournament()
            GA.mutate()
            GA.winner(expectedOutputArray)
        print("Generation %d" % j)
    t1 = time.time()
    runtime = t1-t0
    print("Runtime of %d Generations: %f" % (max_generations, runtime))

if (neural_net == "backprop"):
    hidden_layer_amount = 0
    hidden_nodes_amount = 800
    output_nodes_amount = 1
    epocs = 100000
    activation_type = "s"               #set: "s" for sig,
    converge_test = "test"          #set "epoc", "sum_error", "optimize" or "test"
    sum_error = 1                       #dont change
    opt_hidden = 0                      #dont change
    opt_epocs = 100000                  #dont change
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