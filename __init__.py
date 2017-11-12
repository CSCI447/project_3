from feed_forward import *
from genetic_alg import *
from differencial_evo import *
from evolutionary_strat import *
import csv
import codecs
import time


neural_net = "DE"                           #"BP" or "GA" or "DE" or "ES";  This decides what network to run
data_set = "fert"                          #which data set will be run
inputArray = []                             #Generate initial input array
expectedOutputArray = []                    #Generate expected output array
cross_valid_fold = 8                        #used to determine how many folds in crossvalidation
cross_variable = 0
if data_set == "rb":                        #rosenbrock (used for testing during devolopement)
    with codecs.open('Data_old_fortest/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:              #pull in input data from csv
        csv_input = csv.reader(inputcsvfile, delimiter=",")
        for row in csv_input:
            inputArray.append(row)
    with codecs.open('Data_old_fortest/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:         #pull in verifaction data from csv
        csv_output = csv.reader(outputcsvfile, delimiter=",")
        for row in csv_output:
            expectedOutputArray.append(row)
elif data_set == "wine":                        #wine dataset
    with codecs.open('Data/wine.csv', 'r', encoding='utf-8') as wine:
        csv_input = csv.reader(wine, delimiter=",")
        for row in csv_input:
            example_input = []
            example_output = []
            for element in range(len(row)):
                if element < len(row) - 1:
                    example_input.append(float(row[element]))
                else:
                    example_output.append(float(row[element]))
            inputArray.append(example_input)
            expectedOutputArray.append(example_output)
elif data_set == "fert":                  #fertility dataset
    with codecs.open('Data/fertility.csv', 'r', encoding='utf-8') as fertility:
        csv_input = csv.reader(fertility, delimiter=",")
        for row in csv_input:
            example_input = []
            example_output = []
            for element in range(len(row)):
                if element < len(row) - 1:
                    example_input.append(float(row[element]))
                else:
                    if row[element] == "O":
                        example_output.append(0)        #vectorization of categorical feature
                    else:
                        example_output.append(1)
            inputArray.append(example_input)
            expectedOutputArray.append(example_output)
elif data_set == "glass":                               #glass identificaation datset
    with codecs.open('Data/glass_identification.csv', 'r', encoding='utf-8') as glass:
        csv_input = csv.reader(glass, delimiter=",")
        for row in csv_input:
            example_input = []
            example_output = []
            for element in range(len(row)):
                if element < len(row) - 1:
                     example_input.append(float(row[element]))
                else:
                     example_output.append(float(row[element]))
            inputArray.append(example_input)
            expectedOutputArray.append(example_output)
elif data_set == "cmc":                              #contraceptive method choice dataset
    with codecs.open('Data/contraceptive_method_choice.csv', 'r', encoding='utf-8') as cmc:
        csv_input = csv.reader(cmc, delimiter=",")
        for row in csv_input:
            example_input = []
            example_output = []
            for element in range(len(row)):
                if element < len(row) - 1:
                     example_input.append(float(row[element]))
                else:
                     example_output.append(float(row[element]))
            inputArray.append(example_input)
            expectedOutputArray.append(example_output)
elif data_set == "abl":                          #abalone dataset
    with codecs.open('Data/abalone.csv', 'r', encoding='utf-8') as abl:
        csv_input = csv.reader(abl, delimiter=",")
        for row in csv_input:
            example_input = []
            example_output = []
            for element in range(len(row)):
                if element < len(row) - 1:
                    if element == 0:
                        if row[element] == 'M':            #vectorization of categorical feature
                            example_input.append(0.0)
                        elif row[element] == "I":
                            example_input.append(0.5)
                        elif row[element] == "F":
                            example_input.append(1.0)
                    else:
                     example_input.append(float(row[element]))
                else:
                     example_output.append(float(row[element]))
            inputArray.append(example_input)
            expectedOutputArray.append(example_output)


t0 = time.time()                                        #set timing for algorithms
len_in_outs = len(inputArray)                           #helps with crossvalidation

#############################EVOLUTIONALARY STRATEGY ALGORITHM#######################################

if (neural_net == "ES"):                                #init code for Evolutionary Strategy
    hidden_nodes_amount = 20                            #starting varibale for ES
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
        ES.initialize_ES()                              #initalizing weights
        for j in range(max_generations):
            train_inputs = []                           #variables used for cross validation
            train_outputs = []                          #variables used for cross validation
            test_inputs = []                            #variables used for cross validation
            test_outputs = []                           #variables used for cross validation
            ES.zero_error()
            for i in range(len_in_outs - int((len_in_outs / cross_valid_fold))):                    #this uses an iterator and the modulo function to splice different variable in the total set to test and training data
                train_inputs.append(inputArray[cross_valid_variable % len_in_outs])                 #""
                train_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])       #""
                cross_valid_variable += 1                                                           #""
            for i in range(int((len_in_outs / cross_valid_fold))):                                  #""
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])                  #""
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])        #""
                cross_valid_variable += 1                                                           #""
            ES.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)                #updates training and test datasets in alg
            for individual in range(pop_size):                                                      #start training, iterate through individuals
                for i in range(len(train_inputs)):                                                  #iterate over test data
                    output[individual] = ES.feed_ES(i, individual)                                  #redord output of one datapoint through the net
                    ES.calc_fit(individual, i, output[individual])                                  #Calculate fitness of that datapoint
            ES.evolve()                                                                             #deals with the evolution of the chromosomes
            ES.winner(j)                                                                            #checks to see if a chromosome meets threshold req

            test_final = 0                                                                          #variable for testing data
            for i in range(len(test_inputs)):                                                       #ranging over test inputs
                output[i] = ES.feed_testing_ES(i, 0)                                                #get output for test data in trained network
                test_final += int(test_outputs[i][0]) - output[i]                                   #calc fitness for this test datapoin
            test_final = test_final * test_final                                                    #make fitness positive
            test_final = math.sqrt(test_final)
            print("Test Sum Error = %f" % test_final)
            print("Generation %d" % j)

#############################FINISH CONDITION#######################################

        best_chrome = ES.sort_all_runs()                                                            #all runs sorted to get best chromosome
        final_chrome = []
        for i in range(hidden_nodes_amount):                                                        #creating the chromosome from the weight matrices
            for j in range(len(inputArray[0][0])):
                final_chrome.append(best_chrome[0][i][j])
        for i in range(len(expectedOutputArray[0])):
            for j in range(hidden_nodes_amount):
                final_chrome.append(best_chrome[1][i][j])
        print("best chrome")                                                                        #displaying final chromosome and time
        print(final_chrome)
        t1 = time.time()
        runtime = t1 - t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))

#############################DIFFERENCIAL CONDITION#######################################

elif (neural_net == "DE"):                              #for Differencial Evolution network
    hidden_nodes_amount = 20                            #DE start variable
    output_nodes_amount = 1
    max_generations = 1000
    pop_size = 20
    run_condition = "epocs"
    output = numpy.ones(shape=(pop_size, 1))
    mutation_rate = .25
    crossover_rate = .75
    cross_valid_variable = 0

    if(run_condition == "epocs"):
        DE = DE(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, pop_size, mutation_rate, crossover_rate)         #init DE
        DE.initialize_de()                              #initialize DE weights
        for j in range(max_generations):
            train_inputs = []                           #variables used for cross validation
            train_outputs = []                          #variables used for cross validation
            test_inputs = []                            #variables used for cross validation
            test_outputs = []                           #variables used for cross validation
            DE.zero_error()
            for i in range(len_in_outs - int((len_in_outs / cross_valid_fold))):                        #this uses an iterator and the modulo function to splice different variable in the total set to test and training data
                train_inputs.append(inputArray[cross_valid_variable % len_in_outs])                     #""
                train_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])           #""
                cross_valid_variable += 1                                                               #""
            for i in range(int((len_in_outs / cross_valid_fold))):                                      #""
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])                      #""
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])            #""
                cross_valid_variable += 1                                                               #""
            DE.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)                    #updates training and test datasets in alg
            for individual in range(pop_size):                                                          #training, iterate over population size
                for i in range(len(train_inputs)):                                                      #iterate over datapoints
                    output[individual] = DE.feed_DE(i, individual)                                      #create output for one datapoint
                    DE.calc_fit(individual, i, output[individual])                                      #setting fitness for that datapoint
            DE.evolve()                                                                 #process chromosome evolution
            DE.mutate()                                                                 #mutate chrome
            DE.winner(j)                                                                #determin if a chome meet reqs

            test_final = 0
            for i in range(len(test_inputs)):                                           #Runnig cross validation esting data and printing
                output[i] = DE.feed_testing_DE(i, 0)
                test_final += int(test_outputs[i][0]) - output[i]
            test_final = test_final * test_final
            test_final = math.sqrt(test_final)
            print("Test Sum Error = %f" % test_final)
            print("Generation %d" % j)

#############################FINISH CONDITION#######################################

        best_chrome = DE.sort_all_runs()                                                #all runs sorted to get best chromosome
        final_chrome = []
        for i in range(hidden_nodes_amount):                                            #creating final chomosome, printing
            for j in range(len(inputArray[0][0])):
                final_chrome.append(best_chrome[0][i][j])
        for i in range(len(expectedOutputArray[0])):
            for j in range(hidden_nodes_amount):
                final_chrome.append(best_chrome[1][i][j])
        print("best chrome")
        print(final_chrome)
        t1 = time.time()
        runtime = t1 - t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))

#############################GENEATIC ALGORITHM#######################################

elif (neural_net == "GA"):                                                      #for genetic algorithm, init variables
    hidden_nodes_amount = 20
    output_nodes_amount = 2
    activation_type = "s"
    cross_valid_variable = 0
    max_generations = 1000
    pop_size = 20
    num_tournament_participants = 10
    num_tournament_victors = 5
    mutation_rate = .25
    crossover_rate = .75
    output = numpy.ones(shape=(pop_size, output_nodes_amount))
    run_condition = "epocs"
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    if (run_condition == "epocs"):
        GA = GA(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, pop_size, num_tournament_participants, num_tournament_victors, mutation_rate, crossover_rate)            #building GA
        GA.initialize_ga()                                                                          #init weights for ga
        for j in range(max_generations):
            train_inputs = []                                                                       #setting up cross validation variables
            train_outputs = []
            test_inputs = []
            test_outputs = []
            GA.zero_error()
            for i in range(len_in_outs - int((len_in_outs/cross_valid_fold))):                      #this uses an iterator and the modulo function to splice different variable in the total set to test and training data
                train_inputs.append(inputArray[cross_valid_variable%len_in_outs])                   #""
                train_outputs.append(expectedOutputArray[cross_valid_variable%len_in_outs])         #""
                cross_valid_variable += 1                                                           #""
            for i in range(int((len_in_outs/cross_valid_fold))):                                    #""
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])                  #""
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])        #""
                cross_valid_variable += 1                                                           #""
            GA.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)                #updates training and test datasets in alg
            for individual in range(pop_size):                                                      #begin training, very similar to past algs
                for i in range(len(train_inputs)):
                    output[individual] = GA.feed_GA(i, individual)
                    GA.calc_fit(individual, i, output[individual])
            GA.tournament()                                                                         #tournament to choose chromes to breed
            GA.mutate()                                                                             #mutate children
            GA.winner(j)                                                                            #determin if chrome meets final criteria

            test_final = 0
            for i in range(len(test_inputs)):                                                       #again, processing testing data and printing
                output[i] = GA.feed_testing_GA(i, 0)
                test_final += int(test_outputs[i][0]) - output[i]
            test_final = test_final * test_final
            test_final = math.sqrt(test_final)
            print("Test Sum Error = %f" % test_final)

            print("Generation %d" % j)

#############################FINISH CONDITION#######################################

        best_chrome = GA.sort_all_runs()
        final_chrome = []
        for i in range(hidden_nodes_amount):                                                        #creating final chrome if max genrations reached + printing
            for j in range(len(inputArray[0][0])):
                final_chrome.append(best_chrome[0][i][j])
        for i in range(len(expectedOutputArray[0])):
            for j in range(hidden_nodes_amount):
                final_chrome.append(best_chrome[1][i][j])
        print("best chrome")
        print(final_chrome)
        t1 = time.time()
        runtime = t1-t0
        print("Runtime of %d Generations: %f" % (max_generations, runtime))

#############################BACKPROP ALGORITHM#######################################

elif (neural_net == "BP"):              #Backprop alg, setup variable
    hidden_layer_amount = 0
    hidden_nodes_amount = 20
    output_nodes_amount = 1
    epocs = 1
    activation_type = "s"
    converge_test = "epoc"
    sum_error = 1
    opt_hidden = 0
    opt_epocs = 10000
    opt_runtime = 10000
    opt_test_start = 720
    opt_test_end = 740
    opt_test_interval = 1
    cross_valid_variable = 0

    if (converge_test == "epoc"):
        FF = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)         #build bp network
        FF.initialize()                                                                             #init weights
        for j in range(epocs):                                                                      #start epoc and init run varibales
            sum_error = 0
            sum_error_test = 0
            train_inputs = []
            train_outputs = []
            test_inputs = []
            test_outputs = []
            for i in range(len_in_outs - int((len_in_outs/cross_valid_fold))):                      #this uses an iterator and the modulo function to splice different variable in the total set to test and training data
                train_inputs.append(inputArray[cross_valid_variable%len_in_outs])                   #""
                train_outputs.append(expectedOutputArray[cross_valid_variable%len_in_outs])         #""
                cross_valid_variable += 1                                                           #""
            for i in range(int((len_in_outs/cross_valid_fold))):                                    #""
                test_inputs.append(inputArray[cross_valid_variable % len_in_outs])                  #""
                test_outputs.append(expectedOutputArray[cross_valid_variable % len_in_outs])        #""
                cross_valid_variable += 1                                                           #""
            FF.update_in_out(train_inputs, train_outputs, test_inputs, test_outputs)                #updates training and test datasets in alg
            for i in range(len(train_inputs)):                                                      #train network
                output = FF.feed_forward(i)
                sum_error += (int(train_outputs[i][0]) - output)
                FF.backprop(i)
            for i in range(len(test_inputs)):                                                       #test on cross validation variables
                output_test = FF.feed_forward_test(i)
                sum_error_test += (int(test_outputs[i][0]) - output_test)
            sum_error_test = sum_error_test * sum_error_test
            sum_error_test = math.sqrt(sum_error_test)
            print('Generation=%d, Training Error=%.3f, Testing Error=%.3f' % (j, sum_error, sum_error_test))

    t1 = time.time()
    runtime = t1-t0

