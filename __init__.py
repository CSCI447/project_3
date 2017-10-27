from feed_forward import *
import csv
import codecs

inputArray = []
expectedOutputArray = []
with codecs.open('Data_old_fortest/6_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        #print(type(row[0]))
        inputArray.append(row)

with codecs.open('Data_old_fortest/6_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)

hidden_layer_amount = 0
hidden_nodes_amount = 800
output_nodes_amount = 1
epocs = 1000
activation_type = "s"       #set "s" for sig;

feedforward = FF(inputArray, expectedOutputArray, hidden_nodes_amount, output_nodes_amount, activation_type)

feedforward.initialize()
# feedforward.feed_forward(0)           #testing inits
# feedforward.backprop(0)               #testing inits

for j in range(epocs):
    sum_error = 0
    for i in range(len(inputArray)):
        output = feedforward.feed_forward(i)
        sum_error += (int(expectedOutputArray[i][0]) - output)
        feedforward.backprop(i)
    print('>epoch=%d, training error=%.3f' % (j, sum_error))

