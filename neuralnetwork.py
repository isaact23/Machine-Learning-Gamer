# Neural Network Program
# Sourced from Tariq Rashid's 'Make Your Own Neural Network'

import numpy, scipy.special
#import matplotlib

class neuralNetwork():

    # initialization
    def __init__(self, inputnodes=3, hiddennodes1=3, hiddennodes2=3, outputnodes=3, learningrate=0.3):
        
        # number of nodes in each layer
        self.inodes = inputnodes
        self.h1nodes = hiddennodes1
        self.h2nodes = hiddennodes2
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # initialize activation function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function_inverse = lambda x: scipy.special.logit(x)

        # create random link weight matrices
        self.wih1 = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.h1nodes, self.inodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.h1nodes, -0.5), (self.h2nodes, self.h1nodes))
        self.wh2o = numpy.random.normal(0.0, pow(self.h2nodes, -0.5), (self.onodes, self.h2nodes))

        # an alternative expression that uses a flat distribution rather than a normal distribution for randomization
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

    # training the network
    def train(self, inputs_list, targets_list):
        # convert inputs list to correct array type
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # query the first hidden layer
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # query the second hidden layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # query the final layer
        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        # calculate the error
        output_errors = targets - final_outputs
        hidden2_errors = numpy.dot(self.wh2o.T, output_errors)
        hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)

        # backpropagate
        # update the weights for links between hidden2 and output layers
        self.wh2o += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))

        # update the weights for links between hidden1 and hidden2 layers
        self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)), numpy.transpose(hidden1_outputs))

        # update the weights for links between input and hidden1 layers
        self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs))

    # query the network
    def query(self, inputs):

        # calculate the first hidden layer
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # calculate the second hidden layer
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # calculate the final layer
        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # query the network backwards
    def reverseQuery(self, targets_list):
        assert False, "neuralNetwork.reverseQuery() is not ready for use."
        
        final_outputs = numpy.array(targets_list, ndmin=2).T

        final_inputs = self.activation_function_inverse(final_outputs) # Calculate the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)

        hidden_outputs -= numpy.min(hidden_outputs) # Scale to prevent errors with inverse activation function
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.activation_function_inverse(hidden_outputs) # Calculate the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        
        inputs -= numpy.min(inputs) # Scale
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


if __name__ == "__main__":
    
    nn = neuralNetwork(7, 6, 5, 5, learningrate=0.1)

    input_data = [0.9, 0.5, 0.05, 0.02, 0.35, 0.888, 0.2]
    output_data = [0.3141592, 0.2, 0.95, 0.99, 0.36912]

    for i in range(1000):
        nn.train(input_data, output_data)

    print("Expected data: " + str(output_data))
    print("Actual data: " + str(nn.query(input_data)))
