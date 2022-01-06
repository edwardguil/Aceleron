import numpy as np

class Dense:

    def __init__(self, n_neurons, n_input_features):
        self.weights = np.random.randn(n_input_features, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def setWeights(self, weights):
        self.weights = weights

    def setBias(self, bias):
        self.bias = bias

    def forwardPass(self, input):
        self.output = np.dot(input, self.weights) + self.biases
        return self.output
