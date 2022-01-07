import numpy as np

class Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def setWeights(self, weights):
        self.weights = weights

    def setBias(self, bias):
        self.bias = bias

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

class ReLU:

    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output

class Softmax:

    def __init__(self, overflow_prevention=True):
        if overflow_prevention:
            self.power = lambda x: x - np.max(x, axis=1, keepdims=True)
        else:
            self.power = lambda x: x

    def forward(self, input):
        exp = np.exp(self.power(input))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

