import numpy as np
from Losses import *

class Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)#np.ones((n_inputs, n_neurons))#
        self.biases = np.zeros((1, n_neurons))

    def setWeights(self, weights):
        self.weights = weights

    def setBias(self, bias):
        self.bias = bias

    def forward(self, input):
        self.inputs = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, dinputs):
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0, keepdims=True)
        return np.dot(dinputs, self.weights.T)

class ReLU:

    def forward(self, input):
        self.inputs = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, dinputs):
        dinputsCopy = dinputs.copy()
        dinputsCopy[self.inputs <= 0] = 0
        return dinputsCopy

class Softmax:

    def __init__(self, overflow_prevention=True):
        if overflow_prevention:
            self.power = lambda x: x - np.max(x, axis=1, keepdims=True)
        else:
            self.power = lambda x: x

    def forward(self, input):
        self.inputs = input
        exp = np.exp(self.power(input))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output


class SoftmaxCrossEntropy:
    def __init__(self):
        self.softmax = Softmax()
        self.crossEntropy = CategoricalCrossentropy()
        self.loss = 0

    def forward(self, inputs, y_true):
        out = self.softmax.forward(inputs)
        self.loss = self.crossEntropy.calculateLoss(out, y_true)
        return out

    def backward(self, dinputs, y_true):
        samples = len(dinputs)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        dinputCopy = dinputs.copy()
        dinputCopy[range(samples), y_true] -= 1
        return dinputCopy / samples