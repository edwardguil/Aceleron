class Neuron: 

    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

    def setWeights(self, weights):
        self.weights = weights

    def getWeights(self):
        return self.weights

    def setBias(self, bias):
        self.bias = bias

    def getBias(self, bias):
        return self.bias


    def dotProduct(self):
        total = 0
        for e in range(0, len(self.weights)):
            pass
