import numpy as np

class Softmax:

    def __init__(self, overflow_prevention=True):
        if overflow_prevention:
            self.power = lambda x: x - np.max(input, axis=1, keepdims=True)
        else:
            self.power = lambda x: x

    def forward(self, input):
        exp = np.exp(self.power(input))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

