import numpy as np

class ReLU:

    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output