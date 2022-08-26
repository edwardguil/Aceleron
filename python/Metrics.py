from abc import abstractmethod
import numpy as np
from abc import ABC

class Metric(ABC):

    @abstractmethod
    def calculate():
        pass

class Accuracy(Metric):

    def calculate(self, y_true, y_pred):
        prediction = np.argmax(y_pred, axis=1)
        # If y_true are one-hot
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(y_true == prediction)