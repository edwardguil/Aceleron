from abc import abstractmethod
import numpy as np
from abc import ABC

class Loss(ABC):

    @abstractmethod
    def loss():
        pass

    def calculateLoss(self, y_true, y_pred):
        return np.mean(self.loss(y_true, y_pred))


class CategoricalCrossentropy(Loss):

    def loss(self, y_true, y_pred):
        """Calculates the loss. Expects y_true to be one-hot encoded."""
        return -np.log(np.sum(np.multiply(y_true, y_pred), axis=1))

class SparseCategoricalCrossentropy(Loss):

    def loss(self, y_true, y_pred):
        """Calculates the loss. Expects y_true to be a 1D array of integers """
        return -np.log(y_pred[range(len(y_true)), y_true])
