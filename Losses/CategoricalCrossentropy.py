import numpy as np

class CategoricalCrossentropy:

    def loss(y_true, y_pred):
        """Calculates the loss. Expects y_true to be one-hot encoded. 
        
        """
        return -1 * np.sum(np.multiply(y_true, np.log(y_pred)))
