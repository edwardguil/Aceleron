#include "matrix.h"
#include "losses.h"

float Loss::calculateLoss(matrix::Matrix<float> y_true, 
	    matrix::Matrix<float> y_pred) {
    return 1.0; 
}

matrix::Matrix<float> CategoricalCrossentropy::loss(matrix::Matrix<float> y_true, 
	    matrix::Matrix<float> y_pred) {
    // Expects y_true to be one hot encoded
    return matrix::mul_const(matrix::log(matrix::sum(matrix::mul(y_true, 
			y_pred))), (float) -1.0);
}

matrix::Matrix<float> SparseCategoricalCrossentropy::loss(matrix::Matrix<float> y_true, 
	    matrix::Matrix<float> y_pred) {
    // Expects y_true to be 1D array of integers. 
    // NOT IMPLEMENTED CORRECTLY YET.
    // NEEDS TO BE -np.log(y_pred[range(len(y_true)), y_true]):wq 
    return matrix::mul_const(matrix::log(y_true), (float) -1.0);
}
