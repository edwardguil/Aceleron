#include "matrix.h"
#include "losses.h"

double Loss::calculateLoss(matrix::Matrix<double> y_true, 
	    matrix::Matrix<double> y_pred) {
    return matrix::sum(loss(y_true, y_pred), 1, false)[0][0] / y_true.rows; 
}

matrix::Matrix<double> CategoricalCrossentropy::loss(matrix::Matrix<double> y_true, 
	    matrix::Matrix<double> y_pred) {
    // Expects y_true to be one hot encoded
    return matrix::mul_const(matrix::log(matrix::sum(matrix::mul(y_true, 
			y_pred))), (double) -1.0);
}

matrix::Matrix<double> SparseCategoricalCrossentropy::loss(matrix::Matrix<double> y_true, 
	    matrix::Matrix<double> y_pred) {
    // Expects y_true to be 1D array of integers. 
    // NOT IMPLEMENTED CORRECTLY YET.
    // NEEDS TO BE -np.log(y_pred[range(len(y_true)), y_true]):wq 
    return matrix::mul_const(matrix::log(y_true), (double) -1.0);
}
