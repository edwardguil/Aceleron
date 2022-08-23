#include "matrix.h"
#include "losses.h"

// Abstract Class
float Loss::calculateLoss(Matrix<float> y_true, Matrix<float> y_pred) {
    return 1.0; 
}

Matrix<float> CategoricalCrossentropy::loss(Matrix<float> y_true, Matrix<float> y_pred) {
    // Expects y_true to be one hot encoded
    return matrix_mulconst(matrix_log(matrix_sum(matrix_mul(y_true, 
			y_pred))), (float) -1.0);
}

Matrix<float> SparseCategoricalCrossentropy::loss(Matrix<float> y_true, Matrix<float> y_pred) {
	// Expects y_true to be 1D array of integers. 
	// NOT IMPLEMENTED CORRECTLY YET.
	// NEEDS TO BE -np.log(y_pred[range(len(y_true)), y_true]):wq 
	return matrix_mulconst(matrix_log(y_true), (float) -1.0);
}
