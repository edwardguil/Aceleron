#include "matrix.h"

/* Loss::calculateLoss()
* -----
* Inherited function, averages the loss calculated 
* from the loss function. 
*
* @y_true: the true labels
* @y_pred: the predicted labels, i.e. output from the last layer, 
*
* Returns: the calculated loss
*/
template<typename dtype, typename vtype>
double Loss<dtype, vtype>::calculateLoss(matrix::Matrix<dtype, vtype> y_true, 
	    matrix::Matrix<dtype, vtype> y_pred) {
    return matrix::sum(loss(y_true, y_pred), 1, false).get_idx(0) / y_true.rows; 
}

/* CategoricalCrossEntropy::loss()
* -----
* Calculates the loss w.r.t each sample. 
*
* @y_true: the true labels, expects one-hot
* @y_pred: the predicted labels, i.e. output from the last layer.
*
* Returns: the calculated loss for each sample
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> CategoricalCrossentropy<dtype, vtype>::loss(matrix::Matrix<dtype, vtype> y_true, 
	    matrix::Matrix<dtype, vtype> y_pred) {
    // Expects y_true to be one hot encoded
    return matrix::mul_const(matrix::log(matrix::sum(matrix::mul(y_true, y_pred), 1, true)), (double) -1.0);
}

/* SparseCategoricalCrossEntropy::loss()
* -----
* Not currently implemented. Calculates the loss w.r.t each sample. 
*
* @y_true: the true labels, expects 1D array of integers
* @y_pred: the predicted labels, i.e. output from the last layer.
*
* Returns: the calculated loss for each sample
*/
template<typename dtype, typename vtype>
matrix::Matrix<dtype, vtype> SparseCategoricalCrossentropy<dtype, vtype>::loss(matrix::Matrix<dtype, vtype> y_true, 
	    matrix::Matrix<dtype, vtype> y_pred) {
    // NEEDS TO BE -np.log(y_pred[range(len(y_true)), y_true]):wq 
    return matrix::mul_const(matrix::log(y_true), (double) -1.0);
}
