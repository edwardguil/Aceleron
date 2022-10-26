#include "matrix.h"
#include <iostream>
namespace metric
{
    /* accuracy()
	* -----
	* Computes the Accuracy metric, that is: 
	* Accuracy = Number of correct predictions/total number of predictions.
	*
	* @y_true: the true labels
	* @y_pred: the predicted labels, i.e. output from the last layer, 
	*
	* Returns: the calculated accuracy
	*/
	template<typename dtype>
    double accuracy(matrix::Matrix<dtype> y_true, 
			matrix::Matrix<dtype> y_pred) {

		matrix::Matrix<int> prediction = matrix::argmax(y_pred);
		matrix::Matrix<int> tru = matrix::argmax(y_true);
		return (matrix::sum(matrix::equals(tru, prediction), 0, false).get_idx(0) /  
				(double) y_pred.rows);
    }

	template<typename dtype>
	double accuracy(matrix::Matrix<dtype, dtype*> y_true, 
			matrix::Matrix<dtype, dtype*> y_pred) {

		matrix::Matrix<int, int*> prediction = matrix::argmax(y_pred);
		matrix::Matrix<int, int*> tru = matrix::argmax(y_true);
		return (matrix::sum(matrix::equals(tru, prediction), 0, false).get_idx(0) /  
				(double) y_pred.rows);
	}
}
