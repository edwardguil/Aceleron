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
	template<typename dtype, typename vtype>
    double accuracy(matrix::Matrix<dtype, vtype> y_true, 
			matrix::Matrix<dtype, vtype> y_pred) {

		matrix::Matrix<int> prediction = matrix::argmax(y_pred);
		matrix::Matrix<int> tru = matrix::argmax(y_true);
		return (matrix::sum(matrix::equals(tru, prediction), 0, false)[0] /  
				(double) y_pred.rows);
    }
}
