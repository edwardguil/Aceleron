#include "matrix.h"
#include "metrics.h"
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
    double accuracy(matrix::Matrix<double> y_true, 
			matrix::Matrix<double> y_pred) {

		matrix::Matrix<int> prediction = matrix::argmax(y_pred);
		matrix::Matrix<int> tru = matrix::argmax(y_true);
		return (matrix::sum(matrix::equals(tru, prediction), 0, false)[0][0] /  
				(double) y_pred.rows);
    }
}
