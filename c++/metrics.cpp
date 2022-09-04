#include "matrix.h"
#include "metrics.h"
#include <iostream>
namespace metric
{
    
    double accuracy(matrix::Matrix<double> y_true, 
		matrix::Matrix<double> y_pred) {
	matrix::Matrix<int> prediction = matrix::argmax(y_pred);
	matrix::Matrix<int> tru = matrix::argmax(y_true);
	return (matrix::sum(matrix::equals(tru, prediction), 0, false)[0][0] /  
		    (double) y_pred.rows);
    }
}
