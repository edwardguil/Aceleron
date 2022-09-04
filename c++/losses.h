#ifndef LOSSES_H
#define LOSSES_H
#include "matrix.h"
// Abstract Class
class Loss {
public: 
    // Abstract Method
    virtual matrix::Matrix<double> loss(matrix::Matrix<double> y_true, 
		matrix::Matrix<double> y_pred) = 0;
    
    double calculateLoss(matrix::Matrix<double> y_true, 
		matrix::Matrix<double> y_pred);
};


class CategoricalCrossentropy: public Loss {
public: 
    matrix::Matrix<double> loss(matrix::Matrix<double> y_true, 
		matrix::Matrix<double> y_pred);
};

class SparseCategoricalCrossentropy: public Loss {
public: 
    matrix::Matrix<double> loss(matrix::Matrix<double> y_true, 
		matrix::Matrix<double> y_pred);
};
#endif
