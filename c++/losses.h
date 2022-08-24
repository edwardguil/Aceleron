#ifndef LOSSES_H
#define LOSSES_H
#include "matrix.h"
// Abstract Class
class Loss {
public: 
    // Abstract Method
    virtual matrix::Matrix<float> loss(matrix::Matrix<float> y_true, 
		matrix::Matrix<float> y_pred) = 0;
    
    float calculateLoss(matrix::Matrix<float> y_true, 
		matrix::Matrix<float> y_pred);
};


class CategoricalCrossentropy: public Loss {
public: 
    matrix::Matrix<float> loss(matrix::Matrix<float> y_true, 
		matrix::Matrix<float> y_pred);
};

class SparseCategoricalCrossentropy: public Loss {
public: 
    matrix::Matrix<float> loss(matrix::Matrix<float> y_true, 
		matrix::Matrix<float> y_pred);
};
#endif
